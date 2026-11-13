"""Before/after evaluation of chirpdetector.postprocess on a detector bundle.

Usage (through the guard, with the v2-postprocess worktree on the path)::

    scripts/guarded --name pp-eval --mem 24G --vram 0.3 -- uv run python \
        experiments/evaluation/eval_postprocess.py RUN \
        --sim $CHIRP_FAST/datasets/sim_v4

(run then with the v2-postprocess worktree on PYTHONPATH; that code is
merged into chirpdetector main)

RUN is a run directory (``RUN/final/bundle``); output goes to
``RUN/eval_postprocess`` (existing ``RUN/eval`` is not touched).

(a) Real hand labels, v1 benchmark rules (as evaluate_bundle.py /
    evaluate_dualhead.py): `predict_span` over every labelled region of the
    four recordings at score >= 0.05, now with the recordings' tracks and
    post-processing in flag mode (``keep_dropped=True``). "before" = all
    rows (identical to the old predictions), "after" = rows with
    ``drop_reason == ""``.
(b) sim_v4 test tiles (as evaluate_dualhead.py): `predict_dataset`, the
    scene's tracks per tile, `assign_by_frequency` for an emitter head,
    then `flag_detections`. Also: which dropped detections had matched a
    true chirp, the pair rule (tracks withheld), and an oracle check: the
    true boxes of every visible chirp of the test scenes (chirps.parquet,
    perfect detections) must all survive, also chirps in quick succession
    (``dt_nearest_other_chirp_s``).

For both: TP/FP/FN at 0.5, best F1 and its threshold, labels lost
(TP before, FN after) at 0.5 and at the best-F1 threshold, with time,
recording and reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import REC_2019, REC_2022  # noqa: E402


import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from gridtools.io import load_tracks, open_recording

from chirpdetector.bundle import load_bundle
from chirpdetector.data import dataset_ground_truth, load_dataset
from chirpdetector.evaluation import (
    EvaluationConfig,
    GroundTruth,
    assign_by_frequency,
    evaluate,
    parse_thresholds,
    read_ground_truth,
)
from chirpdetector.inference import predict_span
from chirpdetector.postprocess import (
    COLUMNS,
    PostprocessConfig,
    flag_detections,
    summarize,
)
from chirpdetector.prediction import load_detector, predict_dataset

log = logging.getLogger("eval_postprocess")

LABEL_FILES = (
    str(REC_2019 / "2019-12-09-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2019 / "2020-06-03-10_10/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-05-12-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-06-01-10_00/traces-grid1-editable-labels.csv"),
)
TRACK_DIRS = {
    "2019-12-09-10_00": str(REC_2019 / "2019-12-09-10_00"),
    "2020-06-03-10_10": str(REC_2019 / "2020-06-03-10_10"),
    "2022-05-12-10_00": str(REC_2022 / "interrim/2022-05-12-10_00"),
    "2022-06-01-10_00": str(REC_2022 / "interrim/2022-06-01-10_00"),
}
LOW_THRESHOLD = 0.05
REFERENCE = 0.5
THRESHOLDS = parse_thresholds("0.05:0.95:0.05")
CONFIG = PostprocessConfig()


def _kept(flagged: pd.DataFrame) -> pd.DataFrame:
    keep = flagged["drop_reason"].astype(str).to_numpy() == ""
    return flagged.loc[keep].drop(columns=list(COLUMNS)).reset_index(drop=True)


def _all(flagged: pd.DataFrame) -> pd.DataFrame:
    return flagged.drop(columns=list(COLUMNS)).reset_index(drop=True)


def _eval(preds, truth, tracks, reference: float, provenance=None):
    config = EvaluationConfig(
        thresholds=THRESHOLDS, reference_threshold=reference, n_bootstrap=0
    )
    return evaluate(preds, truth, config=config, tracks=tracks, provenance=provenance)


def _pooled(result) -> pd.DataFrame:
    table = result.summary
    return table[table["level"] == "all"].reset_index(drop=True)


def _row(pooled: pd.DataFrame, threshold: float) -> dict:
    row = pooled.iloc[int(np.argmin(np.abs(pooled["threshold"] - threshold)))]
    keys = ["threshold", "labels", "tp", "fp", "fn", "precision", "recall", "f1"]
    return {k: (float(row[k]) if k in ("threshold", "precision", "recall", "f1") else int(row[k])) for k in keys}


def _best(pooled: pd.DataFrame) -> dict:
    return _row(pooled, float(pooled.loc[pooled["f1"].idxmax(), "threshold"]))


def _label_outcomes(result) -> pd.DataFrame:
    """One row per positive label: outcome (tp/fn) and the matched pred_row."""
    items = result.items
    sel = items[items["label_row"].notna() & items["outcome"].isin(["tp", "fn"])]
    return sel.assign(label_row=sel["label_row"].astype(int)).set_index("label_row")


def _lost(before, after, flagged: pd.DataFrame, truth: GroundTruth) -> pd.DataFrame:
    """Labels found before and missed after, with the dropped prediction's reason."""
    b, a = _label_outcomes(before), _label_outcomes(after)
    common = b.index.intersection(a.index)
    lost = common[(b.loc[common, "outcome"] == "tp") & (a.loc[common, "outcome"] == "fn")]
    gained = common[(b.loc[common, "outcome"] == "fn") & (a.loc[common, "outcome"] == "tp")]
    rows = []
    for k in lost:
        pred = int(b.loc[k, "pred_row"])
        label = truth.labels.iloc[k]
        rows.append(
            {
                "recording": label["recording"],
                "t_center_s": float(label["t_center_s"]),
                "label_f_low_hz": float(label.get("f_low_hz", np.nan)),
                "label_f_high_hz": float(label.get("f_high_hz", np.nan)),
                "pred_row": pred,
                "pred_score": float(flagged.loc[pred, "score"]),
                "reason": str(flagged.loc[pred, "drop_reason"]),
                "harmonic_of_id": flagged.loc[pred, "harmonic_of_id"],
                "duplicate_of": flagged.loc[pred, "duplicate_of"],
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_gained"] = len(gained)
    return out


def _dropped_matches(before, flagged: pd.DataFrame, truth: GroundTruth) -> pd.DataFrame:
    """Every dropped detection with what it matched before (at the reference)."""
    items = before.items
    by_pred = items[items["pred_row"].notna()].assign(pred_row=lambda d: d["pred_row"].astype(int))
    by_pred = by_pred.drop_duplicates("pred_row").set_index("pred_row")
    dropped = flagged.index[flagged["drop_reason"].astype(str) != ""]
    rows = []
    for r in dropped:
        rec = {
            "pred_row": int(r),
            "recording": flagged.loc[r, "recording"],
            "t_center_s": 0.5 * float(flagged.loc[r, "t_start_s"] + flagged.loc[r, "t_end_s"]),
            "f_low_hz": float(flagged.loc[r, "f_low_hz"]),
            "f_high_hz": float(flagged.loc[r, "f_high_hz"]),
            "score": float(flagged.loc[r, "score"]),
            "emitter_hz": float(flagged.loc[r, "emitter_hz"]) if "emitter_hz" in flagged else np.nan,
            "reason": str(flagged.loc[r, "drop_reason"]),
            "harmonic_of_id": flagged.loc[r, "harmonic_of_id"],
            "duplicate_of": flagged.loc[r, "duplicate_of"],
            "outcome": "",
            "label": "",
            "label_reason": "",
        }
        if r in by_pred.index:
            it = by_pred.loc[r]
            rec["outcome"] = str(it["outcome"])
            rec["label"] = "" if pd.isna(it["label"]) else str(it["label"])
            if pd.notna(it["label_row"]) and "reason" in truth.labels.columns:
                reason = truth.labels.iloc[int(it["label_row"])]["reason"]
                rec["label_reason"] = "" if pd.isna(reason) else str(reason)
        rows.append(rec)
    return pd.DataFrame(rows)


_BEFORE: dict[tuple[str, float], object] = {}
"""Evaluations of all rows, per (name, reference threshold)."""


def _compare(name: str, flagged, truth, tracks, out: Path, provenance=None) -> dict:
    """Before/after at the reference, at the best F1, and lost labels."""
    out.mkdir(parents=True, exist_ok=True)
    before = _eval(_all(flagged), truth, tracks, REFERENCE, provenance)
    after = _eval(_kept(flagged), truth, tracks, REFERENCE, provenance)
    before.write(out / f"{name}_before", plot=False)
    after.write(out / f"{name}_after", plot=False)
    pb, pa = _pooled(before), _pooled(after)
    summary = {
        "counts": summarize(flagged),
        "counts_at_reference": summarize(flagged[flagged["score"] >= REFERENCE]),
        "before": {"at_0.5": _row(pb, REFERENCE), "best_f1": _best(pb)},
        "after": {"at_0.5": _row(pa, REFERENCE), "best_f1": _best(pa)},
    }
    lost_tables = []
    for label, t in (("at_0.5", REFERENCE), ("best_f1_after", summary["after"]["best_f1"]["threshold"]), ("lowest", LOW_THRESHOLD)):
        b = before if t == REFERENCE else _eval(_all(flagged), truth, tracks, t)
        _BEFORE[(name, t)] = b
        a = after if t == REFERENCE else _eval(_kept(flagged), truth, tracks, t)
        lost = _lost(b, a, flagged, truth)
        summary[f"lost_{label}"] = {"threshold": t, "n_lost": len(lost), "n_gained": lost.attrs["n_gained"]}
        if len(lost):
            lost_tables.append(lost.assign(threshold=t, at=label))
        if label == "lowest":
            dropped = _dropped_matches(b, flagged, truth)
            dropped.to_csv(out / f"{name}_dropped.csv", index=False)
            summary["dropped_by_outcome"] = (
                dropped.groupby(["reason", "outcome", "label_reason"]).size().rename("n").reset_index().to_dict(orient="records")
                if len(dropped) else []
            )
            summary["dropped_tp"] = int((dropped["outcome"] == "tp").sum()) if len(dropped) else 0
    lost_all = pd.concat(lost_tables, ignore_index=True) if lost_tables else pd.DataFrame()
    lost_all.to_csv(out / f"{name}_lost_tp.csv", index=False)
    pb.assign(stage="before").to_csv(out / f"{name}_before_pooled.csv", index=False)
    pa.assign(stage="after").to_csv(out / f"{name}_after_pooled.csv", index=False)
    return summary


# --------------------------------------------------------------------- real


def real(bundle, out: Path, device: str, reuse: Path | None) -> dict:
    tracks = {name: load_tracks(path) for name, path in TRACK_DIRS.items()}
    parts, frames, provenance = [], [], None
    if reuse is None:
        detector = load_detector(bundle, device=device)
    for path in map(Path, LABEL_FILES):
        truth = read_ground_truth(path)
        name = truth.recordings[0]
        parts.append(truth)
        if reuse is not None:
            continue
        recording = open_recording(path.parent)
        for region in truth.regions.itertuples(index=False):
            spans = predict_span(
                bundle,
                recording,
                float(region.t_start_s),
                float(region.t_end_s),
                name=name,
                score_threshold=LOW_THRESHOLD,
                detector=detector,
                device=device,
                tracks=tracks[name],
                postprocess=CONFIG,
                keep_dropped=True,
            )
            provenance = spans.attrs.get("provenance")
            frames.append(spans)
        log.info("%s: %d regions", name, len(truth.regions))
    truth = GroundTruth.concat(parts)
    if reuse is not None:
        preds = pd.read_parquet(reuse)
        if "emitter_hz" in preds.columns:
            preds = assign_by_frequency(preds, tracks, CONFIG.fundamental_tolerance_hz, max_gap_s=CONFIG.max_gap_s)
        # spans were flagged one at a time; regions do not overlap
        flagged = flag_detections(preds, tracks, CONFIG)
    else:
        flagged = pd.concat(frames, ignore_index=True)
    flagged.to_parquet(out / "real_predictions_flagged.parquet")
    summary = _compare(
        "real",
        flagged,
        truth,
        tracks,
        out,
        provenance={"inputs": [{"role": "labels", "path": p} for p in LABEL_FILES], "tracks": TRACK_DIRS},
    )
    # dropped detections at >= 0.5 inside / outside the labelled bands
    regions = truth.regions
    inside = np.zeros(len(flagged), dtype=bool)
    for r in regions.itertuples(index=False):
        c = 0.5 * (flagged["t_start_s"] + flagged["t_end_s"])
        inside |= (
            (flagged["recording"] == r.recording).to_numpy()
            & (c >= r.t_start_s).to_numpy()
            & (c <= r.t_end_s).to_numpy()
            & (flagged["f_low_hz"] < r.f_high_hz).to_numpy()
            & (flagged["f_high_hz"] > r.f_low_hz).to_numpy()
        )
    high = flagged["score"].to_numpy() >= REFERENCE
    reason = flagged["drop_reason"].astype(str).to_numpy()
    summary["dropped_at_0.5_by_band"] = {
        r: {"in_band": int(np.sum(high & inside & (reason == r))), "outside_band": int(np.sum(high & ~inside & (reason == r)))}
        for r in ("harmonic", "duplicate")
    }
    summary["harmonic_pair_flagged_at_0.5"] = int(np.sum(high & flagged["harmonic_pair_of"].notna().to_numpy()))
    summary["provenance_example"] = provenance
    return summary


# ---------------------------------------------------------------------- sim


def _scene_oracle(dataset, sources: list[str]) -> dict:
    """True boxes of every visible chirp (and its harmonic copy) as detections."""
    rows, tracks = [], {}
    for source in sources:
        scene = Path(dataset.tracks_dir(source))
        chirps = pd.read_parquet(scene / "chirps.parquet")
        tracks[source] = load_tracks(scene)
        vis = chirps[chirps["visible"].astype(bool) & chirps["box_t0_s"].notna()]
        for c in vis.itertuples(index=False):
            rows.append(
                {
                    "recording": source, "kind": "chirp", "t_start_s": c.box_t0_s, "t_end_s": c.box_t1_s,
                    "f_low_hz": c.box_f0_hz, "f_high_hz": c.box_f1_hz, "score": 1.0,
                    "emitter_hz": c.baseline_hz, "true_emitter_id": c.emitter_id,
                    "dt_nearest_other_chirp_s": c.dt_nearest_other_chirp_s, "t_center_s": c.t_center_s,
                }
            )
            if np.isfinite(c.harmonic_box_t0_s):
                rows.append(
                    {
                        "recording": source, "kind": "harmonic", "t_start_s": c.harmonic_box_t0_s,
                        "t_end_s": c.harmonic_box_t1_s, "f_low_hz": c.harmonic_box_f0_hz,
                        "f_high_hz": c.harmonic_box_f1_hz, "score": 1.0, "emitter_hz": 2 * c.baseline_hz,
                        "true_emitter_id": c.emitter_id, "dt_nearest_other_chirp_s": np.nan,
                        "t_center_s": c.t_center_s,
                    }
                )
    table = pd.DataFrame(rows)
    # same-emitter interval to the next/previous visible chirp
    table["dt_same_emitter_s"] = np.nan
    for _, g in table[table["kind"] == "chirp"].groupby(["recording", "true_emitter_id"]):
        t = g["t_center_s"].to_numpy()
        order = np.argsort(t)
        d = np.diff(t[order])
        near = np.full(len(t), np.inf)
        near[order[1:]] = np.minimum(near[order[1:]], d)
        near[order[:-1]] = np.minimum(near[order[:-1]], d)
        table.loc[g.index, "dt_same_emitter_s"] = near
    result = {}
    for mode in ("head", "edge"):
        preds = table.drop(columns=["kind", "true_emitter_id", "dt_nearest_other_chirp_s", "dt_same_emitter_s"])
        if mode == "edge":
            preds = preds.drop(columns=["emitter_hz"])
        else:
            preds = assign_by_frequency(preds, tracks, CONFIG.fundamental_tolerance_hz, max_gap_s=CONFIG.max_gap_s)
        flagged = flag_detections(preds, tracks, CONFIG)
        reason = flagged["drop_reason"].astype(str).to_numpy()
        chirp = table["kind"].to_numpy() == "chirp"
        # the simulator keeps >= 0.2 s between two chirps of one fish
        quick = chirp & (table["dt_same_emitter_s"].to_numpy() < 0.5)
        result[mode] = {
            "n_chirps": int(chirp.sum()),
            "chirps_dropped": int(np.sum(chirp & (reason != ""))),
            "chirps_dropped_by_reason": {r: int(np.sum(chirp & (reason == r))) for r in ("harmonic", "duplicate")},
            "n_same_emitter_within_0.5s": int(quick.sum()),
            "n_same_emitter_within_0.3s": int(np.sum(chirp & (table["dt_same_emitter_s"].to_numpy() < 0.3))),
            "min_same_emitter_interval_s": float(np.nanmin(table["dt_same_emitter_s"].to_numpy()[chirp])),
            "quick_dropped": int(np.sum(quick & (reason != ""))),
            "n_any_other_within_0.1s": int(np.sum(chirp & (table["dt_nearest_other_chirp_s"].to_numpy() < 0.1))),
            "n_harmonic_boxes": int((~chirp).sum()),
            "harmonic_boxes_flagged": int(np.sum(~chirp & (reason == "harmonic"))),
        }
        lost = table.loc[chirp & (reason != ""), ["recording", "t_center_s", "emitter_hz", "dt_same_emitter_s"]]
        result[mode]["dropped_examples"] = lost.head(20).to_dict(orient="records")
    return result


def simulated(bundle, dataset_dir: Path, out: Path, device: str, reuse: Path | None) -> dict:
    dataset = load_dataset(dataset_dir, check_files=False)
    ids = list(dataset.manifest.image_ids("test"))
    log.info("simulated test: %d images", len(ids))
    if reuse is not None:
        preds = pd.read_parquet(reuse)
        preds = preds.drop(columns=[c for c in ("emitter_id", "emitter_df_hz") if c in preds.columns])
    else:
        preds = predict_dataset(bundle, dataset, ids, score_threshold=LOW_THRESHOLD, device=device)
    cache, tracks, sources = {}, {}, []
    for image_id in ids:
        source = dataset.manifest.image(image_id).source
        if source not in cache:
            cache[source] = load_tracks(dataset.tracks_dir(source))
            sources.append(source)
        tracks[image_id] = cache[source]
    if "emitter_hz" in preds.columns:
        preds = assign_by_frequency(preds, tracks, CONFIG.fundamental_tolerance_hz, max_gap_s=CONFIG.max_gap_s)
    flagged = flag_detections(preds, tracks, CONFIG)
    flagged.to_parquet(out / "sim_test_predictions_flagged.parquet")
    truth = dataset_ground_truth(dataset, ids)
    summary = _compare("sim_test", flagged, truth, tracks, out)
    # the pair rule alone (tracks withheld), on the same detections
    pairs = flag_detections(preds, None, PostprocessConfig(duplicate=False))
    flagged_pairs = pairs["harmonic_pair_of"].notna().to_numpy()
    before = _BEFORE[("sim_test", LOW_THRESHOLD)]
    items = before.items  # type: ignore[attr-defined][before.items["pred_row"].notna()].drop_duplicates("pred_row")
    kind = pd.Series("unmatched_fp", index=range(len(flagged)), dtype=object)
    for it in items.itertuples(index=False):
        if it.outcome == "tp":
            kind[int(it.pred_row)] = "true_chirp"
        elif it.outcome == "ignored":
            reason = truth.labels.iloc[int(it.label_row)]["reason"]
            kind[int(it.pred_row)] = f"ignore_{reason}"
    high = flagged["score"].to_numpy() >= REFERENCE
    summary["pair_rule_without_tracks"] = {
        "all": kind[flagged_pairs].value_counts().to_dict(),
        "score_ge_0.5": kind[flagged_pairs & high].value_counts().to_dict(),
        "tracked_rule_score_ge_0.5": kind[(flagged["drop_reason"].astype(str) == "harmonic").to_numpy() & high].value_counts().to_dict(),
        "harmonic_ignore_detections_score_ge_0.5": int(np.sum((kind == "ignore_harmonic").to_numpy() & high)),
    }
    summary["oracle_true_boxes"] = _scene_oracle(dataset, sources)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reuse-sim", type=Path, default=None, help="Existing sim predictions (no GPU).")
    parser.add_argument("--reuse-real", type=Path, default=None, help="Existing real predictions (no GPU).")
    parser.add_argument("--skip-sim", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    bundle = load_bundle(args.run / "final" / "bundle")
    out = args.out or args.run / "eval_postprocess"
    out.mkdir(parents=True, exist_ok=True)
    import chirpdetector

    summary = {
        "bundle": str(bundle.path),
        "weights_sha256": bundle.weights_sha256,
        "chirpdetector": chirpdetector.__file__,
        "postprocess": CONFIG.to_dict(),
        "reused_predictions": {"sim": str(args.reuse_sim), "real": str(args.reuse_real)},
    }
    summary["real"] = real(bundle, out, args.device, args.reuse_real)
    if not args.skip_sim:
        summary["sim_test"] = simulated(bundle, args.sim, out, args.device, args.reuse_sim)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    for part in ("real", "sim_test"):
        if part in summary:
            s = summary[part]
            print(part, "before", s["before"], "\n", part, "after ", s["after"])
            print(part, "counts", s["counts"], "at 0.5", s["counts_at_reference"])
            print(part, {k: v for k, v in s.items() if k.startswith("lost_")})


if __name__ == "__main__":
    main()
