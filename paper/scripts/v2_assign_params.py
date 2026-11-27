"""Simulator truth of the simulated test chirps (Fig. 10 G, H: detection, assignment by size).

For every true positive of ``assign_baseline.csv`` (``v2_fig_extract.py
assign-baseline``: the post-processed evaluation of the simulated test
scenes, true positives at score >= 0.5, one row per tile-level match) this
writes the simulated chirp's true parameters, so that Fig. 10 can show the
emitter assignment by the chirp's true 10 % width and height. For every
visible chirp of the same evaluation (its chirp labels, the 9,401 of Fig. 10
A: true positives and false negatives) it writes the same truth and whether
the detector found the chirp at the operating threshold, so that Fig. 10 can
show detection recall by the same true sizes.

The join
--------
1. ``label_row`` -> label. The evaluation's ground truth is
   ``chirpdetector.data.dataset_ground_truth`` of the test tiles of
   ``datasets/sim_v4`` (``GroundTruth`` sorts it by tile and box midpoint).
   Every row of the evaluation's ``items.csv`` that names a label is checked
   against it (tile, label kind, box midpoint, emitter); any mismatch stops
   the script.
2. label -> chirp: a tile belongs to one scene (the manifest's source), and
   the label's ``chirp_id`` is the scene's chirp.
3. chirp -> truth: ``chirpdetector.params.truth.scene_truth`` of the scene,
   the simulator's noise-free per-EOD-period frequency of the emitter
   (height: largest per-period excursion above the local baseline; FWHM and
   10 % width: widths at 50 % and 10 % of it; ``edge_clipped``: a width is a
   lower bound). This is the definition the chirp-parameter decoder is
   trained and evaluated on: where the decoder's evaluation
   (``params/runs/.../eval/predictions.parquet``, Fig. 10 decoder row) has the
   same chirp, its ``true_*`` columns must agree to 1e-9.

Detection at the operating threshold
------------------------------------
The evaluation's ``items.csv`` matches labels and predictions at its
reference threshold (0.5). A chirp label is detected at the operating
threshold if it is a true positive of the evaluation's matching re-run on
the same post-processed predictions (``sim_test_predictions_flagged.parquet``,
the kept rows, as ``experiments/evaluation/eval_postprocess.py`` evaluated
them) with only the boxes of score >= the threshold: the counts of Fig. 10
A's operating point. (A true positive at 0.5 whose box scores below the
threshold may there be matched to another box; filtering the 0.5 matches by
score misses those, 9 of 9,401 at 0.8.) The re-run at the reference
threshold must reproduce ``items.csv`` (outcome and matched box of every
chirp label), else the script stops.

Output in ``$PAPER_OUT/data/processed/v2_figures/``:

``assign_params.csv``
    One row per true positive of ``assign_baseline.csv``, keyed like it by
    ``recording`` (tile), ``label_row`` and ``pred_row``;
    ``assign_params.json``, the counts of the join.
``detect_params.csv``
    One row per chirp label of the evaluation, keyed by ``recording`` and
    ``label_row``: ``outcome_ref`` (tp/fn at the reference threshold),
    ``detected`` (true positive at the operating threshold), the matched box
    there (``pred_row``, ``score``), scene, chirp and truth;
    ``detect_params.json``, the threshold and the counts.

Environment: the upgrade repository's root venv (chirpdetector, gridtools),
run from ``paper/``::

    /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python \\
        scripts/v2_assign_params.py

``--threshold``: the operating threshold; default the run's
(``PAPER_DETECTIONS``: ``meta.json`` of its recordings, the rule of
``v2_fig_common.OPERATING``; 0.8 for rtdetr_gapfill_t080). Fig. 10 refuses
a ``detect_params.json`` made at another threshold than its run's.

CPU only (no model is run): about nine minutes (ground truth 1.5 min, two
matchings of 3.5 min each, the truth of the 289 test scenes that hold a
chirp label 1 min), about 1.7 GB of memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config

log = logging.getLogger("v2_assign_params")

CHIRP_DATA = paper_config.CHIRP_DATA
SIM_V4 = Path(os.environ.get("CHIRP_FAST", str(CHIRP_DATA))).expanduser() / "datasets" / "sim_v4"
PP_EVAL = CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_r1" / "eval_postprocess"
# the recalibrated bundle (v2_fig_common.DECODER_EVAL); its true_* columns are the same
DECODER_EVAL = CHIRP_DATA / "params" / "runs" / "v4-dualhead-sim-v4-r2-recal1" / "bundle" / "eval"

KEYS = ["recording", "label_row", "pred_row"]  # a true positive of assign_baseline.csv
DETECT_KEYS = ["recording", "label_row"]  # a chirp label (a visible simulated chirp)
TRUTH = ["height_hz", "fwhm_s", "width10_s"]  # scene_truth columns used (s, Hz)
TOL = 1e-9  # agreement with the decoder evaluation's truth


def check_items(items: pd.DataFrame, labels: pd.DataFrame) -> int:
    """Check every item row that names a label against the label table.

    ``items``: the evaluation's ``items.csv`` (``recording``, ``label_row``,
    ``label``, ``label_t_center_s``, ``true_emitter_id``); ``labels``: the
    ground truth's label table in its row order (``recording``, ``label``,
    ``t_center_s``, ``emitter_id``). Raises ValueError on any mismatch;
    returns the number of rows checked.
    """
    it = items[items["label_row"].notna()]
    rows = it["label_row"].astype(int).to_numpy()
    if len(rows) and (rows.min() < 0 or rows.max() >= len(labels)):
        raise ValueError(f"label_row outside the {len(labels)} labels")
    lab = labels.iloc[rows]
    checks = {
        "tile": lab["recording"].astype(str).to_numpy() == it["recording"].astype(str).to_numpy(),
        "label kind": lab["label"].astype(str).to_numpy() == it["label"].astype(str).to_numpy(),
        "box midpoint": np.isclose(lab["t_center_s"].to_numpy(float),
                                   it["label_t_center_s"].to_numpy(float), rtol=0, atol=1e-6),
        "emitter": (pd.array(lab["emitter_id"], dtype="Int64").fillna(-1).to_numpy(int)
                    == pd.array(it["true_emitter_id"], dtype="Int64").fillna(-1).to_numpy(int)),
    }
    for what, ok in checks.items():
        if not ok.all():
            raise ValueError(f"{what} differs for {int((~ok).sum())} of {len(ok)} label rows "
                             "of items.csv: the ground truth is not the evaluation's")
    return len(it)


def tp_chirps(tp: pd.DataFrame, labels: pd.DataFrame, sources: Mapping[str, str]) -> pd.DataFrame:
    """Scene and chirp of each true positive.

    ``tp``: rows of ``assign_baseline.csv`` (``KEYS``, ``true_emitter_id``);
    ``labels``: the label table (``recording``, ``label``, ``emitter_id``,
    ``chirp_id``); ``sources``: tile -> scene. Returns ``KEYS``,
    ``source_id``, ``chirp_id``; raises if a true positive's label is not a
    chirp of its tile with its emitter.
    """
    lab = labels.iloc[tp["label_row"].astype(int).to_numpy()]
    bad = (
        (lab["recording"].astype(str).to_numpy() != tp["recording"].astype(str).to_numpy())
        | (lab["label"].astype(str).to_numpy() != "chirp")
        | (pd.array(lab["emitter_id"], dtype="Int64").fillna(-1).to_numpy(int)
           != pd.array(tp["true_emitter_id"], dtype="Int64").fillna(-1).to_numpy(int))
    )
    if bad.any():
        raise ValueError(f"{int(bad.sum())} true positives do not point to a chirp label "
                         "of their tile and emitter")
    missing = sorted(set(tp["recording"].astype(str)) - set(sources))
    if missing:
        raise ValueError(f"tiles without a scene: {missing[:5]}")
    out = tp[KEYS].copy()
    out["source_id"] = tp["recording"].astype(str).map(sources).to_numpy()
    out["chirp_id"] = lab["chirp_id"].astype(int).to_numpy()
    return out


def join_truth(chirps: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Rows naming a chirp (true positives, chirp labels) with its truth.

    ``chirps``: rows with ``source_id`` and ``chirp_id``; ``truth``: one row
    per (``source_id``, ``chirp_id``) with ``TRUTH`` and ``edge_clipped``.
    Every row must find its chirp (a missing chirp raises; NaN truth, e.g. a
    chirp that never rises above baseline, is kept and counted by the
    caller).
    """
    if truth.duplicated(["source_id", "chirp_id"]).any():
        raise ValueError("truth has duplicate (source_id, chirp_id)")
    out = chirps.merge(truth[["source_id", "chirp_id", *TRUTH, "edge_clipped"]],
                       on=["source_id", "chirp_id"], how="left", validate="many_to_one",
                       indicator=True)
    lost = out["_merge"] != "both"
    if lost.any():
        raise ValueError(f"{int(lost.sum())} chirps (rows) without a truth row")
    return out.drop(columns="_merge")


def visible_chirps(items: pd.DataFrame, labels: pd.DataFrame,
                   sources: Mapping[str, str]) -> pd.DataFrame:
    """Every chirp label of an evaluation with its outcome, scene and chirp.

    ``items``: an item table of the evaluation (``items.csv``, or a re-run at
    another threshold): ``recording``, ``outcome``, ``label_row``,
    ``pred_row``, ``label``, ``score``; ``labels``: the ground truth's label
    table in its row order; ``sources``: tile -> scene. Each chirp label of
    ``labels`` must be one item row, a true positive or a false negative, of
    its own tile. Returns ``DETECT_KEYS``, ``outcome``, ``pred_row`` (Int64),
    ``score``, ``source_id``, ``chirp_id``, in label order.
    """
    it = items[items["label"].astype(str) == "chirp"]
    odd = sorted(set(it["outcome"].astype(str)) - {"tp", "fn"})
    if odd:
        raise ValueError(f"chirp labels with outcome {odd}: not a chirp evaluation")
    rows = it["label_row"].astype(int)
    if rows.duplicated().any():
        raise ValueError(f"{int(rows.duplicated().sum())} chirp labels in more than one item row")
    expected = set(np.flatnonzero(labels["label"].astype(str).to_numpy() == "chirp").tolist())
    if set(rows) != expected:
        raise ValueError(f"items name {len(set(rows) - expected)} rows that are no chirp label "
                         f"and miss {len(expected - set(rows))} chirp labels")
    lab = labels.iloc[rows.to_numpy()]
    if (lab["recording"].astype(str).to_numpy() != it["recording"].astype(str).to_numpy()).any():
        raise ValueError("an item's tile is not its label's tile")
    missing = sorted(set(it["recording"].astype(str)) - set(sources))
    if missing:
        raise ValueError(f"tiles without a scene: {missing[:5]}")
    out = pd.DataFrame({
        "recording": it["recording"].astype(str).to_numpy(),
        "label_row": rows.to_numpy(),
        "outcome": it["outcome"].astype(str).to_numpy(),
        "pred_row": pd.array(it["pred_row"], dtype="Int64"),
        "score": it["score"].to_numpy(float),
        "source_id": it["recording"].astype(str).map(sources).to_numpy(),
        "chirp_id": lab["chirp_id"].astype(int).to_numpy(),
    })
    return out.sort_values("label_row").reset_index(drop=True)


def check_rematch(reference: pd.DataFrame, rerun: pd.DataFrame) -> int:
    """The re-run at the evaluation's reference threshold must reproduce it.

    Both: ``visible_chirps`` tables. Every chirp label must have the same
    outcome and the same matched box; raises ValueError otherwise. Returns
    the number of labels compared.
    """
    m = reference.merge(rerun, on=DETECT_KEYS, how="outer", validate="one_to_one",
                        suffixes=("", "_rerun"), indicator=True)
    if (m["_merge"] != "both").any():
        raise ValueError(f"{int((m['_merge'] != 'both').sum())} chirp labels in only one "
                         "of the evaluation and its re-run")
    for col in ("outcome", "pred_row"):
        if col == "pred_row":  # NA (missed) as -1
            a, b = (pd.to_numeric(m[c]).fillna(-1).astype(int).to_numpy()
                    for c in (col, f"{col}_rerun"))
        else:
            a, b = (m[c].astype(str).to_numpy() for c in (col, f"{col}_rerun"))
        if (a != b).any():
            raise ValueError(f"the re-run changes the {col} of {int((a != b).sum())} chirp labels: "
                             "not the evaluation's predictions or matching rule")
    return int(len(m))


def detection_table(reference: pd.DataFrame, at_threshold: pd.DataFrame,
                    threshold: float) -> pd.DataFrame:
    """Per chirp label: its outcome at the reference threshold and whether it
    is detected at ``threshold`` (a true positive of the matching there).

    Both: ``visible_chirps`` tables of the same labels. Returns
    ``DETECT_KEYS``, ``source_id``, ``chirp_id``, ``outcome_ref``,
    ``detected`` and the box matched at ``threshold`` (``pred_row``,
    ``score``; NA/NaN if missed). Raises if a box matched at ``threshold``
    scores below it, or if a label missed at the reference is found at the
    (higher) threshold.
    """
    ref = reference[[*DETECT_KEYS, "source_id", "chirp_id", "outcome"]]
    m = ref.rename(columns={"outcome": "outcome_ref"}).merge(
        at_threshold[[*DETECT_KEYS, "outcome", "pred_row", "score"]], on=DETECT_KEYS,
        how="outer", validate="one_to_one", indicator=True)
    if (m["_merge"] != "both").any():
        raise ValueError(f"{int((m['_merge'] != 'both').sum())} chirp labels in only one table")
    m["detected"] = m["outcome"] == "tp"
    low = m["detected"] & ~(m["score"] >= threshold)
    if low.any():
        raise ValueError(f"{int(low.sum())} boxes matched at {threshold:g} score below it")
    gained = m["detected"] & (m["outcome_ref"] != "tp")
    if gained.any():
        raise ValueError(f"{int(gained.sum())} chirps missed at the reference but found at "
                         f"{threshold:g}: thresholds swapped?")
    return m.drop(columns=["outcome", "_merge"])[
        [*DETECT_KEYS, "source_id", "chirp_id", "outcome_ref", "detected", "pred_row", "score"]]


def check_decoder(truth: pd.DataFrame, decoder: pd.DataFrame) -> dict:
    """Compare with the decoder evaluation's truth (``true_<param>`` columns).

    Returns the number of shared chirps and the largest absolute difference
    per parameter; raises if any exceeds ``TOL``.
    """
    m = truth.merge(decoder[["source_id", "chirp_id", *(f"true_{p}" for p in TRUTH)]],
                    on=["source_id", "chirp_id"], how="inner", validate="one_to_one")
    out: dict = {"n_shared": int(len(m))}
    for p in TRUTH:
        a, b = m[p].to_numpy(float), m[f"true_{p}"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if (np.isfinite(a) != np.isfinite(b)).any():
            raise ValueError(f"{p}: finite here but NaN in the decoder evaluation (or vice versa)")
        diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        if diff > TOL:
            raise ValueError(f"{p}: differs from the decoder evaluation's truth by {diff:g}")
        out[f"max_abs_diff_{p}"] = diff
    return out


# ---------------------------------------------------------------------------
# root environment (chirpdetector, gridtools)
# ---------------------------------------------------------------------------
def ground_truth():
    """The evaluation's ground truth of the test tiles (``GroundTruth``), tile -> scene,
    and scene -> scene directory."""
    from chirpdetector.data import load_dataset
    from chirpdetector.data.contract import dataset_ground_truth

    dataset = load_dataset(SIM_V4, check_files=False)
    ids = list(dataset.manifest.image_ids("test"))
    gt = dataset_ground_truth(dataset, ids)
    sources = {i: dataset.manifest.image(i).source for i in ids}
    scene_dirs = {s: Path(dataset.manifest.sources[s].extra["scene_dir"])
                  for s in set(sources.values())}
    return gt, sources, scene_dirs


def rerun_items(gt, threshold: float, metrics: Mapping) -> tuple[pd.DataFrame, dict]:
    """The evaluation's item table with only the boxes of score >= ``threshold``.

    The kept rows of the saved post-processed predictions (as
    ``eval_postprocess.py``'s ``_kept``), the ground truth ``gt`` and the
    evaluation's matching rule (``metrics``: its ``metrics.json``); no
    tracks (they only give ΔF). Returns the items and the pooled counts at
    ``threshold``.
    """
    from chirpdetector.evaluation import EvaluationConfig, MatchConfig, evaluate
    from chirpdetector.postprocess import COLUMNS

    flagged = pd.read_parquet(PP_EVAL / "sim_test_predictions_flagged.parquet")
    keep = flagged["drop_reason"].astype(str).to_numpy() == ""
    kept = flagged.loc[keep].drop(columns=[c for c in COLUMNS if c in flagged.columns])
    config = EvaluationConfig(match=MatchConfig(**metrics["config"]["match"]),
                              thresholds=(threshold,), reference_threshold=threshold,
                              n_bootstrap=0)
    result = evaluate(kept.reset_index(drop=True), gt, config=config)
    pooled = result.summary[(result.summary["level"] == "all")
                            & np.isclose(result.summary["threshold"], threshold)]
    counts = {k: int(pooled.iloc[0][k]) for k in ("labels", "tp", "fn", "fp")}
    return result.items, counts


def scene_truths(scene_dirs: Mapping[str, Path]) -> pd.DataFrame:
    """``scene_truth`` of every chirp of the scenes (one row per chirp)."""
    from chirpdetector.params.truth import TRUTH_PAD_S, scene_truth

    frames = []
    for i, (source, directory) in enumerate(sorted(scene_dirs.items())):
        t = scene_truth(directory, pad_s=TRUTH_PAD_S)
        t.insert(0, "source_id", source)
        frames.append(t)
        if (i + 1) % 50 == 0:
            log.info("truth of %d / %d scenes", i + 1, len(scene_dirs))
    return pd.concat(frames, ignore_index=True)


def truth_counts(out: pd.DataFrame) -> dict:
    """Rows with finite truth, NaN truth and an edge-clipped width."""
    finite = np.isfinite(out[TRUTH].to_numpy(float)).all(axis=1)
    return {"n_with_truth": int(finite.sum()), "n_truth_nan": int((~finite).sum()),
            "n_edge_clipped": int(out["edge_clipped"].astype(bool).sum())}


def operating_threshold() -> float:
    """The operating threshold of the run (``PAPER_DETECTIONS``): ``meta.json``
    of its recordings, one value (the rule of ``v2_fig_common.OPERATING``)."""
    from v2_fig_extract import RUN

    values = {json.loads(m.read_text())["operating_threshold"]
              for m in sorted(RUN.glob("*/meta.json"))}
    if len(values) != 1:
        raise ValueError(f"{RUN}: operating thresholds {sorted(values)}")
    return float(values.pop())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--threshold", type=float, default=None,
                        help="operating threshold (default: the run's, from its meta.json)")
    args = parser.parse_args()
    if args.threshold is None:
        args.threshold = operating_threshold()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    log.info("operating threshold %g (%s)", args.threshold, paper_config.DETECTIONS)
    out_dir = paper_config.processed_dir("v2_figures")
    tp = pd.read_csv(out_dir / "assign_baseline.csv")
    items = pd.read_csv(PP_EVAL / "sim_test_after" / "items.csv", low_memory=False)
    metrics = json.loads((PP_EVAL / "sim_test_after" / "metrics.json").read_text())
    reference = float(metrics["config"]["reference_threshold"])
    gt, sources, scene_dirs = ground_truth()
    labels = gt.labels
    n_checked = check_items(items, labels)
    log.info("%d label rows of items.csv agree with the ground truth (%d labels)",
             n_checked, len(labels))
    chirps = tp_chirps(tp, labels, sources)

    # every chirp label: detected at the operating threshold?
    visible = visible_chirps(items, labels, sources)
    rerun_ref, counts_ref = rerun_items(gt, reference, metrics)
    n_same = check_rematch(visible, visible_chirps(rerun_ref, labels, sources))
    log.info("the re-run at %g reproduces items.csv for all %d chirp labels", reference, n_same)
    rerun_op, counts_op = rerun_items(gt, args.threshold, metrics)
    detect = detection_table(visible, visible_chirps(rerun_op, labels, sources), args.threshold)
    log.info("at %g: %d of %d chirp labels detected", args.threshold,
             int(detect["detected"].sum()), len(detect))

    tp_scenes = set(chirps["source_id"])
    scenes = sorted(tp_scenes | set(detect["source_id"]))
    truth = scene_truths({s: scene_dirs[s] for s in scenes})
    decoder_all = check_decoder(truth, pd.read_parquet(DECODER_EVAL / "predictions.parquet"))

    # assignment: the true positives of assign_baseline.csv
    tp_truth = truth[truth["source_id"].isin(tp_scenes)]
    out = join_truth(chirps, tp_truth)
    decoder = check_decoder(tp_truth, pd.read_parquet(DECODER_EVAL / "predictions.parquet"))
    log.info("truth agrees with the decoder evaluation on %d shared chirps", decoder["n_shared"])
    c = truth_counts(out)
    meta = {
        "n_tp": int(len(tp)),
        "n_with_truth": c["n_with_truth"],
        "n_truth_nan": c["n_truth_nan"],
        "n_edge_clipped": c["n_edge_clipped"],
        "n_chirps": int(chirps[["source_id", "chirp_id"]].drop_duplicates().shape[0]),
        "n_scenes": len(tp_scenes),
        "n_item_rows_checked": n_checked,
        "decoder_check": decoder,
    }
    out.to_csv(out_dir / "assign_params.csv", index=False)
    (out_dir / "assign_params.json").write_text(json.dumps(meta, indent=1) + "\n")
    log.info("wrote %s: %s", out_dir / "assign_params.csv", meta)

    # detection: every chirp label
    det = join_truth(detect, truth)
    filtered = (visible["outcome"] == "tp") & (visible["score"] >= args.threshold)
    dmeta = {
        "threshold": args.threshold,
        "reference_threshold": reference,
        "n_labels": int(len(det)),
        "n_detected": int(det["detected"].sum()),
        # the same from the reference matches filtered by score (Fig. 10 E's convention)
        "n_tp_reference_score_ge_threshold": int(filtered.sum()),
        "n_tp_reference": int((det["outcome_ref"] == "tp").sum()),
        "counts_reference": counts_ref,
        "counts_threshold": counts_op,
        "n_labels_rematch_checked": n_same,
        **truth_counts(det),
        "n_chirps": int(det[["source_id", "chirp_id"]].drop_duplicates().shape[0]),
        "n_scenes": len(set(det["source_id"])),
        "decoder_check": decoder_all,
    }
    det.to_csv(out_dir / "detect_params.csv", index=False)
    (out_dir / "detect_params.json").write_text(json.dumps(dmeta, indent=1) + "\n")
    log.info("wrote %s: %s", out_dir / "detect_params.csv", dmeta)


if __name__ == "__main__":
    main()
