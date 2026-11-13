"""Evaluate a dual-head bundle: detection and emitter assignment.

Usage (through the guard)::

    scripts/guarded --name eval-dh --mem 24G --vram 0.3 -- uv run python \
        evaluate_dualhead.py BUNDLE --sim DATASET --out OUT [--skip-real]

(a) Simulated test split: `predict_dataset` on the test images; each image's
    predicted ``emitter_hz`` is assigned to the nearest live track of its
    scene (`assign_by_frequency`, the scene's tracks on the scene clock);
    `evaluate` with those tracks gives detection P/R/F1/AP, assignment
    accuracy by ΔF bin and end-to-end metrics. Also the emitter error (Hz)
    of the true positives by ΔF bin, and the same with the box-free
    baseline "emitter = nearest live track to the box's lower edge".
(b) Real hand labels, v1 benchmark rules (as ``evaluate_bundle.py``):
    `predict_span` over every labelled region of the four recordings,
    scored with the benchmark's matching rule. Comparable with v1: P 1.000,
    R 0.764, F1 0.866 at threshold 0.5 on 335 labels. Attached to the bundle.
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
    attach_to_bundle,
    evaluate,
    parse_thresholds,
    read_ground_truth,
)
from chirpdetector.inference import predict_span
from chirpdetector.prediction import load_detector, predict_dataset

log = logging.getLogger("evaluate_dualhead")

LABEL_FILES = (
    str(REC_2019 / "2019-12-09-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2019 / "2020-06-03-10_10/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-05-12-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-06-01-10_00/traces-grid1-editable-labels.csv"),
)
LOW_THRESHOLD = 0.05
TOLERANCE_HZ = 50.0
DF_EDGES = (0.0, 10.0, 25.0, 50.0, 100.0)


def _bin(delta: np.ndarray, n_live: np.ndarray) -> np.ndarray:
    names = ["0-10", "10-25", "25-50", "50-100", "100+"]
    idx = np.clip(np.digitize(np.nan_to_num(delta), DF_EDGES) - 1, 0, 4)
    return np.where(n_live >= 2, np.asarray(names, dtype=object)[idx], "single")


def simulated(bundle, dataset_dir: Path, out: Path, device: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_dir, check_files=False)
    ids = list(dataset.manifest.image_ids("test"))
    log.info("simulated test: %d images", len(ids))
    predictions = predict_dataset(
        bundle, dataset, ids, score_threshold=LOW_THRESHOLD, device=device
    )
    cache: dict[str, object] = {}
    tracks = {}
    for image_id in ids:
        source = dataset.manifest.image(image_id).source
        if source not in cache:
            cache[source] = load_tracks(dataset.tracks_dir(source))
        tracks[image_id] = cache[source]
    # sanity: the label's emitter_id is the scene track at emitter_hz
    truth = dataset_ground_truth(dataset, ids)
    chirps = truth.labels[truth.labels["label"] == "chirp"].head(500)
    gaps = [
        abs(float(tracks[r].track(int(e)).frequency_at(np.array([t]), 1.0)[0]) - hz)
        for r, e, t, hz in zip(
            chirps["recording"],
            chirps["emitter_id"],
            chirps["t_center_s"],
            chirps["emitter_hz"],
            strict=True,
        )
    ]
    log.info("emitter_id vs track frequency: median gap %.3g Hz", np.nanmedian(gaps))
    predictions = assign_by_frequency(predictions, tracks, TOLERANCE_HZ)
    predictions.to_parquet(out / "predictions.parquet")
    config = EvaluationConfig(
        thresholds=parse_thresholds("0.05:0.95:0.05"), n_bootstrap=1000
    )
    result = evaluate(predictions, truth, config=config, tracks=tracks)
    result.write(out, plot=True)
    metrics = result.metrics()

    # emitter error and a box-only baseline, by ΔF bin, on the true positives
    items = result.items
    tp = items[(items["outcome"] == "tp") & (items["label"] == "chirp")].copy()
    rows = tp["pred_row"].astype(int).to_numpy()
    labels = truth.labels.iloc[tp["label_row"].astype(int).to_numpy()]
    tp["true_hz"] = labels["emitter_hz"].to_numpy(float)
    tp["pred_hz"] = predictions["emitter_hz"].to_numpy(float)[rows]
    tp["error_hz"] = np.abs(tp["pred_hz"] - tp["true_hz"])
    # baselines without the emitter head: the nearest live track to the
    # predicted box's lower edge, raw and shifted by the median offset
    # (emitter - f_low) of the training chirps
    train_chirps = dataset.boxes[
        (dataset.boxes["kind"] == "chirp")
        & dataset.boxes["image_id"].isin(set(dataset.manifest.image_ids("train")))
    ]
    offset = float((train_chirps["emitter_hz"] - train_chirps["f_low_hz"]).median())
    log.info("median emitter - f_low offset of the training chirps: %.1f Hz", offset)
    true_ids = tp["true_emitter_id"].astype("Int64").fillna(-2).to_numpy(int)
    for name, shift in (("box_edge", 0.0), ("box_edge_offset", offset)):
        baseline = predictions.iloc[rows].copy().reset_index(drop=True)
        baseline["emitter_hz"] = baseline["f_low_hz"].to_numpy(float) + shift
        baseline = assign_by_frequency(baseline, tracks, 1000.0)
        found = baseline["emitter_id"].astype("Int64").fillna(-1).to_numpy(int)
        tp[f"{name}_correct"] = found == true_ids
    metrics["baseline_offset_hz"] = offset
    tp["correct"] = (
        tp["pred_emitter_id"].astype("Int64").fillna(-1).to_numpy()
        == tp["true_emitter_id"].astype("Int64").fillna(-2).to_numpy()
    )
    tp["bin"] = _bin(
        tp["delta_f_hz"].to_numpy(float), tp["n_live_tracks"].fillna(0).to_numpy(int)
    )
    table = (
        tp.groupby("bin")
        .agg(
            n=("error_hz", "size"),
            emitter_mae_hz=("error_hz", "mean"),
            emitter_median_hz=("error_hz", "median"),
            emitter_p90_hz=("error_hz", lambda v: float(np.quantile(v, 0.9))),
            accuracy=("correct", "mean"),
            box_edge_accuracy=("box_edge_correct", "mean"),
            box_edge_offset_accuracy=("box_edge_offset_correct", "mean"),
        )
        .reset_index()
    )
    pooled = {
        "bin": "all",
        "n": len(tp),
        "emitter_mae_hz": tp["error_hz"].mean(),
        "emitter_median_hz": tp["error_hz"].median(),
        "emitter_p90_hz": float(np.quantile(tp["error_hz"], 0.9))
        if len(tp)
        else np.nan,
        "accuracy": tp["correct"].mean(),
        "box_edge_accuracy": tp["box_edge_correct"].mean(),
        "box_edge_offset_accuracy": tp["box_edge_offset_correct"].mean(),
    }
    table = pd.concat([pd.DataFrame([pooled]), table], ignore_index=True)
    table.to_csv(out / "emitter_by_df_bin.csv", index=False)
    print(table.to_string(index=False))
    metrics["emitter_by_df_bin"] = table.to_dict(orient="records")
    metrics["strata"] = strata(result, truth, tp, out)
    return metrics


def strata(result, truth, tp: pd.DataFrame, out: Path) -> dict:
    """Detection, emitter and assignment by scene kind and by SNR quartile.

    Precision, F1 and AP exist per scene kind (false positives belong to an
    image, hence to a scene); per SNR quartile only recall, emitter error
    and assignment (false positives have no SNR).
    """
    detection = result.metrics()["detection"]
    labels = truth.labels
    items = result.items
    chirp = items[(items["label"] == "chirp") & items["outcome"].isin(["tp", "fn"])]
    chirp_labels = labels.iloc[chirp["label_row"].astype(int).to_numpy()]
    found = (chirp["outcome"].to_numpy() == "tp").astype(float)
    tp_labels = labels.iloc[tp["label_row"].astype(int).to_numpy()]
    rows = []
    for kind in sorted(set(labels["dataset"].astype(str))):
        d = detection.get("datasets", {}).get(kind, {})
        sel = tp_labels["dataset"].astype(str).to_numpy() == kind
        rows.append(
            {
                "stratum": "scene_kind",
                "value": kind,
                "labels": d.get("labels"),
                "precision": d.get("precision"),
                "recall": d.get("recall"),
                "f1": d.get("f1"),
                "ap": d.get("average_precision"),
                "n_tp": int(sel.sum()),
                "emitter_mae_hz": float(tp["error_hz"].to_numpy()[sel].mean()),
                "emitter_median_hz": float(np.median(tp["error_hz"].to_numpy()[sel])),
                "assign_accuracy": float(tp["correct"].to_numpy()[sel].mean()),
            }
        )
    if "snr_db" in labels.columns:
        snr_all = chirp_labels["snr_db"].to_numpy(float)
        edges = np.nanquantile(snr_all, [0, 0.25, 0.5, 0.75, 1.0])
        names = [f"Q{k + 1} [{edges[k]:.1f}, {edges[k + 1]:.1f}] dB" for k in range(4)]
        q_lab = np.clip(np.searchsorted(edges, snr_all, side="right") - 1, 0, 3)
        q_tp = np.clip(
            np.searchsorted(edges, tp_labels["snr_db"].to_numpy(float), side="right")
            - 1,
            0,
            3,
        )
        for k in range(4):
            sel = q_tp == k
            rows.append(
                {
                    "stratum": "snr_quartile",
                    "value": names[k],
                    "labels": int((q_lab == k).sum()),
                    "precision": None,
                    "recall": float(found[q_lab == k].mean()),
                    "f1": None,
                    "ap": None,
                    "n_tp": int(sel.sum()),
                    "emitter_mae_hz": float(tp["error_hz"].to_numpy()[sel].mean()),
                    "emitter_median_hz": float(
                        np.median(tp["error_hz"].to_numpy()[sel])
                    ),
                    "assign_accuracy": float(tp["correct"].to_numpy()[sel].mean()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "strata.csv", index=False)
    print(table.round(3).to_string(index=False))
    return {"rows": table.to_dict(orient="records")}


def real(bundle, out: Path, device: str):
    frames, parts = [], []
    detector = load_detector(bundle, device=device)
    for path in map(Path, LABEL_FILES):
        truth = read_ground_truth(path)
        recording = open_recording(path.parent)
        name = truth.recordings[0]
        for region in truth.regions.itertuples(index=False):
            frames.append(
                predict_span(
                    bundle,
                    recording,
                    float(region.t_start_s),
                    float(region.t_end_s),
                    name=name,
                    score_threshold=LOW_THRESHOLD,
                    detector=detector,
                    device=device,
                )
            )
        parts.append(truth)
        log.info("%s: %d regions", name, len(truth.regions))
    predictions = pd.concat(frames, ignore_index=True)
    path = out / "real_predictions.parquet"
    predictions.to_parquet(path)
    truth = GroundTruth.concat(parts)
    result = evaluate(
        predictions,
        truth,
        config=EvaluationConfig(thresholds=parse_thresholds("0.05:0.95:0.05")),
        provenance={"inputs": [{"role": "labels", "path": p} for p in LABEL_FILES]},
    )
    result.write(out / "real", plot=True)
    return result, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--sim", type=Path, action="append", default=[], help="Simulated dataset(s)."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    bundle = load_bundle(args.bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"bundle": str(bundle.path), "weights_sha256": bundle.weights_sha256}
    for dataset_dir in args.sim:
        name = f"{dataset_dir.name}_test"
        summary[name] = simulated(bundle, dataset_dir, args.out / name, args.device)
    if not args.skip_real:
        result, predictions_path = real(bundle, args.out, args.device)
        summary["real"] = result.metrics()
        attach_to_bundle(result, bundle.path, predictions_path)
        table = result.summary
        pooled = table[table["level"] == "all"]
        print(
            pooled[
                ["threshold", "labels", "tp", "fp", "fn", "precision", "recall", "f1"]
            ].to_string(index=False)
        )
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=1, default=float) + "\n"
    )


if __name__ == "__main__":
    main()
