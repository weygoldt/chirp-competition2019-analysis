"""Evaluate a detector bundle: simulated test split and the real hand labels.

Usage (on the shared GPU, through the guard)::

    scripts/guarded --name eval --mem 24G --vram 0.5 -- uv run python \
        evaluate_bundle.py BUNDLE --sim DATASET --out OUT [--attach real]

(a) Simulated test split: `predict_dataset` on the test images, scored with
    `chirpdetector.evaluation` against the dataset's boxes (each image one
    recording, ignore regions as ``ignore`` labels); recall broken down by
    the box extras (family, SNR, overlap, close ΔF).
(b) Real hand labels (the v1 detector benchmark): `predict_span` over every
    labelled region of the four labelled recordings, scored against the
    audian label files with the benchmark's matching rule (centres within
    100 ms, overlapping frequency ranges, one-to-one; predictions outside a
    region's band dropped). Comparable with v1: P 1.000, R 0.764, F1 0.866
    at threshold 0.5 on 335 labels.

The real-data metrics (the arbiter) are attached to the bundle with
`attach_to_bundle`; both evaluations are written to OUT.
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
from gridtools.io import open_recording

from chirpdetector.bundle import load_bundle
from chirpdetector.data import dataset_ground_truth, load_dataset
from chirpdetector.evaluation import (
    EvaluationConfig,
    GroundTruth,
    attach_to_bundle,
    evaluate,
    parse_thresholds,
    read_ground_truth,
)
from chirpdetector.inference import predict_span
from chirpdetector.prediction import load_detector, predict_dataset

log = logging.getLogger("evaluate_bundle")

LABEL_FILES = (
    str(REC_2019 / "2019-12-09-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2019 / "2020-06-03-10_10/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-05-12-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-06-01-10_00/traces-grid1-editable-labels.csv"),
)
LOW_THRESHOLD = 0.05  # predictions are kept down to this score for the sweeps


def _strata(result, truth: GroundTruth) -> pd.DataFrame:
    """Recall at the reference threshold by the box extras of the labels."""
    items = result.items
    rows = items[items["label_row"].notna() & (items["label"] == "chirp")]
    labels = truth.labels.iloc[rows["label_row"].astype(int).to_numpy()].reset_index(
        drop=True
    )
    found = (rows["outcome"].to_numpy() == "tp").astype(float)
    out = []

    def add(name: str, keys: pd.Series) -> None:
        frame = pd.DataFrame({"key": keys.to_numpy(), "found": found})
        for key, group in frame.groupby("key", dropna=False, sort=True):
            out.append(
                {
                    "stratum": name,
                    "value": str(key),
                    "labels": len(group),
                    "recall": float(group["found"].mean()),
                }
            )

    add("all", pd.Series(["all"] * len(rows)))
    for column in ("family", "template_family", "chirp_family", "kind_of_chirp"):
        if column in labels.columns:
            add(column, labels[column].astype(str))
    for column, edges in (
        ("snr_db", [-np.inf, -10, -5, 0, 5, 10, 20, np.inf]),
        ("delta_f_hz", [0, 10, 25, 50, 100, np.inf]),
        ("excursion_hz", [0, 50, 100, 200, 400, np.inf]),
        ("duration_s", [0, 0.02, 0.05, 0.1, 0.3, np.inf]),
    ):
        if column in labels.columns:
            add(column, pd.cut(labels[column].astype(float), edges, right=False).astype(str))
    for column in labels.columns:
        if "overlap" in column and labels[column].dtype == bool:
            add(column, labels[column].astype(str))
    return pd.DataFrame(out)


def simulated(bundle, dataset_dir: Path, out: Path, device: str) -> dict:
    dataset = load_dataset(dataset_dir, check_files=False)
    ids = list(dataset.manifest.image_ids("test"))
    log.info("simulated test: %d images", len(ids))
    predictions = predict_dataset(
        bundle, dataset, ids, score_threshold=LOW_THRESHOLD, device=device
    )
    predictions.to_parquet(out / "sim_test_predictions.parquet")
    truth = dataset_ground_truth(dataset, ids)
    config = EvaluationConfig(
        thresholds=parse_thresholds("0.05:0.95:0.05"), n_bootstrap=1000
    )
    result = evaluate(predictions, truth, config=config)
    result.write(out / "sim_test", plot=True)
    strata = _strata(result, truth)
    strata.to_csv(out / "sim_test" / "recall_by_stratum.csv", index=False)
    print(strata.to_string(index=False))
    return result.metrics()


def real(bundle, out: Path, device: str):
    parts, frames = [], []
    detector = load_detector(bundle, device=device)
    for path in map(Path, LABEL_FILES):
        truth = read_ground_truth(path)
        recording = open_recording(path.parent)
        name = truth.recordings[0]
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
            )
            frames.append(spans)
        parts.append(truth)
        log.info("%s: %d regions", name, len(truth.regions))
    predictions = pd.concat(frames, ignore_index=True)
    path = out / "real_predictions.parquet"
    predictions.to_parquet(path)
    truth = GroundTruth.concat(parts)
    config = EvaluationConfig(thresholds=parse_thresholds("0.05:0.95:0.05"))
    result = evaluate(
        predictions,
        truth,
        config=config,
        provenance={"inputs": [{"role": "labels", "path": p} for p in LABEL_FILES]},
    )
    result.write(out / "real", plot=True)
    return result, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-sim", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    bundle = load_bundle(args.bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"bundle": str(bundle.path), "weights_sha256": bundle.weights_sha256}
    if not args.skip_sim:
        summary["sim_test"] = simulated(bundle, args.sim, args.out, args.device)
    result, predictions_path = real(bundle, args.out, args.device)
    summary["real"] = result.metrics()
    attach_to_bundle(result, bundle.path, predictions_path)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    table = result.summary
    pooled = table[table["level"] == "all"]
    print(pooled[["threshold", "labels", "tp", "fp", "fn", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
