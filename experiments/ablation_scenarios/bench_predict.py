"""Run one detector bundle on a breakdown-benchmark dataset and evaluate it.

    scripts/guarded --name bench-A --mem 16G --vram 0.3 -- python \
        experiments/ablation_scenarios/bench_predict.py BUNDLE --dataset DATASET --out OUT

The simulated-test path of experiments/evaluation/eval_postprocess.py,
unchanged in substance: `predict_dataset` on every test tile at score >= 0.05,
the scene's tracks per tile, `assign_by_frequency` (emitter head -> nearest
live track), `flag_detections` with the default `PostprocessConfig`, then
`evaluate` before (all rows) and after post-processing (kept rows), with the
default matching (centres within 100 ms, overlapping frequency ranges,
one-to-one) and, for the chirp-rate sweep, a strict variant (25 ms: a third
of the fastest train interval; 99 % of true positives on sim_v4 test lie
within 12 ms).

Writes OUT/predictions_flagged.parquet and OUT/{before,after}_tol{100,025}/
(evaluation tables, items.csv the per-label outcomes).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from gridtools.io import load_tracks

from chirpdetector.bundle import load_bundle
from chirpdetector.data import dataset_ground_truth, load_dataset
from chirpdetector.evaluation import (
    EvaluationConfig,
    assign_by_frequency,
    evaluate,
    parse_thresholds,
)
from chirpdetector.evaluation.matching import MatchConfig
from chirpdetector.postprocess import COLUMNS, PostprocessConfig, flag_detections, summarize
from chirpdetector.prediction import predict_dataset

LOW_THRESHOLD = 0.05
REFERENCE = 0.5
# only the operating point: the per-label outcomes (items.csv) are computed at
# the reference threshold alone, and the full 0.05-0.95 grid made every
# evaluation take ~14 min on 12,150 tiles (4 per arm)
THRESHOLDS = parse_thresholds("0.5")
CONFIG = PostprocessConfig()
TOLERANCES_S = {"100": 0.1, "025": 0.025}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reuse", action="store_true", help="Reuse OUT/predictions_raw.parquet.")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Evaluate only these STAGE_tolTAG (e.g. after_tol100), to run them in parallel; "
        "needs --reuse and an existing OUT/predictions_flagged.parquet; writes summary_<only>.json.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.bundle)
    dataset = load_dataset(args.dataset, check_files=False)
    ids = list(dataset.manifest.image_ids("test"))
    raw_path = args.out / "predictions_raw.parquet"
    if args.reuse and raw_path.is_file():
        preds = pd.read_parquet(raw_path)
    else:
        preds = predict_dataset(bundle, dataset, ids, score_threshold=LOW_THRESHOLD, device=args.device)
        preds.to_parquet(raw_path)
    cache, tracks = {}, {}
    for image_id in ids:
        source = dataset.manifest.image(image_id).source
        if source not in cache:
            cache[source] = load_tracks(dataset.tracks_dir(source))
        tracks[image_id] = cache[source]
    flagged_path = args.out / "predictions_flagged.parquet"
    if args.only:
        assert args.reuse and flagged_path.is_file(), "--only needs --reuse and the flagged predictions"
        flagged = pd.read_parquet(flagged_path)
    else:
        preds = assign_by_frequency(preds, tracks, CONFIG.fundamental_tolerance_hz, max_gap_s=CONFIG.max_gap_s)
        flagged = flag_detections(preds, tracks, CONFIG)
        flagged.to_parquet(flagged_path)
    truth = dataset_ground_truth(dataset, ids)
    keep = flagged["drop_reason"].astype(str).to_numpy() == ""
    stages = {
        "before": flagged.drop(columns=list(COLUMNS)).reset_index(drop=True),
        "after": flagged.loc[keep].drop(columns=list(COLUMNS)).reset_index(drop=True),
    }
    summary = {
        "bundle": str(bundle.path),
        "weights_sha256": bundle.weights_sha256,
        "dataset": str(args.dataset),
        "dataset_sha256": dataset.manifest.content_sha256,
        "n_images": len(ids),
        "postprocess": CONFIG.to_dict(),
        "counts": summarize(flagged),
        "counts_at_reference": summarize(flagged[flagged["score"] >= REFERENCE]),
    }
    for stage, rows in stages.items():
        for tag, tol in TOLERANCES_S.items():
            if args.only and f"{stage}_tol{tag}" not in args.only:
                continue
            cfg = EvaluationConfig(
                thresholds=THRESHOLDS,
                reference_threshold=REFERENCE,
                n_bootstrap=0,
                match=MatchConfig(time_tolerance_s=tol),
            )
            result = evaluate(rows, truth, config=cfg, tracks=tracks)
            result.write(args.out / f"{stage}_tol{tag}", plot=False)
            pooled = result.summary[result.summary["level"] == "all"]
            row = pooled.iloc[(pooled["threshold"] - REFERENCE).abs().argmin()]
            summary[f"{stage}_tol{tag}"] = {k: float(row[k]) for k in ("labels", "tp", "fp", "fn", "precision", "recall", "f1")}
            logging.info("%s tol %s: %s", stage, tag, summary[f"{stage}_tol{tag}"])
    name = "summary.json" if not args.only else f"summary_{'_'.join(args.only)}.json"
    (args.out / name).write_text(json.dumps(summary, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
