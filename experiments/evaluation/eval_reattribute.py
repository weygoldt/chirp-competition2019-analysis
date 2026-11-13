"""Harmonic re-attribution vs the current filter (flag-only), both detectors.

Usage (worktree on the path, CPU only)::

    scripts/guarded --name pp-reattr --mem 24G --vram 0 -- uv run python \
        experiments/evaluation/eval_reattribute.py RUN

(run then with the v2-reattribute worktree on PYTHONPATH; that code is
merged into chirpdetector main)

Reuses the predictions of eval_postprocess.py (``RUN/eval_postprocess/
*_predictions_flagged.parquet``: predict_span on the real regions and
predict_dataset on the sim_v4 test tiles, score >= 0.05, emitters assigned
with the tracks, 10-s gap). Re-attribution depends on the detections that
are reported, so at every threshold t the detections with score >= t are
post-processed and evaluated at t (as predict_span does in production);
the flag-only filter commutes with the threshold and is evaluated the same
way for a like-for-like comparison. Output: ``RUN/eval_postprocess/
reattribute/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import FAST  # noqa: E402


import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from gridtools.io import load_tracks

sys.path.insert(0, str(Path(__file__).parent))
from eval_postprocess import LABEL_FILES, TRACK_DIRS  # noqa: E402

from chirpdetector.data import dataset_ground_truth, load_dataset  # noqa: E402
from chirpdetector.evaluation import (  # noqa: E402
    EvaluationConfig,
    GroundTruth,
    evaluate,
    read_ground_truth,
)
from chirpdetector.postprocess import (  # noqa: E402
    COLUMNS,
    PostprocessConfig,
    flag_detections,
    summarize,
)

THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)]
VARIANTS = {
    "filter": PostprocessConfig(reattribute_harmonics=False),
    "reattribute": PostprocessConfig(),
}
OLD = ("drop_reason", "harmonic_of_id", "harmonic_pair_of", "duplicate_of")

_STATE: dict = {}


def _load(run: Path, part: str, sim: Path):
    flagged = pd.read_parquet(run / "eval_postprocess" / f"{part}_predictions_flagged.parquet")
    preds = flagged.drop(columns=[c for c in (*OLD, *COLUMNS) if c in flagged.columns])
    if part == "real":
        truth = GroundTruth.concat([read_ground_truth(p) for p in LABEL_FILES])
        tracks = {name: load_tracks(path) for name, path in TRACK_DIRS.items()}
    else:
        ds = load_dataset(sim, check_files=False)
        ids = list(ds.manifest.image_ids("test"))
        truth = dataset_ground_truth(ds, ids)
        cache, tracks = {}, {}
        for i in ids:
            s = ds.manifest.image(i).source
            if s not in cache:
                cache[s] = load_tracks(ds.tracks_dir(s))
            tracks[i] = cache[s]
    return preds, truth, tracks


def _init(run: str, part: str, sim: str) -> None:
    _STATE["data"] = _load(Path(run), part, Path(sim))


def _one(args: tuple[str, float]) -> dict:
    variant, t = args
    preds, truth, tracks = _STATE["data"]
    sel = preds.loc[preds["score"].to_numpy() >= t].reset_index(drop=True)
    flagged = flag_detections(sel, tracks, VARIANTS[variant])
    keep = flagged["drop_reason"].astype(str).to_numpy() == ""
    kept = flagged.loc[keep].reset_index(drop=True)
    config = EvaluationConfig(thresholds=(t,), reference_threshold=t, n_bootstrap=0)
    result = evaluate(kept.drop(columns=list(COLUMNS)), truth, config=config)
    pooled = result.summary[result.summary["level"] == "all"].iloc[0]
    row = {"variant": variant, "threshold": t}
    row.update({k: int(pooled[k]) for k in ("labels", "tp", "fp", "fn")})
    row.update({k: float(pooled[k]) for k in ("precision", "recall", "f1")})
    row.update({f"n_{k}": v for k, v in summarize(flagged).items()})
    again = kept["reattributed"].astype(str).to_numpy() == "harmonic"
    items = result.items
    matched = items[items["pred_row"].notna()].drop_duplicates("pred_row")
    outcome = pd.Series("", index=range(len(kept)), dtype=object)
    outcome.iloc[matched["pred_row"].astype(int).to_numpy()] = matched["outcome"].to_numpy()
    out = outcome.to_numpy()[again]
    row["reattributed_tp"] = int(np.sum(out == "tp"))
    row["reattributed_fp"] = int(np.sum(out == "fp"))
    row["reattributed_ignored"] = int(np.sum(out == "ignored"))
    row["reattributed_uncounted"] = int(np.sum(out == ""))
    if variant == "reattribute" and abs(t - 0.5) < 1e-9:
        cols = ["recording", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score",
                "harmonic_of_id", "harmonic_f_low_hz", "harmonic_f_high_hz"]
        cols += [c for c in ("emitter_hz", "harmonic_emitter_hz") if c in kept.columns]
        row["_rows"] = kept.loc[again, cols].assign(outcome=out).to_dict(orient="records")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--sim", type=Path, default=FAST / "datasets/sim_v4")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.run / "eval_postprocess" / "reattribute"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config": VARIANTS["reattribute"].to_dict()}
    for part in ("real", "sim_test"):
        jobs = [(v, t) for v in VARIANTS for t in THRESHOLDS]
        with ProcessPoolExecutor(
            args.workers, initializer=_init, initargs=(str(args.run), part, str(args.sim))
        ) as pool:
            rows = list(pool.map(_one, jobs))
        listing = [r.pop("_rows") for r in rows if "_rows" in r]
        table = pd.DataFrame(rows)
        table.to_csv(out / f"{part}_by_threshold.csv", index=False)
        if listing:
            pd.DataFrame(listing[0]).to_csv(out / f"{part}_reattributed_at_0.5.csv", index=False)
        part_summary = {}
        for variant in VARIANTS:
            v = table[table["variant"] == variant]
            at = v[np.isclose(v["threshold"], 0.5)].iloc[0].to_dict()
            best = v.loc[v["f1"].idxmax()].to_dict()
            part_summary[variant] = {"at_0.5": at, "best_f1": best}
        summary[part] = part_summary
        print(part)
        print(table[np.isclose(table["threshold"], 0.5)].drop(columns=[c for c in table.columns if c.startswith("n_")]).to_string(index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
