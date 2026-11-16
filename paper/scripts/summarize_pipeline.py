"""The chirp pipeline's own evaluations in numbers, for the Methods (no figure).

New step. The templates, background library, simulated datasets, expert
benchmark, real-recording labelling session and production run belong to the
pipeline (chirpdetector, gridtools; ``experiments/`` of this repository) and
wrote their results under ``$CHIRP_DATA``. This step reads those result files
and writes the numbers the Methods cite with write_stats ("pipeline"). The
numbers shown in the three v2 figures are written by the ``plot_v2_*`` scripts.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from manuscript_stats import write_stats
from paper_config import CHIRP_DATA

TEMPLATES = CHIRP_DATA / "templates" / "v4"
BACKGROUND = CHIRP_DATA / "background" / "v2" / "manifest.csv"
DATASET = CHIRP_DATA / "datasets" / "sim_v4" / "diagnostics" / "counts.json"
BENCHMARK = CHIRP_DATA / "labels" / "benchmark" / "report_v2" / "metrics.json"
REAL = CHIRP_DATA / "labels" / "real_v1" / "report"
PRODUCTION = CHIRP_DATA / "detections" / "competition2019_v2_20261113"  # the run all analyses derive from


def read_json(path):
    return json.loads(path.read_text())


def templates() -> dict:
    manifest = read_json(TEMPLATES / "manifest.json")
    splits = read_json(TEMPLATES / "splits_v1.json")["counts"]
    return {
        "total": manifest["n_templates"],
        "chirp_chamber": manifest["counts"]["per_source"]["chirp_chamber"],
        "harvested": manifest["counts"]["per_source"]["competition"],
        "default_selection": manifest["counts"]["cumulative"]["after_decay_interrupted"],
        "split_templates": {s: splits[s]["templates"] for s in ("train", "val", "test")},
        "split_groups": {s: splits[s]["groups"] for s in ("train", "val", "test")},
    }


def scenes() -> dict:
    counts = read_json(DATASET)
    with BACKGROUND.open() as f:
        n_segments = sum(1 for _ in csv.DictReader(f))
    return {
        "background_segments": n_segments,
        "per_split": {s: counts[s]["groups"] for s in ("train", "val", "test")},
        "tiles": {s: counts[s]["images"] for s in ("train", "val", "test")},
        "test_chirp_labels": counts["test"]["chirp_boxes"],  # chirps in overlapping tiles count twice
        "test_unique_chirps": counts["test"]["unique_chirps"],
    }


def benchmark() -> dict:
    m = read_json(BENCHMARK)
    retest = m["retest"]
    return {
        "n_items": m["n_items"],
        "n_true_chirps": m["n_true_chirps"],
        "n_negatives": m["n_negatives"],
        "per_set": m["per_set"],
        "median_time_per_item_s": m["time_s"]["median"],
        "retest": {"n_pairs": retest["n_pairs"], "agreement": retest["decision_agreement"],
                   "kappa": retest["cohen_kappa"], "emitter_agreement": retest["emitter_agreement"],
                   "n_both_chirp": retest["n_both_chirp"]},
    }


def real_session() -> dict:
    m = read_json(REAL / "metrics.json")
    candidates = m["retest"]["candidates"]
    with (REAL / "emitter_agreement.csv").open() as f:
        rows = {(r["source"], r["model"]): r for r in csv.DictReader(f) if r["df_bin"] == "all"}
    # all labelled candidates (2019/2020 and 2022) with the production assignment; the
    # candidates of the 2019/2020 trials with the gap-filled assignment the analyses use;
    # the chirps the expert marked in the windows
    rows = {"candidates": rows["candidates", "t001 (= production)"],
            "candidates_gapfilled": rows["candidates", "gap-filled (2019 trials)"],
            "window_marks": rows["window marks (t>=0.5)", "t001 (= production)"]}
    agreement = {key: {"n_assigned": int(r["assigned"]), "agreement": float(r["agreement"])}
                 for key, r in rows.items()}
    return {
        "candidates": m["progress"]["candidates_labelled"],
        "windows": m["progress"]["windows_done"],
        "marks": m["progress"]["marks"],
        "retest": {"n_pairs": candidates["pairs"], "agreement": candidates["agreement_chirp_vs_not"],
                   "kappa": candidates["kappa"]},
        "emitter_agreement": agreement,
    }


def production() -> dict:
    metas = [read_json(p) for p in sorted(PRODUCTION.glob("*/meta.json"))]
    hours = np.array([m["inputs"]["duration_s"] for m in metas]) / 3600
    factor = np.array([m["timings"]["realtime_factor"] for m in metas])
    full = hours > 1  # one recording is 3 min long; its start-up time dominates
    return {
        "n_recordings": len(metas),
        "hours": hours.sum(),
        "wall_hours": sum(m["timings"]["wall_s"] for m in metas) / 3600,
        "realtime_factor_full_recordings": {"median": np.median(factor[full]),
                                            "range": [factor[full].min(), factor[full].max()]},
    }


def main() -> None:
    stats = {"templates": templates(), "scenes": scenes(), "benchmark": benchmark(),
             "real_session": real_session(), "production": production()}
    print(json.dumps(stats, indent=1, default=float))
    write_stats("pipeline", stats)


if __name__ == "__main__":
    main()
