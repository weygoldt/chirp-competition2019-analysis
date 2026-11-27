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
from paper_config import CHIRP_DATA, pipeline_path

TEMPLATES = CHIRP_DATA / "templates" / "v4"
BACKGROUND = CHIRP_DATA / "background" / "v2" / "manifest.csv"
# the selected pipeline's (paper_config.PIPELINES, by PAPER_DETECTIONS): its
# detector's dataset, blind benchmark, real-recording session, production run
# (the run all analyses derive from), chirp-parameter decoder (recalibrated
# without the octave-error template of its validation split) and training scenes
DATASET = pipeline_path("sim_dataset") / "diagnostics" / "counts.json"
BENCHMARK = pipeline_path("benchmark") / "metrics.json"
REAL = pipeline_path("real_session")
PRODUCTION = pipeline_path("source_run")
DECODER = pipeline_path("decoder")
TRAIN_SCENES = pipeline_path("train_scenes")
# the template-library flags the simulator excludes (scenes' chirps.exclude)
EXCLUDE = ("has_nan", "truncated", "step_flag", "too_short", "brief_cycles", "ripple_flag",
           "multi_peak", "line_in_span", "baseline_drift", "low_snr", "decay_interrupted")
OCTAVE_RATIO = 0.9  # chirp plateau / baseline EODf above this: the trace reads 2x the EODf


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


def octave_templates() -> dict:
    """Big templates whose chirp plateau sits at the baseline EODf above it (octave error).

    Plateau: median frequency excursion between the 50 % crossings, the outer
    fifth on each side left out; ratio to the template's baseline EODf. Over
    the templates the simulator draws from (default selection).
    """
    import pandas as pd

    t = pd.read_parquet(TEMPLATES / "templates.parquet")
    keep = t["valid"] & (t["duplicate_of"] < 0)
    for flag in EXCLUDE:
        keep &= ~t[flag].astype(bool)
    n_default = read_json(TEMPLATES / "manifest.json")["counts"]["cumulative"]["after_decay_interrupted"]
    if int(keep.sum()) != n_default:
        raise SystemExit(f"default selection {int(keep.sum())} != manifest {n_default}")
    big = t[keep & (t["family"] == "big")]
    delta_f = np.load(TEMPLATES / "delta_f_hz.npy", mmap_mode="r")
    centre, rate = delta_f.shape[1] // 2, 20000.0
    ratio = {}
    for k, row in big.iterrows():
        a = int(centre + row["onset_50_s"] * rate)
        b = int(centre + row["offset_50_s"] * rate)
        if b - a < 40 or not np.isfinite(row["baseline_hz"]):
            continue
        m = (b - a) // 5
        ratio[int(row["template_id"])] = float(np.median(delta_f[k, a + m:b - m])) / row["baseline_hz"]
    values = np.array(list(ratio.values()))
    octave = sorted(i for i, r in ratio.items() if r > OCTAVE_RATIO)
    splits = read_json(TEMPLATES / "splits_v1.json")["template_ids"]
    in_split = {s: [i for i in octave if i in set(splits[s])] for s in ("train", "val", "test")}
    train = pd.concat([pd.read_parquet(f, columns=["template_id"])
                       for f in sorted((TRAIN_SCENES / "train").glob("*/chirps.parquet"))])
    on_train = train["template_id"].isin(in_split["train"])
    excursion = t.set_index("template_id").loc[octave, "excursion_hz"]
    return {
        "n_big_checked": len(ratio),
        "n": len(octave),
        "template_ids": octave,
        "per_split": {s: len(v) for s, v in in_split.items()},
        "ratio": [min(ratio[i] for i in octave), max(ratio[i] for i in octave)],  # pairs for #stat-range
        "ratio_others_max": float(values[values <= OCTAVE_RATIO].max()),
        "excursion_hz": [float(excursion.min()), float(excursion.max())],
        "train_scene_chirps": {"n": int(on_train.sum()), "of": len(train),
                               "share": float(on_train.mean())},
    }


def decoder() -> dict:
    bundle = read_json(DECODER / "bundle.json")
    recal = bundle["training"]["recalibration"]
    octave = octave_templates()
    val = [i for i in octave["template_ids"] if str(i) in recal["excluded_boxes"]]
    if sorted(recal["excluded_template_ids"]) != val or octave["per_split"]["val"] != len(val):
        raise SystemExit(f"the bundle left out {recal['excluded_template_ids']}; octave templates: "
                         f"{octave['template_ids']} ({octave['per_split']} per split)")
    return {
        "octave_templates": octave,
        "recalibration": {
            "val_boxes_excluded": sum(recal["excluded_boxes"].values()),
            "val_boxes_before": recal["n_calibration_before"],
            "val_boxes_after": bundle["training"]["n_calibration"],
        },
    }


def main() -> None:
    stats = {"templates": templates(), "scenes": scenes(), "benchmark": benchmark(),
             "real_session": real_session(), "production": production(), "decoder": decoder()}
    print(json.dumps(stats, indent=1, default=float))
    write_stats("pipeline", stats)


if __name__ == "__main__":
    main()
