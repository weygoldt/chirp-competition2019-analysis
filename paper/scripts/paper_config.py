"""Paths and settings of the paper analysis: the one place every script reads.

The scripts in this folder are copies of the analysis behind the chirp
competition manuscript (``~/wrk/analyses/chirp_competition`` at d040123),
changed only where they read detections or write outputs. Environment
variables override the defaults (the machine the results were made on):

``CHIRP_REC_2019``
    The 2019/2020 tube competition recordings (one directory per recording
    with tracks, BORIS annotations, LED files and the chirp detections) and
    ``trial_summary.csv``. Default ``/mnt/data2/2019_tube_competition``.
``PAPER_DETECTIONS``
    Which chirp detections the analysis reads: the suffix of
    ``chirp_times_<suffix>.npy`` / ``chirp_ids_<suffix>.npy`` in each
    recording directory. ``rtdetr`` (default) = pipeline v2 (dual-head
    RT-DETR, run competition2019_v2_20261113); ``rtdetr_sw...`` = its
    single-window successor (run competition2019_v3, ``PIPELINES["sw"]``);
    ``rcnn`` = v1 (the YOLOv8 detections the manuscript used).
``PAPER_OUT``
    Output root. Intermediates go to ``$PAPER_OUT/data/processed`` (the old
    scripts' relative ``data/processed``), model summaries to
    ``$PAPER_OUT/results``, figures to ``$PAPER_OUT/figures``. Default
    ``paper/output/<detections>`` inside the repository (git-ignored; all
    derived data of the paper lives there, raw data and detections stay in
    the dataset).
``PAPER_FIGURE_DIR``
    Figure directory (the old scripts' variable). Default
    ``$PAPER_OUT/figures``. Never the manuscript's ``figures/``: only
    ``paper/export_figures.sh`` copies the primary run's figures there.
``PAPER_PRIMARY``
    The primary run (the detections at the operating threshold), default
    ``rtdetr_sw_gapfill_t080`` (single-window detector, since 2026-11-27;
    before: ``rtdetr_gapfill_t080``); its outputs are ``paper/output/<primary>``. Its
    figures go into the manuscript (``export_figures.sh``), and the window
    around chase offsets is derived from its event rates for every run
    (``offset_window.py``).

The scripts are run with the output root as working directory
(``run_all.sh``), so any relative path the old code still uses lands there
too.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

# Python 3.14 changed the default start method on Linux from "fork" to
# "forkserver", which re-imports the main module in every worker. The old
# scripts run their analysis at module level (no __main__ guard), so pymc's
# parallel chains would re-run the script. Restore the 3.12 default; every
# ported script imports this module before sampling.
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:  # already set in this process
    pass


def _env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


SCRIPTS = Path(__file__).resolve().parent
PAPER = SCRIPTS.parent

DATA_ROOT = _env("CHIRP_REC_2019", "/mnt/data2/2019_tube_competition")
DETECTIONS = os.environ.get("PAPER_DETECTIONS", "rtdetr")
CHIRP_DATA = _env("CHIRP_DATA", "/mnt/data1/chirp_v2")
OUT = _env("PAPER_OUT", PAPER / "output" / DETECTIONS)
PROCESSED = OUT / "data" / "processed"
RESULTS = OUT / "results"
FIGURES = _env("PAPER_FIGURE_DIR", OUT / "figures")
PRIMARY = os.environ.get("PAPER_PRIMARY", "rtdetr_sw_gapfill_t080")
PRIMARY_OUT = PAPER / "output" / PRIMARY

# The two chirp-detection pipelines the paper can run on (paths under
# CHIRP_DATA). "v2" is the released detector (three STFT windows as the RGB
# channels, trained on train_v4; models-v1); "sw" its successor after the
# front-end ablation round 2 (author's decision 2026-11-27): one 205-ms window,
# trained on train_v4r (train_v4 + rapid-chirping scenes), seed 3 of that round,
# with its own decoder (experiments/production_v3/run_switch.sh). A
# PAPER_DETECTIONS name starting with "rtdetr_sw" selects "sw". Per pipeline:
# ``runs``: detections name -> (run directory, chirp table, run holding the
# gap-filled tracks or None); the detector bundle and its simulated-test
# evaluation (eval_postprocess.py) with the dataset it ran on and the training
# scenes; the decoder bundle; the blind benchmark's report and the name of the
# detector in it; the real-data session's report.
PIPELINES = {
    "v2": {
        # the chirp tables of the *_recal1 runs (2026-11-27): the same detections
        # with the decoder recalibrated without the octave-error template
        "runs": {
            "rtdetr": ("competition2019_v2_20261113_recal1", "chirps_rtdetr.parquet", None),
            "rtdetr_gapfill": ("competition2019_v2_20261113_gapfill_recal1",
                               "chirps_rtdetr_gapfill.parquet",
                               "competition2019_v2_20261113_gapfill"),
            "rtdetr_gapfill_t080": ("competition2019_v2_20261113_gapfill_t080_recal1",
                                    "chirps_rtdetr_gapfill_t080.parquet",
                                    "competition2019_v2_20261113_gapfill"),
            "rtdetr_gapfill_t090": ("competition2019_v2_20261113_gapfill_t090_recal1",
                                    "chirps_rtdetr_gapfill_t090.parquet",
                                    "competition2019_v2_20261113_gapfill"),
        },
        "source_run": "detections/competition2019_v2_20261113",
        "detector": "runs/dualhead_r18_sim_v4_r1/final/bundle",
        "sim_eval": "runs/dualhead_r18_sim_v4_r1/eval_postprocess",
        "sim_dataset": "datasets/sim_v4",
        "train_scenes": "scenes/train_v4",
        "decoder": "params/runs/v4-dualhead-sim-v4-r2-recal1/bundle",
        "benchmark": "labels/benchmark/report_v2",
        "benchmark_model": "dualhead",
        "real_session": "labels/real_v1/report",
    },
    "sw": {
        "runs": {
            "rtdetr_sw": ("competition2019_v3", "chirps_rtdetr.parquet", None),
            "rtdetr_sw_gapfill": ("competition2019_v3_gapfill", "chirps_rtdetr_gapfill.parquet",
                                  "competition2019_v3_gapfill"),
            "rtdetr_sw_gapfill_t080": ("competition2019_v3_gapfill_t080",
                                       "chirps_rtdetr_gapfill_t080.parquet",
                                       "competition2019_v3_gapfill"),
            "rtdetr_sw_gapfill_t090": ("competition2019_v3_gapfill_t090",
                                       "chirps_rtdetr_gapfill_t090.parquet",
                                       "competition2019_v3_gapfill"),
        },
        "source_run": "detections/competition2019_v3",
        "detector": "runs/dualhead_r18_single_v4r_s3/final/bundle",
        "sim_eval": "runs/dualhead_r18_single_v4r_s3/eval_postprocess",
        "sim_dataset": "datasets/sim_v4r_longonly",
        "train_scenes": "scenes/train_v4r",
        "decoder": "params/runs/v4r-single-s3-recal1/bundle",
        "benchmark": "labels/benchmark/report_v3",
        "benchmark_model": "single_v3",
        "real_session": "labels/real_v1/report_sw",
    },
}
PIPELINE_NAME = "sw" if DETECTIONS.startswith("rtdetr_sw") else "v2"
PIPELINE = PIPELINES[PIPELINE_NAME]
# every detections name the figure scripts know (both pipelines)
RUNS = {name: run for p in PIPELINES.values() for name, run in p["runs"].items()}


def pipeline_path(key: str) -> Path:
    """A path of the selected pipeline (``PIPELINE[key]``) under CHIRP_DATA."""
    return CHIRP_DATA / PIPELINE[key]
# The author's blind re-labelling of chase onsets / offsets (paper/chase_review): a frozen copy of the
# session (session.json) and the saves (revisions.jsonl), with their sha256 in paper/inputs/README.md.
CHASE_VALIDATION = _env("CHASE_VALIDATION", _env("PAPER_INPUTS", PAPER / "inputs") / "chase_validation")
# Git-ignored data inputs (paper/inputs), or the inputs/ folder of the derived-data package.
INPUTS = _env("PAPER_INPUTS", PAPER / "inputs")

# Seed for scripts that were unseeded in the old analysis (seeded ones keep theirs).
SEED = 42

# The manuscript's figures (submodule; written only by export_figures.sh), the
# former standalone manuscript checkout, and the old analysis repository.
_FORBIDDEN = (
    (PAPER.parent / "manuscript" / "figures").resolve(),
    Path("~/wrk/analyses/chirp_competition_paper/figures").expanduser().resolve(),
    Path("~/wrk/analyses/chirp_competition").expanduser().resolve(),
)


def _check(path: Path) -> Path:
    resolved = path.resolve()
    for bad in _FORBIDDEN:
        if resolved == bad or bad in resolved.parents:
            raise RuntimeError(f"refusing to write into {bad} ({path})")
    return path


for _p in (OUT, PROCESSED, RESULTS, FIGURES):
    _check(_p)


def processed(*parts: str) -> Path:
    """A path under ``$PAPER_OUT/data/processed``; parent directories are created."""
    p = _check(PROCESSED.joinpath(*parts))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(*parts: str) -> Path:
    """A directory under ``$PAPER_OUT/data/processed``; it is created."""
    p = _check(PROCESSED.joinpath(*parts))
    p.mkdir(parents=True, exist_ok=True)
    return p


def results(*parts: str) -> Path:
    """A path under ``$PAPER_OUT/results``; parent directories are created."""
    p = _check(RESULTS.joinpath(*parts))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def figure(filename: str) -> Path:
    """A path in the figure directory; it is created."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return _check(FIGURES / filename)
