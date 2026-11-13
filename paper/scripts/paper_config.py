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
    RT-DETR, run competition2019_v2_20261113); ``rcnn`` = v1 (the YOLOv8
    detections the manuscript used).
``PAPER_OUT``
    Output root. Intermediates go to ``$PAPER_OUT/data/processed`` (the old
    scripts' relative ``data/processed``), model summaries to
    ``$PAPER_OUT/results``, figures to ``$PAPER_OUT/figures``. Default
    ``$CHIRP_DATA/paper_rerun/<detections>`` with ``CHIRP_DATA`` defaulting to
    ``/mnt/data1/chirp_v2``.
``PAPER_FIGURE_DIR``
    Figure directory (the old scripts' variable). Default
    ``$PAPER_OUT/figures``. Never the manuscript's ``figures/``.

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
OUT = _env("PAPER_OUT", CHIRP_DATA / "paper_rerun" / DETECTIONS)
PROCESSED = OUT / "data" / "processed"
RESULTS = OUT / "results"
FIGURES = _env("PAPER_FIGURE_DIR", OUT / "figures")

# Seed for scripts that were unseeded in the old analysis (seeded ones keep theirs).
SEED = 42

# The manuscript's figure directory; nothing may be written there.
_FORBIDDEN = Path("~/wrk/analyses/chirp_competition_paper/figures").expanduser().resolve()
_OLD_REPO = Path("~/wrk/analyses/chirp_competition").expanduser().resolve()


def _check(path: Path) -> Path:
    resolved = path.resolve()
    for bad in (_FORBIDDEN, _OLD_REPO):
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
