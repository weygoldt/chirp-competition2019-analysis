"""Shared definitions of the 2022 labelling session real_v2_2022 (2026-11-19).

Why this session exists: in real_v1 the 2022 sample was spread evenly over the
19 recordings, but the 2022 detections are concentrated in pair 3
(2022-06-20/22/24), so single items carried 60-66 % of a score bin's weight
(``experiments/data2022/README.md``, "In short"). Here every sampling cell is
pooled over its recordings, so all items of a cell have the same inclusion
probability ``n_cell / N_cell``.

The labeller, its view and the item kinds are the ones of real_v1
(``experiments/real_labels/common.py``: ``VIEW``, ``LABELLER_ARGS``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
REAL_V1_CODE = HERE.parent / "real_labels"
sys.path.insert(0, str(HERE.parent))
from paths import DATA, REPO  # noqa: E402

SESSION = "real_v2_2022"
ROOT = DATA / "labels" / SESSION  # truth, pools, review, report (never read by the labeller)
TRUTH = ROOT / "truth" / f"{SESSION}_truth.parquet"
CANDIDATES = DATA / "labels" / "candidates" / f"{SESSION}.parquet"  # what the labeller reads
LABELS_OUT = DATA / "labels" / f"{SESSION}_labels"  # label-emitters --out

REAL_V1_ROOT = DATA / "labels" / "real_v1"
REAL_V1_TRUTH = REAL_V1_ROOT / "truth" / "real_v1_truth.parquet"
REAL_V1_LABELS = DATA / "labels" / "real_v1_labels"

# chirpdetector that serves the session (main = branch v2-real-labels, merged 2026-11-16)
UPGRADE = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
CHIRPDETECTOR_SRC = UPGRADE / "packages" / "chirpdetector" / "src"

PAIR3 = ("2022-06-20-10_00", "2022-06-22-10_00", "2022-06-24-10_00")
GROUPS = ("pair3", "rest")  # rest: the other 15 trials and 2022-07-07-10_00 (no trial)
STATUSES = ("winner", "loser", "other")
SCORE_EDGES = (0.5, 0.7, 0.8, 0.9, 1.0001)
SCORE_NAMES = ("0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
POSITIVE = ("chirp", "chirp_emitter_unclear")


def score_bin(score: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """Score bin name of each score (``"below"`` under 0.5)."""
    s = np.asarray(score, dtype=float)
    k = np.searchsorted(np.asarray(SCORE_EDGES), s, side="right") - 1
    names = np.array([*SCORE_NAMES, "below"])
    k = np.where((k < 0) | (k >= len(SCORE_NAMES)), len(SCORE_NAMES), k)
    return names[k]


def group_of(recording: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """``pair3`` for 2022-06-20/22/24, ``rest`` for every other 2022 recording."""
    return np.where(np.isin(np.asarray(recording, dtype=str), PAIR3), "pair3", "rest")


def split_sizes(n_small: int, n_big: int, n: int, min_each: int) -> tuple[int, int]:
    """Items (small, big) for a cell of ``n`` items.

    Each size gets its share of the cell's pool, but at least ``min_each``
    (or all it has); a size that runs short passes its items to the other.
    A cell whose pool is at most ``n`` is taken whole.
    """
    if n_small + n_big <= n:
        return n_small, n_big
    k_big = max(round(n * n_big / (n_small + n_big)), min(min_each, n_big))
    k_big = min(k_big, n - min(min_each, n_small), n_big)
    k_small = min(n_small, n - k_big)
    k_big = min(n_big, n - k_small)
    return k_small, k_big


def kish_n_eff(weights: npt.ArrayLike) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2``."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum()) if w.size and (w**2).sum() > 0 else 0.0


def cell_n_eff(n_small: int, n_big: int, k_small: int, k_big: int) -> float:
    """Effective sample size of a cell's estimate when (k_small, k_big) items
    are drawn from pools of (n_small, n_big), each item weighted N / k."""
    w = ([n_small / k_small] * k_small if k_small else []) + (
        [n_big / k_big] * k_big if k_big else [])
    return kish_n_eff(w)
