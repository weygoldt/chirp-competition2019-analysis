"""Refine chase offsets with the chaser's deceleration (author, 2026-11-23).

Author: it is the chaser that disengages; at the end of a chase it slows down while the
chasee keeps evading, and the gap opens. Only kinematics are used (never chirps: the
offsets are later used to align chirps).

Per sample, for a chase of ``chaser`` after ``chasee`` (fish ``A`` / ``B`` of a pair table,
``c7_chase_labels_kinematics.pair_frame`` layout: ``speed_A``, ``speed_B``, ``dist``):

    drop_X(t) = mean speed of X over (t - delta, t]  -  mean over (t, t + delta]   (BL/s)
    open(t)   = mean distance over (t, t + delta]  -  mean over (t - delta, t]       (BL)
    cue(t)    = drop_chaser(t) - w_chasee * drop_chasee(t) + w_open * open(t)

and the refined offset is the time of the largest cue within ``window`` around the
detected offset (samples where the two fish were not both seen are skipped; if none is
left, the offset is kept).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DT = 0.08


@dataclass(frozen=True)
class Refine:
    delta_s: float = 0.6
    w_chasee: float = 1.0
    w_open: float = 1.0
    before_s: float = 2.0
    after_s: float = 2.0


def _means(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the n samples up to and including t, and of the n samples after t (NaN-aware)."""
    s = pd.Series(x)
    past = s.rolling(n, min_periods=max(1, n // 2)).mean().to_numpy()
    fut = s[::-1].rolling(n, min_periods=max(1, n // 2)).mean().to_numpy()[::-1]
    fut = np.r_[fut[1:], np.nan]
    return past, fut


def cue(s: pd.DataFrame, chaser: str, chasee: str, r: Refine = Refine()) -> np.ndarray:  # noqa: B008
    n = max(1, round(r.delta_s / DT))
    p_c, f_c = _means(s[f"speed_{chaser}"].to_numpy(float), n)
    p_e, f_e = _means(s[f"speed_{chasee}"].to_numpy(float), n)
    p_d, f_d = _means(s["dist"].to_numpy(float), n)
    out = (p_c - f_c) - r.w_chasee * (p_e - f_e) + r.w_open * (f_d - p_d)
    both = np.isfinite(s["dist"].to_numpy(float))
    out[~both] = np.nan
    return out


def refine(t: np.ndarray, c: np.ndarray, t_off: float, r: Refine = Refine()) -> float:  # noqa: B008
    """The time of the largest cue within [t_off - before_s, t_off + after_s]; t_off if none."""
    w = (t >= t_off - r.before_s) & (t <= t_off + r.after_s) & np.isfinite(c)
    if not w.any():
        return float(t_off)
    k = np.flatnonzero(w)[np.argmax(c[w])]
    return float(t[k])
