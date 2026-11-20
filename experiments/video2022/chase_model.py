"""A chaser-centric chase detector learned from the author's labels (prototype, 2026-11-20).

Author (labelling 15 clips): chases start spontaneously, get interrupted, end in
contact; it is the chaser that decides to engage and to disengage, and a subtle
deceleration of the chaser while the chasee keeps evading marks the end. So the
question per moment is directed: *is fish X chasing fish Y now?*

1. :func:`directed_features`: from two fish's kinematics on the model's frame grid
   (``c7_chase_labels_kinematics.clip_series`` layout, fish A and B), features of
   "X chases Y" for X = A and X = B: does X swim towards Y (``aim``), does Y swim
   away from X (``flee``), both speeds and accelerations and their difference,
   distance and its change, alignment; each also averaged over centred windows of
   ``WINDOWS_S`` and over the second before / after (offline, so the model may look ahead).
2. :class:`Logistic`: L2-regularised logistic regression (scipy; no scikit-learn here)
   on standardised features.
3. :func:`decode`: chase intervals from the per-frame probability: smoothed over
   ``smooth_s``; a chase starts when it reaches ``p_on`` and ends once it has stayed
   below ``p_off`` for ``hold_s`` (so short turns and pauses do not split a chase);
   gaps without both fish seen up to ``max_gap_s`` are bridged; chases shorter than
   ``min_dur_s`` are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DT = 0.08  # s between model frames (every 2nd video frame at 25 frames/s)
WINDOWS_S = (0.5, 1.0, 2.0)
BASE = ("aim", "flee", "speed_x", "speed_y", "rel_speed", "accel_x", "accel_y", "dist", "ddist", "cos")


def _direction(s: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Instantaneous features of "x chases y"."""
    dx, dy = s[f"x_{y}"] - s[f"x_{x}"], s[f"y_{y}"] - s[f"y_{x}"]  # from x to y
    nd = np.hypot(dx, dy)
    vx, vy = s[f"vx_{x}"], s[f"vy_{x}"]
    ux, uy = s[f"vx_{y}"], s[f"vy_{y}"]
    nx, ny = np.hypot(vx, vy), np.hypot(ux, uy)
    return pd.DataFrame({
        "aim": (vx * dx + vy * dy) / (nx * nd + 1e-12),
        "flee": (ux * dx + uy * dy) / (ny * nd + 1e-12),
        "speed_x": s[f"speed_{x}"], "speed_y": s[f"speed_{y}"],
        "rel_speed": s[f"speed_{x}"] - s[f"speed_{y}"],
        "accel_x": s[f"accel_{x}"], "accel_y": s[f"accel_{y}"],
        "dist": s["dist"], "ddist": s["ddist"], "cos": s["cos"],
    }, index=s.index)


def with_context(f: pd.DataFrame) -> pd.DataFrame:
    """The instantaneous features plus their means over centred windows and the second before / after."""
    out = [f]
    for w in WINDOWS_S:
        n = max(1, round(w / DT)) | 1
        out.append(f[list(BASE)].rolling(n, center=True, min_periods=1).mean().add_suffix(f"_c{w:g}"))
    n1 = round(1.0 / DT)
    past = f[["speed_x", "speed_y", "ddist", "aim"]].rolling(n1, min_periods=1).mean()
    fut = f[["speed_x", "speed_y", "ddist", "aim"]][::-1].rolling(n1, min_periods=1).mean()[::-1]
    out.append(past.add_suffix("_past1"))
    out.append(fut.add_suffix("_next1"))
    out.append((fut - past).add_suffix("_change1"))
    g = pd.concat(out, axis=1)
    g.loc[~np.isfinite(f["dist"]).to_numpy()] = np.nan  # nothing where the two fish are not both seen
    return g


def directed_features(s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{"A": features of "A chases B", "B": features of "B chases A"}."""
    return {"A": with_context(_direction(s, "A", "B")), "B": with_context(_direction(s, "B", "A"))}


class Logistic:
    """L2-regularised logistic regression on standardised features (rows with NaN are skipped in fitting)."""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> Logistic:
        from scipy.optimize import minimize

        ok = np.isfinite(X).all(axis=1)
        X, y = X[ok], y[ok].astype(float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w_pos = 0.5 / max(y.mean(), 1e-9)  # balance the classes
        w = np.where(y > 0, w_pos, 0.5 / max(1 - y.mean(), 1e-9))

        def loss(b: np.ndarray) -> tuple[float, np.ndarray]:
            z = Z @ b
            p = 1 / (1 + np.exp(-z))
            ll = (w * (np.logaddexp(0, z) - y * z)).sum() + 0.5 * self.l2 * (b[1:] @ b[1:])
            grad = Z.T @ (w * (p - y)) + self.l2 * np.r_[0.0, b[1:]]
            return ll, grad

        self.coef = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        p = 1 / (1 + np.exp(-(Z @ self.coef)))
        p[~np.isfinite(X).all(axis=1)] = np.nan
        return p


@dataclass(frozen=True)
class Decoder:
    p_on: float = 0.6
    p_off: float = 0.4
    hold_s: float = 0.6
    smooth_s: float = 0.4
    max_gap_s: float = 1.0
    min_dur_s: float = 0.8


def decode(p: np.ndarray, d: Decoder = Decoder()) -> list[tuple[int, int]]:  # noqa: B008 - frozen, immutable
    """(first, last) sample of each chase from per-sample probabilities (NaN = not both fish seen)."""
    p = pd.Series(p).rolling(max(1, round(d.smooth_s / DT)), center=True, min_periods=1).mean().to_numpy()
    out, start, below, gap = [], None, 0, 0
    hold, max_gap = round(d.hold_s / DT), round(d.max_gap_s / DT)
    last_in = None
    for i, v in enumerate(p):
        if start is None:
            if np.isfinite(v) and v >= d.p_on:
                start, below, gap, last_in = i, 0, 0, i
            continue
        if not np.isfinite(v):
            gap += 1
            if gap > max_gap:
                out.append((start, last_in))
                start = None
            continue
        gap = 0
        if v < d.p_off:
            below += 1
            if below >= hold:
                out.append((start, last_in))
                start = None
        else:
            below, last_in = 0, i
    if start is not None:
        out.append((start, last_in))
    return [(a, b) for a, b in out if (b - a + 1) * DT >= d.min_dur_s]
