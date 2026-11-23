"""Fast test: how sharp do the chase-triggered loser-chirp kernels get when close chases are merged?

The same estimator as the 2019 script, without the resampling: loser chirp lags around
each chase onset / offset (centres in the first 3 h), pooled over trials, Gaussian kernel
(sigma 0.3 s), divided by the same with the centres jittered by up to 120 s (``N_JITTER``
draws). Chases: the locked model chases (``model_v2``), consecutive chases of a trial
merged when the gap between them is < G s (G = 0: as now, only overlaps merged). Chirps:
the 2022 dataset folder (>= 0.8, unweighted).

    .venv/bin/python experiments/paper2022/merge_gap_test.py

Output: ``$CHIRP_DATA/paper2022/output/chases_model_v2/figures/merge_gap_test.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_kernel_blur as dk
from paths import DATA

CHASES = DATA / "video2022" / "chases" / "model" / "model_v2" / "events.csv"
DATASET = DATA / "paper2022" / "dataset"
SIGMA = 0.3
T = np.arange(-30, 30, 0.1)
ACTIVE = (0.0, 3 * 3600.0)
N_JITTER = 30
JITTER_S = 120.0
GAPS = (0.0, 2.0, 5.0, 10.0)


def merge(on: np.ndarray, off: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Chases (sorted by onset) merged where the next onset comes < gap s after the current end (or overlaps)."""
    o = np.argsort(on)
    out: list[list[float]] = []
    for a, b in zip(on[o], off[o], strict=True):
        if out and a - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    m = np.array(out) if out else np.zeros((0, 2))
    return m[:, 0], m[:, 1]


def kde(lags: np.ndarray) -> np.ndarray:
    lags = lags[np.abs(lags) < 31]
    if not len(lags):
        return np.zeros_like(T)
    return np.exp(-0.5 * ((T[:, None] - lags[None, :]) / SIGMA) ** 2).sum(axis=1) / (SIGMA * np.sqrt(2 * np.pi))


def lags(centres: np.ndarray, chirps: np.ndarray) -> np.ndarray:
    chirps = np.sort(chirps)
    out = []
    for c in centres:
        a, b = np.searchsorted(chirps, [c - 31, c + 31])
        out.append(chirps[a:b] - c)
    return np.concatenate(out) if out else np.zeros(0)


def kernel(trials: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    """rate around centres / rate around jittered centres (pooled over trials)."""
    obs, base, n = np.zeros_like(T), np.zeros_like(T), 0
    for centres, chirps in trials:
        centres = centres[(centres >= ACTIVE[0]) & (centres < ACTIVE[1])]
        n += len(centres)
        obs += kde(lags(centres, chirps))
        for _ in range(N_JITTER):
            base += kde(lags(centres + rng.uniform(-JITTER_S, JITTER_S, len(centres)), chirps)) / N_JITTER
    return obs / np.maximum(base, 1e-12)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = pd.read_csv(DATASET / "trial_summary.csv")
    ev = pd.read_csv(CHASES)
    data = []
    for r in ts.itertuples():
        t = np.load(DATASET / r.recording / "chirp_times_rtdetr_t080.npy")
        ids = np.load(DATASET / r.recording / "chirp_ids_rtdetr_t080.npy")
        c = ev[ev["recording"] == r.recording]
        data.append((c["t_start"].to_numpy(), c["t_end"].to_numpy(), t[ids == r.lose_ID]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False, constrained_layout=True)
    for col, center in enumerate(("agon", "agoff")):
        ax = axes[col]
        t19, k19 = dk.ratio(dk.RATES_2019 / f"lchirps@{center}.json")
        ax.plot(t19, k19, color="0.3", lw=2, label="2019 (BORIS chases)")
        t22, k22 = dk.ratio(dk.RATES_2022 / f"lchirps@{center}.json")
        ax.plot(t22, k22, color="#fdae6b", lw=1, ls="--", label="2022, 2019 script (overlaps merged)")
        for gap, color in zip(GAPS, ("#fdae6b", "#f16913", "#d94801", "#7f2704"), strict=True):
            trials, n = [], 0
            for on, off, chirps in data:
                mon, moff = merge(on, off, gap)
                n += len(mon)
                trials.append((mon if center == "agon" else moff, chirps))
            k = kernel(trials, np.random.default_rng(1))
            ax.plot(T, k, color=color, lw=1.5, label=f"2022, merged when gap < {gap:g} s ({n} chases)")
            pre = k[(T > -2) & (T < 0)].mean()
            print(f"{center} gap < {gap:g} s: {n} chases, peak {k.max():.2f} at {T[np.argmax(k)]:+.1f} s, "
                  f"mean -2..0 s {pre:.2f}, 0..2 s {k[(T > 0) & (T < 2)].mean():.2f}")
        print(f"{center} 2019: peak {k19.max():.2f}, -2..0 s {k19[(t19 > -2) & (t19 < 0)].mean():.2f}")
        ax.axhline(1, color="0.7", lw=0.8)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlim(-30, 30)
        ax.set_xlabel(f"time from the chase {'onset' if center == 'agon' else 'offset'} (s)")
        ax.set_title(f"loser's chirps around the chase {'onset' if center == 'agon' else 'offset'}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("rate / jittered baseline")
    fig.suptitle("Merging chases separated by short gaps (2022 model chases) vs 2019", fontsize=10)
    out = dk.RUN / "figures" / "merge_gap_test.png"
    fig.savefig(out, dpi=150)
    print(out)


def variants_figure() -> None:
    """2019, 2019 blurred by the model's timing errors, and 2022 variants (merge gap, confidence)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = pd.read_csv(DATASET / "trial_summary.csv")
    ev = pd.read_csv(CHASES)
    loco = pd.read_csv(dk.LOCO)
    m = loco[(loco["method"] == "model") & loco["found"]]
    err = {"agon": m["onset_err_s"].dropna().to_numpy(), "agoff": m["offset_err_s"].dropna().to_numpy()}
    variants = ((0.0, 0.0, "#fdae6b", "2022 as now"), (0.9, 5.0, "#e6550d", "2022, confident, gaps < 5 s merged"),
                (0.9, 10.0, "#7f2704", "2022, confident, gaps < 10 s merged"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for col, center in enumerate(("agon", "agoff")):
        ax = axes[col]
        t19, k19 = dk.ratio(dk.RATES_2019 / f"lchirps@{center}.json")
        ax.plot(t19, k19, color="0.25", lw=2, label="2019 (BORIS chases)")
        ax.plot(t19, dk.blur(t19, k19, err[center]), color="0.25", ls="--", lw=1.3,
                label="2019 blurred by the 2022 model's timing errors")
        for min_p, gap, color, label in variants:
            trials, n = [], 0
            for r in ts.itertuples():
                t = np.load(DATASET / r.recording / "chirp_times_rtdetr_t080.npy")
                ids = np.load(DATASET / r.recording / "chirp_ids_rtdetr_t080.npy")
                c = ev[(ev["recording"] == r.recording) & (ev["p_mean"] >= min_p)]
                mon, moff = merge(c["t_start"].to_numpy(), c["t_end"].to_numpy(), gap)
                n += len(mon)
                trials.append((mon if center == "agon" else moff, t[ids == r.lose_ID]))
            ax.plot(T, kernel(trials, np.random.default_rng(1)), color=color, lw=1.6, label=f"{label} ({n})")
        ax.axhline(1, color="0.7", lw=0.8)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlim(-30, 30)
        ax.set_xlabel(f"time from the chase {'onset' if center == 'agon' else 'offset'} (s)")
        ax.set_title(f"loser's chirps around the chase {'onset' if center == 'agon' else 'offset'}", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel("rate / jittered baseline")
    fig.suptitle("Sharper 2022 kernels: merge close chases (definition) and keep confident ones; "
                 "dashed = what our timing errors alone would do to 2019", fontsize=9)
    out = dk.RUN / "figures" / "merge_variants.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    if "--variants" in sys.argv:
        variants_figure()
    else:
        main()
