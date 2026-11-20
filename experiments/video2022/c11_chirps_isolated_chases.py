"""Loser chirps around isolated chases only (no other chase within 10 s), relative to same-trial random moments.

Chases come in bouts, which smears chirp timing. Here only chases without another chase
of the same source within ``GAP`` s before the onset or after the offset are kept
(trained model ``model_v2``, its confident chases, and the old detector's identified
winner chases), and the loser's chirps are stacked as in ``c10_chirps_model_chases.py``.

    .venv/bin/python experiments/video2022/c11_chirps_isolated_chases.py

Output: ``$CHIRP_DATA/video2022/chases/model/model_v2/chirps_isolated.png``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c10_chirps_model_chases as c10

GAP = 10.0


def isolated(ev: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, g in ev.groupby("recording"):
        g = g.sort_values("t_start")
        s, e = g["t_start"].to_numpy(), g["t_end"].to_numpy()
        prev_end = np.r_[-np.inf, np.maximum.accumulate(e)[:-1]]
        next_start = np.r_[s[1:], np.inf]
        keep.append(g[(s - prev_end > GAP) & (next_start - e > GAP)])
    return pd.concat(keep)


srcs = {"model_v2 (all)": c10.winner_chases("model_v2", "poses_r3"),
        "old detector": c10.winner_chases("old detector", "poses_r3")}
v2 = srcs["model_v2 (all)"]
srcs["model_v2 confident"] = v2[v2["p_mean"] >= 0.9]
recs = sorted(set().union(*(set(e["recording"]) for e in srcs.values())))
base = c10.random_rates(recs, "poses_r3", np.random.default_rng(c10.SEED))
C = c10.CENTERS
rows = []
for name, ev in srcs.items():
    iso = isolated(ev)
    for align, col in (("onset", "t_start"), ("offset", "t_end")):
        y, n = c10.ratio(iso, col, "loser", base)
        rows.append((name, align, n, y))
        print(f"{name:20s} isolated {align:6s} n={n:4d} (of {len(ev)}): -5..-2 s {y[(C > -5) & (C < -2)].mean():.2f}, "
              f"-2..0 s {y[(C > -2) & (C < 0)].mean():.2f}, 0..2 s {y[(C > 0) & (C < 2)].mean():.2f}, "
              f"2..5 s {y[(C > 2) & (C < 5)].mean():.2f}, 5..10 s {y[(C > 5) & (C < 10)].mean():.2f}")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
colors = {"model_v2 (all)": "#fc8d59", "model_v2 confident": "#b30000", "old detector": "0.45"}
for ax, align in zip(axes, ("onset", "offset"), strict=True):
    for name, al, n, y in rows:
        if al == align:
            ax.step(C, y, where="mid", color=colors[name], lw=2 if "confident" in name else 1.3, label=f"{name} (n = {n})")
    ax.axhline(1, color="0.6", lw=0.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(f"time from the chase {align} (s)")
axes[0].set_ylabel("loser's chirps, relative to random\nmoments in the same trials")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Isolated chases only (no other chase within 10 s before the onset or after the offset)", fontsize=10)
fig.savefig("/mnt/data1/chirp_v2/video2022/chases/model/model_v2/chirps_isolated.png", dpi=150)
