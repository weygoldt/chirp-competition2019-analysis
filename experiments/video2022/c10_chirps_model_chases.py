"""Chirps of winner and loser around the onsets and offsets of the model's chases (2022, dark phase).

Chases: ``c9_chase_model_apply.py`` (chaser-centric model on the winner / loser
trajectories), those in which the winner chases, split by the model's confidence
(mean probability over the chase >= ``CONFIDENT`` or below). Chirps: the corrected 0.8
dataset (``c4_chirps_around_chases.CHIRPS``), weighted by ``p_chirp``. Reference: as
many random times with both fish in view (``identity_tracks``). If the less confident
chases were not real chases, the loser's chirping should not change around them.

    .venv/bin/python experiments/video2022/c10_chirps_model_chases.py model_v1

Output: ``$CHIRP_DATA/video2022/chases/model/<name>/chirps_around_chases.png`` / ``.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c4_chirps_around_chases as c4
import c8_chase_model as c8
import identity_tracks as it

CONFIDENT = 0.9
WINDOW_S = 10.0
BIN_S = 0.5
SEED = 20261120


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--run", default="poses_r3")
    args = ap.parse_args()
    root = c8.OUT / args.name
    ev = pd.read_csv(root / "events.csv")
    ev = ev[ev["chaser"] == "winner"]
    edges = np.arange(-WINDOW_S, WINDOW_S + BIN_S / 2, BIN_S)
    centers = (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(SEED)
    groups = {f"confident (p >= {CONFIDENT})": ev[ev["p_mean"] >= CONFIDENT],
              f"less confident (p < {CONFIDENT})": ev[ev["p_mean"] < CONFIDENT]}
    rows = []
    for rec in sorted(ev["recording"].unique()):
        ch = pd.read_parquet(c4.CHIRPS / rec / "chirps_rtdetr_t080.parquet")
        det = pd.read_parquet(it.OUT / args.run / f"{rec}_detections.parquet", columns=["frame", "t_s", "identity"])
        both = det.groupby("frame").agg(t=("t_s", "first"), n=("identity", "nunique"))
        both = both.loc[both["n"] == 2, "t"].to_numpy()
        n_ref = int((ev["recording"] == rec).sum())
        ref = rng.choice(both, min(n_ref, len(both)), replace=False)
        for s in ("loser", "winner"):
            c = ch[ch["status"] == s]
            t, w = c["t_center_s"].to_numpy(), c["p_chirp"].to_numpy()
            rows.append({"recording": rec, "status": s, "group": "random, both fish in view", "align": "onset",
                         "n": len(ref), **dict(enumerate(c4.peri(ref, t, w, edges)))})
            for name, g in groups.items():
                g = g[g["recording"] == rec]
                for align, col in (("onset", "t_start"), ("offset", "t_end")):
                    rows.append({"recording": rec, "status": s, "group": name, "align": align, "n": len(g),
                                 **dict(enumerate(c4.peri(g[col].to_numpy(), t, w, edges)))})
    d = pd.DataFrame(rows)
    d.to_csv(root / "chirps_around_chases.csv", index=False)
    colors = dict(zip(groups, ("#d95f0e", "#fdae6b"), strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey="row", constrained_layout=True)
    for r, s in enumerate(("loser", "winner")):
        ref = d[(d["status"] == s) & (d["group"].str.startswith("random"))]
        ref_rate = ref[list(range(len(centers)))].sum().to_numpy() / ref["n"].sum() / BIN_S
        for col, align in enumerate(("onset", "offset")):
            ax = axes[r, col]
            for name in groups:
                g = d[(d["status"] == s) & (d["group"] == name) & (d["align"] == align)]
                rate = g[list(range(len(centers)))].sum().to_numpy() / max(g["n"].sum(), 1) / BIN_S
                ax.step(centers, rate, where="mid", color=colors[name], lw=2, label=f"{name}, n = {g['n'].sum()}")
            ax.step(centers, ref_rate, where="mid", color="0.5", lw=1.2, label="random times, both fish in view")
            ax.axvline(0, color="k", ls=":", lw=0.8)
            if r == 1:
                ax.set_xlabel(f"time from the chase {align} (s)")
            if col == 0:
                ax.set_ylabel(f"{s}'s chirps per second")
        axes[r, 0].legend(fontsize=7, frameon=False)
    axes[0, 0].set_title("aligned to chase onsets", fontsize=9)
    axes[0, 1].set_title("aligned to chase offsets", fontsize=9)
    fig.suptitle("Chirps around the model's chases in which the winner chases (2022, dark phase, 18 trials)",
                 fontsize=10)
    fig.savefig(root / "chirps_around_chases.png", dpi=150)
    plt.close(fig)
    for name in groups:
        for align in ("onset", "offset"):
            g = d[(d["status"] == "loser") & (d["group"] == name) & (d["align"] == align)]
            rate = g[list(range(len(centers)))].sum().to_numpy() / max(g["n"].sum(), 1) / BIN_S
            print(f"loser, {name}, {align}: 5 s before {rate[(centers > -5) & (centers < 0)].mean():.3f}, "
                  f"5 s after {rate[(centers > 0) & (centers < 5)].mean():.3f} chirps/s")
    ref = d[(d["status"] == "loser") & (d["group"].str.startswith("random"))]
    print(f"loser, random: {(ref[list(range(len(centers)))].sum().to_numpy() / ref['n'].sum() / BIN_S).mean():.3f}")
    print(f"-> {root}")


if __name__ == "__main__":
    main()
