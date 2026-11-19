"""First look: the winner's and loser's chirps around the onsets of video-detected chases (2022, dark phase).

Chirps: the corrected 0.8 dataset on gap-filled tracks
(``competition2022_v2_gapfill_t080_corrected``), each weighted by ``p_chirp`` (its
probability of being a chirp), winner / loser from ``meta.csv``. Chases:
``c1_chases.py`` events. Both are in recording time (the video's frame times come
from the LED synchronisation, ``b1_assess_sleap``).

For every chase onset, the chirps in [-WINDOW_S, +WINDOW_S] around it are counted
in BIN_S bins, summed over events and divided by the number of events and the bin
width (chirps per second per fish). Reference: the same around the same number of
random times at which both fish were in view (from the chase run's tracks).
Which fish leads a chase is not known in terms of winner / loser yet (the video
tracks are not linked to the electric identities).

    .venv/bin/python experiments/video2022/c4_chirps_around_chases.py poses_r3

Output: ``$CHIRP_DATA/video2022/chases/<run>/chirps_around_chases.png`` and ``.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_chases as c1
from paths import DATA

CHIRPS = DATA / "detections" / "competition2022_v2_gapfill_t080_corrected"
WINDOW_S = 10.0
BIN_S = 0.5
SEED = 20261120


def peri(times: np.ndarray, chirp_t: np.ndarray, w: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Weighted chirp counts per bin relative to each time, summed over times."""
    out = np.zeros(len(edges) - 1)
    for t in times:
        sel = (chirp_t >= t + edges[0]) & (chirp_t < t + edges[-1])
        out += np.histogram(chirp_t[sel] - t, bins=edges, weights=w[sel])[0]
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    args = ap.parse_args()
    root = c1.OUT / args.run
    ev = pd.read_csv(root / "events.csv")
    edges = np.arange(-WINDOW_S, WINDOW_S + BIN_S / 2, BIN_S)
    rng = np.random.default_rng(SEED)
    rows = []
    for rec, g in ev.groupby("recording"):
        path = CHIRPS / rec / "chirps_rtdetr_t080.parquet"
        if not path.exists():  # 2022-07-07 is no trial
            continue
        ch = pd.read_parquet(path)
        tr = pd.read_parquet(root / rec / "tracks.parquet", columns=["frame", "t_s"])
        both = tr.groupby("frame")["t_s"].agg(["first", "size"])
        both = both.loc[both["size"] >= 2, "first"].to_numpy()
        onsets = g["t_start"].to_numpy()
        ref = rng.choice(both, len(onsets), replace=False) if len(both) >= len(onsets) else both
        for s in ("winner", "loser"):
            c = ch[ch["status"] == s]
            t, w = c["t_center_s"].to_numpy(), c["p_chirp"].to_numpy()
            rows.append({"recording": rec, "status": s, "n_events": len(onsets), "n_ref": len(ref),
                         **{f"chase_{i}": v for i, v in enumerate(peri(onsets, t, w, edges))},
                         **{f"ref_{i}": v for i, v in enumerate(peri(ref, t, w, edges))}})
    d = pd.DataFrame(rows)
    d.to_csv(root / "chirps_around_chases.csv", index=False)
    centers = (edges[:-1] + edges[1:]) / 2
    nb = len(centers)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, constrained_layout=True)
    for ax, s, col in zip(axes, ("loser", "winner"), ("#d95f0e", "#2b8cbe"), strict=True):
        g = d[d["status"] == s]
        n_ev, n_ref = g["n_events"].sum(), g["n_ref"].sum()
        chase = g[[f"chase_{i}" for i in range(nb)]].sum().to_numpy() / n_ev / BIN_S
        ref = g[[f"ref_{i}" for i in range(nb)]].sum().to_numpy() / n_ref / BIN_S
        ax.step(centers, chase, where="mid", color=col, lw=2, label=f"around {n_ev} chase onsets")
        ax.step(centers, ref, where="mid", color="0.5", lw=1.2, label="around random times, both fish in view")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        n_tr = g["recording"].nunique()
        ax.set_title(f"{s}'s chirps ({n_tr} trial{'' if n_tr == 1 else 's'})", fontsize=10)
        ax.set_xlabel("time from chase onset (s)")
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("chirps per second (expected, p_chirp-weighted)")
    fig.suptitle("Chirps around video-detected chases (2022, dark phase, first look)", fontsize=11)
    fig.savefig(root / "chirps_around_chases.png", dpi=150)
    print(root / "chirps_around_chases.png")


if __name__ == "__main__":
    main()
