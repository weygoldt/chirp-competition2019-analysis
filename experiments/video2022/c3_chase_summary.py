"""Overview figure of the chase events of a run (``c1_chases.py``).

Panels: (a) hours with both fish in view per recording; (b) chase events per hour
with both fish in view; (c) event durations; (d) leader vs follower speed; (e)
event onsets over the dark phase.

    .venv/bin/python experiments/video2022/c3_chase_summary.py poses_r3

Output: ``$CHIRP_DATA/video2022/chases/<run>/summary.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_chases as c1  # noqa: E402


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    args = ap.parse_args()
    root = c1.OUT / args.run
    recs = pd.read_csv(root / "recordings.csv").sort_values("recording").reset_index(drop=True)
    ev = pd.read_csv(root / "events.csv")
    pair = recs["pairing"].str[:-1].where(recs["pairing"] != "control", "")
    colors = [f"C{int(p) - 1}" if p else "0.5" for p in pair]
    labels = [f"{r[5:10]} {p}" for r, p in zip(recs["recording"], recs["pairing"], strict=True)]
    x = np.arange(len(recs))
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    ax.bar(x - 0.2, recs["hours_both_visible"], 0.4, color=colors, alpha=0.45, label="hours both fish in view")
    ax.set_ylabel("hours with both fish in view (of 3 h)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, recs["events_per_hour_both_visible"], 0.4, color=colors, label="chases per hour")
    ax2.set_ylabel("chase events per hour with both fish in view")
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(f"{len(ev)} chase events in {len(recs)} recordings (dark phase). Light bars: hours both fish in view;"
                 f" dark bars: chases per hour of that time", fontsize=9)
    ax = fig.add_subplot(gs[0, 2])
    if len(ev):
        ax.hist(ev["duration_s"], bins=np.arange(1, max(6, ev["duration_s"].max()) + 0.25, 0.25), color="0.4")
    ax.set_xlabel("event duration (s)")
    ax.set_ylabel("events")
    ax.set_title(f"median {ev['duration_s'].median():.1f} s" if len(ev) else "no events", fontsize=9)
    ax = fig.add_subplot(gs[1, 0])
    if len(ev):
        ax.scatter(ev["speed_leader"], ev["speed_follower"], s=6, color="0.3", alpha=0.5, edgecolors="none")
        lim = [0, float(np.nanpercentile(ev[["speed_leader", "speed_follower"]], 99.5)) * 1.05]
        ax.plot(lim, lim, color="C3", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel("leader speed (body lengths/s)")
    ax.set_ylabel("follower speed (body lengths/s)")
    ax.set_title("mean speeds per event", fontsize=9)
    ax = fig.add_subplot(gs[1, 1])
    if len(ev):
        ax.hist(ev["lag_s"], bins=np.arange(0, 2.05, 0.08), color="0.4")
    ax.set_xlabel("follower's lag behind the leader's path (s)")
    ax.set_ylabel("events")
    ax.set_title("how far behind the follower swims", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    if len(ev):
        ax.hist(ev["t_start"] / 60, bins=np.arange(0, 181, 10), color="0.4")
    ax.set_xlabel("event onset, recording time (min)")
    ax.set_ylabel("events")
    ax.set_title("over the dark phase (all recordings)", fontsize=9)
    fig.savefig(root / "summary.png", dpi=150)
    print(root / "summary.png")


if __name__ == "__main__":
    main()
