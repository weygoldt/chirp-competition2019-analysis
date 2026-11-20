"""Loser's chirps around chase onsets and offsets: 2019 (hand-annotated chases) next to 2022 (model chases).

2019: the paper's primary run (read only; nothing in ``paper/`` is changed):
``paper/output/rtdetr_gapfill_t080/data/processed/event_time_rates/lchirps@{agon,agoff}.json``,
the jackknifed loser chirp rate around the BORIS-annotated chase onsets / offsets of
15 trials, and its baseline from event times jittered by up to 120 s (bootstrapped);
ratio = median rate / median baseline.

2022: the locked model run ``model_v2`` (winner chasing), relative to random moments
with both fish in view in the same trial (``c10_chirps_model_chases``): all chases, and
the isolated confident ones (``c11_chirps_isolated_chases``; caution: isolated chases
come from quiet periods by construction, which deepens the dip before the onset and
after the offset).

    .venv/bin/python experiments/video2022/c12_chirps_2019_vs_2022.py

Output: ``$CHIRP_DATA/video2022/chases/model/model_v2/chirps_2019_vs_2022.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c10_chirps_model_chases as c10
import c11_chirps_isolated_chases as c11

PAPER_RATES = Path(__file__).resolve().parents[2] / "paper" / "output" / "rtdetr_gapfill_t080" / "data" / \
    "processed" / "event_time_rates"
RUN = "model_v2"
WINDOWS = ((-10, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10))


def profile_2019(align: str) -> tuple[np.ndarray, np.ndarray]:
    name = "lchirps@agon.json" if align == "onset" else "lchirps@agoff.json"
    path = PAPER_RATES / name
    if not path.exists():  # the main checkout holds the paper outputs
        path = Path("/home/weygoldt/wrk/analyses/chirp_competition_upgrade/paper/output/rtdetr_gapfill_t080/data/"
                    "processed/event_time_rates") / name
    with open(path) as fh:
        d = json.load(fh)
    rate = np.median(np.array(d["jackknifed_eventrate"]), axis=0)
    base = np.median(np.array(d["bootstrapped_baseline_jittered"]), axis=0)
    return np.array(d["conv_time"]), rate / base


def window_means(t: np.ndarray, y: np.ndarray) -> str:
    return ", ".join(f"{a}..{b} s {y[(t > a) & (t < b)].mean():.2f}" for a, b in WINDOWS)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = c10.c8.OUT / RUN
    d22 = pd.read_csv(out / "chirps_around_chases.csv")
    ev = c10.winner_chases(RUN, "poses_r3")
    iso = c11.isolated(ev[ev["p_mean"] >= c10.CONFIDENT])
    base = c10.random_rates(sorted(iso["recording"].unique()), "poses_r3", np.random.default_rng(c10.SEED))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, align, col in zip(axes, ("onset", "offset"), ("t_start", "t_end"), strict=True):
        t19, y19 = profile_2019(align)
        r = d22[(d22["chases"] == RUN) & (d22["who"] == "loser") & (d22["align"] == align)].iloc[0]
        y_all = r[[str(i) for i in range(len(c10.CENTERS))]].to_numpy(float)
        y_iso, n_iso = c10.ratio(iso, col, "loser", base)
        ax.plot(t19, y19, color="0.35", lw=1.6, label="2019: hand-annotated chases (15 trials; baseline: jittered times)")
        ax.step(c10.CENTERS, y_all, where="mid", color="#fc8d59", lw=1.2, label="2022: all model chases (winner chasing)")
        ax.step(c10.CENTERS, y_iso, where="mid", color="#b30000", lw=2,
                label=f"2022: isolated confident chases (n = {n_iso}; no other chase within 10 s)")
        ax.axhline(1, color="0.7", lw=0.8)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlim(-12, 16)
        ax.set_xlabel(f"time from the chase {align} (s)")
        print(f"2019 {align:6s}: {window_means(t19, y19)}")
        print(f"2022 all {align:6s}: {window_means(c10.CENTERS, y_all)}")
        print(f"2022 isolated {align:6s}: {window_means(c10.CENTERS, y_iso)}")
    axes[0].set_ylabel("loser's chirps relative to baseline")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Loser's chirps around chases: 2019 vs 2022 (each relative to its own baseline)", fontsize=10)
    fig.savefig(out / "chirps_2019_vs_2022.png", dpi=150)
    plt.close(fig)
    print(out / "chirps_2019_vs_2022.png")


if __name__ == "__main__":
    main()
