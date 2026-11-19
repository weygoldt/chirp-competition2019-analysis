"""Figures of the 2019/2020 vs 2022 comparison (reads the outputs of ``a1_signal.py``).

1. ``rates.png``: kept detections per hour of dark phase, by score bin, for
   the trial's winner and loser, one dot per analysed trial (2019/2020: the 33
   analysed trials, gap-filled assignment; 2022: the 18 trials of meta.csv);
   2022's pair 3 (2022-06-20/22/24) marked. Message: winners have a similar
   floor of mid-score detections in both datasets, while 2022 losers (pair 3
   aside) have far fewer high-score detections.
2. ``signal.png``: per recording, the trial fish's SNR, the number of
   electrodes within 10 dB of each fish's best one, and the number of
   spectral lines no fish explains.

    .venv/bin/python experiments/data2022/a1_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2019

SIGNAL = DATA / "data2022" / "a1" / "signal"
PAIR3 = {"2022-06-20-10_00", "2022-06-22-10_00", "2022-06-24-10_00"}
EXCLUDED_2019 = {"2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00"}
BINS = ["0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1"]
COLOURS = {"2019/2020": "#525252", "2022": "#2b8cbe"}


def analysed(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_csv(REC_2019 / "trial_summary.csv")
    drop = set(t.loc[t["draw"] == 1, "recording"]) | EXCLUDED_2019
    return df[~df["recording"].isin(drop)]


def rates_figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = analysed(pd.read_csv(SIGNAL / "rates.csv"))
    r = r[r["status"].isin(["winner", "loser"]) & r["score_bin"].isin(BINS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, status in zip(axes, ("winner", "loser"), strict=True):
        for k, b in enumerate(BINS):
            for j, ds in enumerate(("2019/2020", "2022")):
                g = r[(r["status"] == status) & (r["score_bin"] == b) & (r["dataset"] == ds)]
                x = k + (j - 0.5) * 0.36 + np.random.default_rng(k * 7 + j).uniform(-0.1, 0.1, len(g))
                y = g["per_hour"].to_numpy() + 0.3
                p3 = g["recording"].isin(PAIR3).to_numpy()
                ax.scatter(x[~p3], y[~p3], s=12, color=COLOURS[ds], alpha=0.7, edgecolors="none",
                           label=f"{ds} trials" if k == 0 else None)
                if p3.any():
                    ax.scatter(x[p3], y[p3], s=26, facecolors="none", edgecolors="#d95f0e", lw=1.2,
                               label="2022 pair 3 (06-20/22/24)" if k == 0 else None)
                ax.hlines(np.median(y), k + (j - 0.5) * 0.36 - 0.14, k + (j - 0.5) * 0.36 + 0.14,
                          color=COLOURS[ds], lw=2.5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(BINS)), BINS)
        ax.set_xlabel("detector score bin")
        ax.set_title(f"detections assigned to the {status}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, which="both")
    axes[0].set_ylabel("kept detections per hour of dark phase (+0.3)\none dot per trial, bar = median")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("2022 losers have ~18x fewer high-score detections; the winners' mid-score floor is the same",
                 fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def signal_figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = pd.read_csv(SIGNAL / "recordings.csv")
    rec = rec[rec["trial"]]
    panels = [
        (("snr_db_winner", "snr_db_loser"), "fish peak over spectral floor (dB)", "fish SNR (median of 24 segments)"),
        (("n_electrodes_10db_winner", "n_electrodes_10db_loser"), "electrodes within 10 dB of the best",
         "how many electrodes see each fish"),
        (("unexplained_lines_median",), "lines per segment (250-2000 Hz)", "spectral lines no fish explains"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (cols, ylabel, title) in zip(axes, panels, strict=True):
        ticks = []
        for j, ds in enumerate(("2019/2020", "2022")):
            g = rec[rec["dataset"] == ds]
            for i, c in enumerate(cols):
                x0 = j * (len(cols) + 0.6) + i
                y = g[c].to_numpy()
                ax.scatter(x0 + np.random.default_rng(i + 3 * j).uniform(-0.15, 0.15, len(y)), y, s=12,
                           color=COLOURS[ds], alpha=0.7, edgecolors="none")
                ax.hlines(np.nanmedian(y), x0 - 0.25, x0 + 0.25, color="k", lw=2)
                p3 = g["recording"].isin(PAIR3).to_numpy()
                ax.scatter(np.full(p3.sum(), x0 + 0.3), y[p3], s=22, facecolors="none", edgecolors="#d95f0e", lw=1.1)
                name = c.split("_")[-1] if len(cols) > 1 else ""
                ticks.append((x0, f"{ds}\n{name}".strip()))
        ax.set_xticks([t for t, _ in ticks], [s for _, s in ticks], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per trial (one dot each; bar = median; orange ring = 2022 pair 3): in 2022 the fish are as "
                 "strong, but each is seen on fewer electrodes and among more interference lines", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rates_summary(path: Path) -> pd.DataFrame:
    """Median (and 10th / 90th percentile) over trials of kept detections per hour, by dataset, status, score range."""
    r = analysed(pd.read_csv(SIGNAL / "rates.csv"))
    r = r[r["status"].isin(["winner", "loser"])]
    ranges = {"0.5-0.9": ["0.5-0.7", "0.7-0.8", "0.8-0.9"], ">=0.9": ["0.9-1"], ">=0.8": ["0.8-0.9", "0.9-1"],
              "0.7-0.9": ["0.7-0.8", "0.8-0.9"]}
    rows = []
    for (ds, status), g in r.groupby(["dataset", "status"]):
        for name, bins in ranges.items():
            for subset, gg in (("all trials", g), ("without pair 3", g[~g["recording"].isin(PAIR3)])):
                per_trial = gg[gg["score_bin"].isin(bins)].groupby("recording")["per_hour"].sum()
                rows.append({"dataset": ds, "status": status, "scores": name, "trials": subset,
                             "n_trials": len(per_trial), "median_per_hour": per_trial.median(),
                             "p10": per_trial.quantile(0.1), "p90": per_trial.quantile(0.9)})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    s = rates_summary(SIGNAL / "rates_summary.csv")
    print(s.round(1).to_string(index=False))
    rates_figure(SIGNAL / "rates.png")
    signal_figure(SIGNAL / "signal.png")
    print(f"wrote {SIGNAL / 'rates.png'}, {SIGNAL / 'signal.png'}")


if __name__ == "__main__":
    main()
