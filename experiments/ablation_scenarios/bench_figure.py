"""Figure of the breakdown benchmark (bench_analyse.py tables).

    uv run python experiments/ablation_scenarios/bench_figure.py [--arms A B]

Top: per arm, (a) emitter assignment accuracy (labels both arms found) against
the emitter's ΔF to the nearest other fish, (b) recall against the interval
to the neighbouring chirp of the same fish (matching within 25 ms, after
post-processing). Bottom: B - A with the paired scene-bootstrap 95 % CI.
x = the bin's median ΔF / interval (log axes). 17.4 cm wide (the manuscript's full
width), the palette of Figs. 5-7 (indigo, teal), 8 pt text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

ROOT = DATA / "runs/ablation_frontend/scenarios/bench/analysis"
CM = 1 / 2.54
COLORS = {"A": "#2b3a76", "B": "#34858d", "Bc16": "#34858d", "Bc22": "#5ea869"}
LABELS = {
    "A": "A: 3 windows",
    "B": "B: long window only",
    "Bc16": "B, long window, epoch 16",
    "Bc22": "B, long window, epoch 22",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
        }
    )


def panel(ax, x, table, metric, arms, ylabel):
    for k, arm in enumerate(arms):
        y = 100 * table[f"{metric}_{arm}"].to_numpy()
        lo = 100 * table[f"{metric}_{arm}_lo"].to_numpy()
        hi = 100 * table[f"{metric}_{arm}_hi"].to_numpy()
        xs = x * (1 + 0.03 * (k - (len(arms) - 1) / 2))
        ax.errorbar(xs, y, yerr=[y - lo, hi - y], color=COLORS[arm], marker="o", capsize=0, elinewidth=0.8, label=LABELS[arm])
    ax.set_xscale("log")
    ax.set_ylabel(ylabel)


def diff_panel(ax, x, table, metric, arms):
    a, b = arms[0], arms[1]
    d = 100 * table[f"{metric}_d"].to_numpy()
    lo = 100 * table[f"{metric}_d_lo"].to_numpy()
    hi = 100 * table[f"{metric}_d_hi"].to_numpy()
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.errorbar(x, d, yerr=[d - lo, hi - d], color=COLORS[b], marker="o", capsize=0, elinewidth=0.8)
    ax.set_xscale("log")
    short = {"Bc16": "B", "Bc22": "B"}
    ax.set_ylabel(f"{short.get(b, b)} $-$ {short.get(a, a)} (pp)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["A", "B"])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    style()
    df = pd.read_csv(args.root / "sweep_df.csv")
    df = df[(df["by"] == "emitter ΔF") & (df["bin"] != "all")]
    rate = pd.read_csv(args.root / "sweep_rate.csv")
    rate = rate[(rate["by"] == "same-fish interval") & (rate["bin"] != "all") & (rate["tol_ms"] == 25) & (rate["stage"] == "after")]
    rate = rate[np.isfinite(rate["median_dt_ms"])]
    fig, axes = plt.subplots(2, 2, figsize=(17.4 * CM, 10.5 * CM), sharex="col", gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    x_df = df["median_df_hz"].to_numpy()
    panel(axes[0, 0], x_df, df, "assign_both", args.arms, "correct emitter (%)")
    axes[0, 0].axhline(50, color="grey", lw=0.6, ls="--")
    axes[0, 0].text(28, 51, "chance (two fish)", color="grey", fontsize=7, va="bottom", ha="right")
    axes[0, 0].set_title("a  Emitter assignment as two fish approach in EODf")
    axes[0, 0].legend(loc="upper left")
    diff_panel(axes[1, 0], x_df, df, "assign_both", args.arms)
    axes[1, 0].set_xlabel("emitter's ΔF to its nearest other fish (Hz)")
    x_dt = rate["median_dt_ms"].to_numpy()
    panel(axes[0, 1], x_dt, rate, "recall", args.arms, "chirps detected (%)")
    axes[0, 1].axvline(200, color="grey", lw=0.6, ls="--")
    axes[0, 1].text(210, 60, "training\nminimum", color="grey", fontsize=7, va="bottom")
    axes[0, 1].set_title("b  Detection as chirp trains get faster")
    diff_panel(axes[1, 1], x_dt, rate, "recall", args.arms)
    axes[1, 1].set_xlabel("interval to the same fish's nearest chirp (ms)")
    for ax in axes[1]:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1, 0].set_xticks([0.3, 1, 3, 10, 20])
    axes[1, 0].set_xlim(0.22, 30)
    axes[1, 1].set_xticks([75, 100, 150, 200, 300, 500, 1000])
    axes[1, 1].set_xlim(66, 1400)
    out = args.root / f"benchmark_{'_'.join(args.arms)}.png"
    fig.savefig(out, dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()
