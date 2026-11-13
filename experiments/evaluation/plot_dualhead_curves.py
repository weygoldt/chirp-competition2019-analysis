"""Plot a dual-head fit's epoch curves: losses, detection and emitter metrics.

Usage: uv run python plot_dualhead_curves.py RUN/final/curves.csv OUT.png
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    curves = pd.read_csv(sys.argv[1])
    epoch = curves["epoch"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    ax = axes[0]
    for name in ("vfl", "bbox", "giou", "emitter"):
        ax.plot(epoch, curves[f"train/{name}"], label=name)
    ax.set(title="training loss terms (summed over layers)", xlabel="epoch", yscale="log")
    ax.legend(frameon=False)
    ax = axes[1]
    for name in ("precision", "recall", "f1", "ap"):
        ax.plot(epoch, curves[f"val/{name}"], label=name)
    ax.set(title="val detection (F1 etc. at 0.5)", xlabel="epoch", ylim=(0.5, 1.0))
    ax.legend(frameon=False)
    ax = axes[2]
    ax.plot(epoch, curves["val/emitter_mae_hz"], label="mean")
    ax.plot(epoch, curves["val/emitter_median_ae_hz"], label="median")
    ax.set(title="val emitter |error| (Hz), true positives", xlabel="epoch")
    ax.legend(frameon=False)
    ax = axes[3]
    ax.plot(epoch, curves["val/assign_accuracy"], label="assignment accuracy")
    ax.plot(epoch, curves["val/fitness"], label="fitness = AP x accuracy")
    best = curves["val/fitness"].idxmax()
    ax.axvline(curves.loc[best, "epoch"], color="0.6", lw=0.8, ls="--")
    ax.set(title="val assignment (nearest live track)", xlabel="epoch")
    ax.legend(frameon=False)
    fig.savefig(sys.argv[2], dpi=110)


if __name__ == "__main__":
    main()
