"""Validation curves of both arms (final/curves.csv of each run).

    uv run python experiments/ablation/plot_curves.py OUT_PNG
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = {
    "A: 3 windows (released)": (DATA / "runs/dualhead_r18_sim_v4_r1", "#2a78d6"),
    "B: long window x3": (DATA / "runs/dualhead_r18_sim_v4_longonly_r1", "#eb6834"),
}
PANELS = [
    ("val/fitness", "fitness (AP x assignment)"),
    ("val/ap", "AP"),
    ("val/f1", "F1 at 0.5"),
    ("val/assign_accuracy", "assignment accuracy"),
    ("val/emitter_mae_hz", "emitter MAE (Hz)"),
]


def main() -> None:
    out = Path(sys.argv[1])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 3.1), constrained_layout=True)
    for name, (run, color) in RUNS.items():
        c = pd.read_csv(run / "final" / "curves.csv")
        best = int(c.loc[c["val/fitness"].idxmax(), "epoch"])
        for ax, (col, title) in zip(axes, PANELS, strict=True):
            ax.plot(c["epoch"], c[col], color=color, lw=2, label=f"{name}, best epoch {best}")
            b = c[c["epoch"] == best]
            ax.plot(b["epoch"], b[col], "o", ms=8, color=color, mec="white", mew=2, zorder=3)
            ax.set_title(title, fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("epoch", fontsize=9)
        ax.grid(color="#e4e3df", lw=0.8)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("sim_v4 val (2,000 tiles, the trainer's selection set), per epoch; dot = selected checkpoint", fontsize=10)
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
