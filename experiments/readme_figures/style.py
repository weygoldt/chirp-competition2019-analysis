"""Figure style of the package READMEs: one light and one dark variant.

Follows the eeltracker and chirp competition figure conventions: Inter, the
eeltracker font sizes, 17 cm width, no top and right spines, the seaborn
crest/mako families for categories. Spectrograms are drawn in one ink
colour whose opacity rises with power, so the low end is transparent and the
figure sits on any page background. GitHub picks the variant with
``<picture>`` and ``prefers-color-scheme``.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CM = 1 / 2.54
FULL_WIDTH_CM = 17.0
DPI = 300

FONT_SIZES = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
}


@dataclass(frozen=True)
class Theme:
    name: str
    ink: str  # text and the spectrogram's colour
    muted: str  # tick labels, secondary text
    faint: str  # spines, ticks, grid electrodes
    fish: tuple[str, ...]  # one colour per fish


def _shade(color, lightness: float, saturation: float = 1.0) -> str:
    r, g, b = mc.to_rgb(color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return mc.to_hex(colorsys.hls_to_rgb(h, lightness, min(1.0, s * saturation)))


# seaborn's "crest" at 0.18, 0.55 and 0.92 (seaborn is not a dependency here)
_FISH_BASE = ["#72b490", "#2c7e8c", "#294079"]

LIGHT = Theme(
    name="light",
    ink="#1f2933",
    muted="#5d6875",
    faint="#c3cad3",
    fish=tuple(_shade(c, 0.42, 1.05) for c in _FISH_BASE),
)
DARK = Theme(
    name="dark",
    ink="#e6edf3",
    muted="#9aa5b1",
    faint="#3a424c",
    fish=tuple(_shade(c, 0.66, 1.1) for c in _FISH_BASE),
)
THEMES = (LIGHT, DARK)


def apply(theme: Theme) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Inter",
            "mathtext.it": "Inter:italic",
            **FONT_SIZES,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titleweight": "medium",
            "axes.titlepad": 4,
            "axes.linewidth": 0.6,
            "axes.edgecolor": theme.faint,
            "axes.labelcolor": theme.muted,
            "axes.facecolor": "none",
            "text.color": theme.ink,
            "xtick.color": theme.faint,
            "ytick.color": theme.faint,
            "xtick.labelcolor": theme.muted,
            "ytick.labelcolor": theme.muted,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "legend.frameon": False,
            "image.origin": "lower",
        }
    )


def ink_cmap(theme: Theme, gamma: float = 1.6) -> ListedColormap:
    """The theme's ink colour with opacity rising from 0 (low) to 1 (high)."""
    rgba = np.tile(np.r_[mc.to_rgb(theme.ink), 1.0], (256, 1))
    rgba[:, 3] = np.linspace(0, 1, 256) ** gamma
    return ListedColormap(rgba)


def figure(height_cm: float):
    return plt.figure(figsize=(FULL_WIDTH_CM * CM, height_cm * CM), dpi=DPI)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
