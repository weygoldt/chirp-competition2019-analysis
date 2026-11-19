# Ported from chirp_competition d040123 (src_paper/plot_obliterate_receiver_effect.py).
# Changes: inputs from $PAPER_OUT/data/processed, figure to paper_config.figure(); the 'rows affected' percentage (hard-coded 6.1 %, from the v1 data) is computed from obs.csv and also goes to write_stats ("receiver_effect_figure").
# The schematic's window geometry comes from offset_window.py (was drawn for -3, +13 s); the predictor is
# loser_chirped_offset_window (was loser_chirped_fixed_m3p13).
"""
Figure: receiver-effect decomposition (3 panels).

Panel a: chirp-rate landscape on a hybrid time axis — chase duration
         normalised to [0, 1]; time outside the chase in real seconds.
         Analysis windows (chase end, full post-offset) are illustrated
         as horizontal bars above the rate trace. Chase-tail extent is
         shown at the mode of the chase-duration distribution.
Panel b: window-geometry schematic (two successive chases), making the
         clipped-vs-unclipped edge effect visually concrete and marking
         the contamination zone.
Panel c: forest plot of chirp-slope estimates across 11 models grouped
         into primary analysis, marginal window decomposition, and joint
         window decomposition.

Expects:
  slope_summary.csv  — from the stats pipeline; rows keyed by (model,
                       predictor, re, subset).
  normalized_chirp_rate.csv — one row per x on the hybrid axis with
                       columns {x, observed_mean, observed_lo,
                       observed_hi, null_mean, null_lo, null_hi}.
"""

from __future__ import annotations

from pathlib import Path

import os
import json
from dataclasses import dataclass
from pathlib import Path

import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import PROCESSED, figure
from style import (
    letter_subplots,
    set_light_style,
    set_dark_style,  # noqa: F401  — imported for symmetry with other scripts
    winner_color,  # noqa: F401
    loser_color,  # noqa: F401
    baseline_color,
    adjust_alpha,
)


set_light_style()

cm = 1 / 2.54  # for figsize in centimetres


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
filename = Path(__file__).stem
SLOPE_SUMMARY = (
    PROCESSED / "chirp_effect_on_dominant_adaptive_window"
    / "gamma_glmm_binary" / "slope_summary.csv"
)
RATE_DATA = PROCESSED / "loser_chirps_warped_chase" / "normalized_chirp_rate.csv"  # never read
OUT_FIG = figure(f"{filename}.pdf")
OBS_DATA = (
    PROCESSED / "chirp_effect_on_dominant_adaptive_window"
    / "data" / "chirp_effect_on_dominant_obs.csv"
)


def window_disagreement() -> pd.Series:
    """Per chase row: does the post-offset loser-chirp flag differ between the
    clipped and the unclipped window?"""
    obs = pd.read_csv(OBS_DATA)
    return obs["loser_chirped_post_offset_clipped"] != obs["loser_chirped_post_offset_unclipped"]


def contamination_pct() -> float:
    """Share of chase rows whose post-offset flags differ, in percent (was
    hard-coded as 6.1 %: 119 of 1,958 rows of the v1 data)."""
    return 100 * float(window_disagreement().mean())

# PRIMARY_RE = "primary_actor_partner"
PRIMARY_RE = "primary_recording"
SUBSET = "full"

# Panel a: chase duration used to position the (duration-dependent) chase-tail
# window on the hybrid axis. Set to the mode of the chase-duration
# distribution in the dataset; recorded in the figure caption.
MODE_CHASE_DURATION_S = 4.0
# Lengths of the chase-tail and the post-offset window in real seconds: the
# two parts of the offset window (offset_window.py) before and after the offset.
LEFT_S, RIGHT_S = offset_window()
CHASE_TAIL_WINDOW_S = -LEFT_S
POST_OFFSET_WINDOW_S = RIGHT_S

cmap1_ = sns.color_palette("crest_r", as_cmap=True)
loser_color = cmap1_(0.5)
winner_color = adjust_alpha(cmap1_(0.9), 1.2)

# Colour palette (hardcoded hex; swap if you switch to set_dark_style()).
# C_TEXT = "#2C2C2A"
C_TEXT = "k"
# C_NEUTRAL = "#5F5E5A"
C_NEUTRAL = "k"
C_STRUCTURE = "#888780"
# C_POSITIVE = "#0F6E56"  # teal 700
C_POSITIVE = cmap1_(0.5)
# C_POSITIVE_M = "#1D9E75"  # teal 500 (used for the tentatively-positive M3)
C_POSITIVE_M = cmap1_(0.5)
# C_NEGATIVE = "#993C1D"  # coral 700

C_NEGATIVE = "#993C1D"  # coral 700

# C_TEAL_FILL = "#E1F5EE"  # teal 50 (measurement windows)
C_TEAL_FILL = "#E1F5EE"  # teal 50 (measurement windows)
C_CORAL_FILL = "#FAECE7"  # coral 50 (contamination zone, highlight strip)
C_GRAY_FILL = "#F1EFE8"  # gray 50 (chase boxes)


# ---------------------------------------------------------------------------
# Forest plot row specification
# ---------------------------------------------------------------------------
# (model_name, predictor_column, group, display_label, dot/bar colour)
FOREST_ROWS = [
    ("M1_causal", "loser_chirped_offset_window", "primary", "unadjusted", C_NEUTRAL),
    (
        "M2_acausal",
        "loser_chirped_offset_window",
        "primary",
        "acausal (falsification)",
        C_NEUTRAL,
    ),
    (
        "M3_causal_adj_prev",
        "loser_chirped_offset_window",
        "primary",
        "+ prev. ICI",
        C_POSITIVE,
    ),
    (
        "M4_causal_adj_full",
        "loser_chirped_offset_window",
        "primary",
        "+ prev. ICI, + duration",
        C_POSITIVE,
    ),
    ("Mtail", "loser_chirped_chase_tail", "marginal", "chase end only", C_NEUTRAL),
    (
        "Mpost_clipped",
        "loser_chirped_post_offset_clipped",
        "marginal",
        "post-offset, clipped",
        C_POSITIVE,
    ),
    (
        "Mpost_unclipped",
        "loser_chirped_post_offset_unclipped",
        "marginal",
        "post-offset, unclipped",
        C_NEGATIVE,
    ),
    (
        "Mjoint_clipped",
        "loser_chirped_chase_tail",
        "joint",
        "clipped: chase end",
        C_NEUTRAL,
    ),
    (
        "Mjoint_clipped",
        "loser_chirped_post_offset_clipped",
        "joint",
        "clipped: post-offset",
        C_POSITIVE,
    ),
    (
        "Mjoint_unclipped",
        "loser_chirped_chase_tail",
        "joint",
        "unclipped: chase end",
        C_NEUTRAL,
    ),
    (
        "Mjoint_unclipped",
        "loser_chirped_post_offset_unclipped",
        "joint",
        "unclipped: post-offset",
        C_NEGATIVE,
    ),
]

GROUP_LABELS = {
    "primary": "\\textbf{Primary analysis}",
    "marginal": "\\textbf{Decomposition (marginal)}",
    "joint": "\\textbf{Decomposition (joint)}",
}

# Indices into FOREST_ROWS whose rows form the sign-flip contrast pair.
# These rows get the coral highlight strip behind them.
HIGHLIGHT_INDICES = (5, 6)

# Rate-ratio ticks on the multiplier (linear) scale.
RR_TICKS = [0.7, 0.85, 1.0, 1.15, 1.3]
# Multiplier-scale axis limits — a hair outside the most extreme HDI
# endpoints so bars don't touch the axes.
RR_LIMS = (0.58, 1.42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class HybridRates:
    """All three datasets loaded on their native grids (no truncation)."""

    t_onset: np.ndarray
    obs_onset: np.ndarray
    null_onset: np.ndarray

    u: np.ndarray
    obs_warped: np.ndarray
    null_warped: np.ndarray

    t_offset: np.ndarray
    obs_offset: np.ndarray
    null_offset: np.ndarray


def load_hybrid_rates(
    onset_path: Path,
    warped_path: Path,
    offset_path: Path,
) -> HybridRates:
    """Load the three JSONs into a HybridRates container.

    Truncation to the display range happens at plot time, so the caller
    can change T_pre_s / T_post_s without reloading.
    """
    with open(onset_path) as f:
        onset = json.load(f)
    with open(warped_path) as f:
        warped = json.load(f)
    with open(offset_path) as f:
        offset = json.load(f)

    return HybridRates(
        t_onset=np.asarray(onset["conv_time"]),
        obs_onset=np.asarray(onset["jackknifed_eventrate"]),
        null_onset=np.asarray(onset["bootstrapped_baseline_jittered"]),
        u=np.asarray(warped["u_grid"]),
        obs_warped=np.asarray(warped["jackknifed_eventrate"]),
        null_warped=np.asarray(warped["bootstrapped_baseline_jittered"]),
        t_offset=np.asarray(offset["conv_time"]),
        obs_offset=np.asarray(offset["jackknifed_eventrate"]),
        null_offset=np.asarray(offset["bootstrapped_baseline_jittered"]),
    )


def load_slope_summary(path: Path = SLOPE_SUMMARY) -> pd.DataFrame:
    return pd.read_csv(path)


def load_rate_data(path: Path = RATE_DATA) -> pd.DataFrame:
    """Load the hybrid-axis chirp-rate trace for panel a.

    Expected columns:
        x              hybrid-axis position (s pre-onset < 0; 0..1 normalised
                       chase position; x > 1 encodes (x - 1) seconds past
                       offset)
        observed_mean  observed chirp rate (chirps/min) at x
        observed_lo    observed 95 % CI lower bound
        observed_hi    observed 95 % CI upper bound
        null_mean      jitter-null chirp rate at x
        null_lo        null 95 % CI lower bound
        null_hi        null 95 % CI upper bound
    """
    return pd.read_csv(path)


def get_slope(
    df: pd.DataFrame,
    model: str,
    predictor: str,
    re: str = PRIMARY_RE,
    subset: str = SUBSET,
) -> pd.Series:
    """Pick the single slope row matching the (model, predictor, re, subset)."""
    mask = (
        (df["model"] == model)
        & (df["predictor"] == predictor)
        & (df["re"] == re)
        & (df["subset"] == subset)
    )
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(
            f"{len(rows)} rows for {model}/{predictor}/{re}/{subset}; "
            "expected exactly 1."
        )
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
def setup_plot():
    """Build the figure and the three panel axes.

    Three panels stacked vertically. Panel a (rate) and panel b (schematic)
    span most of the figure width; panel c (forest) is narrower in its
    chart area so the left gutter carries row labels and the right gutter
    carries the × ICI / HDI column.
    """

    figsize = (17.4 * cm, 12.0 * cm)
    # fig, axs = plt.subplots(2, 1, figsize=figsize, height_ratios=(0.6, 1.0), dpi=150)
    fig = plt.figure(figsize=figsize, dpi=150)  # eLife two-column
    # axs = axs.flatten()

    # Positions chosen so each panel gets the vertical real estate its
    # content needs: rate (~4.4 cm), schematic (~3.5 cm), forest (~10 cm).
    # ax_rate = fig.add_axes([0.12, 0.78, 0.84, 0.18])
    ax_schematic = fig.add_axes([0.1, 0.7, 0.8, 0.32])
    ax_forest = fig.add_axes([0.25, 0.08, 0.45, 0.58])

    # ax_rate = axs[0]
    # ax_schematic = axs[0]
    # ax_forest = axs[1]

    # Schematic axes: no spines, no ticks (we draw everything by hand).
    ax_schematic.set_xlim(-1.0, 30.0)
    ax_schematic.set_ylim(-0.1, 9.0)

    for sp in ax_schematic.spines.values():
        sp.set_visible(False)
    ax_schematic.set_xticks([])
    ax_schematic.set_yticks([])

    # Rate axes: keep bottom and left spines, trim top and right.
    # for sp in ["top", "right"]:
    #     ax_rate.spines[sp].set_visible(False)

    # Forest axes: keep only the bottom spine (x-axis).
    for sp in ["top", "right", "left"]:
        ax_forest.spines[sp].set_visible(False)
    ax_forest.set_xlim(*RR_LIMS)
    ax_forest.set_yticks([])

    return fig, ax_schematic, ax_forest


# ---------------------------------------------------------------------------
# Panel a: chirp-rate landscape on hybrid axis
# ---------------------------------------------------------------------------
def plot_hybrid_rate(
    ax,
    data: HybridRates,
    T_pre_s: float = 20.0,
    T_post_s: float = 20.0,
    obs_color: str = loser_color,
    null_color: str = baseline_color,
    band_alpha: float = 0.5,
    scale_per_minute: bool = True,
    show_seams: bool = True,
    annotate_ticks: bool = True,
) -> dict:
    """Plot onset/warped/offset rates on a single axis with equal thirds.

    Plot-x layout:
        [0, 1]  pre-chase   ← onset-centered,  t ∈ [-T_pre_s, 0]
        [1, 2]  in-chase    ← warped,          u ∈ [0, 1]
        [2, 3]  post-chase  ← offset-centered, t ∈ [0, T_post_s]

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : HybridRates
    T_pre_s, T_post_s : float
        Display half-windows in seconds for the pre/post regions.
    scale_per_minute : bool
        Multiply rates by 60 so the y-axis reads chirps/min.

    Returns
    -------
    dict
        Layout metadata (seam positions, plot-x extent, T_pre/T_post) so
        downstream code can overlay annotations.
    """
    scale = 60.0 if scale_per_minute else 1.0

    pre = (data.t_onset >= -T_pre_s) & (data.t_onset <= 0.0)
    inc = (data.u >= 0.0) & (data.u <= 1.0)
    post = (data.t_offset >= 0.0) & (data.t_offset <= T_post_s)

    def q(a, mask):
        return np.percentile(a[:, mask], [2.5, 50, 97.5], axis=0) * scale

    obs_pre_q = q(data.obs_onset, pre)
    obs_in_q = q(data.obs_warped, inc)
    obs_post_q = q(data.obs_offset, post)
    null_pre_q = q(data.null_onset, pre)
    null_in_q = q(data.null_warped, inc)
    null_post_q = q(data.null_offset, post)

    seam_on = 1.0
    seam_off = 2.0
    x_end = 3.0

    x_pre = np.linspace(0.0, seam_on, int(pre.sum()))
    x_in = np.linspace(seam_on, seam_off, int(inc.sum()))
    x_post = np.linspace(seam_off, x_end, int(post.sum()))

    x_all = np.concatenate([x_pre, x_in, x_post])
    obs_lo = np.concatenate([obs_pre_q[0], obs_in_q[0], obs_post_q[0]])
    obs_med = np.concatenate([obs_pre_q[1], obs_in_q[1], obs_post_q[1]])
    obs_hi = np.concatenate([obs_pre_q[2], obs_in_q[2], obs_post_q[2]])
    null_lo = np.concatenate([null_pre_q[0], null_in_q[0], null_post_q[0]])
    null_med = np.concatenate([null_pre_q[1], null_in_q[1], null_post_q[1]])
    null_hi = np.concatenate([null_pre_q[2], null_in_q[2], null_post_q[2]])

    ax.fill_between(x_all, null_lo, null_hi, color=null_color, alpha=band_alpha, lw=0)
    ax.plot(x_all, null_med, color=null_color, lw=1, label="null")
    ax.fill_between(x_all, obs_lo, obs_hi, color=obs_color, alpha=band_alpha, lw=0)
    ax.plot(x_all, obs_med, color=obs_color, lw=1, label="observed")

    if show_seams:
        ax.axvline(seam_on, c="k", lw=0.5, ls="--")
        ax.axvline(seam_off, c="k", lw=0.5, ls="--")

    # Not called (the figure has no rate panel any more); the post-offset bar follows the window.
    post_end = seam_off + POST_OFFSET_WINDOW_S / T_post_s * (x_end - seam_off)
    _rounded_rect(ax, (1.1, 2), (7.5, 8), C_TEAL_FILL, C_POSITIVE, zorder=-100)
    _rounded_rect(ax, (2, post_end), (7.5, 8), C_TEAL_FILL, C_POSITIVE, zorder=-100)

    if annotate_ticks:
        ax.set_xticks(
            [
                0.0,
                seam_on / 2,
                seam_on,
                (seam_on + seam_off) / 2,
                seam_off,
                (seam_off + x_end) / 2,
                x_end,
            ]
        )
        ax.set_xticklabels(
            [
                f"-{T_pre_s:g}",
                f"-{T_pre_s / 2:g}",
                "onset",
                "",
                "offset",
                f"+{T_post_s / 2:g}",
                f"+{T_post_s:g}",
            ]
        )

    ax.set_xlim(0, x_end)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Chirps/min)" if scale_per_minute else "Chirp rate (Hz)")

    return {
        "seam_onset": seam_on,
        "seam_offset": seam_off,
        "x_end": x_end,
        "T_pre_s": T_pre_s,
        "T_post_s": T_post_s,
    }


# ---------------------------------------------------------------------------
# Panel b: window-geometry schematic
# ---------------------------------------------------------------------------
def draw_schematic(ax: plt.Axes) -> None:
    # Reference geometry in seconds.
    TIMELINE_Y = 2.4
    CHASE_Y = (1.6, 3.2)
    TAIL_Y = (4.0, 4.55)
    CLIP_Y = (5.0, 5.55)
    UNCLIP_Y = (6.0, 6.55)

    # Chase n ends at OFFSET; its offset window ends at OFFSET_PLUS_RIGHT, and the
    # next chase (6 s long) starts CONTAM_S before that, so that the unclipped
    # window overlaps it (the contamination zone). With the -3, +13 s window
    # this is the original drawing (next chase 18-24 s).
    CHASE_N = (4.0, 10.0)
    OFFSET = 10.0
    CONTAM_S = 5.0
    OFFSET_PLUS_RIGHT = OFFSET + POST_OFFSET_WINDOW_S
    NEXT_ONSET = OFFSET_PLUS_RIGHT - CONTAM_S
    if NEXT_ONSET <= OFFSET:
        raise ValueError(f"the offset window ends {POST_OFFSET_WINDOW_S:g} s after the offset, too "
                         f"close to draw a next chase starting {CONTAM_S:g} s before its end")
    CHASE_NP1 = (NEXT_ONSET, NEXT_ONSET + 6.0)

    # The windows of chase n, clipped as in summarize_submissive_chirp_effect_on_dominant.
    TAIL_SPAN = (max(OFFSET - CHASE_TAIL_WINDOW_S, CHASE_N[0]), OFFSET)
    CLIP_SPAN = (OFFSET, NEXT_ONSET)
    UNCLIP_SHARED_SPAN = (OFFSET, NEXT_ONSET)
    UNCLIP_CONTAM_SPAN = (NEXT_ONSET, OFFSET_PLUS_RIGHT)

    # Timeline
    ax.plot([-0.5, 29.0], [TIMELINE_Y] * 2, color=C_STRUCTURE, lw=0.8, zorder=1)

    # Chase boxes
    for span, label in [(CHASE_N, "chase n"), (CHASE_NP1, "chase n+1")]:
        _rounded_rect(ax, span, CHASE_Y, C_GRAY_FILL, C_NEUTRAL, alpha=1)
        ax.text(
            sum(span) / 2,
            sum(CHASE_Y) / 2,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=C_TEXT,
            zorder=3,
        )

    # Window bars (three rows stacked above timeline)
    _rounded_rect(ax, TAIL_SPAN, TAIL_Y, C_POSITIVE, C_POSITIVE)
    ax.text(
        TAIL_SPAN[1] + 0.3,
        sum(TAIL_Y) / 2,
        "chase end",
        ha="left",
        va="center",
        fontsize=8,
        color=C_TEXT,
    )

    _rounded_rect(ax, CLIP_SPAN, CLIP_Y, C_POSITIVE, C_POSITIVE)
    ax.text(
        CLIP_SPAN[1] + 0.3,
        sum(CLIP_Y) / 2,
        "post-offset, clipped",
        ha="left",
        va="center",
        fontsize=8,
        color=C_TEXT,
    )

    # Unclipped = shared teal part + coral contamination zone.
    _rounded_rect(ax, UNCLIP_SHARED_SPAN, UNCLIP_Y, C_POSITIVE, C_POSITIVE)
    _rounded_rect(ax, UNCLIP_CONTAM_SPAN, UNCLIP_Y, C_NEGATIVE, C_NEGATIVE)
    ax.text(
        UNCLIP_CONTAM_SPAN[1] + 0.3,
        sum(UNCLIP_Y) / 2,
        "post-offset, unclipped",
        ha="left",
        va="center",
        fontsize=8,
        color=C_TEXT,
    )

    # Contamination zone caption, stacked with the proportion annotation.
    zone_mid = sum(UNCLIP_CONTAM_SPAN) / 2
    ax.text(
        zone_mid,
        UNCLIP_Y[1] + 0.95,
        "contamination zone",
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_NEGATIVE,
        fontweight="medium",
    )
    ax.text(
        zone_mid,
        UNCLIP_Y[1] + 0.25,
        f"({contamination_pct():.1f} \\% of rows affected)",
        ha="center",
        va="bottom",
        fontsize=7,
        color=C_NEGATIVE,
        style="italic",
    )

    # Dashed guides from window boundaries down to the timeline.
    for x, y_top in [
        (TAIL_SPAN[0], TAIL_Y[0]),
        (OFFSET, TAIL_Y[0]),
        (NEXT_ONSET, CLIP_Y[0]),
        (OFFSET_PLUS_RIGHT, UNCLIP_Y[0]),
    ]:
        ax.plot(
            [x, x],
            [y_top, TIMELINE_Y - 0.1],
            color=C_STRUCTURE,
            lw=0.8,
            # ls=(0, (1.5, 2)),
            linestyle="dashed",
            alpha=0.55,
            zorder=0.5,
        )

    # Timeline event markers (ticks + labels beneath).
    for x, label in [
        (OFFSET, "offset"),
        (NEXT_ONSET, "next onset"),
        (OFFSET_PLUS_RIGHT, f"previous\noffset + {POST_OFFSET_WINDOW_S:g} s"),
    ]:
        ax.plot(
            [x, x],
            [TIMELINE_Y - 1.0, TIMELINE_Y - 1.25],
            # color=C_STRUCTURE,
            color="k",
            lw=1,
            zorder=1.5,
        )
        ax.text(
            x,
            TIMELINE_Y - 1.45,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color=C_NEUTRAL,
        )


def _rounded_rect(
    ax: plt.Axes,
    xspan: tuple[float, float],
    yspan: tuple[float, float],
    facecolor: str,
    edgecolor: str,
    zorder: float = 2,
    alpha=0.5,
) -> None:
    """Draw a rounded rectangle from (x0, y0) to (x1, y1) on `ax`."""

    rect = mpatches.FancyBboxPatch(
        (xspan[0], yspan[0]),
        xspan[1] - xspan[0],
        yspan[1] - yspan[0],
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.0,
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_patch(rect)
    rect = mpatches.FancyBboxPatch(
        (xspan[0], yspan[0]),
        xspan[1] - xspan[0],
        yspan[1] - yspan[0],
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor=None,
        fill=False,
        edgecolor=edgecolor,
        linewidth=1,
        zorder=zorder,
        alpha=1,
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Panel b: forest plot
# ---------------------------------------------------------------------------
def draw_forest(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Pull slope data for every row.
    slopes = []
    for model, predictor, group, label, color in FOREST_ROWS:
        r = get_slope(df, model, predictor)
        slopes.append(
            {
                "label": label,
                "model_id": model,
                "group": group,
                "color": color,
                "beta": float(r["beta_mean"]),
                "lo": float(r["beta_hdi95_low"]),
                "hi": float(r["beta_hdi95_high"]),
                "rr": float(r["rr_median"]),
                "rr_lo": float(r["rr_hdi95_low"]),
                "rr_hi": float(r["rr_hdi95_high"]),
            }
        )

    # Lay out row y-positions and group-header y-positions.
    # Cursor increments by 1 per row; a new group inserts an extra
    # half-slot for its header.
    y_slots: list[float] = []
    header_slots: dict[str, float] = {}
    cursor = 0.0
    last_group = None
    for s in slopes:
        if s["group"] != last_group:
            header_slots[s["group"]] = cursor + 0.45
            cursor += 1.0
            last_group = s["group"]
        y_slots.append(cursor)
        cursor += 0.7

    # Invert so the first row appears at the top.
    total = cursor
    y_positions = [total - 0.95 - y for y in y_slots]
    header_ys = {g: total - 0.9 - y for g, y in header_slots.items()}

    ax.set_ylim(-1, total)

    # Reference line at RR = 1 (no effect).
    ax.axvline(1.0, color=C_STRUCTURE, lw=0.5, ls="--", zorder=0.5)

    # Highlight strip behind the sign-flip pair.
    if HIGHLIGHT_INDICES is not None:
        y_hi = y_positions[HIGHLIGHT_INDICES[0]]
        y_lo = y_positions[HIGHLIGHT_INDICES[-1]]
        ax.axhspan(y_lo - 0.4, y_hi + 0.4, facecolor=C_CORAL_FILL, alpha=1, zorder=0.25)

    # Blended transform so labels/values sit outside the chart area
    # but remain y-aligned with the rows.
    label_trans = blended_transform_factory(ax.transAxes, ax.transData)

    # Group headers (italicised, slightly heavier).
    for g, y in header_ys.items():
        ax.text(
            -0.02,
            y,
            GROUP_LABELS[g],
            transform=label_trans,
            ha="right",
            va="center",
            fontsize=8,
            color=C_TEXT,
            fontweight="medium",
            style="italic",
        )

    # Right-column headers. The model identifier sits closer to the chart
    # in light grey; the numeric value sits further out in monospace.
    model_key_col = 1.35
    ax.text(
        model_key_col,
        total - 1,
        "model",
        transform=label_trans,
        ha="left",
        va="bottom",
        fontsize=7,
        color=C_STRUCTURE,
        fontweight="medium",
        family="monospace",
    )

    hdi_col_x = 1.02
    ax.text(
        hdi_col_x,
        total - 1,
        "× ICI   [95 \\% HDI]",
        transform=label_trans,
        ha="left",
        va="bottom",
        fontsize=7,
        color=C_NEUTRAL,
        fontweight="medium",
        family="monospace",
    )

    # Data rows: HDI bar + median dot + left label + right value.
    lw_kwargs = dict(lw=0.8, alpha=1, solid_capstyle="round")
    for s, y in zip(slopes, y_positions):
        ax.plot(
            [s["rr_lo"], s["rr_hi"]],
            [y, y],
            color=s["color"],
            **lw_kwargs,
        )
        ax.scatter(
            [s["rr"]],
            [y],
            color=s["color"],
            marker="o",
            edgecolor="k",
            facecolor="w",
            s=9,
            zorder=10,
            lw=0.8,
        )
        ax.text(
            -0.02,
            y,
            s["label"],
            transform=label_trans,
            ha="right",
            va="center",
            fontsize=8.5,
            color=C_TEXT,
        )
        # Model identifier (light grey, monospace; underscores escaped for
        # the LaTeX backend).
        model_id_safe = s["model_id"].replace("_", r"\_")
        ax.text(
            model_key_col,
            y,
            model_id_safe,
            transform=label_trans,
            ha="left",
            va="center",
            fontsize=7.5,
            color=C_STRUCTURE,
            family="monospace",
        )
        rr_text = f"{s['rr']:>4.2f}   [{s['rr_lo']:>4.2f}, {s['rr_hi']:>4.2f}]"
        ax.text(
            hdi_col_x,
            y,
            rr_text,
            transform=label_trans,
            ha="left",
            va="center",
            fontsize=8,
            color=C_NEUTRAL,
            family="monospace",
        )

    # X-axis: linear in RR (multiplier) scale.
    ax.set_xticks(RR_TICKS)
    ax.set_xticklabels([f"{r:g}" for r in RR_TICKS], fontsize=8, color=C_TEXT)
    ax.tick_params(axis="x", length=3, width=0.5, color=C_STRUCTURE, pad=3)
    ax.set_xlabel("Inter-chase-interval multiplier", color=C_TEXT, labelpad=4)
    # ax.set_xlabel("Inter-chasing-interval multiplier", color=C_TEXT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_slope_summary()

    data = load_hybrid_rates(
        onset_path=PROCESSED / "event_time_rates" / "lchirps@agon.json",
        warped_path=PROCESSED / "event_time_rates" / "lchirps@warped_chase.json",
        offset_path=PROCESSED / "event_time_rates" / "lchirps@agoff.json",
    )

    fig, ax_schematic, ax_forest = setup_plot()
    # layout = plot_hybrid_rate(ax_rate, data)
    draw_schematic(ax_schematic)
    draw_forest(ax_forest, df)

    xoffsets = [-0.05, -0.42]
    yoffsets = [0.8, 1.0]
    letter_subplots(xoffset=xoffsets, yoffset=yoffsets)

    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".svg"), dpi=300)
    print(OUT_FIG)
    differs = window_disagreement()
    write_stats("receiver_effect_figure", {"window_disagreement": {
        "n": differs.sum(), "n_rows": len(differs), "fraction": differs.mean()}})
    plt.show()


if __name__ == "__main__":
    main()
