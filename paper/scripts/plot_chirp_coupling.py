# Ported from chirp_competition d040123 (src_paper/plot_chirp_coupling.py).
# Changes: inputs from $PAPER_OUT/data/processed, figure to paper_config.figure(). The window width comes
# from offset_window.py (was the constant 16 s of the -3, +13 s window; columns were *_fixed_m3p13).
"""
Figure: coupling of chirp rate to competition intensity.

Layout:
  Panel a — conditional effect of log(previous ICI) on relative chirp
            rate (ribbon reflects slope uncertainty only; the between-
            fish baseline-rate variance cancels under per-draw grid-mean
            normalisation). Rug of observed predictor values underneath.
            Inset (top-right): KDE + 95 % HDI + posterior-mean multiplier
            per doubling of prev ICI, dashed null at 1.
  Panel b — same structure as a but for chase duration D; inset is
            mounted top-left to avoid the rising curve.
  Panel c — posterior median relative chirp rate over (prev ICI, D)
            from the model.
  Panel d — mean observed chirp rate per 2D bin, rescaled to chirps
            per full offset window (its width, offset_window.py) for
            comparability with the model's window-scale predictions.
            Each row contributes its per-second rate (chirp_count /
            window_width), so clipped-window rows aren't downweighted.
            Bins with no observations are masked.

Expects:
  data/processed/chirp_coupling/negbin/C2_prev_ICI_and_D.nc
  data/processed/chirp_effect_on_dominant_adaptive_window/data/
      chirp_effect_on_dominant_obs.csv
"""

from __future__ import annotations

from pathlib import Path

import os
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde

from offset_window import offset_window
from paper_config import PROCESSED, figure
from style import (
    letter_subplots,
    set_light_style,
    set_dark_style,  # noqa: F401
    winner_color,  # noqa: F401
    loser_color,  # noqa: F401
)

set_light_style()

cm = 1 / 2.54


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEATMAP_ALPH = 1
C2_TRACE = PROCESSED / "chirp_coupling" / "negbin" / "C2_prev_ICI_and_D.nc"
OBS_DATA = (
    PROCESSED / "chirp_effect_on_dominant_adaptive_window"
    / "data" / "chirp_effect_on_dominant_obs.csv"
)


formats = ["pdf", "png", "svg"]
filename = Path(__file__).stem
targetpath = figure(filename)

# The full width of the counting window (offset_window.py) around the chase offset.
LEFT_S, RIGHT_S = offset_window()
WINDOW_WIDTH_S = RIGHT_S - LEFT_S

# Raw-heatmap binning (count of bins per axis; edges computed from data).
RAW_N_BINS_PREV = 12
RAW_N_BINS_D = 12
MIN_N_PER_BIN = 1

# Colour palette.
C_TEXT = "#2C2C2A"
C_NEUTRAL = "#5F5E5A"
C_STRUCTURE = "#888780"
C_PREV = "#185FA5"  # blue 600 — prev-ICI effect
C_PREV_FILL = "#B5D4F4"  # blue 100
C_DUR = "#0F6E56"  # teal 700 — duration effect
C_DUR_FILL = "#E1F5EE"  # teal 50
C_RUG = "#888780"
C_NULL_LINE = "#888780"
C_RAW_MASK = "#E7E5DC"  # gray, for masked bins in panel d

RATE_YLIM = (0.0, 2.5)
CLOUD_XLIM = (0.70, 1.55)

# Heatmap padding: fraction of log-range added beyond the observed min/max
# so the outermost data points don't sit exactly on the axis boundary.
HEATMAP_PAD = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_c2_posterior(path: Path = C2_TRACE) -> az.InferenceData:
    return az.from_netcdf(path)


def load_observations(path: Path = OBS_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "n_loser_chirps_offset_window": "chirp_count",
            "offset_window_duration": "window_width",
            "time_since_previous_chasing": "prev_ICI",
            "chasing_duration": "D",
        }
    )
    df = df.dropna(subset=["chirp_count", "window_width", "prev_ICI", "D"]).copy()
    df = df[df["window_width"] > 0].copy()
    return df


def observed_log_range(
    values: np.ndarray,
    pad: float = HEATMAP_PAD,
) -> tuple[float, float]:
    """Log-range of observed values, padded outward by `pad` of log-span."""
    log_v = np.log10(values)
    lo, hi = log_v.min(), log_v.max()
    span = hi - lo
    return 10 ** (lo - pad * span), 10 ** (hi + pad * span)


# ---------------------------------------------------------------------------
# Posterior summaries on a grid
# ---------------------------------------------------------------------------
def posterior_rate_curve(
    idata: az.InferenceData,
    predictor_grid: np.ndarray,
    predictor_centre: float,
    predictor_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior of *relative* chirp rate on a 1D predictor grid."""
    post = idata.posterior
    b0 = post["Intercept"].values.ravel()
    if predictor_name == "prev_ICI":
        slope = post["log_time_since_previous_chasing_c"].values.ravel()
    elif predictor_name == "D":
        slope = post["log_chasing_duration_c"].values.ravel()
    else:
        raise ValueError(f"unknown predictor {predictor_name!r}")

    log_grid_c = np.log(predictor_grid) - predictor_centre
    eta = b0[:, None] + slope[:, None] * log_grid_c[None, :]
    rate = np.exp(eta)
    rate = rate / rate.mean(axis=1, keepdims=True)

    median = np.quantile(rate, 0.5, axis=0)
    hdi = np.stack([az.hdi(rate[:, i], hdi_prob=0.95) for i in range(rate.shape[1])])
    return median, hdi[:, 0], hdi[:, 1]


def posterior_rate_grid(
    idata: az.InferenceData,
    prev_grid: np.ndarray,
    d_grid: np.ndarray,
    log_prev_centre: float,
    log_d_centre: float,
) -> np.ndarray:
    """2D posterior median relative rate on a (prev, D) grid."""
    post = idata.posterior
    b0 = post["Intercept"].values.ravel()
    b_p = post["log_time_since_previous_chasing_c"].values.ravel()
    b_d = post["log_chasing_duration_c"].values.ravel()

    log_prev_c = np.log(prev_grid) - log_prev_centre
    log_d_c = np.log(d_grid) - log_d_centre

    draws = np.random.default_rng(0).choice(
        len(b0), size=min(len(b0), 500), replace=False
    )
    b0, b_p, b_d = b0[draws], b_p[draws], b_d[draws]

    eta = (
        b0[:, None, None]
        + b_p[:, None, None] * log_prev_c[None, None, :]
        + b_d[:, None, None] * log_d_c[None, :, None]
    )
    rate = np.exp(eta)
    rate = rate / rate.mean(axis=(1, 2), keepdims=True)
    return np.median(rate, axis=0)


# ---------------------------------------------------------------------------
# Raw-data heatmap binning
# ---------------------------------------------------------------------------
def raw_mean_rate_grid(
    obs: pd.DataFrame,
    prev_edges: np.ndarray,
    d_edges: np.ndarray,
    window_width_s: float = WINDOW_WIDTH_S,
    min_n: int = MIN_N_PER_BIN,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Mean observed chirp rate per 2D bin, rescaled to chirps per
    `window_width_s` seconds.

    Each row in `obs` contributes its per-second rate
    (chirp_count / window_width) to the bin's mean, rather than a raw
    count. That makes the bin value rate-based even when individual
    rows have clipped windows (window_width < WINDOW_WIDTH_S), matching
    the offset formulation of the Bayesian model and the colourbar
    label 'mean chirps per {WINDOW_WIDTH_S:.0f}-s window'. Uses
    np.histogram2d so out-of-range observations are dropped cleanly.

    The returned array has shape (n_d, n_p) — first axis chase
    duration, second axis previous ICI — ready for
    `pcolormesh(prev_edges, d_edges, result)`.
    """
    prev = obs["prev_ICI"].values
    d = obs["D"].values
    per_sec_rate = obs["chirp_count"].values / obs["window_width"].values

    counts, _, _ = np.histogram2d(
        d,
        prev,
        bins=[d_edges, prev_edges],
    )
    rate_sums, _, _ = np.histogram2d(
        d,
        prev,
        bins=[d_edges, prev_edges],
        weights=per_sec_rate,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_rate = np.where(counts > 0, rate_sums / counts, np.nan)

    # Rescale per-second rates to chirps per standard window.
    mean_per_window = mean_rate * window_width_s

    mask = counts < min_n
    return np.ma.array(mean_per_window, mask=mask), counts.astype(int)


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
def setup_plot(
    prev_xlim: tuple[float, float],
    d_xlim: tuple[float, float],
) -> dict[str, plt.Axes]:
    fig = plt.figure(figsize=(17.4 * cm, 12.0 * cm), dpi=200)

    right = 0.93

    # Single 2×2 gridspec for the whole figure: prev/d curves on top,
    # heatmap+cbar pairs below, so each bottom cell inherits the column
    # width of the conditional-effect panel above it.
    gs_bm = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=right,
        top=0.95,
        bottom=0.1,
        height_ratios=[1, 1.52],
        wspace=0.40,
        hspace=0.40,
    )
    ax_prev = fig.add_subplot(gs_bm[0, 0])
    ax_d = fig.add_subplot(gs_bm[0, 1])

    sub_l = gs_bm[1, 0].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.04)
    sub_r = gs_bm[1, 1].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.04)

    ax_model = fig.add_subplot(sub_l[0, 0])
    ax_model_cbar = fig.add_subplot(sub_l[0, 1])
    ax_raw = fig.add_subplot(sub_r[0, 0])
    ax_raw_cbar = fig.add_subplot(sub_r[0, 1])

    # Ridge insets: top-right of prev (curve falls rightward, empty there),
    # top-left of d (curve rises rightward, empty on the left).
    ax_ridge_prev = ax_prev.inset_axes([0.5, 0.8, 0.4, 0.2])
    ax_ridge_d = ax_d.inset_axes([0.1, 0.8, 0.4, 0.2])

    for ax in (ax_prev, ax_d):
        ax.set_ylim(*RATE_YLIM)
        ax.set_xscale("log")

    ax_prev.set_xlim(*prev_xlim)
    ax_d.set_xlim(*d_xlim)

    for ax in (ax_model, ax_raw):
        ax.set_xscale("log")
        ax.set_yscale("log")

    return {
        "fig": fig,
        "prev": ax_prev,
        "d": ax_d,
        "ridge_prev": ax_ridge_prev,
        "ridge_d": ax_ridge_d,
        "model": ax_model,
        "raw": ax_raw,
        "model_cbar": ax_model_cbar,
        "raw_cbar": ax_raw_cbar,
    }


# ---------------------------------------------------------------------------
# Panel a / b: conditional-effect curves
# ---------------------------------------------------------------------------
def draw_conditional_effect(
    ax: plt.Axes,
    grid: np.ndarray,
    median_rel: np.ndarray,
    lo_rel: np.ndarray,
    hi_rel: np.ndarray,
    rug_values: np.ndarray,
    color_line: str,
    color_fill: str,
    xlabel: str,
) -> None:
    ax.axhline(1.0, color=C_STRUCTURE, lw=0.5, ls=(0, (2, 2)), zorder=1.5)
    ax.fill_between(
        grid, lo_rel, hi_rel, color=color_fill, alpha=0.3, linewidth=0, zorder=2
    )
    ax.plot(grid, median_rel, color=color_line, lw=1.6, zorder=3)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(
        rug_values,
        ymin=0.0,
        ymax=0.03,
        transform=trans,
        color=C_RUG,
        lw=0.3,
        alpha=0.5,
        zorder=1,
    )

    ax.set_xlabel(xlabel, fontsize=9, color=C_TEXT)
    ax.set_ylabel(r"chirp rate ($\times$ grid mean)", fontsize=9, color=C_TEXT)
    ax.tick_params(axis="both", length=3, width=0.5, color=C_STRUCTURE, labelsize=8)


# ---------------------------------------------------------------------------
# Ridge inset (mounted on the a/b panels)
# ---------------------------------------------------------------------------
def draw_ridge_inset(
    ax: plt.Axes,
    draws: np.ndarray,
    color_line: str,
    color_fill: str,
) -> None:
    """Compact one-row cloud: KDE + 95 % HDI bar + posterior-mean dot,
    dashed null at 1. Intended for an inset on the conditional-effect
    panels."""
    mult = 2**draws  # per-doubling multiplier

    kde_x = np.linspace(*CLOUD_XLIM, 400)
    kde = gaussian_kde(mult, bw_method=0.25)
    dens = kde(kde_x)
    dens = dens / dens.max() * 0.75

    y_row = 1.0
    ax.fill_between(
        kde_x,
        y_row,
        y_row + dens,
        color=color_fill,
        alpha=0.3,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        [1.0, 1.0],
        [0, y_row + np.max(dens)],
        color=C_NULL_LINE,
        lw=0.5,
        ls="dashed",
        zorder=1,
    )

    hdi_lo, hdi_hi = az.hdi(mult, hdi_prob=0.95)
    ax.plot(
        [hdi_lo, hdi_hi],
        [y_row - 0.3, y_row - 0.3],
        color="k",
        lw=0.8,
        alpha=1,
        solid_capstyle="round",
    )
    mean = float(np.median(mult))  # point estimate: posterior median
    ax.scatter(
        [mean],
        [y_row - 0.3],
        marker="o",
        edgecolor="k",
        facecolor="w",
        s=9,
        zorder=10,
        lw=0.8,
    )

    ax.text(
        0.98,
        0.98,
        f"{mean:.2f} [{hdi_lo:.2f}, {hdi_hi:.2f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=C_TEXT,
        family="monospace",
    )

    ax.set_xlim(*CLOUD_XLIM)
    ax.set_ylim(0.45, y_row + 1.5)
    ax.set_yticks([])
    ax.set_xlabel(
        "rate multiplier per doubling",
        fontsize=7,
        color=C_TEXT,
        labelpad=2,
    )
    ax.tick_params(
        axis="x", length=2, width=0.4, color=C_STRUCTURE, labelsize=6.5, pad=1
    )
    ax.set_facecolor("white")


# ---------------------------------------------------------------------------
# Panel c: model heatmap
# ---------------------------------------------------------------------------
def draw_model_heatmap(
    ax: plt.Axes,
    cax: plt.Axes,
    fig: plt.Figure,
    prev_grid: np.ndarray,
    d_grid: np.ndarray,
    rate_grid: np.ndarray,
    obs_prev: np.ndarray,
    obs_d: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    lo_clip = float(np.quantile(rate_grid, 0.02))
    hi_clip = float(np.quantile(rate_grid, 0.98))
    cmap = sns.color_palette("crest", as_cmap=True)
    mesh = ax.pcolormesh(
        prev_grid,
        d_grid,
        rate_grid,
        cmap=cmap,
        shading="auto",
        vmin=lo_clip,
        vmax=hi_clip,
        zorder=1,
        alpha=HEATMAP_ALPH,
        rasterized=True,
    )

    ax.scatter(
        obs_prev,
        obs_d,
        s=2.5,
        facecolor=C_TEXT,
        alpha=1,
        edgecolors="none",
        zorder=2,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("previous inter-chase interval (s)", fontsize=9, color=C_TEXT)
    ax.set_ylabel("chase duration (s)", fontsize=9, color=C_TEXT)
    ax.tick_params(axis="both", length=3, width=0.5, color=C_STRUCTURE, labelsize=8)

    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
    cbar.set_label(
        r"predicted chirp rate ($\times$ grid mean)",
        fontsize=7.5,
        color=C_TEXT,
        labelpad=5,
    )
    cbar.ax.tick_params(length=2, width=0.4, color=C_STRUCTURE, labelsize=7)
    cbar.outline.set_visible(False)


# ---------------------------------------------------------------------------
# Panel d: raw-data heatmap (mean chirp count per window)
# ---------------------------------------------------------------------------
def draw_raw_heatmap(
    ax: plt.Axes,
    cax: plt.Axes,
    fig: plt.Figure,
    mean_counts: np.ma.MaskedArray,
    prev_edges: np.ndarray,
    d_edges: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    cmap = sns.color_palette("crest", as_cmap=True)
    cmap.set_bad(C_RAW_MASK)

    mesh = ax.pcolormesh(
        prev_edges,
        d_edges,
        mean_counts,
        cmap=cmap,
        shading="flat",
        vmin=0,
        vmax=float(np.nanquantile(mean_counts.compressed(), 0.95)),
        zorder=1,
        alpha=HEATMAP_ALPH,
        rasterized=True,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("previous inter-chase interval (s)", fontsize=9, color=C_TEXT)
    ax.set_ylabel("chase duration (s)", fontsize=9, color=C_TEXT)
    ax.tick_params(axis="both", length=3, width=0.5, color=C_STRUCTURE, labelsize=8)

    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
    cbar.set_label(
        f"mean chirp count per {WINDOW_WIDTH_S:.0f}s window",
        fontsize=7.5,
        color=C_TEXT,
        labelpad=5,
    )
    cbar.ax.tick_params(length=2, width=0.4, color=C_STRUCTURE, labelsize=7)
    cbar.outline.set_visible(False)


# ---------------------------------------------------------------------------
# Summary stats for the shell
# ---------------------------------------------------------------------------
def print_summary_stats(
    idata: az.InferenceData,
    obs: pd.DataFrame,
    window_width_s: float = WINDOW_WIDTH_S,
) -> None:
    """Absolute chirp-rate anchors and 10->90 percentile effect sizes."""
    n_windows = len(obs)
    mean_per_window = float(obs["chirp_count"].mean())
    chirps_per_s = mean_per_window / window_width_s
    chirps_per_min = chirps_per_s * 60.0

    print("Dataset anchors:")
    print(f"  n = {n_windows:,} chasing events (windows of {window_width_s:.0f} s)")
    print(f"  mean chirps per chase:   {mean_per_window:.3f}")
    print(f"  mean chirps per minute:  {chirps_per_min:.3f}")
    print(f"  mean chirps per second:  {chirps_per_s:.4f}")

    print("10->90 percentile effect (posterior mean [95% HDI]):")
    for label, col, param in [
        ("prev ICI      ", "prev_ICI", "log_time_since_previous_chasing_c"),
        ("chase duration", "D", "log_chasing_duration_c"),
    ]:
        p10, p90 = np.quantile(obs[col], [0.10, 0.90])
        slope = idata.posterior[param].values.ravel()
        ratio = np.exp(slope * (np.log(p90) - np.log(p10)))
        mean_r = float(np.median(ratio))
        lo, hi = az.hdi(ratio, hdi_prob=0.95)
        print(
            f"  {label}: {p10:.2f} -> {p90:.2f} s  =>  "
            f"x{mean_r:.2f}  [{lo:.2f}, {hi:.2f}]"
        )

    # Absolute expected chirp count per window at the four p10/p90 corners
    # of (prev ICI, D). Uses the model's posterior draws directly rather
    # than applying the multipliers to the empirical grid mean, so the
    # reported uncertainty reflects the joint intercept + slope posterior.
    b0 = idata.posterior["Intercept"].values.ravel()
    b_p = idata.posterior["log_time_since_previous_chasing_c"].values.ravel()
    b_d = idata.posterior["log_chasing_duration_c"].values.ravel()
    log_prev_centre = float(np.log(obs["prev_ICI"]).mean())
    log_d_centre = float(np.log(obs["D"]).mean())

    prev_p10, prev_p90 = np.quantile(obs["prev_ICI"], [0.10, 0.90])
    d_p10, d_p90 = np.quantile(obs["D"], [0.10, 0.90])

    def corner_chirps(p_val: float, d_val: float) -> np.ndarray:
        rate = np.exp(
            b0
            + b_p * (np.log(p_val) - log_prev_centre)
            + b_d * (np.log(d_val) - log_d_centre)
        )
        return rate * window_width_s

    # The two predictors push the rate in opposite directions, so the
    # useful anchors are the "chirp-rich" corner (short prev ICI + long D)
    # and the "chirp-poor" corner (long prev ICI + short D); same-direction
    # diagonals partially cancel and aren't reported.
    chirps_hi = corner_chirps(prev_p10, d_p90)
    chirps_lo = corner_chirps(prev_p90, d_p10)

    def fmt(arr: np.ndarray) -> str:
        m = float(np.median(arr))
        lo, hi = az.hdi(arr, hdi_prob=0.95)
        return f"{m:5.2f}  [{lo:4.2f}, {hi:4.2f}]"

    # Empirical neighbourhoods around each corner: ±1-decile band on
    # each predictor (p5-p15 for "low", p85-p95 for "high") so there are
    # enough observations to summarise while staying close to the corner
    # point the model was evaluated at.
    q_p = np.quantile(obs["prev_ICI"], [0.05, 0.15, 0.85, 0.95])
    q_d = np.quantile(obs["D"], [0.05, 0.15, 0.85, 0.95])
    prev_lo_band, prev_hi_band = (q_p[0], q_p[1]), (q_p[2], q_p[3])
    d_lo_band, d_hi_band = (q_d[0], q_d[1]), (q_d[2], q_d[3])

    def band_stats(prev_band: tuple, d_band: tuple) -> dict:
        mask = obs["prev_ICI"].between(*prev_band) & obs["D"].between(*d_band)
        sub = obs.loc[mask, "chirp_count"].values
        if len(sub) == 0:
            return {"n": 0}
        return {
            "n": len(sub),
            "mean": float(sub.mean()),
            "min": float(sub.min()),
            "p5": float(np.quantile(sub, 0.05)),
            "median": float(np.quantile(sub, 0.50)),
            "p95": float(np.quantile(sub, 0.95)),
            "max": float(sub.max()),
        }

    def emp_fmt(s: dict) -> str:
        if s["n"] == 0:
            return "no observations in band"
        return (
            f"n={s['n']:3d}  mean={s['mean']:.2f}  median={s['median']:.1f}  "
            f"p5={s['p5']:.1f}  p95={s['p95']:.1f}  "
            f"range=[{s['min']:.0f}, {s['max']:.0f}]"
        )

    rich_emp = band_stats(prev_lo_band, d_hi_band)
    poor_emp = band_stats(prev_hi_band, d_lo_band)

    print(
        f"Expected chirps per {window_width_s:.0f}-s window "
        "(model: posterior mean [95% HDI]; empirical: obs in ±1-decile band):"
    )
    print("  chirp-rich corner:")
    print(
        f"    model     (prev={prev_p10:5.2f} s, D={d_p90:5.2f} s):  {fmt(chirps_hi)}"
    )
    print(
        f"    empirical (prev in [{prev_lo_band[0]:.2f},{prev_lo_band[1]:.2f}] s, "
        f"D in [{d_hi_band[0]:.2f},{d_hi_band[1]:.2f}] s):"
    )
    print(f"              {emp_fmt(rich_emp)}")
    print("  chirp-poor corner:")
    print(
        f"    model     (prev={prev_p90:6.2f} s, D={d_p10:5.2f} s):  {fmt(chirps_lo)}"
    )
    print(
        f"    empirical (prev in [{prev_hi_band[0]:.2f},{prev_hi_band[1]:.2f}] s, "
        f"D in [{d_lo_band[0]:.2f},{d_lo_band[1]:.2f}] s):"
    )
    print(f"              {emp_fmt(poor_emp)}")
    model_ratio_str = fmt(chirps_hi / chirps_lo)
    emp_ratio_str = (
        f"x{rich_emp['mean'] / poor_emp['mean']:.2f}"
        if poor_emp.get("mean", 0) > 0
        else "n/a"
    )
    print(
        f"  contrast (rich / poor):  model x{model_ratio_str}   "
        f"empirical {emp_ratio_str}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    idata = load_c2_posterior()
    obs = load_observations()

    print_summary_stats(idata, obs)

    log_prev_centre = float(np.log(obs["prev_ICI"]).mean())
    log_d_centre = float(np.log(obs["D"]).mean())

    # Tighten the model-prediction grids to the observed data range
    # (with the same padding as the heatmap limits). This keeps the
    # posterior_rate_* functions from extrapolating into empty space
    # and matches the heatmap axes.
    prev_lo, prev_hi = observed_log_range(obs["prev_ICI"].values)
    d_lo, d_hi = observed_log_range(obs["D"].values)
    prev_grid = np.logspace(np.log10(prev_lo), np.log10(prev_hi), 200)
    d_grid = np.logspace(np.log10(d_lo), np.log10(d_hi), 200)

    # Raw-heatmap bin edges on the same log-padded range.
    raw_bins_prev = np.logspace(
        np.log10(prev_lo), np.log10(prev_hi), RAW_N_BINS_PREV + 1
    )
    raw_bins_d = np.logspace(np.log10(d_lo), np.log10(d_hi), RAW_N_BINS_D + 1)

    med_prev, lo_prev, hi_prev = posterior_rate_curve(
        idata, prev_grid, log_prev_centre, "prev_ICI"
    )
    med_d, lo_d, hi_d = posterior_rate_curve(idata, d_grid, log_d_centre, "D")
    model_heat = posterior_rate_grid(
        idata, prev_grid, d_grid, log_prev_centre, log_d_centre
    )
    mean_rate_grid, raw_counts = raw_mean_rate_grid(obs, raw_bins_prev, raw_bins_d)
    n_bins_total = raw_counts.size
    n_bins_populated = int((raw_counts >= MIN_N_PER_BIN).sum())
    n_obs_in_bins = int(raw_counts.sum())
    print(
        f"Raw heatmap: {n_bins_populated}/{n_bins_total} bins populated "
        f"(n >= {MIN_N_PER_BIN}); {n_obs_in_bins}/{len(obs)} observations "
        f"fall inside the padded range."
    )

    # Heatmap limits: observed data range with a hair of log-padding.
    heatmap_xlim = (prev_lo, prev_hi)
    heatmap_ylim = (d_lo, d_hi)

    axes = setup_plot(prev_xlim=heatmap_xlim, d_xlim=heatmap_ylim)
    fig = axes["fig"]

    draw_conditional_effect(
        axes["prev"],
        grid=prev_grid,
        median_rel=med_prev,
        lo_rel=lo_prev,
        hi_rel=hi_prev,
        rug_values=obs["prev_ICI"].values,
        color_line=loser_color,
        color_fill=loser_color,
        xlabel="previous inter-chase interval (s)",
    )
    draw_conditional_effect(
        axes["d"],
        grid=d_grid,
        median_rel=med_d,
        lo_rel=lo_d,
        hi_rel=hi_d,
        rug_values=obs["D"].values,
        color_line=loser_color,
        color_fill=loser_color,
        xlabel="chase duration (s)",
    )

    draw_ridge_inset(
        axes["ridge_prev"],
        draws=idata.posterior["log_time_since_previous_chasing_c"].values.ravel(),
        color_line=loser_color,
        color_fill=loser_color,
    )
    draw_ridge_inset(
        axes["ridge_d"],
        draws=idata.posterior["log_chasing_duration_c"].values.ravel(),
        color_line=loser_color,
        color_fill=loser_color,
    )

    draw_model_heatmap(
        axes["model"],
        axes["model_cbar"],
        fig,
        prev_grid=prev_grid,
        d_grid=d_grid,
        rate_grid=model_heat,
        obs_prev=obs["prev_ICI"].values,
        obs_d=obs["D"].values,
        xlim=heatmap_xlim,
        ylim=heatmap_ylim,
    )
    draw_raw_heatmap(
        axes["raw"],
        axes["raw_cbar"],
        fig,
        mean_counts=mean_rate_grid,
        prev_edges=raw_bins_prev,
        d_edges=raw_bins_d,
        xlim=heatmap_xlim,
        ylim=heatmap_ylim,
    )

    axes["d"].set_ylabel("")
    axes["raw"].set_ylabel("")

    # Offset-capped spines. Per-axes because each panel needs a different
    # set of spines kept:
    #   prev / d / model / raw — default (bottom + left)
    #   ridge insets           — bottom only
    #   cbars                  — right only
    for ax in (axes["prev"], axes["d"], axes["model"], axes["raw"]):
        sns.despine(ax=ax, offset=5, trim=False)

    for ax in (axes["ridge_prev"], axes["ridge_d"]):
        sns.despine(ax=ax, left=True, offset=2, trim=False)

    for cax in (axes["model_cbar"], axes["raw_cbar"]):
        sns.despine(ax=cax, left=True, top=True, bottom=True)

    letter_axes = [axes["prev"], axes["d"], axes["model"], axes["raw"]]
    xoffsets = [-0.2, -0.15] * 2
    yoffsets = [1.1, 1.1, 1.0, 1.0]
    letter_subplots(axes=letter_axes, xoffset=xoffsets, yoffset=yoffsets)

    for fmt in formats:
        print(targetpath.with_suffix(f".{fmt}"))
        fig.savefig(
            targetpath.with_suffix(f".{fmt}"),
            dpi=300,
        )

    plt.show()


if __name__ == "__main__":
    main()
