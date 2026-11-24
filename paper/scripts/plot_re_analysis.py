# Ported from chirp_competition d040123 (src_paper/plot_re_analysis.py).
# Changes: inputs from $PAPER_OUT/data/processed, figure to paper_config.figure(); the design summary,
# HDI overlaps and grouping levels of the readout also go to write_stats ("re_analysis"). The predictor is
# loser_chirped_offset_window (offset_window.py; was loser_chirped_fixed_m3p13).
"""
Random-effects justification — figure + terminal readout.

Single-source script that:

  * Prints an 8-section readout from slope_summary.csv,
    variance_summary.csv, and the raw events CSV, focused on whether
    (1|recording) is defensible as the random-effects structure of the
    primary loser-chirp / ICI model.

  * Renders a two-panel supplementary figure showing the same evidence:
    panel a — RR(loser_chirped) across three RE structures; panel b —
    variance components for every RE term in those fits, with the
    HDI lower bound surfaced as the diagnostic column (rows whose HDI
    extends to or below 0.10 are flagged in coral).

The readout runs first so missing cells surface before plotting. If any
cell needed by the figure is missing, the figure step is skipped with a
clear message.

Expects (produced by stats_chirp_affects_chasing.py):
  slope_summary.csv     rows keyed by (model, predictor, re, subset)
  variance_summary.csv  rows keyed by (model, subset, re, re_term)
  chirp_effect_on_dominant_obs.csv  raw events for design-summary stats
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde

from manuscript_stats import write_stats
from paper_config import PROCESSED, figure
from style import (
    letter_subplots,
    set_light_style,
    set_dark_style,  # noqa: F401  — imported for symmetry with other scripts
    winner_color,  # noqa: F401
    loser_color,  # noqa: F401
    baseline_color,  # noqa: F401
    adjust_alpha,  # noqa: F401
)


set_light_style()

cm = 1 / 2.54  # for figsize in centimetres


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
filename = Path(__file__).stem
_GAMMA = PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "gamma_glmm_binary"
SLOPE_SUMMARY = _GAMMA / "slope_summary.csv"
VARIANCE_SUMMARY = _GAMMA / "variance_summary.csv"
EVENTS_DATA = (
    PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data"
    / "chirp_effect_on_dominant_obs.csv"
)
# Directory holding the InferenceData NetCDF files written by the
# stats pipeline. Filenames follow the {re}__{subset}__{model}.nc
# convention. Optional: if a trace is missing, the figure falls back
# to bars-only for that row.
TRACE_DIR = _GAMMA
OUT_FIG = figure(f"{filename}.pdf")


# ---------------------------------------------------------------------------
# Single source of truth: model + RE specs
# ---------------------------------------------------------------------------
# Fixed-effect spec held constant across the comparison; only RE varies.
MODEL = "M4_causal_adj_full"
PREDICTOR = "loser_chirped_offset_window"
SUBSET = "full"


@dataclass(frozen=True)
class RESpec:
    re: str
    short: str  # math label for the figure's row label / panel-a column
    long: str  # math label for panel-b group header
    is_primary: bool
    in_figure: bool = True  # included in the supplementary figure


# Order is narrative: chosen spec first, then richer-but-weakly-identified,
# then minimal fallback. The fallback is kept in RE_SPECS so the readout
# can quote it (the primary-vs-fallback contrast is the cleanest evidence
# that recording-id is absorbing real structure), but it is excluded from
# the figure: panel a's argument is "primary vs. richer crossed model",
# and adding the fallback there only invites a side conversation about
# recording-loser confounding.
RE_SPECS: tuple[RESpec, ...] = (
    RESpec(
        re="primary_recording",
        short=r"$(1\,|\,\mathrm{recording})$",
        long=r"$(1\,|\,\mathrm{recording})$",
        is_primary=True,
    ),
    RESpec(
        re="sensitivity_loserid_winnerid_recording",
        short=(
            r"$(1\,|\,\mathrm{loser}) + (1\,|\,\mathrm{winner}) + "
            r"(1\,|\,\mathrm{recording})$"
        ),
        long=(
            r"$(1\,|\,\mathrm{loser}) + (1\,|\,\mathrm{winner}) + "
            r"(1\,|\,\mathrm{recording})$"
        ),
        is_primary=False,
    ),
    RESpec(
        re="fallback_loser_only",
        short=r"$(1\,|\,\mathrm{loser})$",
        long=r"$(1\,|\,\mathrm{loser})$",
        is_primary=False,
        in_figure=False,
    ),
)

# Display order for sigma terms within each block.
RE_TERM_ORDER = ("recording", "loser_id", "winner_id")
RE_TERM_LABELS = {
    "recording": r"$\sigma_{\mathrm{rec}}$",
    "loser_id": r"$\sigma_{\mathrm{loser}}$",
    "winner_id": r"$\sigma_{\mathrm{winner}}$",
}

# Diagnostic thresholds.
# A variance component whose HDI lower bound is at or below this value is
# considered weakly identified -- the posterior leaves serious mass at zero.
# This is a visual marker only; it does not drive a refusal-to-plot.
HDI_LOW_FLAG = 0.10
RHAT_FLAG = 1.01
ESS_FLAG = 400

# KDE cloud styling. One-sided (top-half) clouds sitting above each row's
# bar with a small gap; the bar + median dot are drawn separately below.
KDE_HEIGHT = 0.30  # peak amplitude as fraction of row spacing
KDE_GAP = 0.05  # vertical offset between the bar and the cloud baseline
KDE_ALPHA = 0.30  # cloud fill alpha
KDE_GRID = 200  # density-evaluation grid resolution

# Axis limits — None means derive from data; pin to a tuple to lock in.
RR_LIMS: tuple[float, float] | None = None
SIGMA_LIMS: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Colour palette (kept consistent with plot_obliterate_receiver_effect.py)
# ---------------------------------------------------------------------------
cmap1_ = sns.color_palette("crest_r", as_cmap=True)
C_TEXT = "k"
C_NEUTRAL = "k"
C_STRUCTURE = "#888780"
C_POSITIVE = cmap1_(0.5)
C_NEGATIVE = "#D55E00"  # Okabe-Ito vermillion; flags weakly-identified variance components


def _spec_color(spec: RESpec) -> str:
    return C_POSITIVE if spec.is_primary else C_NEUTRAL


# ===========================================================================
# Data loading + slicing  (shared by readout and figure)
# ===========================================================================
def load_slope_summary() -> pd.DataFrame:
    if not SLOPE_SUMMARY.exists():
        raise FileNotFoundError(f"missing: {SLOPE_SUMMARY}")
    return pd.read_csv(SLOPE_SUMMARY)


def load_variance_summary() -> pd.DataFrame:
    if not VARIANCE_SUMMARY.exists():
        raise FileNotFoundError(f"missing: {VARIANCE_SUMMARY}")
    return pd.read_csv(VARIANCE_SUMMARY)


def load_events() -> pd.DataFrame | None:
    """Raw events table for design-summary stats. Returns None if the file
    isn't present so the readout still runs against the CSVs alone."""
    if not EVENTS_DATA.exists():
        return None
    return pd.read_csv(EVENTS_DATA)


def slice_slope(df: pd.DataFrame, re: str) -> pd.Series | None:
    """Single slope row for (MODEL, PREDICTOR, re, SUBSET); None if absent."""
    mask = (
        (df["model"] == MODEL)
        & (df["predictor"] == PREDICTOR)
        & (df["re"] == re)
        & (df["subset"] == SUBSET)
    )
    rows = df[mask]
    return rows.iloc[0] if len(rows) == 1 else None


def slice_variance(df: pd.DataFrame, re: str) -> pd.DataFrame:
    """Variance-component rows for (MODEL, re, SUBSET) in display order."""
    mask = (df["model"] == MODEL) & (df["re"] == re) & (df["subset"] == SUBSET)
    sub = df[mask].copy()
    order = {t: i for i, t in enumerate(RE_TERM_ORDER)}
    sub["__ord"] = sub["re_term"].map(lambda t: order.get(t, len(order)))
    return sub.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)


def all_cells_present(slope_df: pd.DataFrame, var_df: pd.DataFrame) -> bool:
    """True iff every figure RE_SPECS row has a slope and at least one sigma.

    Only checks specs with `in_figure=True`; missing fallback rows do not
    block the figure since the figure does not render them.
    """
    for s in RE_SPECS:
        if not s.in_figure:
            continue
        if slice_slope(slope_df, s.re) is None:
            return False
        if not len(slice_variance(var_df, s.re)):
            return False
    return True


# --- trace access (figure-only; readout doesn't need this) ----------------
_TRACE_CACHE: dict[str, az.InferenceData | None] = {}


def load_trace(re_label: str) -> az.InferenceData | None:
    """Load the InferenceData for one (re, SUBSET, MODEL); cache results.
    Returns None if the .nc file is missing so the figure can fall back
    to bars-only for that row."""
    if re_label in _TRACE_CACHE:
        return _TRACE_CACHE[re_label]
    nc_path = TRACE_DIR / f"{re_label}__{SUBSET}__{MODEL}.nc"
    if not nc_path.exists():
        print(
            f"  [warn] trace not found: {nc_path.name}; "
            "row will render without KDE cloud."
        )
        _TRACE_CACHE[re_label] = None
        return None
    _TRACE_CACHE[re_label] = az.from_netcdf(nc_path)
    return _TRACE_CACHE[re_label]


def extract_rr_draws(trace: az.InferenceData) -> np.ndarray | None:
    """Pull the focal predictor's posterior and exponentiate to RR scale.
    Returns a flat 1-D array of draws, or None if the variable is absent."""
    if PREDICTOR not in trace.posterior.data_vars:
        print(f"  [warn] predictor '{PREDICTOR}' not in trace; skipping cloud.")
        return None
    beta = trace.posterior[PREDICTOR].values.ravel()
    return np.exp(beta)


def extract_sigma_draws(
    trace: az.InferenceData, sigma_var_name: str
) -> np.ndarray | None:
    """Pull a variance-component's posterior by exact variable name.
    The name is taken verbatim from variance_summary.csv, which the
    pipeline populates via _find_sigma_var()."""
    if sigma_var_name not in trace.posterior.data_vars:
        print(f"  [warn] sigma var '{sigma_var_name}' not in trace; skipping cloud.")
        return None
    return trace.posterior[sigma_var_name].values.ravel()


def _draw_kde_cloud(
    ax: plt.Axes,
    draws: np.ndarray,
    y_center: float,
    color,
    height: float = KDE_HEIGHT,
    alpha: float = KDE_ALPHA,
    n_grid: int = KDE_GRID,
    gap: float = KDE_GAP,
    xmin: float | None = None,
) -> None:
    """One-sided KDE cloud sitting above the row's bar.

    Density is rendered on the current x-axis range and scaled so the
    peak reaches `height` above `y_center + gap`. With `xmin` set, the
    grid is left-truncated at that value so the cloud cannot extend
    below a hard support boundary (e.g. xmin=0 for sigma).
    """
    if draws is None or draws.size < 2:
        return
    if not np.isfinite(draws).all() or draws.std() == 0:
        return
    x_lo, x_hi = ax.get_xlim()
    if xmin is not None:
        x_lo = max(x_lo, xmin)
    if x_lo >= x_hi:
        return
    kde = gaussian_kde(draws)
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    dens = kde(x_grid)
    peak = float(dens.max())
    if peak == 0.0:
        return
    scaled = dens / peak * height
    base = y_center + gap
    ax.fill_between(
        x_grid,
        base,
        base + scaled,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=2,
    )


# ===========================================================================
# Terminal readout
# ===========================================================================
WIDTH = 118


def _rule(char: str = "=", n: int = WIDTH) -> None:
    print(char * n)


def _section(num: int, title: str) -> None:
    print()
    _rule("=")
    print(f"  {num}. {title}")
    _rule("=")


def _hdi(lo: float, hi: float, fmt: str = "{:>5.2f}") -> str:
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


# --- 1. design summary -----------------------------------------------------
def _replication_profile(df: pd.DataFrame, group_col: str) -> dict[int, int]:
    """For each level of group_col, count distinct recordings; return
    {n_recordings: n_levels_with_that_count}."""
    per_group = df.groupby(group_col)["recording"].nunique()
    return per_group.value_counts().sort_index().to_dict()


def _format_profile(profile: dict[int, int]) -> str:
    """Format {1: 6, 2: 3, 3: 1} as '6x1, 3x2, 1x3'."""
    if not profile:
        return "(no data)"
    return ", ".join(f"{count}x{n}" for n, count in profile.items())


def print_design_summary(events: pd.DataFrame | None) -> dict:
    """Print the design summary; return its numbers (empty without the events table)."""
    _section(1, "Design summary")
    if events is None:
        print(f"  Events file not found at {EVENTS_DATA}.")
        print("  Skipping design summary; downstream sections still run.")
        return {}

    # Restrict to the subset used by the model so the counts here match
    # n_fit reported elsewhere. The events CSV may carry rows that the
    # model dropped; without filtering, replication profiles can look
    # richer than they actually are at the inference stage.
    needed = {"recording", "loser_id", "winner_id"}
    if not needed.issubset(events.columns):
        print(
            f"  Events table missing one of {needed}; got "
            f"{sorted(events.columns)[:8]} ... -- skipping."
        )
        return {}

    n_rec = events["recording"].nunique()
    n_losers = events["loser_id"].nunique()
    n_winners = events["winner_id"].nunique()
    loser_profile = _replication_profile(events, "loser_id")
    winner_profile = _replication_profile(events, "winner_id")

    n_losers_replicated = sum(v for k, v in loser_profile.items() if k >= 2)
    n_winners_replicated = sum(v for k, v in winner_profile.items() if k >= 2)

    print(f"  Recordings: {n_rec}")
    print(
        f"  Losers:     {n_losers}  "
        f"(replication profile: {_format_profile(loser_profile)})"
    )
    print(
        f"  Winners:    {n_winners}  "
        f"(replication profile: {_format_profile(winner_profile)})"
    )
    print()
    print(
        "  Subjects contributing within-subject replication "
        "(appear in >= 2 recordings):"
    )
    print(f"    losers:  {n_losers_replicated} / {n_losers}")
    print(f"    winners: {n_winners_replicated} / {n_winners}")
    print()
    print(
        f"  Reading note: with {n_rec} recordings and "
        f"{n_losers_replicated + n_winners_replicated} multiply-recorded "
        f"subjects across both roles, individual-level variance components "
        f"are estimated from a small replication base."
    )
    return {
        "n_recordings": n_rec,
        "losers": {"n": n_losers, "n_replicated": n_losers_replicated,
                   "max_recordings": max(loser_profile)},
        "winners": {"n": n_winners, "n_replicated": n_winners_replicated,
                    "max_recordings": max(winner_profile)},
    }


# --- 2. cell audit ---------------------------------------------------------
def print_cell_audit(slope_df: pd.DataFrame, var_df: pd.DataFrame) -> None:
    _section(2, "Cell audit")
    print(f"  Model:     {MODEL}")
    print(f"  Predictor: {PREDICTOR}")
    print(f"  Subset:    {SUBSET}")
    print()
    print(f"  {'RE structure':<44}  {'slope row':>10}  {'sigma rows':>12}")
    print(f"  {'-' * 44}  {'-' * 10}  {'-' * 12}")
    for spec in RE_SPECS:
        s = slice_slope(slope_df, spec.re)
        v = slice_variance(var_df, spec.re)
        slope_state = "OK" if s is not None else "MISSING"
        var_state = f"{len(v)}" if len(v) else "MISSING"
        print(f"  {spec.re:<44}  {slope_state:>10}  {var_state:>12}")


# --- 3. fixed-effect estimates ---------------------------------------------
def print_fixed_effect_table(slope_df: pd.DataFrame) -> None:
    _section(3, "Fixed-effect estimates: rate-ratio of loser_chirped")
    print(
        f"  {'RE structure':<44}  {'n_fit':>5}  "
        f"{'beta':>7}  {'beta HDI95':>16}  "
        f"{'RR':>5}  {'RR HDI95':>14}  {'P(b<0)':>7}"
    )
    print("  " + "-" * (WIDTH - 2))
    for spec in RE_SPECS:
        s = slice_slope(slope_df, spec.re)
        if s is None:
            print(f"  {spec.re:<44}  -- missing --")
            continue
        print(
            f"  {spec.re:<44}  {int(s['n_fit']):>5d}  "
            f"{s['beta_mean']:>+7.3f}  "
            f"{_hdi(s['beta_hdi95_low'], s['beta_hdi95_high'], '{:>+5.2f}'):>16}  "
            f"{s['rr_median']:>5.2f}  "
            f"{_hdi(s['rr_hdi95_low'], s['rr_hdi95_high'], '{:>4.2f}'):>14}  "
            f"{s['prob_beta_lt_0']:>7.3f}"
        )


# --- 4. fixed-effect robustness --------------------------------------------
def print_fixed_effect_robustness(slope_df: pd.DataFrame) -> dict:
    """Print the robustness readout; return the pairwise HDI overlaps (fractions)."""
    _section(4, "Fixed-effect robustness across RE structures")
    rrs = []
    for spec in RE_SPECS:
        s = slice_slope(slope_df, spec.re)
        if s is None:
            continue
        rrs.append(
            (
                spec.re,
                float(s["rr_median"]),
                float(s["rr_hdi95_low"]),
                float(s["rr_hdi95_high"]),
            )
        )
    if len(rrs) < 2:
        print("  Not enough fits to compare.")
        return {}

    overlaps = {}
    medians = [r[1] for r in rrs]
    rr_min, rr_max = min(medians), max(medians)
    print(
        f"  Median RR range across {len(rrs)} RE structures: "
        f"{rr_min:.3f} -- {rr_max:.3f}  (spread {rr_max - rr_min:.3f})"
    )
    print()
    print("  Pairwise HDI overlap (fraction of shorter HDI inside the other):")
    for i, (re_i, _m_i, lo_i, hi_i) in enumerate(rrs):
        for re_j, _m_j, lo_j, hi_j in rrs[i + 1 :]:
            ov_lo = max(lo_i, lo_j)
            ov_hi = min(hi_i, hi_j)
            ov = max(ov_hi - ov_lo, 0.0)
            shorter = min(hi_i - lo_i, hi_j - lo_j)
            frac = ov / shorter if shorter > 0 else float("nan")
            print(f"    {re_i:<44} vs  {re_j:<44}  {frac:>5.0%}")
            overlaps[f"{re_i}__vs__{re_j}"] = frac
    return {"hdi_overlap": overlaps}


# --- 5. variance components ------------------------------------------------
def print_variance_summary(var_df: pd.DataFrame) -> dict:
    """Print the variance components; return the levels of each grouping term."""
    _section(5, "Variance components (sigma on linear-predictor scale)")
    levels: dict = {}
    print(
        f"  {'RE structure':<44}  {'term':<10}  {'lvls':>4}  "
        f"{'sigma':>5}  {'sigma HDI95':>14}  {'HDI low':>7}"
    )
    print("  " + "-" * (WIDTH - 2))
    last_re = None
    for spec in RE_SPECS:
        sub = slice_variance(var_df, spec.re)
        if not len(sub):
            print(f"  {spec.re:<44}  -- missing --")
            continue
        for _, r in sub.iterrows():
            if last_re is not None and spec.re != last_re:
                print("  " + "-" * (WIDTH - 2))
            last_re = spec.re
            levels.setdefault(spec.re, {})[r["re_term"]] = int(r["n_levels"])
            lo_marker = " <-- weak" if r["sigma_hdi95_low"] <= HDI_LOW_FLAG else ""
            print(
                f"  {spec.re:<44}  {r['re_term']:<10}  "
                f"{int(r['n_levels']):>4d}  "
                f"{r['sigma_median']:>5.2f}  "
                f"{_hdi(r['sigma_hdi95_low'], r['sigma_hdi95_high'], '{:>4.2f}'):>14}  "
                f"{r['sigma_hdi95_low']:>7.2f}"
                f"{lo_marker}"
            )
    return {"n_levels": levels}


# --- 6. identification diagnostics -----------------------------------------
def print_identification_summary(var_df: pd.DataFrame) -> None:
    _section(6, "Identification diagnostics for variance components")
    print("  Variance components by HDI lower bound:")
    print(
        f"    HDI low > {HDI_LOW_FLAG:.2f}        : "
        "well-identified (mass clearly away from zero)"
    )
    print(
        f"    HDI low <= {HDI_LOW_FLAG:.2f}       : "
        "weakly identified (HDI extends to or near zero)"
    )
    print()
    well_id: list[tuple[str, str, float, float]] = []
    weakly_id: list[tuple[str, str, float, float]] = []
    for spec in RE_SPECS:
        sub = slice_variance(var_df, spec.re)
        for _, r in sub.iterrows():
            entry = (
                spec.re,
                r["re_term"],
                float(r["sigma_hdi95_low"]),
                float(r["sigma_hdi95_high"]),
            )
            (weakly_id if r["sigma_hdi95_low"] <= HDI_LOW_FLAG else well_id).append(
                entry
            )

    print("  Well-identified:")
    if not well_id:
        print("    none")
    for re, term, lo, hi in well_id:
        print(f"    {re:<44}  {term:<10}  HDI95=[{lo:>4.2f}, {hi:>4.2f}]")
    print()
    print("  Weakly identified (HDI extends to or near zero):")
    if not weakly_id:
        print("    none")
    for re, term, lo, hi in weakly_id:
        print(f"    {re:<44}  {term:<10}  HDI95=[{lo:>4.2f}, {hi:>4.2f}]")


# --- 7. sampling diagnostics -----------------------------------------------
def print_sampling_diagnostics(slope_df: pd.DataFrame) -> None:
    _section(7, "Sampling diagnostics")
    print(
        f"  {'RE structure':<44}  "
        f"{'div':>4}  {'max R-hat':>9}  "
        f"{'min ESS bulk':>12}  {'min ESS tail':>12}  {'flag':>6}"
    )
    print("  " + "-" * (WIDTH - 2))
    for spec in RE_SPECS:
        s = slice_slope(slope_df, spec.re)
        if s is None:
            continue
        div = int(s["diag_divergent"])
        rhat = float(s["diag_max_rhat"])
        ess_b = float(s["diag_min_ess_bulk"])
        ess_t = float(s["diag_min_ess_tail"])
        flag = div > 0 or rhat > RHAT_FLAG or ess_b < ESS_FLAG or ess_t < ESS_FLAG
        flag_str = " [!]" if flag else "  ok"
        print(
            f"  {spec.re:<44}  {div:>4d}  {rhat:>9.4f}  "
            f"{ess_b:>12.0f}  {ess_t:>12.0f}  {flag_str:>6}"
        )


# --- 8. key numbers --------------------------------------------------------
def print_key_numbers(slope_df: pd.DataFrame, var_df: pd.DataFrame) -> None:
    _section(8, "Key numbers for the Methods paragraph")
    s_primary = slice_slope(slope_df, "primary_recording")
    s_sens = slice_slope(slope_df, "sensitivity_loserid_winnerid_recording")
    s_fall = slice_slope(slope_df, "fallback_loser_only")

    if s_primary is not None:
        print("  Primary fit (recording-only RE):")
        print(f"    RR median  = {s_primary['rr_median']:.3f}")
        print(
            f"    RR HDI95   = [{s_primary['rr_hdi95_low']:.3f}, "
            f"{s_primary['rr_hdi95_high']:.3f}]"
        )
        print(f"    n_fit      = {int(s_primary['n_fit'])}")
    print()

    medians = [
        float(r["rr_median"]) for r in (s_primary, s_sens, s_fall) if r is not None
    ]
    if len(medians) >= 2:
        spread = max(medians) - min(medians)
        print(
            f"  Spread of median RR across the {len(medians)} RE structures: "
            f"{spread:.3f}"
        )
        if s_primary is not None:
            ref = float(s_primary["rr_median"])
            print(
                f"  Max abs deviation from primary: "
                f"{max(abs(m - ref) for m in medians):.3f}"
            )
    print()

    sub_sens = slice_variance(var_df, "sensitivity_loserid_winnerid_recording")
    if len(sub_sens):
        print("  Variance components in the fully crossed sensitivity fit:")
        for _, r in sub_sens.iterrows():
            print(
                f"    sigma_{r['re_term']:<10}  "
                f"median={r['sigma_median']:>5.2f}  "
                f"HDI95=[{r['sigma_hdi95_low']:>4.2f}, "
                f"{r['sigma_hdi95_high']:>4.2f}]  "
                f"levels={int(r['n_levels'])}"
            )
    print()

    # Recording-vs-fallback comparison: the cleanest evidence that
    # recording-id is doing real work. Drop it and loser-id absorbs the
    # structure; the fixed effect attenuates.
    sub_primary = slice_variance(var_df, "primary_recording")
    sub_fallback = slice_variance(var_df, "fallback_loser_only")
    if (
        len(sub_primary)
        and len(sub_fallback)
        and s_primary is not None
        and s_fall is not None
    ):
        sigma_rec_primary = sub_primary[sub_primary["re_term"] == "recording"].iloc[0]
        sigma_loser_fallback = sub_fallback[sub_fallback["re_term"] == "loser_id"].iloc[
            0
        ]

        print(
            "  Recording-vs-fallback comparison "
            "(evidence that recording-id absorbs real structure):"
        )
        print(
            f"    primary  : sigma_recording = "
            f"{sigma_rec_primary['sigma_median']:.2f}, "
            f"RR = {s_primary['rr_median']:.2f} "
            f"[{s_primary['rr_hdi95_low']:.2f}, "
            f"{s_primary['rr_hdi95_high']:.2f}]"
        )
        print(
            f"    fallback : sigma_loser     = "
            f"{sigma_loser_fallback['sigma_median']:.2f}, "
            f"RR = {s_fall['rr_median']:.2f} "
            f"[{s_fall['rr_hdi95_low']:.2f}, "
            f"{s_fall['rr_hdi95_high']:.2f}]"
        )
        d_sigma = float(sigma_loser_fallback["sigma_median"]) - float(
            sigma_rec_primary["sigma_median"]
        )
        d_rr = float(s_fall["rr_median"]) - float(s_primary["rr_median"])
        print(
            f"    Removing recording-id: sigma absorbed by loser-id shifts "
            f"by {d_sigma:+.2f}; RR shifts by {d_rr:+.2f}."
        )


def print_full_readout(
    slope_df: pd.DataFrame,
    var_df: pd.DataFrame,
    events: pd.DataFrame | None,
) -> dict:
    """Print the readout; return the numbers of sections 1, 4 and 5 (for write_stats)."""
    stats = {"design": print_design_summary(events)}
    print_cell_audit(slope_df, var_df)
    print_fixed_effect_table(slope_df)
    stats.update(print_fixed_effect_robustness(slope_df))
    stats.update(print_variance_summary(var_df))
    print_identification_summary(var_df)
    print_sampling_diagnostics(slope_df)
    print_key_numbers(slope_df, var_df)
    print()
    return stats


# ===========================================================================
# Figure
# ===========================================================================
def _pad_lims(lo: float, hi: float, frac: float = 0.06) -> tuple[float, float]:
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    pad = span * frac
    return (lo - pad, hi + pad)


def _nice_ticks(
    lims: tuple[float, float], target_n: int = 5, anchor: float | None = None
) -> list[float]:
    """~target_n round ticks spanning lims, optionally including anchor."""
    lo, hi = lims
    raw_step = (hi - lo) / max(target_n - 1, 1)
    mag = 10 ** np.floor(np.log10(raw_step))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw_step:
            break
    start = np.floor(lo / step) * step
    ticks = list(np.arange(start, hi + step / 2, step))
    if anchor is not None and not any(abs(t - anchor) < step / 4 for t in ticks):
        ticks.append(anchor)
    ticks = sorted(t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9)
    return ticks


def _derive_axis_limits(
    slope_df: pd.DataFrame, var_df: pd.DataFrame
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Derive RR (panel a) and sigma (panel b) axis limits.

    Prefer trace-based percentiles (0.5%--99.5%) so KDE cloud tails are
    visible; fall back to HDI bounds when traces are missing.
    """
    fig_specs = [s for s in RE_SPECS if s.in_figure]

    if RR_LIMS is not None:
        rr_lims = RR_LIMS
    else:
        rr_lo_candidates: list[float] = []
        rr_hi_candidates: list[float] = []
        for s in fig_specs:
            trace = load_trace(s.re)
            draws = extract_rr_draws(trace) if trace is not None else None
            if draws is not None:
                lo, hi = np.percentile(draws, [0.5, 99.5])
            else:
                row = slice_slope(slope_df, s.re)
                lo, hi = float(row["rr_hdi95_low"]), float(row["rr_hdi95_high"])
            rr_lo_candidates.append(float(lo))
            rr_hi_candidates.append(float(hi))
        rr_lo = min(rr_lo_candidates + [0.95])
        rr_hi = max(rr_hi_candidates + [1.05])
        rr_lims = _pad_lims(rr_lo, rr_hi, frac=0.10)

    if SIGMA_LIMS is not None:
        sigma_lims = SIGMA_LIMS
    else:
        sigma_hi = 0.0
        for s in fig_specs:
            trace = load_trace(s.re)
            sub = slice_variance(var_df, s.re)
            for _, r in sub.iterrows():
                draws = (
                    extract_sigma_draws(trace, r["sigma_var_name"])
                    if trace is not None
                    else None
                )
                if draws is not None:
                    sigma_hi = max(sigma_hi, float(np.percentile(draws, 99.5)))
                else:
                    sigma_hi = max(sigma_hi, float(r["sigma_hdi95_high"]))
        sigma_lims = _pad_lims(0.0, max(sigma_hi, 0.5), frac=0.06)

    return rr_lims, sigma_lims


def setup_plot(rr_lims: tuple[float, float], sigma_lims: tuple[float, float]):
    """Two stacked panels with shared left + right gutters."""
    figsize = (17.4 * cm, 10.0 * cm)  # eLife two-column
    fig = plt.figure(figsize=figsize, dpi=150)

    left = 0.34
    right = 0.62
    width = right - left

    # Larger gap between panels A and B than before (0.12 -> 0.20).
    ax_fixed = fig.add_axes([left, 0.70, width, 0.22])
    ax_var = fig.add_axes([left, 0.10, width, 0.40])

    for sp in ["top", "right", "left"]:
        ax_fixed.spines[sp].set_visible(False)
        ax_var.spines[sp].set_visible(False)

    ax_fixed.set_xlim(*rr_lims)
    ax_fixed.set_yticks([])
    ax_var.set_xlim(*sigma_lims)
    ax_var.set_yticks([])

    return fig, ax_fixed, ax_var


def _diagnostics_text_fixed(r: pd.Series) -> str:
    return (
        f"div={int(r['diag_divergent']):>3d}  "
        f"$\\hat{{R}}$={r['diag_max_rhat']:.3f}  "
        f"ESS={r['diag_min_ess_bulk']:>4.0f}"
    )


def draw_fixed_forest(ax: plt.Axes, slope_df: pd.DataFrame) -> None:
    rows = []
    for spec in RE_SPECS:
        if not spec.in_figure:
            continue
        r = slice_slope(slope_df, spec.re)
        if r is None:
            raise RuntimeError(
                f"slope row for {spec.re} missing -- rerun pipeline or fix paths."
            )
        trace = load_trace(spec.re)
        draws = extract_rr_draws(trace) if trace is not None else None
        rows.append(
            {
                "label": spec.short,
                "color": _spec_color(spec),
                "rr": float(r["rr_median"]),
                "rr_lo": float(r["rr_hdi95_low"]),
                "rr_hi": float(r["rr_hdi95_high"]),
                "diag": _diagnostics_text_fixed(r),
                "draws": draws,
            }
        )

    n = len(rows)
    y_positions = list(range(n - 1, -1, -1))
    ax.set_ylim(-0.7, n - 0.1)

    ax.axvline(1.0, color=C_STRUCTURE, lw=0.5, ls="--", zorder=0.5)

    label_trans = blended_transform_factory(ax.transAxes, ax.transData)

    y_hdr = n - 0.15
    hdi_col_x = 1.02
    diag_col_x = 1.60
    ax.text(
        -0.02,
        y_hdr,
        r"\textbf{Random-effects structure}",
        transform=label_trans,
        ha="right",
        va="bottom",
        fontsize=8,
        color=C_TEXT,
        style="italic",
    )
    ax.text(
        hdi_col_x,
        y_hdr,
        r"$\times$ ICI [95\,\% HDI]",
        transform=label_trans,
        ha="left",
        va="bottom",
        fontsize="small",
        color=C_NEUTRAL,
        family="monospace",
    )
    ax.text(
        diag_col_x,
        y_hdr,
        r"sampling diagnostics",
        transform=label_trans,
        ha="left",
        va="bottom",
        fontsize="small",
        color=C_STRUCTURE,
        family="monospace",
    )

    lw_kwargs = dict(lw=0.8, alpha=1, solid_capstyle="round")
    for r, y in zip(rows, y_positions):
        # Cloud first so the bar + dot land on top.
        _draw_kde_cloud(ax, r["draws"], y, r["color"])
        ax.plot([r["rr_lo"], r["rr_hi"]], [y, y], color=r["color"], **lw_kwargs)
        ax.scatter(
            [r["rr"]],
            [y],
            color=r["color"],
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
            r["label"],
            transform=label_trans,
            ha="right",
            va="center",
            fontsize=8,  # row labels: 8 pt, as tick labels
            color=C_TEXT,
        )
        rr_text = f"{r['rr']:>4.2f}   [{r['rr_lo']:>4.2f}, {r['rr_hi']:>4.2f}]"
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
        ax.text(
            diag_col_x,
            y,
            r["diag"],
            transform=label_trans,
            ha="left",
            va="center",
            fontsize="small",
            color=C_STRUCTURE,
            family="monospace",
        )

    rr_ticks = _nice_ticks(ax.get_xlim(), target_n=5, anchor=1.0)
    ax.set_xticks(rr_ticks)
    ax.set_xticklabels([f"{r:g}" for r in rr_ticks], fontsize=8, color=C_TEXT)
    ax.tick_params(axis="x", length=3, width=0.5, color=C_STRUCTURE, pad=3)
    ax.set_xlabel("Loser chirp effect on ICI", color=C_TEXT, labelpad=4)


def draw_variance_forest(ax: plt.Axes, var_df: pd.DataFrame) -> None:
    """One row per (RE spec × RE term); specs grouped under italic headers.

    Diagnostic column: number of group levels and the sigma HDI lower
    bound. Rows whose lower bound is at or below HDI_LOW_FLAG are rendered
    in coral; rows whose HDI runs to zero get an additional coral tick at
    sigma=0 on the bar itself.
    """
    blocks = []  # list of (spec, list-of-row-dicts)
    for spec in RE_SPECS:
        if not spec.in_figure:
            continue
        sub = slice_variance(var_df, spec.re)
        trace = load_trace(spec.re)
        rs = []
        for _, row in sub.iterrows():
            draws = (
                extract_sigma_draws(trace, row["sigma_var_name"])
                if trace is not None
                else None
            )
            rs.append(
                {
                    "term": row["re_term"],
                    "label": RE_TERM_LABELS.get(row["re_term"], row["re_term"]),
                    "color": _spec_color(spec),
                    "med": float(row["sigma_median"]),
                    "lo": float(row["sigma_hdi95_low"]),
                    "hi": float(row["sigma_hdi95_high"]),
                    "n_levels": int(row["n_levels"]),
                    "draws": draws,
                }
            )
        blocks.append((spec, rs))

    HEADER_SLOT = 0.55
    ROW_STEP = 0.7
    BLOCK_GAP = 0.45

    y_slots: list[float] = []
    header_ys_raw: list[float] = []
    cursor = 0.0
    for i, (_spec, rs) in enumerate(blocks):
        if i > 0:
            cursor += BLOCK_GAP
        header_ys_raw.append(cursor + 0.1)
        cursor += HEADER_SLOT
        for _ in rs:
            y_slots.append(cursor)
            cursor += ROW_STEP
    total = cursor

    y_positions = [total - 0.35 - y for y in y_slots]
    header_ys = [total - 0.35 - y for y in header_ys_raw]
    ax.set_ylim(-0.7, total + 0.1)

    ax.axvline(0.0, color=C_STRUCTURE, lw=0.5, ls="--", zorder=0.5)

    label_trans = blended_transform_factory(ax.transAxes, ax.transData)

    y_hdr = total - 0.05
    hdi_col_x = 1.02
    diag_col_x = 1.60
    ax.text(
        -0.02,
        y_hdr,
        r"\textbf{Variance components}",
        transform=label_trans,
        ha="right",
        va="bottom",
        fontsize=8,
        color=C_TEXT,
        style="italic",
    )
    ax.text(
        hdi_col_x,
        y_hdr,
        r"$\sigma$ [95\,\% HDI]",
        transform=label_trans,
        ha="left",
        va="bottom",
        fontsize="small",
        color=C_NEUTRAL,
        family="monospace",
    )
    ax.text(
        diag_col_x,
        y_hdr,
        r"levels   HDI low",
        transform=label_trans,
        ha="left",
        va="bottom",
        fontsize="small",
        color=C_STRUCTURE,
        family="monospace",
    )

    for (spec, _rs), y in zip(blocks, header_ys):
        ax.text(
            -0.02,
            y,
            spec.long,
            transform=label_trans,
            ha="right",
            va="center",
            fontsize=8,
            color=C_TEXT,
            style="italic",
        )

    lw_kwargs = dict(lw=0.8, alpha=1, solid_capstyle="round")
    flat_rows = [r for _spec, rs in blocks for r in rs]
    for r, y in zip(flat_rows, y_positions):
        # Cloud first so the bar + dot land on top.
        # xmin=0.0 because sigma is non-negative.
        _draw_kde_cloud(ax, r["draws"], y, r["color"], xmin=0.0)
        ax.plot([r["lo"], r["hi"]], [y, y], color=r["color"], **lw_kwargs)
        ax.scatter(
            [r["med"]],
            [y],
            color=r["color"],
            marker="o",
            edgecolor="k",
            facecolor="w",
            s=9,
            zorder=10,
            lw=0.8,
        )

        # Coral tick at sigma=0 when the HDI extends to (practically) zero.
        # Sampled lower bounds are small but never exactly 0 (e.g. 4e-4).
        if r["lo"] <= 0.01:
            ax.scatter(
                [0.0], [y], color=C_NEGATIVE, marker="|", s=24, zorder=11, lw=1.2
            )

        ax.text(
            -0.02,
            y,
            r["label"],
            transform=label_trans,
            ha="right",
            va="center",
            fontsize=8,  # row labels: 8 pt, as tick labels
            color=C_TEXT,
        )
        sigma_text = f"{r['med']:>4.2f}   [{r['lo']:>4.2f}, {r['hi']:>4.2f}]"
        ax.text(
            hdi_col_x,
            y,
            sigma_text,
            transform=label_trans,
            ha="left",
            va="center",
            fontsize=8,
            color=C_NEUTRAL,
            family="monospace",
        )

        # HDI lower bound is the diagnostic; rows at or below the flag
        # threshold are rendered in coral.
        diag_color = C_NEGATIVE if r["lo"] <= HDI_LOW_FLAG else C_STRUCTURE
        diag_text = f"{r['n_levels']:>3d}   $\\sigma_\\mathrm{{lo}}$={r['lo']:>4.2f}"
        ax.text(
            diag_col_x,
            y,
            diag_text,
            transform=label_trans,
            ha="left",
            va="center",
            fontsize="small",
            color=diag_color,
            family="monospace",
        )

    sigma_ticks = _nice_ticks(ax.get_xlim(), target_n=5, anchor=0.0)
    ax.set_xticks(sigma_ticks)
    ax.set_xticklabels([f"{s:g}" for s in sigma_ticks], fontsize=8, color=C_TEXT)
    ax.tick_params(axis="x", length=3, width=0.5, color=C_STRUCTURE, pad=3)
    ax.set_xlabel(
        r"Group-level SD ($\sigma$, linear-predictor scale)", color=C_TEXT, labelpad=4
    )


def render_figure(slope_df: pd.DataFrame, var_df: pd.DataFrame) -> None:
    rr_lims, sigma_lims = _derive_axis_limits(slope_df, var_df)
    fig, ax_fixed, ax_var = setup_plot(rr_lims, sigma_lims)
    draw_fixed_forest(ax_fixed, slope_df)
    draw_variance_forest(ax_var, var_df)

    xoffsets = [-1, -1.0]
    yoffsets = [1.20, 1.1]
    letter_subplots(xoffset=xoffsets, yoffset=yoffsets)

    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".svg"), dpi=300)
    print(f"  Figure saved: {OUT_FIG}")
    plt.show()


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    slope_df = load_slope_summary()
    var_df = load_variance_summary()
    events = load_events()

    write_stats("re_analysis", print_full_readout(slope_df, var_df, events))

    _section(9, "Figure")
    if not all_cells_present(slope_df, var_df):
        print("  One or more cells missing (see section 2) -- skipping figure.")
        return
    render_figure(slope_df, var_df)


if __name__ == "__main__":
    main()
