"""Figure (main text, Fig. 8): loser chirps come with dense chasing, not with a change in the re-attack rate.

A   interval from the previous chase's offset to this chase's onset (``time_since_previous_chasing``) for
    chasing events with and without a loser chirp in the offset window (``loser_chirped_offset_window``, the
    predictor of the acausal Gamma GLMM M2 of ``stats_chirp_affects_chasing.py``): cumulative fraction on a log
    time axis, medians as dots. All chases with a previous one: the previous interval cannot depend on the
    response (the next one), so no complete-window restriction is needed.
B   the winner's re-attack rate (next-chase onsets per minute at risk) in each baseline piece of time since the
    offset, for time at risk after (teal) and before or without (black) a loser chirp (the time-varying
    ``chirped`` of the hazard model, ``reattack_hazard.py``): dots observed, lines and bands the primary hazard
    model (``stats_reattack_hazard.py``): its rates for the same time at risk with the chirp covariate set to 1
    and to 0, averaged over competitions as the dots (``standardized_rates``; posterior median and 95 % HDI;
    the rates of a typical competition at the covariates' geometric means lie below the dots, because rates
    vary between competitions and intervals). Time at risk is split where the covariate switches, never
    the intervals: survival curves split by a covariate that switches during follow-up would credit the time
    before a chirp to the chirp group (immortal-time bias).
C   per competition, the observed rate ratio (with / without a chirp; Mantel-Haenszel over the pieces of B),
    sorted, on a log scale, with the primary model's hazard ratio (median, band: 95 % HDI); dotted line at 1.

Within competitions, as the models compare. A: the curves average each competition's own cumulative fractions
with the same weight in both groups (``n_chirp * n_none / (n_chirp + n_none)``; competitions without both groups
carry none). B: in each piece, the competitions' rates are averaged with the same weight in both groups,
``X1 X0 / (X1 + X0)`` for the time at risk X1 with and X0 without a chirp (``reattack_hazard.mh_rates``), so
the ratio of the two plotted rates is the piece's Mantel-Haenszel rate ratio. B and C use the primary model's
intervals (those with a previous interval, the same chases as A).

    paper/run.sh plot_receiver_before_after.py   # chain reattack; needs chains interchase and the hazard fits
"""

from __future__ import annotations

import json

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

import reattack_hazard as rh
from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import FIGURES, PROCESSED
from style import letter_subplots, loser_color, set_light_style

NAME = "plot_receiver_before_after"
OBS_DATA = (PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data"
            / "chirp_effect_on_dominant_obs.csv")
HAZARD_DIR = PROCESSED / "reattack_hazard"  # stats_reattack_hazard.py
PRIMARY_FIT = "primary"

PREDICTOR = "loser_chirped_offset_window"  # M2's predictor (A)
PREV = "time_since_previous_chasing"  # previous chase's offset -> this chase's onset [s]
COMP = "recording"  # one competition (the models' grouping)
COVARIATE = "chirped"  # B, C: the hazard model's time-varying covariate

# Palette (Figs. 5-7): with a loser chirp in the loser teal, without in black.
CHIRP_C, NONE_C, MUTED, INK = loser_color, "k", "#6e6e6e", "k"
BAND_ALPHA = 0.18  # B, C: posterior bands
REF = {"color": INK, "lw": 0.6, "ls": ":", "zorder": 0.5}  # C: ratio 1
CURVE_LW = 1.1
MODEL_LW = {0: 0.6, 1: 1.6}  # B: model lines without (thin, on top) and with a chirp (wide, below)
ANNOT_FS = "small"  # 6.7 pt at the 8-pt base

# Layout [cm]. One row, 17.4 cm wide; A and B the same size, each with its own y-axis, C a narrow strip.
W_CM, H_CM = 17.4, 6.45
LEFT, RIGHT = 1.35, 17.2  # A's y-axis label; C's right edge
BOTTOM, AX_H = 1.15, 4.75  # lower edge of the axes (tick labels and x-axis label below), their height
AB_GAP, BC_GAP, C_W = 1.55, 1.75, 3.0  # the gaps hold B's and C's y tick labels and y-axis labels
LETTER_DX = {"A": LEFT - 0.25, "B": 1.3, "C": 1.5}  # centred this far left of their axes' left edge
LETTER_DY = 0.25  # and this far above the top
XLIM_S = (1.0, 2000.0)  # A, B (seconds, log); the stats count intervals outside
XTICKS_S = [1, 10, 100, 1000]
RATE_YLIM = (0.02, 5.0)  # B (re-attacks per min, log); a zero rate is drawn open on the lower edge
RATE_TICKS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
RATIO_RANGE = (0.25, 4.0)  # C (log): the axis covers this range and every ratio
RATIO_TICKS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
RATIO_HEADROOM = 1.7  # C: room above the highest ratio for the count


# ---------------------------------------------------------------------------
# data: A (chasing events)
# ---------------------------------------------------------------------------
def analysed_events(obs: pd.DataFrame) -> pd.DataFrame:
    """The chasing events with a previous interval (present and positive), with a boolean column ``chirp``
    (M2's predictor). The first chase of a recording has no previous interval."""
    required = [COMP, PREDICTOR, PREV]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise KeyError(f"columns missing from the per-chase table: {missing}")
    keep = obs[PREV].notna() & (obs[PREV] > 0)
    ev = obs.loc[keep, required].copy()
    if ev[PREDICTOR].isna().any():
        raise ValueError(f"{PREDICTOR} has missing values")
    ev["chirp"] = ev[PREDICTOR].astype(bool)
    return ev


def group_counts(ev: pd.DataFrame) -> pd.DataFrame:
    """Chasing events per competition with (column True) and without (False) a loser chirp."""
    return ev.groupby([COMP, "chirp"]).size().unstack(fill_value=0).reindex(columns=[False, True], fill_value=0)


def competition_weights(ev: pd.DataFrame) -> pd.Series:
    """Each competition's weight, the same for both groups: ``n1 * n0 / (n1 + n0)`` (0 without both groups)."""
    n = group_counts(ev)
    n1, n0 = n[True].astype(float), n[False].astype(float)
    return (n1 * n0 / (n1 + n0)).where((n1 > 0) & (n0 > 0), 0.0)


def event_weights(ev: pd.DataFrame) -> pd.Series:
    """Per event: its competition's weight shared among the competition's events of its group, normalised so
    that the weights of each group sum to 1. The weighted cumulative fraction of a group is then the weighted
    average of the competitions' own cumulative fractions."""
    w_c = competition_weights(ev)
    if w_c.sum() <= 0:
        raise ValueError("no competition has chasing events with and without a loser chirp")
    n_cg = ev.groupby([COMP, "chirp"])[COMP].transform("size").astype(float)
    return ev[COMP].map(w_c).astype(float) / n_cg / w_c.sum()


def weighted_ecdf(x, w) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values and the cumulative fraction (weights normalised to 1) at each of them."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    keep = w > 0
    x, w = x[keep], w[keep]
    return x, np.cumsum(w) / w.sum()


def weighted_median(x, w) -> float:
    """The smallest value at which the weighted cumulative fraction reaches one half."""
    xs, F = weighted_ecdf(x, w)
    return float(xs[np.searchsorted(F, 0.5 - 1e-12)])


def competition_ratios(ev: pd.DataFrame, col: str = PREV) -> pd.Series:
    """Per competition with both groups: median interval with / without a loser chirp."""
    med = ev.groupby([COMP, "chirp"])[col].median().unstack()
    if True not in med.columns or False not in med.columns:
        return pd.Series(dtype=float)
    return (med[True] / med[False]).dropna()


def summarise_previous(ev: pd.DataFrame) -> dict:
    """Panel A's numbers (write_stats-ready)."""
    w = event_weights(ev)
    used = w > 0
    n = group_counts(ev[used])
    rel = ev[PREV] / ev.groupby(COMP)[PREV].transform("median")
    g = {"chirp": ev["chirp"] & used, "no_chirp": ~ev["chirp"] & used}
    curve = {k: weighted_median(ev.loc[m, PREV], w[m]) for k, m in g.items()}
    pooled = {k: float(ev.loc[m, PREV].median()) for k, m in g.items()}
    normed = {k: float(rel[m].median()) for k, m in g.items()}
    r = competition_ratios(ev[used])
    return {
        "n_events": {"chirp": int(n[True].sum()), "no_chirp": int(n[False].sum()), "total": int(used.sum())},
        "n_events_all_competitions": int(len(ev)),
        "n_competitions": int((competition_weights(ev) > 0).sum()),
        "n_competitions_all": int(ev[COMP].nunique()),
        # medians of the plotted (competition-averaged) curves
        "median_s": curve, "median_ratio": curve["chirp"] / curve["no_chirp"],
        # every event counted once (trial composition included)
        "pooled_median_s": pooled, "pooled_median_ratio": pooled["chirp"] / pooled["no_chirp"],
        # intervals divided by their competition's median interval
        "relative_median": normed, "relative_median_ratio": normed["chirp"] / normed["no_chirp"],
        "competitions": {"n": int(len(r)), "n_below_1": int((r < 1).sum()), "n_above_1": int((r > 1).sum()),
                         "median_ratio": float(r.median()), "min_ratio": float(r.min()),
                         "max_ratio": float(r.max())},
        "n_outside_axis": int(((ev.loc[used, PREV] < XLIM_S[0]) | (ev.loc[used, PREV] > XLIM_S[1])).sum()),
    }


# ---------------------------------------------------------------------------
# data: B, C (time at risk)
# ---------------------------------------------------------------------------
def primary_rows(pp: pd.DataFrame) -> pd.DataFrame:
    """The person-period rows of the primary model (intervals with a previous interval)."""
    for col in (COMP, "interval", "piece", "event", "exposure", COVARIATE, "log_prev"):
        if col not in pp.columns:
            raise KeyError(f"column {col!r} missing from the person-period table")
    return pp[pp["log_prev"].notna()]


def piece_centres(pieces=rh.PIECES_S, xlim=XLIM_S) -> np.ndarray:
    """Where B draws each piece's observed rate: the log centre of the piece within the axis."""
    p = np.asarray(pieces, float)
    lo, hi = np.maximum(p[:-1], xlim[0]), np.minimum(p[1:], xlim[1])
    return np.sqrt(lo * hi)


def summarise_rates(pp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """B's observed rates per piece, C's per-competition rate ratios, and their numbers for the stats."""
    cells = rh.cell_table(pp, COVARIATE)
    rates = rh.mh_rates(cells).reindex(range(len(rh.PIECES_S) - 1))
    comp = rh.competition_rate_ratios(cells)
    r = comp["ratio"].dropna()
    n_int = int(pp["interval"].nunique())
    stats = {
        "n_intervals": n_int, "n_events": int(pp["event"].sum()), "exposure_s": float(pp["exposure"].sum()),
        "rate_per_min": {"chirp": (60 * rates["rate1"]).fillna(0).tolist(),
                         "no_chirp": (60 * rates["rate0"]).fillna(0).tolist()},
        "events": {"chirp": rates["E1_all"].fillna(0).astype(int).tolist(),
                   "no_chirp": rates["E0_all"].fillna(0).astype(int).tolist()},
        "exposure_min": {"chirp": (rates["X1_all"].fillna(0) / 60).tolist(),
                         "no_chirp": (rates["X0_all"].fillna(0) / 60).tolist()},
        "n_competitions": rates["n_comp"].fillna(0).astype(int).tolist(),
        "competitions": {"n": int(len(r)), "n_without_ratio": int(comp["ratio"].isna().sum()),
                         "n_below_1": int((r < 1).sum()), "n_above_1": int((r > 1).sum()),
                         "median_ratio": float(r.median()), "min_ratio": float(r.min()),
                         "max_ratio": float(r.max()),
                         # the Mantel-Haenszel rate ratio over competitions and pieces (strata)
                         "pooled_mh_ratio": float(comp["num"].sum() / comp["den"].sum())},
    }
    return rates, comp, stats


def load_hr(fit: str = PRIMARY_FIT) -> dict:
    """The hazard ratio of a fit (``hr_summary.csv`` of ``stats_reattack_hazard.py``)."""
    hr = pd.read_csv(HAZARD_DIR / "hr_summary.csv").set_index("fit")
    if fit not in hr.index:
        raise ValueError(f"no fit {fit!r} in {HAZARD_DIR / 'hr_summary.csv'}")
    hr = hr.loc[fit]
    return {"median": float(hr["hr"]), "hdi": [float(hr["hdi_lo"]), float(hr["hdi_hi"])],
            "p_lt_1": float(hr["p_lt_1"])}


def standardized_rates(trace, pp: pd.DataFrame, covariates=("log_prev_c", "log_dur_c"),
                       chunk: int = 500) -> pd.DataFrame:
    """The model's rates for the observed time at risk, with the chirp covariate set to 0 and to 1.

    Per draw, competition c and piece k: the time-at-risk-weighted mean of the rows' hazards
    exp(alpha_k + u_c + (beta + b_c) g + covariates) over every row of the cell (chirped or not), then
    averaged over competitions with the dots' weights X1 X0 / (X1 + X0). The same population and the
    same averaging as the observed rates; the two lines differ only by the competitions' chirp effects.
    Returns piece_label, piece, chirp, median, lo, hi (95 % HDI) in events per min.
    """
    post = trace.posterior
    labels = rh.piece_labels()
    k_post = {str(v): i for i, v in enumerate(post["piece_label"].coords["piece_label_dim"].values)}
    c_dim = [d for d in post[f"1|{COMP}"].dims if d not in ("chain", "draw")][0]
    c_post = {str(v): i for i, v in enumerate(post[f"1|{COMP}"].coords[c_dim].values)}
    rows = rh.aggregate(pp, [COMP, "piece", COVARIATE, *covariates])
    cells = rh.cell_table(pp, COVARIATE)
    T = cells["X0"] + cells["X1"]
    w = (cells["X0"] * cells["X1"] / T).where((cells["X0"] > 0) & (cells["X1"] > 0), 0.0)
    w_norm = w / w.groupby(level="piece").transform("sum")
    cell = pd.MultiIndex.from_arrays([rows[COMP], rows["piece"]])
    x_cell = rows.groupby([COMP, "piece"])["exposure"].transform("sum").to_numpy()
    # A[r, k]: row r's share of its cell's time at risk times its competition's weight in piece k
    a = rows["exposure"].to_numpy() / x_cell * w_norm.reindex(cell).fillna(0.0).to_numpy()
    A = np.zeros((len(rows), len(labels)))
    A[np.arange(len(rows)), rows["piece"].to_numpy()] = a

    def flat(name):
        v = post[name].stack(s=("chain", "draw"))
        return v.transpose("s", ...).values

    alpha = flat("piece_label")[:, [k_post[lab] for lab in labels]]  # (S, pieces), in time order
    u = flat(f"1|{COMP}")
    b = flat(f"{COVARIATE}|{COMP}") if f"{COVARIATE}|{COMP}" in post else np.zeros_like(u)
    beta = flat(COVARIATE)
    ci = np.array([c_post[c] for c in rows[COMP]])
    gamma = {c: flat(c) for c in covariates}
    out = {0: [], 1: []}
    for s0 in range(0, len(beta), chunk):  # draws in chunks: (draws x rows) arrays stay small
        sl = slice(s0, s0 + chunk)
        eta0 = alpha[sl][:, rows["piece"].to_numpy()] + u[sl][:, ci]
        for c in covariates:
            eta0 = eta0 + np.outer(gamma[c][sl], rows[c].to_numpy())
        for g in (0, 1):
            eta = eta0 + g * (beta[sl, None] + b[sl][:, ci])
            out[g].append(60 * np.exp(eta) @ A)
    res = []
    for g in (0, 1):
        r = np.concatenate(out[g])  # (S, pieces)
        for k, lab in enumerate(labels):
            h = az.hdi(r[:, k], hdi_prob=0.95)
            res.append({"piece_label": lab, "piece": k, "chirp": g, "median": float(np.median(r[:, k])),
                        "lo": float(h[0]), "hi": float(h[1])})
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def _rc_tweaks() -> None:  # as plot_chase_validation (the v2 figures' helpers)
    mpl.rcParams.update({"lines.markeredgewidth": 0.5, "xtick.major.size": 3, "ytick.major.size": 3,
                         "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "axes.linewidth": 0.8, "legend.handlelength": 1.6,
                         "legend.fontsize": ANNOT_FS})


def _axes_cm(fig, x, y, w, h):
    W, H = fig.get_size_inches() * 2.54
    return fig.add_axes((x / W, y / H, w / W, h / H))


def _letters_cm(fig, axes_and_offsets):
    W, H = fig.get_size_inches() * 2.54
    axs, xo, yo, lets = [], [], [], []
    for ax, let, dx, dy in axes_and_offsets:
        bb = ax.get_position()
        axs.append(ax)
        lets.append(let)
        xo.append(-dx / (bb.width * W))
        yo.append(1 + dy / (bb.height * H))
    letter_subplots(axs, letters=lets, xoffset=xo, yoffset=yo)


def axes_extents(fig, named_axes) -> str:
    """Each axes' extent in figure cm (to check the alignment)."""
    W, H = fig.get_size_inches() * 2.54
    lines = [f"{'axes':<6}{'left':>7}{'right':>7}{'bottom':>8}{'top':>7}{'width':>7}{'height':>8}"
             f"  [cm; figure {W:.2f} x {H:.2f}]"]
    for name, ax in named_axes:
        bb = ax.get_position()
        lines.append(f"{name:<6}{bb.x0 * W:7.3f}{bb.x1 * W:7.3f}{bb.y0 * H:8.3f}{bb.y1 * H:7.3f}"
                     f"{bb.width * W:7.3f}{bb.height * H:8.3f}")
    return "\n".join(lines)


def layout(fig):
    w = (RIGHT - LEFT - AB_GAP - BC_GAP - C_W) / 2
    return (_axes_cm(fig, LEFT, BOTTOM, w, AX_H),
            _axes_cm(fig, LEFT + w + AB_GAP, BOTTOM, w, AX_H),
            _axes_cm(fig, RIGHT - C_W, BOTTOM, C_W, AX_H))


def _plain(v, _pos=None) -> str:
    return f"{v:g}"


def _log_time_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_S)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS_S))
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _log_y(ax, ticks, lo, hi) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def interval_panel(ax, ev: pd.DataFrame, w: pd.Series) -> dict:
    """Competition-averaged cumulative fractions of the previous interval for both groups; medians as dots."""
    medians = {}
    for chirp, color in ((False, NONE_C), (True, CHIRP_C)):
        m = (ev["chirp"] == chirp) & (w > 0)
        xs, F = weighted_ecdf(ev.loc[m, PREV], w[m])
        # steps from 0 at the axis' left edge to 1 at its right edge
        ax.step(np.r_[XLIM_S[0], xs, XLIM_S[1]], np.r_[0, F, 1], where="post", color=color, lw=CURVE_LW,
                zorder=3 if chirp else 2, solid_joinstyle="miter")
        med = weighted_median(ev.loc[m, PREV], w[m])
        ax.plot([med], [0.5], "o", ms=3.6, mfc=color, mec="white", mew=0.6, zorder=4)
        medians[chirp] = med
    _log_time_axis(ax)
    ax.set_ylim(0, 1.01)  # the line at 1 not cut by the axes' top
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "", "0.5", "", "1"])
    # medians, lower right (the curves run from lower left to upper right)
    ax.text(0.98, 0.13, f"median {medians[True]:.0f} s", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ANNOT_FS, color=CHIRP_C)
    ax.text(0.98, 0.03, f"median {medians[False]:.0f} s", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ANNOT_FS, color=NONE_C)
    ax.set_xlabel("Previous inter-chase interval [s]")
    ax.set_ylabel("Cumulative fraction of chases")
    return medians


def rate_panel(ax, rates: pd.DataFrame, model: pd.DataFrame) -> None:
    """Observed rates per piece (dots) and the model's piecewise-constant rates (steps, 95 % HDI bands)."""
    edges = np.clip(np.asarray(rh.PIECES_S, float), *XLIM_S)
    x = piece_centres()
    lo_edge = RATE_YLIM[0]
    for g, color, col in ((0, NONE_C, "rate0"), (1, CHIRP_C, "rate1")):
        m = model[model["chirp"] == g].sort_values("piece")
        med, lo, hi = (np.r_[m[k].to_numpy(), m[k].to_numpy()[-1]] for k in ("median", "lo", "hi"))
        ax.fill_between(edges, lo, hi, step="post", color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
        # where the two model lines coincide (HR near 1), the thin black one shows on the wide teal one
        ax.step(edges, med, where="post", color=color, lw=MODEL_LW[g], zorder=3 - g)
        y = 60 * rates[col].to_numpy(float)
        pos = y > 0
        ax.plot(x[pos], y[pos], "o", ms=3.6, mfc=color, mec="white", mew=0.6, zorder=4 + g)
        # a piece without events: open marker on the lower edge
        ax.plot(x[~pos & np.isfinite(y)], np.full(int((~pos & np.isfinite(y)).sum()), lo_edge), "o", ms=3.2,
                mfc="white", mec=color, mew=0.7, zorder=4 + g, clip_on=False)
    _log_time_axis(ax)
    _log_y(ax, RATE_TICKS, *RATE_YLIM)
    ax.set_xlabel("Time since chase offset [s]")
    ax.set_ylabel("Re-attacks per min at risk")
    keys = [Line2D([], [], ls="", marker="o", ms=3.6, mfc=MUTED, mec="white", mew=0.6),
            (Patch(color=MUTED, alpha=BAND_ALPHA, lw=0), Line2D([], [], color=MUTED, lw=MODEL_LW[0]))]
    ax.legend(keys, ["observed", "model"], loc="upper right", frameon=False, borderaxespad=0.1,
              handletextpad=0.5)


def ratio_panel(ax, comp: pd.DataFrame, hr: dict) -> None:
    """Per competition, the observed rate ratio (sorted), with the model's hazard ratio as a band."""
    r = comp["ratio"].dropna().sort_values()
    x = np.arange(1, len(r) + 1)
    xlim = (0.2, len(r) + 0.8)
    ax.axhline(1, **REF)
    ax.axhspan(*hr["hdi"], color=CHIRP_C, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.axhline(hr["median"], color=CHIRP_C, lw=CURVE_LW * 0.8, zorder=2)
    ax.plot(x, r.to_numpy(), "o", ms=3.2, mfc=INK, mec="white", mew=0.5, zorder=3)
    above = int((r > 1).sum())
    ax.text(0.5, 0.99, f"{above} of {len(r)} above 1", transform=ax.transAxes, ha="center", va="top",
            fontsize=ANNOT_FS, color=MUTED)
    lo = min(RATIO_RANGE[0] / 1.1, float(r.min()) / 1.15)
    hi = max(RATIO_RANGE[1], float(r.max()) * RATIO_HEADROOM)
    _log_y(ax, RATIO_TICKS, lo, hi)
    ax.set_xlim(*xlim)
    ax.set_xticks([])
    ax.set_xlabel("Competitions (sorted)")
    ax.set_ylabel("Re-attack rate ratio\n(with / without chirp)")


def draw(ev: pd.DataFrame, rates: pd.DataFrame, comp: pd.DataFrame, model: pd.DataFrame, hr: dict):
    set_light_style()
    _rc_tweaks()
    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
    ax_a, ax_b, ax_c = layout(fig)
    w = event_weights(ev)
    interval_panel(ax_a, ev, w)
    n = group_counts(ev[w > 0])
    keys = [Line2D([], [], color=CHIRP_C, lw=CURVE_LW), Line2D([], [], color=NONE_C, lw=CURVE_LW)]
    # two lines per entry: one-line labels reach into the curves near 1
    ax_a.legend(keys, [f"with loser chirp\n($n$ = {int(n[True].sum())})",
                       f"without\n($n$ = {int(n[False].sum())})"],
                loc="upper left", frameon=False, borderaxespad=0.1, handletextpad=0.5)
    rate_panel(ax_b, rates, model)
    ratio_panel(ax_c, comp, hr)
    for ax in (ax_a, ax_b, ax_c):
        sns.despine(ax=ax, offset=4, trim=False)
    ax_c.spines["bottom"].set_visible(False)  # no x ticks: a rank axis
    _letters_cm(fig, [(ax, let, LETTER_DX[let], LETTER_DY) for ax, let in zip((ax_a, ax_b, ax_c), "ABC")])
    return fig, (ax_a, ax_b, ax_c)


def main() -> None:
    window = offset_window()
    ev = analysed_events(pd.read_csv(OBS_DATA))
    pp = primary_rows(pd.read_parquet(HAZARD_DIR / "person_period.parquet"))
    rates, comp, rate_stats = summarise_rates(pp)
    model = standardized_rates(az.from_netcdf(HAZARD_DIR / f"{PRIMARY_FIT}.nc"), pp)
    hr = load_hr()
    plotted = {g: model[model["chirp"] == g].sort_values("piece") for g in (0, 1)}
    stats = {"window_s": list(window), "previous": summarise_previous(ev), "rates": rate_stats,
             "model_hr": hr,
             "model_rate_per_min": {name: {k: plotted[g][k].tolist() for k in ("median", "lo", "hi")}
                                    for g, name in ((1, "chirp"), (0, "no_chirp"))}}

    fig, axes = draw(ev, rates, comp, model, hr)
    print(axes_extents(fig, list(zip("ABC", axes))))
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{NAME}.{ext}" for ext in ("pdf", "png")]
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    print(json.dumps(stats, indent=1))
    print("\n".join(map(str, paths)))
    write_stats("receiver_before_after", stats)


if __name__ == "__main__":
    main()
