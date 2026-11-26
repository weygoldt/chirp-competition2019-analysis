"""Figure (main text, Fig. 8): after the loser chirps, the winner attacks again no later than at matched moments.

A   the data and how the re-attack hazard model splits time: ``N_ROWS`` inter-chase intervals sampled at random
    (seed ``SEED``) from the primary model's intervals, offset at 0, sorted by length (shortest on top). Per row:
    the chase that just ended (grey) and its last 5 s (light teal; the offset window's left edge: a loser chirp
    there switches the hazard model's ``chirped`` on at the offset), the loser's chirps (ticks; grey before the
    counting range), the first counted chirp (teal dot), the waiting time before it (black) and after it (teal),
    and the next chase onset (the re-attack, black dot); rows longer than the axis end in an arrow.
B   fraction of intervals in which the winner has not yet attacked again, against time since the loser's first
    chirp after the offset (teal) and since matched moments without a chirp (gray): the same time since the
    offset in other intervals of the same competition in which, at that moment, the winner had not yet
    attacked again and the loser had not yet chirped; matched moments stop counting when their own loser chirps
    (``stats_reattack_matched.py``). Weighted Kaplan-Meier; bands 2.5-97.5 % of resamples of the competitions
    (bootstrap). Dotted: the matched curve raised to the power 0.5, i.e. what a chirp that halved the
    re-attack rate would give. Text: the primary hazard model's HR (``reattack_hazard.json``).
C   interval from the previous chase's offset to this chase's onset, for chases with and without a loser chirp
    in the offset window (``loser_chirped_offset_window``), cumulative fraction on a log axis, medians as
    dots; within competitions: each competition's own curves averaged with the same weight in both groups,
    ``n_chirp * n_none / (n_chirp + n_none)`` (competitions without both groups carry none).

    paper/run.sh plot_receiver_before_after.py   # chain reattack, after stats_reattack_matched.py
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

import reattack_matched as rm
from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import FIGURES, OUT, PROCESSED, SEED
from style import letter_subplots, loser_color, set_light_style

NAME = "plot_receiver_before_after"
OBS_DATA = (PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data"
            / "chirp_effect_on_dominant_obs.csv")
HAZARD_DIR = PROCESSED / "reattack_hazard"  # stats_reattack_hazard.py: intervals.csv
MATCHED_DIR = PROCESSED / "reattack_matched"  # stats_reattack_matched.py: curves, summary, loser chirps
HAZARD_STATS = OUT / "stats" / "reattack_hazard.json"
PRIMARY_FIT = "primary"

PREDICTOR = "loser_chirped_offset_window"  # C: M2's predictor
PREV = "time_since_previous_chasing"  # previous chase's offset -> this chase's onset [s]
COMP = "recording"  # one competition (the models' grouping)

# Palette (Figs. 5-7): after a loser chirp in the loser teal, before or without in black; matched moments gray.
CHIRP_C, NONE_C, MUTED, INK = loser_color, "k", "#6e6e6e", "k"
CTRL_C, BAND_GREY = "0.35", "grey"
BAND_ALPHA = 0.35  # as Fig. 6's bands
CHASE_C = "0.85"
TAIL_C = tuple(0.32 * np.asarray(to_rgb(loser_color)) + 0.68)  # A: the chase's last 5 s, teal blended with white
LW_WAIT, LW_CHIRP, LW_CHASE = 0.85, 1.9, 3.2  # A
CURVE_LW = 1.1
APPEASE_HR = 0.5  # B: the dotted reference, a chirp that halves the re-attack rate
APPEASE_LS = (0, (1.2, 1.2))
ANNOT_FS = "small"  # 6.7 pt at the 8-pt base

N_ROWS = 44  # A
RASTER_XLIM = (-10.0, 60.0)

# Layout [cm]. A on top with its key on the right; B and C below, the same size.
W_CM, H_CM = 17.4, 13.3
ROW2_BOTTOM, ROW2_H = 1.15, 3.7
ROW1_BOTTOM, ROW1_H = 6.55, 6.1
A_X, A_W = 1.25, 9.6
KEY_X = 11.25
B_X, C_X, BC_W = 1.25, 10.1, 7.1
LETTER_DY = 0.35
XLIM_S = (1.0, 2000.0)  # C (seconds, log)
XTICKS_S = [1, 10, 100, 1000]


# ---------------------------------------------------------------------------
# data: C (chasing events)
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
    """Panel C's numbers (write_stats-ready)."""
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
# data: A (intervals, chirps), B (matched curves), the model's HR
# ---------------------------------------------------------------------------
def primary_intervals(iv: pd.DataFrame) -> pd.DataFrame:
    """The primary hazard model's intervals (those with a previous interval), as the matched comparison."""
    for col in (COMP, "onset", "offset", "length", "event", "first_chirp", "prev_interval"):
        if col not in iv.columns:
            raise KeyError(f"column {col!r} missing from the interval table")
    return iv[iv["prev_interval"].notna()]


def loser_chirps(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """recording -> sorted loser chirp times (``loser_chirps.csv`` of stats_reattack_matched)."""
    return {rec: np.sort(g["time_s"].to_numpy(float)) for rec, g in table.groupby(COMP)}


def load_matched(matched_dir=MATCHED_DIR) -> tuple[dict, dict]:
    """The curves (``curves.npz``) and numbers (``summary.json``) of stats_reattack_matched.py."""
    for f in ("curves.npz", "summary.json"):
        if not (matched_dir / f).exists():
            raise FileNotFoundError(f"{matched_dir / f} missing: run stats_reattack_matched.py (chain reattack)")
    with np.load(matched_dir / "curves.npz") as z:
        curves = {k: z[k] for k in z.files}
    return curves, json.loads((matched_dir / "summary.json").read_text())


def model_hr(path=HAZARD_STATS, fit: str = PRIMARY_FIT) -> dict:
    """The hazard ratio of a fit, as the text cites it (``write_stats("reattack_hazard")``)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run stats_reattack_hazard.py (chain reattack)")
    fits = json.loads(path.read_text())["values"]["fits"]
    if fit not in fits:
        raise ValueError(f"no fit {fit!r} in {path}")
    hr = fits[fit]["hr"]
    return {"median": float(hr["median"]), "hdi": [float(x) for x in hr["hdi"]]}


def raster_rows(iv: pd.DataFrame, n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """A random sample of ``n`` intervals (seeded), sorted by length (shortest first)."""
    if n > len(iv):
        raise ValueError(f"{n} rows requested from {len(iv)} intervals")
    pick = np.random.default_rng(seed).choice(len(iv), size=n, replace=False)
    return iv.iloc[pick].sort_values("length", kind="stable")


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


def _plain(v, _pos=None) -> str:
    return f"{v:g}"


def _log_time_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_S)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS_S))
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def raster_panel(ax, rows: pd.DataFrame, chirps: dict, left_edge: float) -> dict:
    """A. Row 0 at the top. ``left_edge``: the offset window's left edge (s, negative)."""
    x0, x1 = RASTER_XLIM
    counts = {"n_rows": len(rows), "n_clipped": 0, "n_chirped_at_offset": 0, "n_chirped_after_offset": 0,
              "n_no_chirp": 0, "n_censored": 0}
    sw_all = rm.switch_times(rows)
    for y, ((_, r), sw) in enumerate(zip(rows.iterrows(), sw_all)):
        off, on, L = float(r["offset"]), float(r["onset"]) - float(r["offset"]), float(r["length"])
        tail0 = max(left_edge, on)
        # the chase that ended at 0 (grey) and its counting tail (light teal)
        if on < tail0:
            ax.plot([max(on, x0), tail0], [y, y], color=CHASE_C, lw=LW_CHASE, solid_capstyle="butt", zorder=1)
        ax.plot([tail0, 0], [y, y], color=TAIL_C, lw=LW_CHASE, solid_capstyle="butt", zorder=1)
        # waiting: black until the first counted chirp, teal after
        end = min(L, x1)
        key = "n_no_chirp" if not np.isfinite(sw) else "n_chirped_at_offset" if sw == 0 else "n_chirped_after_offset"
        counts[key] += 1
        a = min(sw, end)
        if a > 0:
            ax.plot([0, a], [y, y], color=NONE_C, lw=LW_WAIT, solid_capstyle="butt", zorder=2)
        if a < end:
            ax.plot([a, end], [y, y], color=CHIRP_C, lw=LW_CHIRP, solid_capstyle="butt", zorder=2)
        # loser chirps up to the next onset; grey before the counting range
        c = np.asarray(chirps.get(r[COMP], []), float) - off
        c = c[(c >= x0) & (c < end)]
        counted = c >= tail0
        for sel, color in ((counted, "k"), (~counted, "0.6")):
            ax.plot(c[sel], np.full(int(sel.sum()), y), ls="", marker="|", ms=4.2, mew=0.6, color=color, zorder=3)
        fc = float(r["first_chirp"]) - off
        if np.isfinite(fc) and fc < x1:
            ax.plot([fc], [y], ls="", marker="o", ms=3.4, mfc=CHIRP_C, mec="white", mew=0.5, zorder=4)
        # the end: re-attack, censoring or beyond the axis
        if L >= x1:
            counts["n_clipped"] += 1
            ax.plot([x1], [y], ls="", marker=">", ms=2.8, mew=0, color=CHIRP_C if sw < x1 else NONE_C,
                    zorder=4, clip_on=False)
        elif int(r["event"]) == 1:
            ax.plot([L], [y], ls="", marker="o", ms=3.0, mfc="k", mec="white", mew=0.4, zorder=4)
        else:
            counts["n_censored"] += 1
            ax.plot([L], [y], ls="", marker="o", ms=3.0, mfc="white", mec="k", mew=0.6, zorder=4)
    n = len(rows)
    ax.axvline(0, color="k", lw=0.5, ls="--", zorder=0.5)
    ax.axvline(left_edge, color=MUTED, lw=0.5, ls=":", zorder=0.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Time since chase offset [s]")
    ax.set_ylabel(f"{n} inter-chase intervals, by length")
    ax.text(0, -1.3, "offset", ha="center", va="bottom", fontsize=ANNOT_FS)
    ax.text(left_edge, -1.3, rf"$-${-left_edge:g} s", ha="center", va="bottom", fontsize=ANNOT_FS, color=MUTED)
    return counts


def raster_key(fig, x_cm: float, top_cm: float, left_edge: float):
    """A's symbols, and what the hazard model makes of them."""
    W, H = fig.get_size_inches() * 2.54
    handles = [
        Patch(color=CHASE_C, lw=0), Patch(color=TAIL_C, lw=0),
        Line2D([], [], ls="", marker="|", ms=4.2, mew=0.6, color="k"),
        Line2D([], [], ls="", marker="o", ms=3.4, mfc=CHIRP_C, mec="white", mew=0.5),
        Line2D([], [], color=NONE_C, lw=LW_WAIT), Line2D([], [], color=CHIRP_C, lw=LW_CHIRP),
        Line2D([], [], ls="", marker="o", ms=3.0, mfc="k", mec="white", mew=0.4),
        Line2D([], [], ls="", marker=">", ms=2.8, mew=0, color=NONE_C),
    ]
    labels = [
        "chase",
        f"its last {-left_edge:g} s: a chirp here\ncounts from the offset",
        "loser chirp (grey: not counted)",
        "first counted chirp",
        "waiting, no loser chirp yet",
        "waiting after the first chirp",
        "next chase onset (re-attack)",
        f"continues beyond {RASTER_XLIM[1]:g} s",
    ]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x_cm / W, top_cm / H), frameon=False,
               fontsize=ANNOT_FS, handlelength=1.6, handletextpad=0.6, labelspacing=0.55, borderaxespad=0,
               borderpad=0)
    fig.text(x_cm / W, (ROW1_BOTTOM + 0.05) / H,
             "Hazard model: re-attacks per minute\nof teal vs. black waiting time, at the\n"
             "same time since the offset, within\ncompetitions (HR = teal / black).",
             ha="left", va="bottom", fontsize=ANNOT_FS, color=MUTED, linespacing=1.25)


def survival_panel(ax, curves: dict, design: str = "first_chirp", appease: bool = True) -> None:
    """Weighted Kaplan-Meier "not yet re-attacked" after the trigger (teal) and matched moments (gray)."""
    t = curves["grid"]
    for g, line_c, band_c in ((rm.CONTROL, CTRL_C, BAND_GREY), (rm.TRIGGER, CHIRP_C, CHIRP_C)):
        k = f"{design}_survival_{g}"
        ax.fill_between(t, curves[k + "_lo"], curves[k + "_hi"], color=band_c, alpha=BAND_ALPHA, lw=0, zorder=1,
                        step="post")
        ax.step(t, curves[k], where="post", color=line_c, lw=1.0, zorder=3 if g == rm.TRIGGER else 2)
    if appease:
        ax.plot(t, curves[f"{design}_survival_{rm.CONTROL}"] ** APPEASE_HR, color=CHIRP_C, lw=0.8, ls=APPEASE_LS,
                zorder=2.5)
    ax.axvline(0, c="k", lw=0.5, ls="--")
    ax.set_xlim(0, float(t[-1]))
    ax.set_xticks(np.arange(0, float(t[-1]) + 0.1, 10))
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "", "0.5", "", "1"])
    ax.set_ylabel("Not yet re-attacked")


def matched_legend(ax, summary: dict, design: str, trigger_label: str, loc: str = "lower left",
                   appease: bool = True) -> None:
    """Legend of a matched comparison, with the numbers of triggers and matched moments."""
    d = summary[design]
    handles = matched_handles(appease)
    labels = [f"{trigger_label} ($n$ = {d['n_triggers']:,})",
              f"matched moments, no chirp yet ($n$ = {d['n_matched_moments']:,})",
              f"if a chirp halved the re-attack rate (HR {APPEASE_HR:g})"][:len(handles)]
    ax.legend(handles, labels, loc=loc, frameon=False, fontsize=ANNOT_FS, borderaxespad=0.2, handlelength=1.5,
              handletextpad=0.5, labelspacing=0.35)


def matched_handles(appease: bool = True) -> list:
    """Legend symbols of a matched comparison: trigger (teal band and line), matched (gray), HR 0.5 (dotted)."""
    handles = [(Patch(color=CHIRP_C, alpha=BAND_ALPHA, lw=0), Line2D([], [], color=CHIRP_C, lw=1.0)),
               (Patch(color=BAND_GREY, alpha=BAND_ALPHA, lw=0), Line2D([], [], color=CTRL_C, lw=1.0))]
    if appease:
        handles.append(Line2D([], [], color=CHIRP_C, lw=0.8, ls=APPEASE_LS))
    return handles


def interval_panel(ax, ev: pd.DataFrame, w: pd.Series) -> dict:
    """C: competition-averaged cumulative fractions of the previous interval for both groups; medians as dots."""
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
    ax.text(0.98, 0.13, f"median {medians[True]:.0f} s", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ANNOT_FS, color=CHIRP_C)
    ax.text(0.98, 0.03, f"median {medians[False]:.0f} s", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ANNOT_FS, color=NONE_C)
    ax.set_xlabel("Previous inter-chase interval [s]")
    ax.set_ylabel("Cumulative fraction of chases")
    return medians


def draw(rows, chirps, left_edge, curves, summary, hr, ev):
    set_light_style()
    _rc_tweaks()
    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
    ax_a = _axes_cm(fig, A_X, ROW1_BOTTOM, A_W, ROW1_H)
    ax_b = _axes_cm(fig, B_X, ROW2_BOTTOM, BC_W, ROW2_H)
    ax_c = _axes_cm(fig, C_X, ROW2_BOTTOM, BC_W, ROW2_H)
    counts = raster_panel(ax_a, rows, chirps, left_edge)
    raster_key(fig, KEY_X, ROW1_BOTTOM + ROW1_H + 0.1, left_edge)

    survival_panel(ax_b, curves)
    ax_b.set_xlabel("Time since the loser's first chirp or matched moment [s]")
    matched_legend(ax_b, summary, "first_chirp", "after the loser's first chirp")
    ax_b.text(0.98, 0.98, f"hazard model: HR {hr['median']:.2f} [{hr['hdi'][0]:.2f}, {hr['hdi'][1]:.2f}]",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=ANNOT_FS, color=INK)

    w = event_weights(ev)
    interval_panel(ax_c, ev, w)
    n = group_counts(ev[w > 0])
    ax_c.legend([Line2D([], [], color=CHIRP_C, lw=CURVE_LW), Line2D([], [], color=NONE_C, lw=CURVE_LW)],
                [f"loser chirped\n($n$ = {int(n[True].sum()):,})", f"no chirp\n($n$ = {int(n[False].sum()):,})"],
                loc="upper left", frameon=False, borderaxespad=0.1, handletextpad=0.5)
    for ax in (ax_b, ax_c):
        sns.despine(ax=ax, offset=4, trim=False)
    sns.despine(ax=ax_a, offset=4, trim=False, left=True)
    _letters_cm(fig, [(ax_a, "A", 0.75, 0.55), (ax_b, "B", 0.95, LETTER_DY), (ax_c, "C", 1.25, LETTER_DY)])
    return fig, {"A": ax_a, "B": ax_b, "C": ax_c}, counts


def main() -> None:
    window = offset_window()
    curves, summary = load_matched()
    if summary["window_s"] != list(window):
        raise ValueError(f"reattack_matched was computed with the window {summary['window_s']}, not {window}")
    iv = primary_intervals(pd.read_csv(HAZARD_DIR / "intervals.csv"))
    if len(iv) != summary["n_intervals"]:
        raise ValueError(f"{len(iv)} intervals vs {summary['n_intervals']} in reattack_matched (rerun it)")
    chirps = loser_chirps(pd.read_csv(MATCHED_DIR / "loser_chirps.csv"))
    rows = raster_rows(iv)
    ev = analysed_events(pd.read_csv(OBS_DATA))
    hr = model_hr()

    fig, axes, counts = draw(rows, chirps, window[0], curves, summary, hr, ev)
    print(axes_extents(fig, list(axes.items())))
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{NAME}.{ext}" for ext in ("pdf", "png")]
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    stats = {"window_s": list(window), "previous": summarise_previous(ev),
             "raster": {**counts, "seed": SEED, "n_intervals": int(len(iv)), "xlim_s": list(RASTER_XLIM)},
             "model_hr_shown": hr}
    print(json.dumps(stats, indent=1))
    print("\n".join(map(str, paths)))
    write_stats("receiver_before_after", stats)


if __name__ == "__main__":
    main()
