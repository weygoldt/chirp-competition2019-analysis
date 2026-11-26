"""Figure (main text, new Fig. 8): loser chirps come with dense chasing, not with a later re-attack.

The receiver-effect models (``stats_chirp_affects_chasing.py``, Gamma GLMMs with a random intercept per
recording) find the loser-chirp predictor strongly associated with the interval to the previous chase (M2) and
not with the interval to the next one (M1, M4). This figure shows the same comparison in the data ("before vs
after"); the window schematic and the forest plot of the model slopes (``plot_obliterate_receiver_effect``)
are in the supplement. Descriptive only: inference stays with the models.

A   interval from the previous chase's offset to this chase's onset (``time_since_previous_chasing``) for
    chasing events with and without a loser chirp in the offset window (``loser_chirped_offset_window``, the
    models' predictor): cumulative fraction on a log time axis, medians as dots
B   the same for the interval from this chase's offset to the next onset (``time_to_next_chasing``), same axes
C   per competition, the ratio of the median previous and of the median next interval (with / without chirp),
    paired per competition, on a log ratio scale; dotted line at 1

Rows: the models' complete-window chasing events (their subset "untruncated": the offset window,
``offset_window.py``, is complete: the chase lasts at least the window's part before the offset, and the next
chase starts at least the window's part after it later) with both intervals (the rows of M3/M4 there; M1 and M2
use the few extra events that lack the other interval). In all chasing events the window is clipped at the
next onset, so a quick re-attack leaves less time for a chirp: the next-interval comparison would show that
coupling, not the winner. Conditioning on complete windows truncates both groups' next interval at the
window's post part; B shades the excluded range.

Within competitions, as the models compare: the curves of A and B are averages of each competition's own
cumulative fractions, with the same weight for both groups (``n_chirp * n_none / (n_chirp + n_none)``, the
information the competition carries on a difference between the groups; competitions without both groups
carry none). Differences between competitions (some chase densely and chirp a lot) therefore appear equally in
both curves, and the distance between the curves is a weighted average of within-competition differences. The
pooled curves (every event counted once) and intervals divided by each competition's median are computed for
comparison and go to the stats (``receiver_before_after``) with the full set of chasing events (all windows).

    paper/run.sh plot_receiver_before_after.py   # chain receiver; needs chain interchase and the offset window
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import FIGURES, PROCESSED
from style import letter_subplots, loser_color, set_light_style

NAME = "plot_receiver_before_after"
OBS_DATA = (PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data"
            / "chirp_effect_on_dominant_obs.csv")

PREDICTOR = "loser_chirped_offset_window"  # the models' primary predictor
PREV = "time_since_previous_chasing"  # previous chase's offset -> this chase's onset [s]
NEXT = "time_to_next_chasing"  # this chase's offset -> next chase's onset [s]
COMP = "recording"  # one competition (the models' random intercept)
INTERVALS = {"previous": PREV, "next": NEXT}
SUBSETS = ("complete", "full")  # plotted; for the report only

# Palette (Figs. 5-7): chasing events with a loser chirp in the loser teal, without in black.
CHIRP_C, NONE_C, MUTED, INK = loser_color, "k", "#6e6e6e", "k"
SHADE_C = "#ececec"  # B: next intervals excluded by the complete window
LINK_C = "#b4b4b4"  # C: the line joining a competition's two ratios
REF = {"color": INK, "lw": 0.6, "ls": ":", "zorder": 0.5}  # C: ratio 1
CURVE_LW = 1.1
ANNOT_FS = "small"  # 6.7 pt at the 8-pt base

# Layout [cm]. One row, 17.4 cm wide. A and B (equal sizes) form a group: B shares A's y-axis and has no
# y tick labels, so a narrow gap; C is set apart by a wider gap that also holds its two-line y-axis label.
W_CM, H_CM = 17.4, 6.45
LEFT, RIGHT = 1.35, 17.2  # A's y-axis label; C's right edge
BOTTOM, AX_H = 1.15, 4.75  # lower edge of the axes (tick labels and x-axis label below), their height
AB_GAP, BC_GAP, C_W = 0.45, 2.15, 3.3
LETTER_DX = {"A": LEFT - 0.25, "B": 0.25, "C": 1.65}  # centred this far left of their axes' left edge
LETTER_DY = 0.25  # and this far above the top
XLIM_S = (1.0, 2000.0)  # A, B (seconds, log); the stats count intervals outside
XTICKS_S = [1, 10, 100, 1000]
RATIO_RANGE = (0.25, 4.0)  # C (log): the axis covers this range and every ratio
RATIO_TICKS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]  # C: those within the axis
RATIO_HEADROOM = 1.7  # C: the axis reaches this factor above the highest ratio (room for the counts)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def analysed_events(obs: pd.DataFrame, window: tuple[float, float], subset: str) -> pd.DataFrame:
    """The chasing events of a subset that have both intervals, with a boolean column ``chirp``.

    ``complete``: the offset window is complete (``offset_window_duration`` equals its width, the models'
    "untruncated" subset), ``full``: all chasing events. Both intervals must be present and positive (the
    first chase of a recording has no previous one, the last no next one).
    """
    required = [COMP, PREDICTOR, PREV, NEXT, "offset_window_duration"]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise KeyError(f"columns missing from the per-chase table: {missing}")
    left, right = window
    if subset == "complete":
        keep = obs["offset_window_duration"] >= (right - left) - 1e-6
    elif subset == "full":
        keep = pd.Series(True, index=obs.index)
    else:
        raise ValueError(f"unknown subset {subset!r} (known: {SUBSETS})")
    for col in (PREV, NEXT):
        keep &= obs[col].notna() & (obs[col] > 0)
    ev = obs.loc[keep, [COMP, PREDICTOR, PREV, NEXT, "offset_window_duration"]].copy()
    if ev[PREDICTOR].isna().any():
        raise ValueError(f"{PREDICTOR} has missing values")
    ev["chirp"] = ev[PREDICTOR].astype(bool)
    if subset == "complete" and (ev[NEXT] < right - 1e-6).any():
        raise ValueError(f"complete windows with a next interval below the window's post part ({right:g} s)")
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


def competition_ratios(ev: pd.DataFrame, col: str) -> pd.Series:
    """Per competition with both groups: median interval with / without a loser chirp."""
    med = ev.groupby([COMP, "chirp"])[col].median().unstack()
    if True not in med.columns or False not in med.columns:
        return pd.Series(dtype=float)
    return (med[True] / med[False]).dropna()


def summarise(ev: pd.DataFrame) -> dict:
    """The descriptive numbers of one subset (write_stats-ready)."""
    w = event_weights(ev)
    used = w > 0
    n = group_counts(ev[used])
    out = {
        "n_events": {"chirp": int(n[True].sum()), "no_chirp": int(n[False].sum()), "total": int(used.sum())},
        "n_events_all_competitions": int(len(ev)),
        "n_competitions": int((competition_weights(ev) > 0).sum()),
        "n_competitions_all": int(ev[COMP].nunique()),
    }
    for name, col in INTERVALS.items():
        rel = ev[col] / ev.groupby(COMP)[col].transform("median")
        g = {"chirp": ev["chirp"] & used, "no_chirp": ~ev["chirp"] & used}
        curve = {k: weighted_median(ev.loc[m, col], w[m]) for k, m in g.items()}
        pooled = {k: float(ev.loc[m, col].median()) for k, m in g.items()}
        normed = {k: float(rel[m].median()) for k, m in g.items()}
        r = competition_ratios(ev[used], col)
        out[name] = {
            # medians of the plotted (competition-averaged) curves
            "median_s": curve, "median_ratio": curve["chirp"] / curve["no_chirp"],
            # every event counted once (trial composition included)
            "pooled_median_s": pooled, "pooled_median_ratio": pooled["chirp"] / pooled["no_chirp"],
            # intervals divided by their competition's median interval
            "relative_median": normed, "relative_median_ratio": normed["chirp"] / normed["no_chirp"],
            "competitions": {"n": int(len(r)), "n_below_1": int((r < 1).sum()), "n_above_1": int((r > 1).sum()),
                             "median_ratio": float(r.median()),
                             "min_ratio": float(r.min()), "max_ratio": float(r.max())},
            "n_outside_axis": int(((ev.loc[used, col] < XLIM_S[0]) | (ev.loc[used, col] > XLIM_S[1])).sum()),
        }
    return out


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


def interval_panel(ax, ev: pd.DataFrame, w: pd.Series, col: str) -> dict:
    """Competition-averaged cumulative fractions of one interval for both groups; medians as dots."""
    medians = {}
    for chirp, color in ((False, NONE_C), (True, CHIRP_C)):
        m = (ev["chirp"] == chirp) & (w > 0)
        xs, F = weighted_ecdf(ev.loc[m, col], w[m])
        # steps from 0 at the axis' left edge to 1 at its right edge
        ax.step(np.r_[XLIM_S[0], xs, XLIM_S[1]], np.r_[0, F, 1], where="post", color=color, lw=CURVE_LW,
                zorder=3 if chirp else 2, solid_joinstyle="miter")
        med = weighted_median(ev.loc[m, col], w[m])
        ax.plot([med], [0.5], "o", ms=3.6, mfc=color, mec="white", mew=0.6, zorder=4)
        medians[chirp] = med
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_S)
    ax.set_ylim(0, 1.01)  # the line at 1 not cut by the axes' top
    ax.xaxis.set_major_locator(FixedLocator(XTICKS_S))
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "", "0.5", "", "1"])
    # medians, lower right (the curves run from lower left to upper right)
    ax.text(0.98, 0.13, f"median {medians[True]:.0f} s", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ANNOT_FS, color=CHIRP_C)
    ax.text(0.98, 0.03, f"median {medians[False]:.0f} s", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ANNOT_FS, color=NONE_C)
    return medians


def ratio_panel(ax, ev: pd.DataFrame, used: pd.Series) -> None:
    """Per competition: median-interval ratios (with / without chirp), previous and next, joined."""
    r = pd.concat({k: competition_ratios(ev[used], col) for k, col in INTERVALS.items()}, axis=1).dropna()
    ax.axhline(1, **REF)
    for _, row in r.iterrows():
        ax.plot([0, 1], [row["previous"], row["next"]], color=LINK_C, lw=0.6, zorder=1)
    for i, k in enumerate(INTERVALS):
        ax.plot(np.full(len(r), i), r[k], "o", ms=3.2, mfc=INK, mec="white", mew=0.5, zorder=3)
        below, n = int((r[k] < 1).sum()), len(r)
        # the count of each column at the top of the axes, above the highest ratio (RATIO_HEADROOM)
        ax.text(i, 0.99, f"{below} of {n}\nbelow 1", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=ANNOT_FS, color=MUTED, linespacing=1.0)
    ax.set_yscale("log")
    lo = min(RATIO_RANGE[0] / 1.1, float(r.to_numpy().min()) / 1.15)
    hi = max(RATIO_RANGE[1], float(r.to_numpy().max()) * RATIO_HEADROOM)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator([t for t in RATIO_TICKS if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Previous", "Next"])
    ax.set_xlabel("Inter-chase interval")
    ax.set_ylabel("Median interval ratio\n(with / without chirp)")


def draw(ev: pd.DataFrame, window: tuple[float, float]):
    set_light_style()
    _rc_tweaks()
    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
    ax_a, ax_b, ax_c = layout(fig)
    w = event_weights(ev)
    used = w > 0

    interval_panel(ax_a, ev, w, PREV)
    interval_panel(ax_b, ev, w, NEXT)
    ax_a.set_xlabel("Previous inter-chase interval [s]")
    ax_b.set_xlabel("Next inter-chase interval [s]")
    ax_a.set_ylabel("Cumulative fraction of chases")
    ax_b.tick_params(labelleft=False)

    # B: complete windows exclude next intervals shorter than the window's post part, in both groups
    post = window[1]
    ax_b.axvspan(XLIM_S[0], post, color=SHADE_C, lw=0, zorder=0)
    ax_b.text(np.sqrt(XLIM_S[0] * post), 0.5, f"excluded: next chase\nwithin {post:g} s", rotation=90,
              ha="center", va="center", fontsize=ANNOT_FS, color=MUTED, linespacing=1.0)

    n = group_counts(ev[used])
    keys = [Line2D([], [], color=CHIRP_C, lw=CURVE_LW), Line2D([], [], color=NONE_C, lw=CURVE_LW)]
    ax_a.legend(keys, [f"with loser chirp ($n$ = {int(n[True].sum())})",
                       f"without ($n$ = {int(n[False].sum())})"],
                loc="upper left", frameon=False, borderaxespad=0.1, handletextpad=0.5)

    ratio_panel(ax_c, ev, used)

    for ax in (ax_a, ax_b, ax_c):
        sns.despine(ax=ax, offset=4, trim=False)
    _letters_cm(fig, [(ax, let, LETTER_DX[let], LETTER_DY) for ax, let in zip((ax_a, ax_b, ax_c), "ABC")])
    return fig, (ax_a, ax_b, ax_c)


def main() -> None:
    window = offset_window()
    obs = pd.read_csv(OBS_DATA)
    events = {s: analysed_events(obs, window, s) for s in SUBSETS}
    stats = {"window_s": list(window)}  # complete: plotted; full: for comparison
    stats.update({s: summarise(ev) for s, ev in events.items()})

    fig, axes = draw(events["complete"], window)
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
