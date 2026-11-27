# Ported from chirp_competition d040123 (src_paper/plot_event_time_rates_winner_loser_chirp.py; 17.4 cm, prints the extrema the manuscript quotes).
# Changes: JSONs from $PAPER_OUT/data/processed/event_time_rates, figures via paper_config.figure (png, pdf, svg as before),
# written as <stem>_src_paper so it does not overwrite the paper version; rate-axis limits and ticks extended
# (loser 8 -> 16, winner 4 -> 6 /min) because the v2 rates would be clipped; the printed extrema
# and trial counts also go to write_stats ("event_rates").
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from style import (
    letter_subplots,
    set_light_style,
    set_dark_style,
    winner_color,
    loser_color,
)
from IPython import embed
from manuscript_stats import write_stats
from paper_config import PROCESSED, figure


filename = Path(__file__).stem
targetpath = figure(filename)
formats = ["png", "pdf", "svg"]
mode = "light"  # "dark"
pval_keys = {
    "n.s.": 1.0,
    "*": 0.05,
    "**": 0.01,
    "***": 0.001,
}

if mode == "light":
    cmap = sns.color_palette("mako", as_cmap=True)
    cmap2 = sns.color_palette("flare", as_cmap=True)
    set_light_style()
    cols = sns.color_palette("deep")
    # chirpratec = cols[0]
    wchirpratec = cmap(0.5)
    lchirpratec = cmap2(0.25)
    baselinec = "grey"
    eventc = "grey"
    centermarkerc = "k"
else:
    set_dark_style()
    chirpratec = "#DE5252"
    baselinec = "lightgray"
    eventc = "white"
    centermarkerc = "lightgray"


maxtime = 30

eventplotkwargs = dict(
    marker="|",
    markersize=0.5,
    color="k",
    alpha=0.2,
    markeredgewidth=0.1,
    lw=0,
)

signifindicatorkwargs = dict(
    color="darkgrey",
    alpha=1,
)

# Marker styles used when --plot-significant is enabled.
peak_marker_kwargs = dict(
    marker="^",
    markersize=4,
    markerfacecolor="none",
    markeredgecolor="k",
    markeredgewidth=0.8,
    linestyle="none",
    zorder=10,
)
trough_marker_kwargs = dict(peak_marker_kwargs, marker="v")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot event-time chirp rates for winners and losers, and report "
            "all local extrema in the event rate that fall outside the "
            "bootstrapped baseline 95% CI (before and after t=0)."
        )
    )
    p.add_argument(
        "--plot-significant",
        action="store_true",
        help=(
            "Overlay markers at every local extremum of the event rate that "
            "exceeds the bootstrapped baseline CI (▲ peak = above upper CI, "
            "▼ trough = below lower CI)."
        ),
    )
    return p.parse_args()


def setup_plot():
    cm = 1 / 2.54
    dpi = 300
    figsize = (17.4 * cm, 11 * cm)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    meta_spacings = {
        "left": 0.09,
        "right": 0.98,
        "top": 0.92,
        "bottom": 0.13,
        "hspace": 0.35,
        "wspace": 0.2,
    }
    sub_spacings = {
        "hspace": 0.2,
        "wspace": 0.2,
    }
    # Extra horizontal gap between column 0 and columns 1-3.
    # Bump up for more room, down for less. ~0.167 reproduces the original gap.
    first_col_gap = 0.2

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], **meta_spacings)

    # Split each meta row into a 1-column left block and a 3-column right
    # block. width_ratios=[1, 3.8] makes all four visible columns equal width
    # (3 cols + 2 inner gaps of 0.4*c = 3.8*c). The right block keeps the
    # original sub_spacings, so gaps between columns 2-4 are unchanged.
    gs1_outer = gs[0].subgridspec(1, 2, width_ratios=[1, 3.5], wspace=first_col_gap)
    gs1_left = gs1_outer[0, 0].subgridspec(
        2, 1, height_ratios=[1, 0.5], hspace=sub_spacings["hspace"]
    )
    gs1_right = gs1_outer[0, 1].subgridspec(
        2, 3, height_ratios=[1, 0.5], **sub_spacings
    )

    gs2_outer = gs[1].subgridspec(1, 2, width_ratios=[1, 3.5], wspace=first_col_gap)
    gs2_left = gs2_outer[0, 0].subgridspec(
        2, 1, height_ratios=[1, 0.5], hspace=sub_spacings["hspace"]
    )
    gs2_right = gs2_outer[0, 1].subgridspec(
        2, 3, height_ratios=[1, 0.5], **sub_spacings
    )

    top_rate_axes = [fig.add_subplot(gs1_left[0, 0])] + [
        fig.add_subplot(gs1_right[0, i]) for i in range(3)
    ]
    top_event_axes = [fig.add_subplot(gs1_left[1, 0])] + [
        fig.add_subplot(gs1_right[1, i]) for i in range(3)
    ]
    bottom_rate_axes = [fig.add_subplot(gs2_left[0, 0])] + [
        fig.add_subplot(gs2_right[0, i]) for i in range(3)
    ]
    bottom_event_axes = [fig.add_subplot(gs2_left[1, 0])] + [
        fig.add_subplot(gs2_right[1, i]) for i in range(3)
    ]
    axs = top_rate_axes + top_event_axes + bottom_rate_axes + bottom_event_axes
    for ax in top_rate_axes + bottom_rate_axes:
        ax.axvline(0, c="k", lw=0.5, ls="--")
        ax.patch.set_facecolor("none")
    # ... everything below this line is unchanged ...
    texts = [
        "Other's chirps",
        "Chasing \\textit{onset}",
        "Chasing \\textit{offset}",
        "Contact",
    ]
    for i, ax in enumerate(axs[:4]):
        ax.text(
            0.5,
            1.3,
            texts[i],
            transform=ax.transAxes,
            fontsize="medium",
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )
    axs[0].set_ylabel("Chirps/min")
    axs[4].set_ylabel("Trial")
    axs[8].set_ylabel("Chirps/min")
    axs[12].set_ylabel("Trial")
    axs[14].text(
        -0.15,
        -0.95,
        "Time from event [s]",
        transform=axs[14].transAxes,
        verticalalignment="center",
        horizontalalignment="center",
    )
    axs[0].text(
        0.0,
        1,
        "\\textbf{Loser}",
        transform=axs[0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize="medium",
        color=loser_color,
        fontweight="bold",
    )
    axs[8].text(
        0.0,
        1,
        "\\textbf{Winner}",
        transform=axs[8].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize="medium",
        color=winner_color,
        fontweight="bold",
    )
    for i, ax in enumerate(axs):
        if i < 12:
            ax.set_xticklabels([])
        if i in [0, 1, 4, 5, 8, 9, 12, 13, 14, 15]:
            ax.set_xticks([-30, -15, 0, 15, 30])
        if i in [14, 15]:
            ax.set_xticks([-30, -15, 0, 15, 30])

        ax.set_yticklabels([])

    for top_ax, bottom_ax in zip(top_event_axes[1:], bottom_event_axes[1:]):
        top_ax.set_yticks([0, 15])
        bottom_ax.set_yticks([0, 15])

    for i in [4, 12]:
        axs[i].set_yticks([0, 15, 30])
        axs[i].set_yticklabels([0, 15, 30])

    for i in [5, 13]:
        axs[i].set_yticks([0, 15])
        axs[i].set_yticklabels([0, 15])

    # Ticks extended with the rate-axis limits for the v2 detections (were [0, 4, 8] and [0, 2, 4]).
    axs[0].set_yticks([0, 4, 8, 12, 16])
    axs[0].set_yticklabels([0, 4, 8, 12, 16])

    axs[1].set_yticks([0, 4, 8, 12, 16])
    axs[1].set_yticklabels([0, 4, 8, 12, 16])

    axs[8].set_yticks([0, 2, 4, 6])
    axs[8].set_yticklabels([0, 2, 4, 6])

    axs[9].set_yticks([0, 2, 4, 6])
    axs[9].set_yticklabels([0, 2, 4, 6])

    fig.align_ylabels()
    return fig, axs, texts


def load_data():
    datapath = PROCESSED / "event_time_rates"

    lchirps_chase_wildcard = "lchirps@ag*.json"
    lchirps_contact_wildcard = "lchirps@con*.json"
    lchirps_wchirps_wildcard = "lchirps@wchirps*.json"
    lfiles = list(datapath.glob(lchirps_wchirps_wildcard))
    lfiles.extend(sorted(list(datapath.glob(lchirps_chase_wildcard)))[::-1])
    lfiles.extend(list(datapath.glob(lchirps_contact_wildcard)))

    wchirps_chase_wildcard = "wchirps@ag*.json"
    wchirps_contact_wildcard = "wchirps@con*.json"
    wchirps_lchirps_wildcard = "wchirps@lchirps*.json"
    wfiles = list(datapath.glob(wchirps_lchirps_wildcard))
    wfiles.extend(sorted(list(datapath.glob(wchirps_chase_wildcard)))[::-1])
    wfiles.extend(list(datapath.glob(wchirps_contact_wildcard)))

    ldata = []
    wdata = []
    for f in lfiles:
        with open(f) as file:
            ldata.append(json.load(file))
    for f in wfiles:
        with open(f) as file:
            wdata.append(json.load(file))

    ldata.extend(wdata)
    trials = [d["trial_identifiers"] for d in ldata]
    unique_trials = np.sort(np.unique(np.concatenate(trials)))[::-1]
    return ldata, unique_trials


def plot_eventplot(ax, data, eventplotkwargs, trials, ypos):
    centered_times = data["centered_times"]
    found_trials = data["trial_identifiers"]
    if len(centered_times) != len(found_trials):
        raise ValueError("Mismatch between trial identifiers and centered times")

    # find indices of found trials in all trials
    trial_indices = [np.where(trials == t)[0][0] for t in found_trials]
    found_ypos = ypos[trial_indices]

    filtered_ctimes = []
    filtered_ypos = []
    for j in range(len(centered_times)):
        ctimes = np.array(centered_times[j])
        ctimes = ctimes[(ctimes < maxtime) & (ctimes > -maxtime)]
        # if len(ctimes) == 0:
        #     continue
        filtered_ctimes.append(ctimes)
        filtered_ypos.append(found_ypos[j])

    for j, ctimes in enumerate(filtered_ctimes):
        ax.plot(
            ctimes,
            found_ypos[j] * np.ones_like(ctimes),
            **eventplotkwargs,
        )

    return len(filtered_ctimes)


def find_significant_extrema(
    time, rate, baseline_lo, baseline_med, baseline_hi, min_samples=2
):
    """Find the strongest deviation per (before/after × peak/trough) category.

    For every contiguous stretch where the rate exits the baseline 95% CI, one
    extremum is identified (argmax for above-CI regions, argmin for below-CI
    regions). These candidates are then grouped into four categories --
    (before, peak), (before, trough), (after, peak), (after, trough) -- and
    only the single strongest deviation per category (largest |Δ| from the
    nearest CI boundary) is returned. So at most four extrema come back.

    Two multiplicative effect sizes are reported per extremum:
      ``fold_point``  = rate[idx]            / baseline_med[idx]
      ``fold_window`` = mean(rate[s:e])      / mean(baseline_med[s:e])
    i.e. the factor that scales the baseline into the observed rate at the
    extremum and averaged across the significant window, respectively.

    Regions shorter than ``min_samples`` points are discarded as numerical
    boundary noise on a smooth convolved rate. Pass ``min_samples=1`` to
    include them.

    Returns a list sorted by time (up to 4 entries), each a dict with keys:
    ``type``, ``time``, ``rate``, ``delta``, ``baseline_lo``, ``baseline_med``,
    ``baseline_hi``, ``side``, ``region_start``, ``region_end``,
    ``region_width``, ``fold_point``, ``fold_window``.
    """
    time = np.asarray(time)
    rate = np.asarray(rate)
    baseline_lo = np.asarray(baseline_lo)
    baseline_med = np.asarray(baseline_med)
    baseline_hi = np.asarray(baseline_hi)

    def _regions(mask):
        if not mask.any():
            return
        m = mask.astype(np.int8)
        d = np.diff(m)
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if m[0]:
            starts.insert(0, 0)
        if m[-1]:
            ends.append(len(m))
        for s, e in zip(starts, ends):
            if e - s >= min_samples:
                yield s, e

    eps = 1e-12

    def _fold(num, denom):
        return float(num) / float(denom) if denom > eps else float("inf")

    candidates = []
    for kind, mask, pick in [
        ("peak", rate > baseline_hi, np.argmax),
        ("trough", rate < baseline_lo, np.argmin),
    ]:
        for s, e in _regions(mask):
            idx = s + int(pick(rate[s:e]))
            t = float(time[idx])
            r = float(rate[idx])
            if kind == "peak":
                delta = r - float(baseline_hi[idx])
            else:
                delta = r - float(baseline_lo[idx])
            bl_pt = float(baseline_med[idx])
            bl_win = float(np.mean(baseline_med[s:e]))
            rate_win = float(np.mean(rate[s:e]))
            candidates.append(
                {
                    "type": kind,
                    "time": t,
                    "rate": r,
                    "delta": delta,
                    "baseline_lo": float(baseline_lo[idx]),
                    "baseline_med": bl_pt,
                    "baseline_hi": float(baseline_hi[idx]),
                    "side": "before" if t < 0 else "after",
                    "region_start": float(time[s]),
                    "region_end": float(time[e - 1]),
                    "region_width": float(time[e - 1] - time[s]),
                    "fold_point": _fold(r, bl_pt),
                    "fold_window": _fold(rate_win, bl_win),
                }
            )

    # Keep only the strongest deviation per (side, type) category.
    best = {}
    for c in candidates:
        key = (c["side"], c["type"])
        if key not in best or abs(c["delta"]) > abs(best[key]["delta"]):
            best[key] = c

    return sorted(best.values(), key=lambda d: d["time"])


def split_extrema(time, rate, baseline_lo, baseline_med, baseline_hi, min_samples=2):
    """``find_significant_extrema`` on each side of the event separately (before:
    t < 0, after: t ≥ 0).

    ``find_significant_extrema`` gives a stretch outside the CI that crosses the
    event time to the side of its overall extremum, so the other side falls back to
    whatever smaller stretch it has. Here such a stretch is cut at the event time and
    each side gets the extremum of its own part; the region of that part starts (after)
    or ends (before) at the event time. Example, winner chirps around loser chirps at
    threshold 0.8: one stretch from -14.2 to +6.9 s, highest at -0.7 s; unsplit, the
    after peak was a 0.4-s blip at +9.9 s, split it is +0.6 s with the region 0 to 6.9 s.
    """
    time = np.asarray(time)
    arrays = [np.asarray(a) for a in (rate, baseline_lo, baseline_med, baseline_hi)]
    out = []
    for side in (time < 0, time >= 0):
        out.extend(
            find_significant_extrema(
                time[side], *(a[side] for a in arrays), min_samples=min_samples
            )
        )
    return out


def report_extrema(group_label, event_label, extrema):
    """Pretty-print the strongest extremum per (side × type) category.

    Always prints all four category slots; missing ones show ``(none)``.
    Each present extremum prints over two lines: identity + significance on
    the first, region extent + multiplicative effect size on the second.
    """
    # strip LaTeX emphasis for terminal readability
    clean_event = (
        event_label.replace("\\textit{", "").replace("}", "").replace("\\textbf{", "")
    )
    header = f" {group_label}  ·  {clean_event}"
    bar = "─" * 78
    print()
    print(bar)
    print(header)
    print(bar)

    def _fmt_fold(x):
        if not np.isfinite(x):
            return "  ∞ "
        return f"{x:5.2f}"

    by_key = {(e["side"], e["type"]): e for e in extrema}

    for side_key, side_label in [
        ("before", "Before event  (t < 0)"),
        ("after", "After event   (t ≥ 0)"),
    ]:
        print(f"  {side_label}")
        for kind, sym, boundary in [
            ("peak", "▲", "above upper CI"),
            ("trough", "▼", "below lower CI"),
        ]:
            e = by_key.get((side_key, kind))
            if e is None:
                print(f"    {sym} {kind:<6}  (none)")
                continue
            # line 1: identity + significance (unchanged)
            print(
                f"    {sym} {kind:<6}  "
                f"t = {e['time']:+7.2f} s    "
                f"rate = {e['rate']:5.2f} /min    "
                f"Δ = {e['delta']:+5.2f} {boundary}"
            )
            # line 2: region + multiplicative effect size, indented to align
            # with the start of the data columns on line 1
            print(
                f"              "
                f"region [{e['region_start']:+6.2f}, {e['region_end']:+6.2f}] s "
                f"(w = {e['region_width']:5.2f} s)    "
                f"fold = ×{_fmt_fold(e['fold_point'])} (at {kind}), "
                f"×{_fmt_fold(e['fold_window'])} (over window)"
            )


def plot_conv_rates(
    ax,
    data,
    chirpratec=wchirpratec,
    baselinec=baselinec,
    mark_significant=False,
):
    time = np.array(data["conv_time"])
    pct_bl = np.percentile(
        data["bootstrapped_baseline_jittered"], [2.5, 50, 97.5], axis=0
    )
    pct_bl *= 60
    baseline_lo, baseline_med, baseline_hi = pct_bl
    ax.plot(time, baseline_med, c=baselinec, lw=1)
    ax.fill_between(time, baseline_lo, baseline_hi, alpha=0.5, color=baselinec, lw=0)

    pct_rate = np.percentile(data["jackknifed_eventrate"], [2.5, 50, 97.5], axis=0)
    pct_rate *= 60
    rate_lo, rate_med, rate_hi = pct_rate
    ax.plot(time, rate_med, c=chirpratec, lw=1)
    ax.fill_between(time, rate_lo, rate_hi, alpha=0.5, color=chirpratec, lw=0)

    extrema = find_significant_extrema(
        time, rate_med, baseline_lo, baseline_med, baseline_hi
    )
    split = split_extrema(time, rate_med, baseline_lo, baseline_med, baseline_hi)

    if mark_significant:
        for e in extrema:
            kw = peak_marker_kwargs if e["type"] == "peak" else trough_marker_kwargs
            ax.plot(e["time"], e["rate"], **kw)

    return extrema, split


def plot_conv_rates_heatmap(ax, data):
    time = np.array(data["conv_time"])
    per_trial_rates = np.array(data["per_trial_rates"])

    # simple normalization
    per_trial_rates = per_trial_rates / (
        np.max(per_trial_rates, axis=1)[:, None] + 1e-6
    )

    n_zero = int(np.sum(np.sum(per_trial_rates, axis=1) == 0))
    n_tot = per_trial_rates.shape[0]
    print(f"N obs: {n_tot}")
    print(f"N trials with 0 chirps: {n_zero}")

    text = (
        r"\textit{" + str(n_tot) + r" trials,}" + "\n"
        r"\textit{" + str(n_zero) + r" without}" + "\n"
        r"\textit{chirps}"
    )
    ax.text(
        0.98,
        2.8,
        text,
        va="bottom",
        ha="right",
        transform=ax.transAxes,
        color="grey",
        fontsize="x-small",
    )

    ax.imshow(
        per_trial_rates * 60,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[time[0], time[-1], 0, len(per_trial_rates)],
        interpolation="none",
    )

    n = len(per_trial_rates)
    return n, n_zero


# The events of the four panels per group, in the order of load_data().
PANEL_EVENTS = ["other_chirps", "chase_onset", "chase_offset", "contact"]


def extrema_stats(extrema, split):
    """The extrema of one panel for write_stats: side (before/after) -> type (peak/trough),
    and ``peak``, the strongest peak on either side (largest excess over the CI), which
    can move across the event time between detection thresholds; ``split_at_event``:
    side -> type of ``split_extrema`` (stretches cut at the event time), for what
    happens after (before) the event when one stretch spans it."""

    def entry(e):
        return {
            "time_s": e["time"],
            "rate_per_min": e["rate"],
            "baseline_per_min": e["baseline_med"],
            "delta_per_min": e["delta"],
            "region_s": [e["region_start"], e["region_end"]],
            "region_width_s": e["region_width"],
            "fold_point": e["fold_point"],
            "fold_window": e["fold_window"],
        }

    out = {}
    for e in extrema:
        out.setdefault(e["side"], {})[e["type"]] = entry(e)
    peaks = [e for e in extrema if e["type"] == "peak"]
    if peaks:
        out["peak"] = entry(max(peaks, key=lambda e: e["delta"]))
    for e in split:
        side = out.setdefault("split_at_event", {}).setdefault(e["side"], {})
        side[e["type"]] = entry(e)
    return out


def main():
    args = parse_args()

    data, trials = load_data()

    print(data[0].keys())
    xlims = -30, 30
    # Extended for the v2 detections (was [(0, 8)] * 4 + [(0, 4)] * 4): the upper bands reach
    # 12.9 /min (loser, A1) and 4.6 /min (winner, D2); new tops are the next multiple of the tick step.
    yratelims = [(0, 16)] * 4 + [(0, 6)] * 4
    yeventlims = -0.5, np.max([len(d["centered_times"]) for d in data])

    fig, axs, texts = setup_plot()

    # For dataset 1 subplots 0+4, for dataset 2 subplots 1+5, etc.
    rate_plot_indices = [0, 1, 2, 3, 8, 9, 10, 11]
    event_plot_indices = [4, 5, 6, 7, 12, 13, 14, 15]
    full_dataset_eventplot = [4, 12]

    all_extrema = []  # collected for the final summary
    stats = {"loser": {}, "winner": {}}
    for i, (rp, ep) in enumerate(zip(rate_plot_indices, event_plot_indices)):
        d = data[i]
        if i < 4:
            chirpratec = loser_color
            group = "Loser"
        else:
            chirpratec = winner_color
            group = "Winner"
        event_label = texts[i % 4]

        extrema, split = plot_conv_rates(
            axs[rp],
            d,
            chirpratec=chirpratec,
            mark_significant=args.plot_significant,
        )
        report_extrema(group, event_label, extrema)
        report_extrema(group, event_label + " (split at t = 0)", split)
        all_extrema.append((group, event_label, extrema))

        n_trials, n_zero = plot_conv_rates_heatmap(axs[ep], d)
        stats[group.lower()][PANEL_EVENTS[i % 4]] = {
            "n_trials": n_trials,
            "n_trials_without_chirps": n_zero,
            **extrema_stats(extrema, split),
        }

        # add n to top right corner
        # axs[rp].text(
        #     1,
        #     0.95,
        #     f"n=workinprogress",
        #     transform=axs[rp].transAxes,
        #     verticalalignment="top",
        #     horizontalalignment="right",
        #     fontsize=6,
        #     color="k",
        # )

        axs[rp].set_xlim(*xlims)
        axs[rp].set_ylim(*yratelims[i])

        axs[ep].set_xlim(*xlims)

        if ep in full_dataset_eventplot:
            axs[ep].set_ylim(*yeventlims)
        else:
            axs[ep].set_ylim(0, n_trials)

        axs[rp].set_xticks([])
        sns.despine(
            offset=5,
            trim=True,
            top=True,
            right=True,
            bottom=True,
            left=False,
            ax=axs[rp],
        )
        sns.despine(
            offset=5,
            trim=True,
            top=True,
            right=True,
            bottom=False,
            left=False,
            ax=axs[ep],
        )

    # Compact summary: the strongest Δ in each of the 4 categories per panel.
    print()
    print("═" * 78)
    print(" Summary: strongest Δ (rate − CI boundary) per category, per panel")
    print("═" * 78)
    col = f"{'▲ before':>11}{'▼ before':>11}{'▲ after':>11}{'▼ after':>11}"
    print(f"  {'group':<7}{'event':<22}" + col)
    print("  " + "-" * 74)
    for group, event_label, ex in all_extrema:
        clean_event = (
            event_label.replace("\\textit{", "")
            .replace("}", "")
            .replace("\\textbf{", "")
        )
        by_key = {(e["side"], e["type"]): e for e in ex}

        def _cell(side, kind):
            e = by_key.get((side, kind))
            return f"{e['delta']:+.2f}" if e is not None else "—"

        cells = (
            f"{_cell('before', 'peak'):>11}"
            f"{_cell('before', 'trough'):>11}"
            f"{_cell('after', 'peak'):>11}"
            f"{_cell('after', 'trough'):>11}"
        )
        print(f"  {group:<7}{clean_event:<22}{cells}")
    print()
    write_stats("event_rates", stats)

    xoffsets = [-0.4, -0.15, -0.15, -0.15] * 2
    yoffsets_top = [1.25] * 4
    yoffsets_bottom = [1.15] * 4

    yoffsets = np.concatenate([yoffsets_top, yoffsets_bottom])

    letters = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]
    letter_subplots(
        letters=letters,
        axes=np.array(axs)[rate_plot_indices],
        xoffset=xoffsets,
        yoffset=yoffsets,
        fontsize="large",
        fontweight="bold",
    )

    for fmt in formats:
        plt.savefig(targetpath.with_suffix(f".{fmt}"), dpi=300)
        pass
    plt.show()


if __name__ == "__main__":
    main()
