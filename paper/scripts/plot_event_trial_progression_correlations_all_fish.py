import os
import argparse
import json
from pathlib import Path
from IPython import embed

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from style import (
    baseline_color,
    letter_subplots,
    loser_color,
    other_color,
    set_light_style,
    winner_color,
)

set_light_style()

formats = ["pdf", "png", "svg"]
filename = Path(__file__).stem
FIGURE_DIR = Path(os.environ.get("PAPER_FIGURE_DIR", "/home/weygoldt/wrk/analyses/chirp_competition_paper/figures"))

rateplot_linekwargs = dict(lw=1, alpha=1, zorder=100)
rateplot_fillkwargs = dict(alpha=0.5, zorder=99, lw=0)
trial_rateplot_linekwargs = dict(lw=0.5, alpha=0.5, color="lightgrey", zorder=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def remove_outliers(timestamps, hdi_prob=0.95):
    timestamps = np.array(timestamps)
    timestamps = timestamps[~np.isnan(timestamps)]
    if len(timestamps) <= 1:
        raise ValueError("Not enough data points to calculate HDI")
    hdi = az.hdi(timestamps, hdi_prob)
    return timestamps[(timestamps > hdi[0]) & (timestamps < hdi[1])]


def load_data(path):
    datasets = {}
    for file in path.iterdir():
        if file.suffix == ".json":
            with open(file) as f:
                datasets[file.stem] = json.load(f)
    return datasets


def plot_cumdist(baseline_qrts, observed_qrts, time, ax, bl_color, obs_color):
    ax.fill_between(
        time,
        baseline_qrts[0],
        baseline_qrts[2],
        **rateplot_fillkwargs,
        color=bl_color,
    )
    ax.plot(
        time,
        baseline_qrts[1],
        label="Randomized",
        **rateplot_linekwargs,
        color=bl_color,
    )
    ax.fill_between(
        time,
        observed_qrts[0],
        observed_qrts[2],
        **rateplot_fillkwargs,
        color=obs_color,
    )
    ax.plot(
        time,
        observed_qrts[1],
        label="Observed",
        **rateplot_linekwargs,
        color=obs_color,
    )


# ---------------------------------------------------------------------------
# Figure layouts
# ---------------------------------------------------------------------------
def setup_main_plot():
    """Main figure: barplots (top), rate panels (middle), timing PDFs (bottom)."""
    cm = 1 / 2.54
    fig = plt.figure(figsize=(17.4 * cm, 12 * cm), dpi=200)

    outer_gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[2, 1],
        hspace=0.2,
    )

    top_gs = outer_gs[0].subgridspec(
        nrows=2,
        ncols=4,
        hspace=1,
        height_ratios=[0.5, 1],
    )
    bottom_gs = outer_gs[1].subgridspec(
        nrows=2,
        ncols=4,
        hspace=0.2,
        height_ratios=[0.5, 1],
    )

    bar_axes = [fig.add_subplot(top_gs[0, c]) for c in range(4)]
    rate_axes = [fig.add_subplot(top_gs[1, c]) for c in range(4)]
    timing_axes = [fig.add_subplot(bottom_gs[1, c]) for c in range(4)]

    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.5,
        top=0.92,
        right=0.98,
        left=0.1,
        bottom=0.14,
    )

    return fig, bar_axes, rate_axes, timing_axes


def setup_supplement_plot():
    """Supplement figure: cumulative allocation curves."""
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 4, figsize=(17.4 * cm, 4.5 * cm), dpi=200)

    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.5,
        top=0.91,
        right=0.98,
        left=0.1,
        bottom=0.25,
    )

    return fig, axes


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_barplots(axes, data):
    """Row of per-trial event-count barplots."""
    keys = ["chasing", "contact", "chirp$_{lose}$", "chirp$_{win}$"]
    colors = [other_color, other_color, loser_color, winner_color]
    labels = ["Chasing", "Contact", "Loser chirps", "Winner chirps"]
    counts = []

    for ax, key, color, label in zip(axes, keys, colors, labels):
        ybar = np.array(data[key]["event_count"])
        xbar = np.arange(len(ybar)) + 0.5
        ax.bar(xbar, ybar, color=color, alpha=0.5)
        counts.append(np.sum(ybar))

        ax.text(
            0.5,
            1.1,
            f"{label}\n$n_{{\\text{{obs}}}}={np.sum(ybar)}$",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

    for i, ax in enumerate(axes):
        ax.set_xlabel("Trial")
        if i == 0:
            ax.set_ylabel("Count")
        if i < 2:
            ax.set_xticks([0, 5, 10, 15])
        else:
            ax.set_xticks([0, 15, 30])

    return counts


def plot_rates(axes, data, counts):
    """Row of normalised-rate panels with per-trial traces and quantile bands."""
    keys = ["chasing", "contact", "chirp$_{lose}$", "chirp$_{win}$"]
    colors_main = [other_color, other_color, loser_color, winner_color]
    labels = ["Chasing", "Contact", "Loser chirps", "Winner chirps"]

    for ax, key, color, label, count in zip(
        axes,
        keys,
        colors_main,
        labels,
        counts,
    ):
        kde_time = np.array(data[key]["kde_times"]) / 3600
        kde_qts = np.quantile(
            data[key]["kde_ratios"],
            [0.025, 0.5, 0.975],
            axis=0,
        )

        # for trial_trace in data[key]["kde_ratios"]:
            # ax.plot(kde_time, trial_trace, **trial_rateplot_linekwargs)

        ax.fill_between(
            kde_time,
            kde_qts[0],
            kde_qts[2],
            **rateplot_fillkwargs,
            color=color,
        )
        ax.plot(
            kde_time,
            kde_qts[1],
            label=label,
            **rateplot_linekwargs,
            color=color,
        )

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 6)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_ylim(0, 6)
        ax.set_yticks([0, 2, 4, 6])
        # ax.axhline(1, lw=0.5, ls="-", c="grey", zorder=0)
        # ax.axvspan(3, 6, color="grey", alpha=0.3, lw=0, zorder=0)
        ax.axvline(3, color="k", linestyle="dashed", lw=0.75, zorder=10000)
        ax.set_xlabel("Time [h]")
        if i == 0:
            ax.set_ylabel("Norm. rate")
        else:
            ax.set_yticklabels([])


def plot_timing_pdfs(axes, timing_data):
    """Row of KDE-based PDFs for event durations and intervals."""
    data_info = [
        ("chasing_durations", 0),
        ("inter_chasing_intervals", 1),
        ("inter_contact_intervals", 2),
        ("winner_inter_chirp_intervals", 3),
        ("loser_inter_chirp_intervals", 3),
    ]
    colors = [other_color, other_color, other_color, winner_color, loser_color]
    units = ["s", "s", "s", "s", "s"]

    # collect data across trials
    all_data = [[] for _ in data_info]
    for trial in timing_data:
        for j, (key, _) in enumerate(data_info):
            all_data[j].extend(trial[key])

    maxx = []
    stats = {}

    for j, arr in enumerate(all_data):
        # Median and IQR of all pooled values, as reported in the text
        # (computed before trimming outliers for the density estimate).
        med = float(np.median(arr))
        q1, q3 = (float(q) for q in np.percentile(arr, [25, 75]))
        stats[j] = (med, q1, q3)
        arr = remove_outliers(arr, hdi_prob=0.95)
        current_max = np.max(arr)

        kde = gaussian_kde(arr)
        xpad = 0.1 * (np.max(arr) - np.min(arr))
        xkde = np.linspace(np.min(arr) - xpad, np.max(arr) + xpad, 1000)
        xkde = xkde[xkde >= 0]
        ykde = kde(xkde)

        ax_idx = data_info[j][1]
        # if ax_idx == 3:
        #     embed()
        #     exit()
        axes[ax_idx].fill_between(
            xkde,
            ykde,
            **rateplot_fillkwargs,
            color=colors[j],
        )
        # Interquartile range: darker shading under the density
        in_iqr = (xkde >= q1) & (xkde <= q3)
        axes[ax_idx].fill_between(
            xkde[in_iqr],
            ykde[in_iqr],
            **rateplot_fillkwargs,
            color=colors[j],
        )
        # Median: vertical line up to the density
        axes[ax_idx].plot(
            [med, med], [0, float(kde(med)[0])], color="k", lw=0.8, zorder=101
        )

        u = units[j]
        if j < 3:
            maxx.append(current_max)
            axes[ax_idx].text(
                0.95,
                0.95,
                f"median {med:.1f} {u}\nIQR {q1:.1f}–{q3:.1f} {u}",
                ha="right",
                va="top",
                fontsize="small",
                transform=axes[ax_idx].transAxes,
            )
        elif j == 3:
            maxx.append(current_max)
        else:
            (wm, w1, w3), (lm, l1, l3) = stats[3], stats[4]
            axes[ax_idx].text(
                1.02,
                0.95,
                f"median (IQR) [{u}]\nW {wm:.1f} ({w1:.1f}–{w3:.1f})"
                f"\nL {lm:.1f} ({l1:.1f}–{l3:.1f})",
                ha="right",
                va="top",
                fontsize="small",
                transform=axes[ax_idx].transAxes,
            )

    xticks = [[0, 10, 20], [0, 150, 300], [0, 500, 1000], [0, 20, 40]]
    xlabels = (
        "Chasing duration [s]",
        "Inter-chasing\ninterval [s]",
        "Inter-contact\ninterval [s]",
        "Inter-chirp\ninterval [s]",
    )
    for i, ax in enumerate(axes):
        ax.set_xlim(-0.1 * maxx[i], xticks[i][-1])
        ax.set_xticks(xticks[i])
        ax.set_xlabel(xlabels[i])
        if i == 0:
            ax.set_ylabel("PDF")


def plot_cumulative(axes, data):
    """Supplement: cumulative allocation curves."""
    keys = ["chasing", "contact", "chirp$_{lose}$", "chirp$_{win}$"]
    obs_colors = [other_color, other_color, loser_color, winner_color]

    for ax, key, obs_color in zip(axes, keys, obs_colors):
        bl = np.nanpercentile(data[key]["cumdist_baseline"], [2.5, 50, 97.5], axis=0)
        obs = np.nanpercentile(data[key]["cumdist_observed"], [2.5, 50, 97.5], axis=0)
        time = np.array(data[key]["cumdist_time"]) / 3600
        plot_cumdist(bl, obs, time, ax, baseline_color, obs_color)

    axes[0].legend(
        frameon=False,
        fontsize="small",
        bbox_to_anchor=(-0.05, 1.08),
        loc="upper left",
        handlelength=1,
        handleheight=1,
    )

    for i, ax in enumerate(axes):
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 1.5, 3])
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Cumulative")
        if i != 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")


def print_summary(data, timing_data):
    """Print manuscript-ready summary statistics to the terminal."""
    dark_period_min = 3 * 60  # 3 h dark period in minutes

    # --- Per-trial event counts and rates ---
    keys = ["chasing", "contact", "chirp$_{lose}$", "chirp$_{win}$"]
    names = ["Chasing", "Contact", "Loser chirps", "Winner chirps"]

    print("\n" + "=" * 70)
    print("MANUSCRIPT SUMMARY STATISTICS")
    print("=" * 70)

    per_trial_counts = {}
    for key, name in zip(keys, names):
        counts = np.array(data[key]["event_count"])
        total = int(np.sum(counts))
        rates = counts / dark_period_min  # events per minute per trial
        per_trial_counts[key] = counts
        print(f"\n--- {name} ---")
        print(f"  Total events:       {total}")
        print(f"  Trials:             {len(counts)}")
        print(
            f"  Per-trial count:    median={np.median(counts):.0f}, "
            f"range=[{np.min(counts):.0f}, {np.max(counts):.0f}]"
        )
        print(
            f"  Rate [events/min]:  median={np.median(rates):.2f}, "
            f"range=[{np.min(rates):.2f}, {np.max(rates):.2f}], "
            f"IQR=[{np.percentile(rates, 25):.2f}, {np.percentile(rates, 75):.2f}]"
        )

    # --- Rate dynamics (normalized rate at start vs end of dark period) ---
    print("\n--- Normalised rate: first hour vs last hour of dark period ---")
    for key, name in zip(keys, names):
        kde_time = np.array(data[key]["kde_times"]) / 3600
        kde_ratios = np.array(data[key]["kde_ratios"])
        first_hour = kde_ratios[:, kde_time <= 1]
        last_hour = kde_ratios[:, (kde_time >= 2) & (kde_time <= 3)]
        med_first = np.median(np.median(first_hour, axis=1))
        med_last = np.median(np.median(last_hour, axis=1))
        print(
            f"  {name:20s}  first_h median={med_first:.2f}, "
            f"last_h median={med_last:.2f}"
        )

    # --- Timing distributions (pooled across trials) ---
    timing_keys = [
        ("chasing_durations", "Chasing durations [s]"),
        ("inter_chasing_intervals", "Inter-chasing intervals [s]"),
        ("inter_contact_intervals", "Inter-contact intervals [s]"),
        ("winner_inter_chirp_intervals", "Winner inter-chirp intervals [s]"),
        ("loser_inter_chirp_intervals", "Loser inter-chirp intervals [s]"),
    ]

    print("\n--- Timing distributions (pooled across all trials) ---")
    for tkey, label in timing_keys:
        pooled = []
        for trial in timing_data:
            pooled.extend(trial[tkey])
        pooled = np.array(pooled)
        kde = gaussian_kde(pooled)
        xkde = np.linspace(max(0, np.min(pooled)), np.max(pooled), 5000)
        mode = xkde[np.argmax(kde(xkde))]
        print(f"\n  {label}")
        print(
            f"    n={len(pooled)}, min={np.min(pooled):.1f}, max={np.max(pooled):.1f}"
        )
        print(
            f"    median={np.median(pooled):.1f}, "
            f"mean={np.mean(pooled):.1f}, "
            f"mode≈{mode:.1f}"
        )
        print(
            f"    IQR=[{np.percentile(pooled, 25):.1f}, "
            f"{np.percentile(pooled, 75):.1f}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(annotated_only=False):
    # Choose input data and output figure names based on mode.
    if annotated_only:
        data_subdir = "trial_event_progressions_annotated_subset"
        timing_file = Path(
            "data/processed/agonistic_behavior_timing_annotated_subset.json"
        )
        name_suffix = "_annotated_subset"
    else:
        data_subdir = "trial_event_progressions_all_fish"
        timing_file = Path("data/processed/agonistic_behavior_timing.json")
        name_suffix = ""

    targetpath = FIGURE_DIR / f"{filename}{name_suffix}"

    data = load_data(Path("data/processed") / data_subdir)
    with open(timing_file) as f:
        timing = json.load(f)

    # --- Summary statistics for manuscript ---
    print_summary(data, timing)

    # --- Main figure ---
    fig, bar_axes, rate_axes, timing_axes = setup_main_plot()

    counts = plot_barplots(bar_axes, data)
    plot_rates(rate_axes, data, counts)
    plot_timing_pdfs(timing_axes, timing)

    all_axes = bar_axes + rate_axes
    xoffsets = [-0.47, -0.2, -0.2, -0.2] * 2
    yoffsets = [1.3] * 4 + [1.1] * 4
    letters = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]
    letter_subplots(axes=all_axes, letters=letters, xoffset=xoffsets, yoffset=yoffsets)

    letters_t = ["E", "F", "G", "H"]
    letter_subplots(
        axes=timing_axes,
        letters=letters_t,
        xoffset=[-0.47, -0.2, -0.2, -0.2],
        yoffset=1.05,
    )

    fig.align_ylabels()
    sns.despine(fig=fig, offset=5, trim=True)

    for fmt in formats:
        fig.savefig(targetpath.with_suffix(f".{fmt}"), dpi=300)

    # --- Supplement figure ---
    fig_supp, cum_axes = setup_supplement_plot()

    plot_cumulative(cum_axes, data)

    letter_subplots(
        axes=list(cum_axes),
        letters=["A", "B", "C", "D"],
        xoffset=[-0.47, -0.2, -0.2, -0.2],
        yoffset=1.05,
    )

    fig_supp.align_ylabels()
    sns.despine(fig=fig_supp, offset=5, trim=True)

    targetpath_suff = targetpath.parent / f"{targetpath.name}_supplement"
    for fmt in formats:
        fig_supp.savefig(
            targetpath_suff.with_suffix(f".{fmt}"),
            dpi=300,
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot temporal allocation of events (chirps, contacts, chasing) "
            "across trials. Default mode reads from "
            "data/processed/trial_event_progressions_all_fish/ and produces "
            "the main figure with all available trials. Use --annotated-only "
            "to read from the paired-subset directory and save figures with "
            "an '_annotated_subset' suffix."
        )
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help=(
            "Plot the paired-subset figure: reads from "
            "data/processed/trial_event_progressions_annotated_subset/ and "
            "agonistic_behavior_timing_annotated_subset.json."
        ),
    )
    args = parser.parse_args()
    main(annotated_only=args.annotated_only)
