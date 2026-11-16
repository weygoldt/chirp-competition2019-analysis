# Ported from chirp_competition d040123 (src_paper/plot_event_trial_progression_correlations_15_fish.py).
# Changes: inputs from PROCESSED; figure (pdf/png/svg) via paper_config.figure(); print_summary (a
# byte-identical copy of the all-fish script's) imported from there, its numbers go to write_stats
# ("allocation_15_fish").
import os
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from manuscript_stats import write_stats
from paper_config import PROCESSED, figure, processed
from plot_event_trial_progression_correlations_all_fish import print_summary
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
targetpath = figure(filename)

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
        hspace=0.15,
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
        hspace=0.3,
        wspace=0.5,
        top=0.92,
        right=0.98,
        left=0.1,
        bottom=0.12,
    )

    return fig, bar_axes, rate_axes, timing_axes


def setup_supplement_plot():
    """Supplement figure: cumulative allocation curves."""
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 4, figsize=(16 * cm, 4.5 * cm), dpi=200)

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
        ax.set_xticks([0, 5, 10, 15])

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

        for trial_trace in data[key]["kde_ratios"]:
            ax.plot(kde_time, trial_trace, **trial_rateplot_linekwargs)

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
        ax.axhline(1, lw=0.5, ls="-", c="grey", zorder=0)
        ax.axvspan(3, 6, color="grey", alpha=0.3, lw=0, zorder=0)
        ax.axvline(3, color="k", linestyle="dashed", lw=0.5, zorder=10000)
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
    w_mode = l_mode = None

    for j, arr in enumerate(all_data):
        min_data, max_data = np.min(arr), np.max(arr)
        arr = remove_outliers(arr, hdi_prob=0.95)
        current_max = np.max(arr)

        kde = gaussian_kde(arr)
        xpad = 0.1 * (np.max(arr) - np.min(arr))
        xkde = np.linspace(np.min(arr) - xpad, np.max(arr) + xpad, 1000)
        xkde = xkde[xkde >= 0]
        ykde = kde(xkde)
        mode_value = xkde[np.argmax(ykde)]

        ax_idx = data_info[j][1]
        axes[ax_idx].fill_between(
            xkde,
            ykde,
            **rateplot_fillkwargs,
            color=colors[j],
        )

        u = units[j]
        if j < 3:
            maxx.append(current_max)
            axes[ax_idx].text(
                0.95,
                0.95,
                f"mode={mode_value:.1f} {u}\nmin={min_data:.1f} {u}\nmax={max_data:.1f} {u}",
                ha="right",
                va="top",
                fontsize="small",
                transform=axes[ax_idx].transAxes,
            )
        elif j == 3:
            maxx.append(current_max)
            w_mode = mode_value
        else:
            l_mode = mode_value
            axes[ax_idx].text(
                0.95,
                0.95,
                rf"$\text{{mode}}_{{w}}={w_mode:.1f}$ {u}"
                + "\n"
                + rf"$\text{{mode}}_{{l}}={l_mode:.1f}$ {u}",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data = load_data(PROCESSED / "trial_event_progressions_annotated_subset")
    with open(processed("agonistic_behavior_timing.json")) as f:
        timing = json.load(f)

    # --- Summary statistics for manuscript ---
    write_stats("allocation_15_fish", print_summary(data, timing))

    # --- Main figure ---
    fig, bar_axes, rate_axes, timing_axes = setup_main_plot()

    counts = plot_barplots(bar_axes, data)
    plot_rates(rate_axes, data, counts)

    plot_cumulative(timing_axes, data)

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

    plt.show()


if __name__ == "__main__":
    main()
