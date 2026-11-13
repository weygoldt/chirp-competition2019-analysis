# Ported from chirp_competition d040123 (src/plot_event_time_rates_winner_loser_chirp.py; made the paper PDF, 16 x 11 cm).
# Changes: JSONs from $PAPER_OUT/data/processed/event_time_rates, figures via paper_config.figure (png, pdf, svg as before);
# rate-axis limits extended (loser 10 -> 15, winner 4 -> 6 /min) because the v2 rates would be clipped.
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


def setup_plot():
    cm = 1 / 2.54
    dpi = 300
    figsize = (16 * cm, 11 * cm)
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
        if i in [0, 1, 4, 5, 8, 9, 12, 13]:
            continue
        ax.set_yticklabels([])
    for top_ax, bottom_ax in zip(top_event_axes[1:], bottom_event_axes[1:]):
        top_ax.set_yticks([0, 15])
        bottom_ax.set_yticks([0, 15])
    axs[4].set_yticks([0, 15, 30])
    axs[12].set_yticks([0, 15, 30])
    fig.align_ylabels()
    return fig, axs, texts


# def setup_plot():
#     cm = 1 / 2.54
#     dpi = 300
#     figsize = (16 * cm, 11 * cm)
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#
#     meta_spacings = {
#         "left": 0.09,
#         "right": 0.98,
#         "top": 0.92,
#         "bottom": 0.13,
#         "hspace": 0.35,
#         "wspace": 0.3,
#     }
#     sub_spacings = {
#         "hspace": 0.2,
#         "wspace": 0.4,
#     }
#
#     gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], **meta_spacings)
#     gs1 = gs[0].subgridspec(2, 4, height_ratios=[1, 0.5], **sub_spacings)
#     gs2 = gs[1].subgridspec(2, 4, height_ratios=[1, 0.5], **sub_spacings)
#
#     top_rate_axes = [fig.add_subplot(gs1[0, i]) for i in range(gs1.ncols)]
#     top_event_axes = [fig.add_subplot(gs1[1, i]) for i in range(gs1.ncols)]
#     bottom_rate_axes = [fig.add_subplot(gs2[0, i]) for i in range(gs2.ncols)]
#     bottom_event_axes = [fig.add_subplot(gs2[1, i]) for i in range(gs2.ncols)]
#
#     axs = top_rate_axes + top_event_axes + bottom_rate_axes + bottom_event_axes
#
#     for ax in top_rate_axes + bottom_rate_axes:
#         ax.axvline(0, c="k", lw=0.5, ls="--")
#         ax.patch.set_facecolor("none")
#
#     # add descriptor text in top left corner
#     texts = [
#         "Other's chirps",
#         "Chasing \\textit{onset}",
#         "Chasing \\textit{offset}",
#         "Contact",
#     ]
#
#     for i, ax in enumerate(axs[:4]):
#         ax.text(
#             0.5,
#             1.3,
#             texts[i],
#             transform=ax.transAxes,
#             fontsize="medium",
#             verticalalignment="top",
#             horizontalalignment="center",
#             bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
#         )
#
#     # y label to left plots
#     # axs[0].set_ylabel("Chirp rate [$min^{-1}$]")
#     axs[0].set_ylabel("Chirps/min")
#     axs[4].set_ylabel("Trial")
#     axs[8].set_ylabel("Chirps/min")
#     # axs[8].set_ylabel("Chirp rate [$min^{-1}$]")
#     axs[12].set_ylabel("Trial")
#
#     # x label to lower plots
#     axs[14].text(
#         -0.15,
#         -0.95,
#         "Time from event [s]",
#         transform=axs[14].transAxes,
#         verticalalignment="center",
#         horizontalalignment="center",
#     )
#
#     # Add loser/winner indicator to first plot of each row
#     axs[0].text(
#         0.0,
#         1,
#         "\\textbf{Loser}",
#         transform=axs[0].transAxes,
#         verticalalignment="top",
#         horizontalalignment="left",
#         fontsize="medium",
#         color=loser_color,
#         fontweight="bold",
#     )
#     axs[8].text(
#         0.0,
#         1,
#         "\\textbf{Winner}",
#         transform=axs[8].transAxes,
#         verticalalignment="top",
#         horizontalalignment="left",
#         fontsize="medium",
#         color=winner_color,
#         fontweight="bold",
#     )
#
#     # disable unnecessary ticklabels
#     for i, ax in enumerate(axs):
#         if i < 12:
#             ax.set_xticklabels([])
#         if i in [0, 1, 4, 5, 8, 9, 12, 13]:
#             continue
#         ax.set_yticklabels([])
#
#     for top_ax, bottom_ax in zip(top_event_axes[1:], bottom_event_axes[1:]):
#         top_ax.set_yticks([0, 15])
#         bottom_ax.set_yticks([0, 15])
#
#     axs[4].set_yticks([0, 15, 30])
#     axs[12].set_yticks([0, 15, 30])
#
#     fig.align_ylabels()
#     return fig, axs, texts


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


def plot_conv_rates(ax, data, chirpratec=wchirpratec, baselinec=baselinec):
    # print(data.keys())
    time = np.array(data["conv_time"])
    pct = np.percentile(data["bootstrapped_baseline_jittered"], [2.5, 50, 97.5], axis=0)
    pct *= 60
    ax.plot(time, pct[1], c=baselinec, lw=1)
    ax.fill_between(time, pct[0], pct[2], alpha=0.5, color=baselinec, lw=0)

    bl = np.mean(pct[1])

    pct = np.percentile(data["jackknifed_eventrate"], [2.5, 50, 97.5], axis=0)
    pct *= 60
    ax.plot(time, pct[1], c=chirpratec, lw=1)
    ax.fill_between(time, pct[0], pct[2], alpha=0.5, color=chirpratec, lw=0)

    pos_rate = pct[1][time > 0]
    pos_time = time[time > 0]
    peak_time = pos_time[np.argmax(pos_rate)]
    peak_val = pos_rate[np.argmax(pos_rate)]

    print(f"Peak time: {peak_time}")
    print(f"Peak value: {peak_val}")
    print(f"Baseline: {bl}")
    print(f"Diff: {peak_val - bl}")


def plot_conv_rates_heatmap(ax, data):
    time = np.array(data["conv_time"])
    per_trial_rates = np.array(data["per_trial_rates"])

    # simple normalization
    per_trial_rates = per_trial_rates / (
        np.max(per_trial_rates, axis=1)[:, None] + 1e-6
    )

    n_zero = np.sum(np.sum(per_trial_rates, axis=1) == 0)
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
    return n


def main():
    data, trials = load_data()

    print(data[0].keys())
    xlims = -30, 30
    # Extended for the v2 detections (was [(0, 10)] * 4 + [(0, 4)] * 4): the upper bands reach
    # 12.9 /min (loser, A1) and 4.6 /min (winner, D2); new tops are the next multiple of the tick step.
    yratelims = [(0, 15)] * 4 + [(0, 6)] * 4
    yeventlims = -0.5, np.max([len(d["centered_times"]) for d in data])

    fig, axs, texts = setup_plot()

    # For dataset 1 subplots 0+4, for dataset 2 subplots 1+5, etc.
    rate_plot_indices = [0, 1, 2, 3, 8, 9, 10, 11]
    event_plot_indices = [4, 5, 6, 7, 12, 13, 14, 15]
    full_dataset_eventplot = [4, 12]

    stats = []
    stat = None
    for i, (rp, ep) in enumerate(zip(rate_plot_indices, event_plot_indices)):
        d = data[i]
        if i < 4:
            chirpratec = loser_color
            stat = "loser"
        else:
            chirpratec = winner_color
            stat = "winner"

        print(f"Status: {stat}")
        if i < 4:
            print(texts[i])
        else:
            print(print(texts[i - 4]))

        plot_conv_rates(axs[rp], d, chirpratec=chirpratec)

        n_trials = plot_conv_rates_heatmap(axs[ep], d)

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
        # pass
    plt.show()


if __name__ == "__main__":
    main()
