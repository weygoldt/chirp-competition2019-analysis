# Ported from chirp_competition d040123 (src/plot_competition_trial_overview.py).
# Changes: input JSON from PROCESSED (rebuilt by make_competition_overview.py, chirps from
# PAPER_DETECTIONS); figures via paper_config.figure().
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from thunderlab.powerspectrum import decibel

from paper_config import figure, processed
from style import (
    set_light_style,
    loser_color,
    winner_color,
    adjust_alpha,
    letter_subplots,
)

set_light_style()
filename = Path(__file__).stem
targetpath = figure(filename)


def setup_plot():
    cm = 1 / 2.54
    figsize = (17.4 * cm, 8 * cm)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        height_ratios=(1, 0.1, 0.1, 0.001),
        dpi=300,
    )

    ticklabels = [
        ["Contact", "Chasing"],
        # [
        #     "$\\text{Rises}_{\\text{loser}}$",
        #     "$\\text{Rises}_{\\text{winner}}$",
        # ],
        [
            "$\\text{Chirps}_{\\text{loser}}$",
            "$\\text{Chirps}_{\\text{winner}}$",
        ],
    ]
    for i, a in enumerate(axes[1:-1]):
        a.set_xticks([])
        a.spines["bottom"].set_visible(False)
        a.set_ylim(0, 1)
        a.set_yticks([0, 1])
        a.tick_params(pad=50)
        a.set_yticklabels(ticklabels[i], fontsize="small", ha="left")

    axes[-1].set_xlabel("Time [h]")
    axes[-1].spines["left"].set_visible(False)
    axes[-1].set_yticks([])

    axes[0].set_ylabel("Frequency [Hz]", labelpad=10)
    axes[0].spines["bottom"].set_visible(False)
    axes[0].set_xticks([])

    plt.subplots_adjust(
        hspace=0.3,
        wspace=0.1,
        top=0.97,
        bottom=0.12,
        left=0.13,
        right=0.98,
    )
    return fig, axes


def main():
    fig, axes = setup_plot()

    datapath = processed("competition_overview_2020-06-09-10_00.json")

    with open(datapath) as f:
        data = json.load(f)

    fund_v = np.array(data["fund_v"])
    ident_v = np.array(data["ident_v"])
    idx_v = np.array(data["idx_v"])
    times = np.array(data["times"])
    spec = np.array(data["spec"])
    win_id = np.array(data["win_id"])
    lose_id = np.array(data["lose_id"])
    contact_t_GRID = np.array(data["contact_t_GRID"])
    ag_on_off_t_GRID = np.array(data["ag_on_off_t_GRID"])
    rise_idx_int = data["rise_idx_int"]
    chirp_times = data["chirp_times"]
    recording = data["recording"]

    chirp_times = [np.array(arr) for arr in chirp_times]

    Wc = winner_color
    Lc = loser_color

    min_f, max_f = (
        np.min(fund_v[~np.isnan(ident_v)]),
        np.nanmax(fund_v[~np.isnan(ident_v)]),
    )
    axes[0].set_ylim(min_f - 50, max_f + 50)

    for a in axes:
        a.set_xlim(0, np.max(times) / 3600)

    cmap = sns.color_palette("mako", as_cmap=True)
    axes[0].imshow(
        decibel(spec)[::-1],
        aspect="auto",
        extent=[times[0] / 3600, times[-1] / 3600, 0, 2000],
        cmap=cmap,
        # cmap="Greys_r",
        vmin=np.min(decibel(spec)) + np.abs(0.12 * np.min(decibel(spec))),
        vmax=np.max(decibel(spec)) - np.abs(0.3 * np.max(decibel(spec))),
        origin="upper",
        rasterized=True,
        interpolation="bilinear",
    )

    print(Wc, Lc)

    # axes[0].plot(
    #     times[idx_v[ident_v == win_id]] / 3600,
    #     fund_v[ident_v == win_id],
    #     color=adjust_alpha(Wc, 0.8),
    #     lw=0.5,
    #     alpha=1,
    # )
    # axes[0].plot(
    #     times[idx_v[ident_v == lose_id]] / 3600,
    #     fund_v[ident_v == lose_id],
    #     color=adjust_alpha(Lc, 0.8),
    #     lw=0.5,
    #     alpha=1,
    # )
    # axes[0].axvspan(
    #     3,
    #     6,
    #     color="white",
    #     alpha=0.2,
    #     label="light",
    #     lw=0,
    #     zorder=1000,
    # )
    axes[0].axvline(3, color="white", lw=0.5, zorder=1000, ls="--")

    markerkwargs = dict(
        markersize=10,
        markeredgewidth=0.5,
        alpha=1,
    )
    axes[1].plot(
        contact_t_GRID / 3600,
        np.ones_like(contact_t_GRID) * 0,
        "|",
        **markerkwargs,
        color="k",
    )
    axes[1].plot(
        ag_on_off_t_GRID[:, 0] / 3600,
        np.ones_like(ag_on_off_t_GRID[:, 0]) * 1,
        "|",
        **markerkwargs,
        color="k",
    )

    """
    axes[2].plot(
        times[rise_idx_int[0]] / 3600,
        np.ones_like(rise_idx_int[0]) * 1,
        "|",
        **markerkwargs,
        color=Wc,
    )
    axes[2].plot(
        times[rise_idx_int[1]] / 3600,
        np.ones_like(rise_idx_int[1]) * 0,
        "|",
        **markerkwargs,
        color=Lc,
    )
    """

    axes[2].plot(
        chirp_times[0] / 3600,
        np.ones_like(chirp_times[0]) * 1,
        "|",
        markeredgecolor=Wc,
        **markerkwargs,
    )
    axes[2].plot(
        chirp_times[1] / 3600,
        np.ones_like(chirp_times[1]) * 0,
        "|",
        markeredgecolor=Lc,
        **markerkwargs,
    )

    recording = recording.replace("_", ":")
    spl = recording.split("-")
    rec = "-".join(spl[:-1])
    rec = rec + " " + spl[-1]
    axes[0].text(
        0.99,
        0.98,
        "\\textit{" + f"{rec}" + "}",
        fontsize="x-small",
        color="grey",
        transform=axes[0].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        alpha=0.5,
    )

    axes[0].text(
        0.75,
        0.925,
        "\\textit{Inactive period (light on)}",
        fontsize=7,
        color="white",
        transform=axes[0].transAxes,
        verticalalignment="center",
        horizontalalignment="center",
    )

    axes[0].text(
        0.25,
        0.925,
        "\\textit{Active period (light off)}",
        fontsize=7,
        color="white",
        transform=axes[0].transAxes,
        verticalalignment="center",
        horizontalalignment="center",
        # fontproperties=fp,
    )

    axes[0].text(
        0.01,
        0.19,
        "Winner EOD$f$",
        fontsize=7,
        color=adjust_alpha(Wc, 0.5),
        transform=axes[0].transAxes,
        verticalalignment="center",
        horizontalalignment="left",
    )

    axes[0].text(
        0.01,
        0.61,
        "Loser EOD$f$",
        fontsize=7,
        color=adjust_alpha(Lc, 0.5),
        transform=axes[0].transAxes,
        verticalalignment="center",
        horizontalalignment="left",
        # fontproperties=fp,
    )

    letter_subplots(axes[:-2], xoffset=-0.14, yoffset=[1.0, 1.45])
    fig.align_ylabels(axes[1:-1])

    fmts = ["pdf", "png", "svg"]
    for fmt in fmts:
        plt.savefig(
            targetpath.with_suffix(f".{fmt}"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


if __name__ == "__main__":
    main()
