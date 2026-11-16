# Ported from chirp_competition d040123 (src/plot_chirp_distributions_manuscript.py).
# Changes: boxes from the v2 parquet (chirps.load_chirp_table, rows with keep) instead of the v1
# chirpdetector_bboxes.csv: width = t_end_s - t_start_s, height = f_high_hz - track_hz,
# trough = -(track_hz - f_low_hz) (track_hz replaces v1 emitter_eodf); recordings visited in sorted
# order; data root DATA_ROOT; figure via paper_config.figure(). Limits extended for v2 data that the
# old ones clipped: xlims 600 -> 900 ms, ylims/yticks 700 -> 900 Hz, hexbin LogNorm vmax 3000 -> 7000.
# Recordings are found by the parquet of PAPER_DETECTIONS; the plotted distributions (median, central
# 90 %, range) and counts go to write_stats ("chirp_distributions").
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

from chirps import load_chirp_table
from manuscript_stats import write_stats
from paper_config import DATA_ROOT, DETECTIONS, figure
from style import circle_annotate, set_light_style, letter_subplots

scriptname = Path(__file__).stem
targetpath = figure(scriptname)
set_light_style()

cmap = sns.color_palette("mako", as_cmap=True)
cmap_dens = sns.color_palette("mako", as_cmap=True)
# cmap_dens = sns.color_palette("magma", as_cmap=True)

xlims = (0, 900)  # old (0, 600): v2 box widths reach 859 ms
ylims = (0, 900)  # old (0, 700): v2 heights reach 892 Hz


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
def setup_plot():
    cm = 1 / 2.54
    fig = plt.figure(figsize=(17.4 * cm, 8 * cm), dpi=300)

    gs = GridSpec(
        nrows=2,
        ncols=5,
        figure=fig,
        width_ratios=[1, 0.1, 0.2, 1, 0.1],
        height_ratios=[0.1, 1],
    )

    # Left panel: scatter
    scatter_ax = fig.add_subplot(gs[1, 0])
    scatter_top = fig.add_subplot(gs[0, 0])
    scatter_right = fig.add_subplot(gs[1, 1])

    # Right panel: hexbin (no marginals — density is encoded in colour)
    hex_ax = fig.add_subplot(gs[1, 3])
    hex_top = fig.add_subplot(gs[0, 3])
    hex_right = fig.add_subplot(gs[1, 4])
    hex_top.set_visible(False)
    hex_right.set_visible(False)

    # Colorbar insets
    scatter_cb = scatter_ax.inset_axes(bounds=[0.73, 0.24, 0.25, 0.03])
    hex_cb = hex_ax.inset_axes(bounds=[0.73, 0.24, 0.25, 0.03])

    # Configure marginals
    for marginal in (scatter_top, scatter_right):
        marginal.set_xticks([])
        marginal.set_yticks([])

    scatter_top.spines["left"].set_visible(False)
    scatter_right.spines["bottom"].set_visible(False)

    scatter_ax.set_xlabel("Chirp width [ms]")
    scatter_ax.set_ylabel("Chirp height [Hz]")
    hex_ax.set_xlabel("Chirp width [ms]")

    plt.subplots_adjust(
        top=0.99,
        bottom=0.12,
        left=0.09,
        right=0.99,
        wspace=0.2,
        hspace=0.2,
    )

    scatter_axes = (scatter_ax, scatter_top, scatter_right, scatter_cb)
    hex_axes = (hex_ax, hex_cb)
    return fig, scatter_axes, hex_axes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_data():
    path1 = DATA_ROOT
    # path2 = Path("data/raw/remote/leonie/")
    paths = [
        path1,
        # path2,
    ]
    dfs = []
    for path in paths:
        dirs = sorted(path.iterdir())
        for dir in dirs:
            if dir.is_file():
                continue
            if not (dir / f"chirps_{DETECTIONS}.parquet").exists():
                print(f"WARNING: No data found in {dir}")
                continue
            # v2 detections; `keep` = the chirps of the legacy arrays
            df = load_chirp_table(dir)
            df = df[df["keep"]]
            dfs.append(df)
            print(f"Found data in {dir}, dataframe length: {len(df)}")

    n_datasets = len(dfs)
    if len(dfs) == 0:
        print("")
        print(
            "--> ERROR!!!!! Datasets not mounted at the expected locations, please open the file and specify the mountpoint or mount the data correctly"
        )
        print("")
        exit()
    df = pd.concat(dfs)

    # Restrict to the analysed dataset: winner and loser chirps of the 33
    # competitions (excluding draws and recordings whose baseline EODfs were
    # too close for reliable assignment), as reported in the manuscript.
    summary = pd.read_csv(path1 / "trial_summary.csv", index_col=0)
    bad_recordings = [
        "2020-05-21-10_00",
        "2020-05-25-10_00",
        "2020-05-28-10_00",
        "2020-06-10-10_00",
    ]
    summary = summary[(summary["draw"] != 1) & ~summary["recording"].isin(bad_recordings)]
    fish = {
        (r.recording, float(i))
        for r in summary.itertuples()
        for i in (r.win_ID, r.lose_ID)
    }
    keep = [
        (rec, tid) in fish for rec, tid in zip(df["recording"], df["track_id"].astype(float))
    ]
    df = df[np.array(keep)]
    print(f"Restricted to {summary['recording'].nunique()} competitions: {len(df)} chirps")

    # drop rows where track_id is nan
    old_len = len(df)
    # df = df[df["emitter_eodf"].notna()]
    df = df[df["track_id"].notna()]
    new_len = len(df)

    print(
        f"Dropped {old_len - new_len} of all {old_len} chirps due to missing assigned EODf"
    )
    print(f"Loaded data from {n_datasets} datasets in total")

    # get widths and heights (v1: t2 - t1, f2 - emitter_eodf, emitter_eodf - f1)
    widths = df.t_end_s - df.t_start_s
    heights = df.f_high_hz - df.track_hz
    troughs = df.track_hz - df.f_low_hz
    recordings = df.recording

    widths = np.array(widths)
    heights = np.array(heights)
    troughs = -np.array(troughs)
    recordings = np.array(recordings)

    # remove all where trough is greater than 3 times the height
    # maxtrough = -(heights * 0.75)
    # picker = troughs > maxtrough
    # widths = widths[picker]
    # heights = heights[picker]
    # troughs = troughs[picker]
    # recordings = recordings[picker]

    # print(f"Removed {len(picker)-np.sum(picker)} bboxes where trough is below thresh")

    print(f"Finished retrieving data, got {len(widths)} datapoints")
    print(f"n_competitions = {summary['recording'].nunique()}")

    return widths, heights, troughs, recordings


def distribution(values: np.ndarray) -> dict:
    """Median, central 90 % and range of a plotted quantity."""
    return {"median": np.median(values), "p5_p95": np.percentile(values, [5, 95]),
            "range": [values.min(), values.max()]}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _format_axes(main_ax, top_ax, right_ax):
    """Apply shared axis limits and ticks to a main + marginals trio."""
    main_ax.set_xlim(xlims)
    main_ax.set_ylim(ylims)
    main_ax.set_yticks(np.arange(0, ylims[1] + 1, 100))
    top_ax.set_xlim(xlims)
    right_ax.set_ylim(ylims)


def plot_scatter(axes, widths, heights, troughs):
    main_ax, top_ax, right_ax, cb_ax = axes

    sorter = np.argsort(troughs)[::-1]
    sns.scatterplot(
        x=widths[sorter],
        y=heights[sorter],
        hue=troughs[sorter],
        palette=cmap,
        ax=main_ax,
        s=1,
        legend=False,
        edgecolor="k",
        linewidth=0,
    )

    circle_annotate(main_ax, (250, 100), (1, 0), "2")
    circle_annotate(main_ax, (90, 500), (1, 0), "1")
    circle_annotate(main_ax, (90, 600), (1, 0), "3")
    circle_annotate(main_ax, (400, 520), (0, 0), "4")

    main_ax.text(
        0.2,
        1,
        f"n={len(widths)}",
        transform=main_ax.transAxes,
        ha="center",
        va="top",
        fontsize="small",
    )

    kde_kwargs = dict(color=cmap(0.5), fill=True, linewidth=0, alpha=0.5)
    sns.kdeplot(x=widths, ax=top_ax, legend=False, **kde_kwargs)
    sns.kdeplot(y=heights, ax=right_ax, legend=False, **kde_kwargs)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(troughs)
    cbar = plt.colorbar(sm, cax=cb_ax, orientation="horizontal")
    cbar.set_label("Trough [Hz]", fontsize="small")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.set_xticks([-300, 0])

    _format_axes(main_ax, top_ax, right_ax)


def plot_hexbin(axes, widths, heights):
    main_ax, cb_ax = axes

    # hb = main_ax.hexbin(
    #     widths,
    #     heights,
    #     gridsize=60,
    #     cmap=cmap_dens,
    #     mincnt=1,
    #     linewidths=0.1,
    #     edgecolors="none",
    # )
    hb = main_ax.hexbin(
        widths,
        heights,
        gridsize=60,
        cmap=cmap_dens,
        mincnt=1,
        linewidths=0.0,
        edgecolors="none",
        norm=LogNorm(vmin=10, vmax=7000),  # old vmax 3000; v2 max count per hexagon 6551
        rasterized=True,
    )

    # main_ax.text(
    #     0.2,
    #     1,
    #     f"n={len(widths)}",
    #     transform=main_ax.transAxes,
    #     ha="center",
    #     va="top",
    #     fontsize="small",
    # )

    cbar = plt.colorbar(hb, cax=cb_ax, orientation="horizontal")
    cbar.set_label("Count", fontsize="small")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    # cbar.ax.xaxis.set_ticks([0, 2000])
    cbar.ax.xaxis.set_ticks([10, 100, 1000])

    main_ax.set_xlim(xlims)
    main_ax.set_ylim(ylims)
    main_ax.set_yticks(np.arange(0, ylims[1] + 1, 100))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    widths, heights, troughs, recordings = get_data()
    widths *= 1000
    write_stats("chirp_distributions", {
        "n_competitions": len(set(recordings)), "n_chirps": len(widths),
        "box_width_ms": distribution(widths), "box_height_hz": distribution(heights),
        "box_trough_hz": distribution(troughs),
    })

    fig, scatter_axes, hex_axes = setup_plot()

    plot_scatter(scatter_axes, widths, heights, troughs)
    plot_hexbin(hex_axes, widths, heights)

    letter_subplots(
        [scatter_axes[0], hex_axes[0]], xoffset=[-0.2, -0.2], yoffset=[1.1, 1.1]
    )

    for fmt in [
        # "png",
        "pdf",
        # "svg",
    ]:
        fig.savefig(targetpath.with_suffix(f".{fmt}"), dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
