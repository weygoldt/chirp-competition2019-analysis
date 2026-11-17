# Ported from chirp_competition d040123 (src/plot_chirp_distributions_manuscript.py).
# v1 plotted the spectrogram boxes (width t2 - t1, height f2 - emitter EODf, trough f1 - emitter
# EODf). v2 boxes span the long spectrogram window around every chirp (widths 200-700 ms), so the
# figure now shows the chirp-parameter decoder's calibrated estimates (the values the Methods
# report): FWHM, height and undershoot, on log axes, of the chirps inside the decoder's validated
# domain (params_ok; the others are counted). Rows: v2 parquet of PAPER_DETECTIONS
# (chirps.load_chirp_table, rows with keep), winner and loser chirps of the 33 analysed competitions.
# The plotted distributions (median, central 90 %, range) and counts go to write_stats
# ("chirp_distributions").
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from chirps import load_chirp_table
from manuscript_stats import write_stats
from paper_config import DATA_ROOT, DETECTIONS, figure
from style import circle_annotate, letter_subplots, set_light_style

scriptname = Path(__file__).stem
targetpath = figure(scriptname)
set_light_style()

cmap = sns.color_palette("mako", as_cmap=True)

xlims = (1, 500)  # FWHM [ms]; decoder estimates 1.5-380 ms
ylims = (10, 1500)  # height [Hz]; decoder estimates 14-1300 Hz

# the 33 analysed competitions: no draws, and not the recordings whose
# baseline EODfs were too close for reliable assignment
BAD_RECORDINGS = ["2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00"]


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

    # Colorbar insets (lower right: no chirps are long and small)
    scatter_cb = scatter_ax.inset_axes(bounds=[0.68, 0.2, 0.28, 0.03])
    hex_cb = hex_ax.inset_axes(bounds=[0.68, 0.2, 0.28, 0.03])

    # Configure marginals
    for marginal in (scatter_top, scatter_right):
        marginal.set_xticks([])
        marginal.set_yticks([])
    scatter_top.spines["left"].set_visible(False)
    scatter_right.spines["bottom"].set_visible(False)

    for ax in (scatter_ax, hex_ax):
        ax.set_xlabel("Chirp FWHM [ms]")
    scatter_ax.set_ylabel("Chirp height [Hz]")

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
def get_data() -> pd.DataFrame:
    """Kept, assigned chirps of the winners and losers of the analysed competitions."""
    dfs = []
    for recording in sorted(DATA_ROOT.iterdir()):
        if not (recording / f"chirps_{DETECTIONS}.parquet").exists():
            continue
        df = load_chirp_table(recording)
        dfs.append(df[df["keep"]])
    if not dfs:
        raise FileNotFoundError(f"no chirps_{DETECTIONS}.parquet under {DATA_ROOT}")
    df = pd.concat(dfs)

    summary = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    summary = summary[(summary["draw"] != 1) & ~summary["recording"].isin(BAD_RECORDINGS)]
    fish = {(r.recording, float(i)) for r in summary.itertuples() for i in (r.win_ID, r.lose_ID)}
    analysed = [(rec, float(tid)) in fish
                for rec, tid in zip(df["recording"], df["track_id"], strict=True)]
    df = df[np.array(analysed)]
    print(f"{summary['recording'].nunique()} competitions: {len(df)} chirps, "
          f"{(~df['params_ok']).sum()} outside the decoder's validated domain")
    return df


def distribution(values: np.ndarray) -> dict:
    """Median, central 90 % and range of a plotted quantity."""
    return {"median": np.median(values), "p5_p95": np.percentile(values, [5, 95]),
            "range": [values.min(), values.max()]}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def format_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


def plot_scatter(axes, fwhm, height, undershoot):
    main_ax, top_ax, right_ax, cb_ax = axes

    order = np.argsort(undershoot)  # large undershoots on top
    points = main_ax.scatter(fwhm[order], height[order], c=undershoot[order], cmap=cmap,
                             s=1, linewidths=0, rasterized=True)
    format_axes(main_ax)

    circle_annotate(main_ax, (20, 70), (0, 0), "2")
    circle_annotate(main_ax, (2, 420), (0, 0), "1")
    circle_annotate(main_ax, (2, 900), (0, 0), "3")
    circle_annotate(main_ax, (60, 250), (0, 0), "4")

    main_ax.text(0.2, 1, f"n={len(fwhm)}", transform=main_ax.transAxes,
                 ha="center", va="top", fontsize="small")

    kde_kwargs = dict(color=cmap(0.5), fill=True, linewidth=0, alpha=0.5, log_scale=True)
    sns.kdeplot(x=fwhm, ax=top_ax, legend=False, **kde_kwargs)
    sns.kdeplot(y=height, ax=right_ax, legend=False, **kde_kwargs)
    top_ax.set_xlim(xlims)
    right_ax.set_ylim(ylims)
    for marginal in (top_ax, right_ax):
        marginal.minorticks_off()
        marginal.set_xticks([])
        marginal.set_yticks([])

    cbar = plt.colorbar(points, cax=cb_ax, orientation="horizontal")
    cbar.set_label("Undershoot [Hz]", fontsize="small")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")


def plot_hexbin(axes, fwhm, height):
    main_ax, cb_ax = axes
    hb = main_ax.hexbin(fwhm, height, xscale="log", yscale="log", gridsize=60, cmap=cmap,
                        mincnt=1, linewidths=0.0, edgecolors="none", norm=LogNorm(),
                        rasterized=True)
    format_axes(main_ax)

    cbar = plt.colorbar(hb, cax=cb_ax, orientation="horizontal")
    cbar.set_label("Count", fontsize="small")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.set_ticks([1, 10, 100], labels=["1", "10", "100"])
    cbar.ax.minorticks_off()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = get_data()
    shown = df[df["params_ok"]]
    fwhm = shown["fwhm_s"].to_numpy() * 1000
    height = shown["height_hz"].to_numpy()
    undershoot = shown["undershoot_hz"].to_numpy()
    write_stats("chirp_distributions", {
        "n_competitions": df["recording"].nunique(), "n_chirps": len(df),
        "n_outside_domain": int((~df["params_ok"]).sum()), "n_shown": len(shown),
        "fwhm_ms": distribution(fwhm), "height_hz": distribution(height),
        "undershoot_hz": distribution(undershoot),
    })

    fig, scatter_axes, hex_axes = setup_plot()
    plot_scatter(scatter_axes, fwhm, height, undershoot)
    plot_hexbin(hex_axes, fwhm, height)
    letter_subplots(
        [scatter_axes[0], hex_axes[0]], xoffset=[-0.2, -0.2], yoffset=[1.1, 1.1]
    )
    fig.savefig(targetpath.with_suffix(".pdf"), dpi=300)


if __name__ == "__main__":
    main()
