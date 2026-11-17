# Ported from chirp_competition d040123 (src/plot_chirp_distributions_manuscript.py).
# v1 plotted the spectrogram boxes (width t2 - t1, height f2 - emitter EODf, trough f1 - emitter
# EODf). v2 boxes span the long spectrogram window around every chirp (widths 200-700 ms), so the
# figure now shows the chirp-parameter decoder's calibrated estimates (the values the Methods
# report): width at 10 % of the height, height and undershoot (as the frequency below the baseline, -undershoot), on
# linear axes, of the chirps inside the decoder's validated domain (params_ok; the others are
# counted). Axes and color scale end at a high quantile of the shown chirps (AXIS_QUANTILE),
# rounded up to a round number; the chirps beyond them are counted. Rows: v2 parquet of PAPER_DETECTIONS
# (chirps.load_chirp_table, rows with keep), winner and loser chirps of the 33 analysed competitions.
# The plotted distributions (median, central 90 %, range) and counts go to write_stats
# ("chirp_distributions").
#
# Panel C: one example chirp per region 1-3 of A, chosen by rule (REGIONS, the clean-candidate
# thresholds below), not by hand: the clean chirp nearest the region's median in log width and
# log height. Each is shown as a spectrogram cutout of the raw recording (one electrode: the one
# where the emitter's baseline EOD stands out most from everything else in the cutout's band), on
# the same time and frequency span (relative to the emitter's baseline EODf) and the same level
# scale. The examples and the rule go to write_stats ("chirp_distributions.examples",
# "chirp_distributions.example_rule").
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram, welch
from thunderlab.dataloader import DataLoader

from chirps import load_chirp_table
from manuscript_stats import write_stats
from paper_config import DATA_ROOT, DETECTIONS, figure
from style import circle_annotate, letter_subplots, set_light_style

scriptname = Path(__file__).stem
targetpath = figure(scriptname)
set_light_style()

cmap = sns.color_palette("mako", as_cmap=True)

# Axes (width, height) and color scale (undershoot) end at this quantile of the shown chirps,
# rounded up to a round number: the few extreme estimates do not set the scale.
AXIS_QUANTILE = 0.995
ROUND_NUMBERS = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)

# the 33 analysed competitions: no draws, and not the recordings whose
# baseline EODfs were too close for reliable assignment
BAD_RECORDINGS = ["2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00"]

# ---------------------------------------------------------------------------
# Panel C: example chirps, chosen by rule
# ---------------------------------------------------------------------------
# The regions of A as bounds [low, high) on width at 10 % height (ms) and height (Hz). Heights:
# the short chirps' heights have a mode near 70 Hz and a band at 350-480 Hz with a trough at
# 225-300 Hz between them, so small < 250 Hz and large >= 300 Hz. Widths: the large chirps'
# widths peak at 10-15 ms and have fallen off by 30 ms, followed by a flat tail; short < 30 ms,
# and long >= 60 ms (twice that, so that short and long do not touch). The upper bounds are the
# ends of the axes of A (checked in main): only chirps visible in A are considered.
REGIONS = {
    1: {"width10_ms": (0, 30), "height_hz": (300, 500)},  # large and short
    2: {"width10_ms": (0, 30), "height_hz": (0, 250)},  # small and short
    3: {"width10_ms": (60, 150), "height_hz": (300, 500)},  # large and long
}
# Clean candidates: well detected and assigned (operating threshold of the run: 0.8), both fish
# tracked, no other chirp of either fish within ISOLATION_S (so the cutout shows one chirp), no
# clicks in the detection box (the cutout is not click-cleaned), and the other fish's baseline
# EODf at least OTHER_FISH_MARGIN_HZ below the cutout (so the cutout shows the emitter alone).
MIN_SCORE = 0.9
MIN_ASSIGN_CONFIDENCE = 0.99
ISOLATION_S = 0.5
OTHER_FISH_MARGIN_HZ = 50
# On the electrode shown, nothing in the cutout's band comes within MIN_CONTRAST_DB of the
# emitter's baseline EOD, apart from the baseline itself (e.g. the other fish's second harmonic
# would otherwise cross the cutout). Checked on the raw recording, candidate by candidate, from
# the most typical one on.
MIN_CONTRAST_DB = 25

# Cutouts: time around the detection's centre, frequency relative to the emitter's baseline EODf
# (wide enough for the long chirps of region 3 and the large chirps' heights); one Hann window of
# 25.6 ms (512 samples at 20 kHz: short enough for 10-ms chirps, narrow enough in frequency to
# separate a small chirp from the baseline), 1-ms hop, zero-padded to a 4.9-Hz grid; power in dB
# relative to the emitter's baseline (median over the cutout), shown from -DB_RANGE to 0 dB: 5 dB
# below MIN_CONTRAST_DB, so that what the contrast rule lets through stays near black, and above
# the window's first sidelobe (-31.5 dB), which would draw faint lines beside the baseline.
CUTOUT_HALF_S = 0.15
CUTOUT_HZ = (-100, 550)
SPEC_WINDOW_S = 0.0256
SPEC_HOP_S = 0.001
SPEC_NFFT = 4096
BASELINE_LOBE_HZ = 2 / SPEC_WINDOW_S  # half width of the window's main lobe (78 Hz)
DB_RANGE = 30
EXAMPLE_MARKER = dict(marker="D", s=9, facecolor="#e2655c", edgecolor="k", linewidths=0.4)


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
def setup_plot():
    cm = 1 / 2.54
    fig = plt.figure(figsize=(17.4 * cm, 8 * cm), dpi=300)

    gs = GridSpec(
        nrows=2,
        ncols=6,
        figure=fig,
        width_ratios=[1, 0.1, 0.2, 1, 0.17, 0.5],
        height_ratios=[0.1, 1],
    )

    # Left panel: scatter
    scatter_ax = fig.add_subplot(gs[1, 0])
    scatter_top = fig.add_subplot(gs[0, 0])
    scatter_right = fig.add_subplot(gs[1, 1])

    # Middle panel: hexbin (no marginals — density is encoded in colour)
    hex_ax = fig.add_subplot(gs[1, 3])

    # Right column: one spectrogram cutout per region of A
    example_gs = gs[1, 5].subgridspec(len(REGIONS), 1, hspace=0.15)
    example_axes = [fig.add_subplot(example_gs[i]) for i in range(len(REGIONS))]

    # Colorbar insets (lower right: no chirps are long and small)
    scatter_cb = scatter_ax.inset_axes(bounds=[0.68, 0.2, 0.28, 0.03])
    hex_cb = hex_ax.inset_axes(bounds=[0.62, 0.2, 0.34, 0.03])

    # Configure marginals
    for marginal in (scatter_top, scatter_right):
        marginal.set_xticks([])
        marginal.set_yticks([])
    scatter_top.spines["left"].set_visible(False)
    scatter_right.spines["bottom"].set_visible(False)

    for ax in (scatter_ax, hex_ax):
        ax.set_xlabel("Chirp width at 10 \\% height [ms]")  # usetex: \\%
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
    return fig, scatter_axes, hex_axes, example_axes


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
        df["isolation_s"] = isolation_s(df)
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


def isolation_s(table: pd.DataFrame) -> pd.Series:
    """Time from each chirp to the nearest other chirp of its recording (either fish).

    Chirps are the detector's chirps after post-processing (``chirp``), assigned or not;
    other rows get NaN.
    """
    centres = table.loc[table["chirp"], "t_center_s"].sort_values()
    gaps = centres.diff()
    return np.fmin(gaps, gaps.shift(-1)).reindex(table.index)  # previous or next


def bound(values: np.ndarray) -> float:
    """AXIS_QUANTILE of the values, rounded up to a round number (e.g. 476 -> 500)."""
    x = np.quantile(values, AXIS_QUANTILE)
    scale = 10 ** np.floor(np.log10(x))
    return float(scale * next(m for m in ROUND_NUMBERS if m * scale >= x))


def distribution(values: np.ndarray) -> dict:
    """Median, central 90 % and range of a plotted quantity."""
    return {"median": np.median(values), "p5_p95": np.percentile(values, [5, 95]),
            "range": [values.min(), values.max()]}


# ---------------------------------------------------------------------------
# Example chirps of the regions (panel C)
# ---------------------------------------------------------------------------
def in_region(chirps: pd.DataFrame, region: dict) -> pd.Series:
    (w0, w1), (h0, h1) = region["width10_ms"], region["height_hz"]
    return (chirps["width10_ms"].between(w0, w1, inclusive="left")
            & chirps["height_hz"].between(h0, h1, inclusive="left"))


def other_fish_hz(chirps: pd.DataFrame) -> pd.Series:
    """Baseline EODf of the other live track: of the two tracks nearest to the emitter
    estimate, the one the chirp was not assigned to."""
    return chirps["second_track_hz"].where(chirps["nearest_track_id"] == chirps["track_id"],
                                           chirps["nearest_track_hz"])


def is_clean(chirps: pd.DataFrame) -> pd.Series:
    """The clean-candidate rule of the examples (see the constants above)."""
    return ((chirps["score"] >= MIN_SCORE)
            & (chirps["assign_confidence"] >= MIN_ASSIGN_CONFIDENCE)
            & (chirps["n_live_tracks"] == 2)
            & (chirps["isolation_s"] >= ISOLATION_S)
            & (chirps["click_fraction_box"] == 0)
            & (chirps["other_hz"] <= chirps["track_hz"] + CUTOUT_HZ[0] - OTHER_FISH_MARGIN_HZ))


def read_cutout(chirp: pd.Series) -> tuple[np.ndarray, float, float]:
    """Raw voltages (samples x electrodes) of the cutout around a chirp, half a spectrogram
    window wider on each side, their sampling rate and the time of the first sample."""
    with DataLoader(str(DATA_ROOT / chirp["recording"])) as data:
        rate = data.rate
        pad = CUTOUT_HALF_S + SPEC_WINDOW_S / 2
        start = int(round((chirp["t_center_s"] - pad) * rate))
        stop = int(round((chirp["t_center_s"] + pad) * rate))
        voltages = np.asarray(data[start:stop, :], dtype=float)
    return voltages - voltages.mean(axis=0), rate, start / rate


def cutout_contrast_db(voltages: np.ndarray, rate: float, chirp: pd.Series) -> np.ndarray:
    """Per electrode: power at the emitter's baseline EODf over the strongest power elsewhere
    in the cutout's band (outside the baseline's main lobe), in dB.

    Median Welch spectra with the cutout's window (the median ignores the chirp itself).
    """
    n = round(SPEC_WINDOW_S * rate)
    freqs, psd = welch(voltages, rate, window="hann", nperseg=n, nfft=SPEC_NFFT,
                       average="median", axis=0)
    rel_freqs = freqs - chirp["track_hz"]
    baseline = psd[np.argmin(np.abs(rel_freqs))]
    elsewhere = ((rel_freqs >= CUTOUT_HZ[0]) & (rel_freqs <= CUTOUT_HZ[1])
                 & (np.abs(rel_freqs) > BASELINE_LOBE_HZ))
    return 10 * np.log10(baseline / psd[elsewhere].max(axis=0))


def choose_example(shown: pd.DataFrame, region: dict) -> dict:
    """The most typical clean chirp of a region and its cutout.

    Typical: nearest to the median of all shown chirps of the region in (log width, log height).
    Clean: ``is_clean``, then (on the raw recording) an electrode where the cutout's band holds
    nothing within MIN_CONTRAST_DB of the emitter's baseline; the candidates are checked from
    the most typical on.
    """
    members = shown[in_region(shown, region)].assign(other_hz=other_fish_hz)
    log_params = np.log(members[["width10_ms", "height_hz"]])
    centre = log_params.median()
    distance = np.hypot(*(log_params - centre).to_numpy().T)
    candidates = members.assign(distance=distance)[is_clean(members)].sort_values("distance")
    for rank, (_, chirp) in enumerate(candidates.iterrows(), start=1):
        voltages, rate, t_start = read_cutout(chirp)
        contrast = cutout_contrast_db(voltages, rate, chirp)
        electrode = int(np.argmax(contrast))
        if contrast[electrode] >= MIN_CONTRAST_DB:
            break
    else:
        raise ValueError(f"no clean chirp in region {region}")
    print(f"region {region}: {len(members)} chirps, {len(candidates)} clean by the table, "
          f"candidate {rank} passed: {chirp['recording']} {chirp['t_center_s']:.3f} s, "
          f"electrode {electrode} ({contrast[electrode]:.1f} dB)")
    return {"chirp": chirp, "signal": voltages[:, electrode], "rate": rate, "t_start": t_start,
            "electrode": electrode, "contrast_db": contrast[electrode], "rank": rank,
            "n_members": len(members), "n_candidates": len(candidates),
            "median": np.exp(centre)}


def cutout_spectrogram(example: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time (ms re the detection's centre), frequency (Hz re the emitter's baseline EODf) and
    power (dB re the emitter's baseline) of an example's cutout."""
    chirp, rate = example["chirp"], example["rate"]
    n, hop = round(SPEC_WINDOW_S * rate), round(SPEC_HOP_S * rate)
    freqs, times, power = spectrogram(example["signal"], rate, window="hann", nperseg=n,
                                      noverlap=n - hop, nfft=SPEC_NFFT)
    rel_freqs = freqs - chirp["track_hz"]
    baseline = np.median(power[np.argmin(np.abs(rel_freqs))])
    band = (rel_freqs >= CUTOUT_HZ[0]) & (rel_freqs <= CUTOUT_HZ[1])
    rel_times = (times + example["t_start"] - chirp["t_center_s"]) * 1000
    return rel_times, rel_freqs[band], 10 * np.log10(power[band] / baseline)


def example_stats(example: dict) -> dict:
    chirp = example["chirp"]
    year, month, day = (int(x) for x in chirp["recording"][:10].split("-"))  # one recording a day
    return {
        "recording_date": {"year": year, "month": month, "day": day},
        "chirp_id": chirp["chirp_id"], "t_center_s": chirp["t_center_s"],
        "width10_ms": chirp["width10_ms"], "height_hz": chirp["height_hz"],
        "undershoot_hz": chirp["undershoot_hz"], "score": chirp["score"],
        "assign_confidence": chirp["assign_confidence"], "baseline_hz": chirp["track_hz"],
        "other_fish_hz": chirp["other_hz"], "electrode": example["electrode"],
        "contrast_db": example["contrast_db"],
        "n_in_region": example["n_members"], "n_clean_by_table": example["n_candidates"],
        "candidate_rank": example["rank"],
        "region_median": {"width10_ms": example["median"]["width10_ms"],
                          "height_hz": example["median"]["height_hz"]},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_scatter(axes, limits, width, height, undershoot, examples):
    main_ax, top_ax, right_ax, cb_ax = axes

    below = -undershoot  # Hz relative to the baseline: strong undershoots dark, none bright
    order = np.argsort(below)[::-1]  # strong undershoots on top
    points = main_ax.scatter(width[order], height[order], c=below[order], cmap=cmap,
                             norm=Normalize(-limits["undershoot"], 0), s=1, linewidths=0,
                             rasterized=True)
    main_ax.set_xlim(0, limits["width"])
    main_ax.set_ylim(0, limits["height"])

    circle_annotate(main_ax, (40, 130), (0, 0), "2")  # small and short
    circle_annotate(main_ax, (25, 480), (0, 0), "1")  # large and short
    circle_annotate(main_ax, (100, 330), (0, 0), "3")  # large and long

    # the example chirps of C
    main_ax.scatter([e["chirp"]["width10_ms"] for e in examples.values()],
                    [e["chirp"]["height_hz"] for e in examples.values()],
                    zorder=1000, **EXAMPLE_MARKER)

    main_ax.text(0.97, 0.5, f"n={len(width)}", transform=main_ax.transAxes,
                 ha="right", va="center", fontsize="small")

    kde_kwargs = dict(color=cmap(0.5), fill=True, linewidth=0, alpha=0.5)
    sns.kdeplot(x=width, ax=top_ax, legend=False, **kde_kwargs)
    sns.kdeplot(y=height, ax=right_ax, legend=False, **kde_kwargs)
    top_ax.set_xlim(0, limits["width"])
    right_ax.set_ylim(0, limits["height"])
    for marginal in (top_ax, right_ax):
        marginal.set_xticks([])
        marginal.set_yticks([])

    cbar = plt.colorbar(points, cax=cb_ax, orientation="horizontal", extend="min")
    cbar.set_label("Undershoot [Hz]", fontsize="small")
    cbar.set_ticks([-limits["undershoot"], 0])
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")


def plot_hexbin(axes, limits, width, height):
    main_ax, cb_ax = axes
    # the hexagons cover the axes; chirps beyond them are not binned
    hb = main_ax.hexbin(width, height, gridsize=60, extent=(0, limits["width"], 0, limits["height"]),
                        cmap=cmap, mincnt=1, linewidths=0.0, edgecolors="none", norm=LogNorm(),
                        rasterized=True)
    main_ax.set_xlim(0, limits["width"])
    main_ax.set_ylim(0, limits["height"])

    cbar = plt.colorbar(hb, cax=cb_ax, orientation="horizontal")
    cbar.set_label("Count", fontsize="small")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.set_ticks([1, 10, 100], labels=["1", "10", "100"])
    cbar.ax.minorticks_off()


def plot_examples(axes, examples):
    for ax, (number, example) in zip(axes, examples.items(), strict=True):
        times, freqs, power_db = cutout_spectrogram(example)
        ax.pcolormesh(times, freqs, power_db, cmap=cmap, vmin=-DB_RANGE, vmax=0,
                      shading="nearest", rasterized=True)
        ax.set_xlim(-CUTOUT_HALF_S * 1000, CUTOUT_HALF_S * 1000)
        ax.set_ylim(*CUTOUT_HZ)
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([0, 250, 500])
        circle_annotate(ax, (-CUTOUT_HALF_S * 1000 + 30, CUTOUT_HZ[1] - 85), (0, 0), str(number))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time [ms]")
    axes[len(axes) // 2].set_ylabel("Frequency rel. to EOD$f$ [Hz]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = get_data()
    shown = df[df["params_ok"]].assign(width10_ms=lambda d: d["width10_s"] * 1000)
    width = shown["width10_ms"].to_numpy()  # width at 10 % of the height
    height = shown["height_hz"].to_numpy()
    undershoot = shown["undershoot_hz"].to_numpy()
    limits = {"width": bound(width), "height": bound(height), "undershoot": bound(undershoot)}
    for region in REGIONS.values():  # the examples are chirps visible in A
        if region["width10_ms"][1] > limits["width"] or region["height_hz"][1] > limits["height"]:
            raise ValueError(f"region {region} reaches beyond the axes {limits}")
    examples = {number: choose_example(shown, region) for number, region in REGIONS.items()}
    write_stats("chirp_distributions", {
        "n_competitions": df["recording"].nunique(), "n_chirps": len(df),
        "n_outside_domain": int((~df["params_ok"]).sum()), "n_shown": len(shown),
        "axis_quantile": AXIS_QUANTILE,
        "axis_limits": {"width10_ms": limits["width"], "height_hz": limits["height"],
                        "undershoot_hz": limits["undershoot"]},
        "n_beyond_axes": int(((width > limits["width"]) | (height > limits["height"])).sum()),
        "n_beyond_color_scale": int((undershoot > limits["undershoot"]).sum()),
        "width10_ms": distribution(width), "height_hz": distribution(height),
        "undershoot_hz": distribution(undershoot),
        "example_rule": {
            "regions": {f"region_{k}": r for k, r in REGIONS.items()},
            "min_score": MIN_SCORE, "min_assign_confidence": MIN_ASSIGN_CONFIDENCE,
            "isolation_s": ISOLATION_S, "other_fish_margin_hz": OTHER_FISH_MARGIN_HZ,
            "min_contrast_db": MIN_CONTRAST_DB,
            "cutout": {"half_width_ms": CUTOUT_HALF_S * 1000, "freq_hz": CUTOUT_HZ,
                       "window_ms": SPEC_WINDOW_S * 1000, "hop_ms": SPEC_HOP_S * 1000,
                       "nfft": SPEC_NFFT, "db_range": DB_RANGE},
        },
        "examples": {f"region_{k}": example_stats(e) for k, e in examples.items()},
    })

    fig, scatter_axes, hex_axes, example_axes = setup_plot()
    plot_scatter(scatter_axes, limits, width, height, undershoot, examples)
    plot_hexbin(hex_axes, limits, width, height)
    plot_examples(example_axes, examples)
    # C at the height of A and B: their letters sit 10 % of B's height above its top
    hex_box, top_box = hex_axes[0].get_position(), example_axes[0].get_position()
    letter_subplots(
        [scatter_axes[0], hex_axes[0], example_axes[0]], xoffset=[-0.2, -0.2, -0.45],
        yoffset=[1.1, 1.1, 1 + 0.1 * hex_box.height / top_box.height],
    )
    fig.savefig(targetpath.with_suffix(".pdf"), dpi=300)


if __name__ == "__main__":
    main()
