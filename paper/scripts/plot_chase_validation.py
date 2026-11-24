"""Figure (supplement): timing of the BORIS chase annotations against the author's re-labeling.

A   shifts of the re-labeled onsets and offsets relative to BORIS (re-labeled - annotated): kernel density
    (Gaussian, bandwidth KDE_BW_S), interquartile range shaded, median as a line
B   video motion around onsets: BORIS onsets of all analysed chases (dashed), BORIS (black) and re-labeled
    (indigo) onsets of the matched chases (mean, 95 % bootstrap band over chases)
C   the same around offsets
D   loser chirp rate around chase onsets: Fig. 6's panel B1 (identical numbers: the JSON it draws), zoomed to
    RATE_XLIM; the same rate for every chase shifted by the median re-labeling shift (dashed) and for the
    re-labeled onsets of the matched chases (indigo); Fig. 6's jittered null in grey
E   the same around chase offsets (Fig. 6, C1)

D, E: medians and 2.5th-97.5th percentiles of the leave-out resamples, as in Fig. 6 (``chase_event_rates``);
the shifted and re-labeled rates come from ``chase_validation_rates.py``. Inputs: ``chase_validation/pairs.csv``,
``motion.npz``, ``event_rates.npz`` and ``event_time_rates/lchirps@agon.json``, ``lchirps@agoff.json``. The
numbers the caption cites go to write_stats ("chase_validation_figure").
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import chase_event_rates as cer
from manuscript_stats import write_stats
from paper_config import FIGURES, PROCESSED
from style import letter_subplots, loser_color, other_color, set_light_style

# The figure helpers of v2_fig_common, copied: importing that module needs the run to be one of its pipeline
# runs (v2_fig_common.RUNS), so the figure failed in the sensitivity run at threshold 0.9.
_CM = 1 / 2.54
_ANNOT_FS = "small"


def _rc_tweaks() -> None:
    mpl.rcParams.update({"lines.markeredgewidth": 0.5, "xtick.major.size": 3, "ytick.major.size": 3,
                         "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "axes.linewidth": 0.8, "legend.handlelength": 1.8,
                         "legend.fontsize": _ANNOT_FS, "axes.titlesize": "medium", "axes.titlepad": 3})


def _axes_cm(fig, x, y, w, h):
    W, H = fig.get_size_inches() / _CM
    return fig.add_axes((x / W, y / H, w / W, h / H))


def _letters_cm(fig, axes_and_offsets):
    W, H = fig.get_size_inches() / _CM
    axs, xo, yo, lets = [], [], [], []
    for ax, let, dx, dy in axes_and_offsets:
        bb = ax.get_position()
        axs.append(ax)
        lets.append(let)
        xo.append(-dx / (bb.width * W))
        yo.append(1 + dy / (bb.height * H))
    letter_subplots(axs, letters=lets, xoffset=xo, yoffset=yo)


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"]
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    return paths


C = SimpleNamespace(CM=_CM, INK="k", MUTED="#6e6e6e", rc_tweaks=_rc_tweaks, axes_cm=_axes_cm,
                    letters_cm=_letters_cm, save=_save)

NAME = "plot_chase_validation"
# Palette of Figs. 5-7: the re-labeled chases in the chasing indigo, everywhere; the annotated chases in black
# (video motion) or, for the loser's chirp rate, in Fig. 6's loser teal; Fig. 6's grey null.
RELABEL_C, ANNOT_C, ALL_C, LOSER_C, NULL_C = other_color, C.INK, C.MUTED, loser_color, "grey"
REF = {"color": C.INK, "lw": 0.5, "ls": ":", "zorder": 0}  # the event time (reference line)
SEED = 42
W_CM, H_CM = 17.4, 10.8
SHIFT_XLIM = (-6, 3)  # A; the shifts outside are counted in the stats
KDE_BW_S = 0.25
MOTION_XLIM, MOTION_XTICKS = (-6, 6), [-6, -4, -2, 0, 2, 4, 6]
# D, E: Fig. 6 shows +-30 s; +-10 s keeps the loser's modulations around the chase ends that a shift of about
# 1 s can move (the dip before onsets, the rise and plateau after them, the peak at offsets), at a
# scale where a shift of 1 s is visible
RATE_XLIM, RATE_XTICKS = (-10, 10), [-10, -5, 0, 5, 10]
RATE_YLIM, RATE_YTICKS = (0, 15), [0, 5, 10, 15]  # Fig. 6's loser rate axis
WIN_S = 2.0  # the stats' mean rates before / after the event
TROUGH_S = 3.0  # the stats' lowest rate within +-TROUGH_S of the event (Fig. 6's dip before onsets)


def boot_band(M: np.ndarray, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(M), (n, len(M)))
    bs = np.array([np.nanmean(M[i], 0) for i in idx])
    return np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)


def motion_curve(ax, x, M, color, lw=1.0, ls="-", band=True):
    m = np.nanmean(M, 0)
    if band:
        lo, hi = boot_band(M)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(x, m, color=color, lw=lw, ls=ls)
    return m


def kde(d: np.ndarray, x: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian kernel density of ``d`` on ``x`` (integrates to 1 over the real line)."""
    return np.exp(-0.5 * ((x[:, None] - d[None, :]) / bw) ** 2).sum(1) / (len(d) * bw * np.sqrt(2 * np.pi))


def shift_ridges(ax, pairs: pd.DataFrame) -> dict:
    """A: density of the shifts (one scale for both ends), IQR shaded darker, median line and value."""
    x = np.linspace(*SHIFT_XLIM, 901)
    dens = {end: kde(pairs[f"d_{end}_s"].dropna().to_numpy(), x, KDE_BW_S) for end in cer.ENDS}
    scale = 0.72 / max(d.max() for d in dens.values())  # one scale for both ridges
    out = {}
    for y0, end in ((1.0, "onset"), (0.0, "offset")):
        d = pairs[f"d_{end}_s"].dropna().to_numpy()
        q25, q50, q75 = np.percentile(d, [25, 50, 75])
        y = y0 + scale * dens[end]
        iqr = (x >= q25) & (x <= q75)
        ax.fill_between(x, y0, y, color=RELABEL_C, alpha=0.18, lw=0)
        ax.fill_between(x[iqr], y0, y[iqr], color=RELABEL_C, alpha=0.4, lw=0)
        ax.plot(x, y, color=RELABEL_C, lw=1.0)
        ax.plot(SHIFT_XLIM, [y0, y0], color=RELABEL_C, lw=0.5)
        ax.plot([q50, q50], [y0, y0 + scale * np.interp(q50, x, dens[end])], color=C.INK, lw=1.0)
        ax.text(SHIFT_XLIM[0], y0 + 0.12, f"{end}\nn = {len(d)}", fontsize=_ANNOT_FS, ha="left", va="bottom")
        ax.text(q50 - 0.15, y0 + scale * np.interp(q50, x, dens[end]) + 0.04, f"{q50:.2f} s".replace("-", "$-$"),
                fontsize=_ANNOT_FS, ha="right", va="bottom")
        out[end] = {"n": int(len(d)), "n_outside": int(np.sum((d < SHIFT_XLIM[0]) | (d > SHIFT_XLIM[1]))),
                    "median_s": float(q50), "iqr_s": [float(q25), float(q75)]}
    ax.axvline(0, **REF)
    ax.set_xlim(SHIFT_XLIM[0] - 0.2, SHIFT_XLIM[1] + 0.2)
    ax.set_xticks([-6, -4, -2, 0, 2])
    ax.set_ylim(-0.08, 1.98)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Re-labeled $-$ annotated [s]")
    return {"x_range_s": list(SHIFT_XLIM), "kde_bandwidth_s": KDE_BW_S, **out}


def window_means(t: np.ndarray, rate: np.ndarray) -> dict:
    t = np.round(t, 6)
    return {"rate_2s_before": float(rate[(t >= -WIN_S) & (t < 0)].mean()),
            "rate_2s_after": float(rate[(t >= 0) & (t < WIN_S)].mean())}


def rate_panel(ax, end: str, fig6: dict, rates) -> dict:
    """Fig. 6's loser rate around ``end`` (its JSON, its percentiles) and the shifted and re-labeled rates."""
    t = np.asarray(fig6["conv_time"])
    if not np.array_equal(t, rates["conv_time"]):
        raise ValueError("event_rates.npz is on another time grid than Fig. 6; rerun chase_validation_rates.py")
    null = cer.per_minute_band(fig6["bootstrapped_baseline_jittered"])
    annotated = cer.per_minute_band(fig6["jackknifed_eventrate"])
    shifted = cer.per_minute_band(rates[f"{end}_shifted_resamples"])
    relabeled = cer.per_minute_band(rates[f"{end}_relabeled_resamples"])
    subset = cer.per_minute_band(rates[f"{end}_annotated_subset_resamples"])
    sel = (t >= RATE_XLIM[0] - 0.5) & (t <= RATE_XLIM[1] + 0.5)
    ax.fill_between(t[sel], null[0][sel], null[2][sel], color=NULL_C, alpha=0.5, lw=0)
    ax.plot(t[sel], null[1][sel], color=NULL_C, lw=1)
    ax.fill_between(t[sel], annotated[0][sel], annotated[2][sel], color=LOSER_C, alpha=0.5, lw=0)
    ax.plot(t[sel], annotated[1][sel], color=LOSER_C, lw=1)
    ax.fill_between(t[sel], relabeled[0][sel], relabeled[2][sel], color=RELABEL_C, alpha=0.2, lw=0)
    ax.plot(t[sel], relabeled[1][sel], color=RELABEL_C, lw=1)
    ax.plot(t[sel], shifted[1][sel], color=LOSER_C, lw=1, ls=(0, (3, 1.5)))
    ax.axvline(0, **REF)
    ax.set_xlim(*RATE_XLIM)
    ax.set_xticks(RATE_XTICKS)
    ax.set_ylim(*RATE_YLIM)
    ax.set_yticks(RATE_YTICKS)
    ax.set_xlabel(f"Time relative to chase {end} [s]")

    shift = float(rates[f"{end}_shift_s"])
    in_x = (np.round(t, 6) >= RATE_XLIM[0]) & (np.round(t, 6) <= RATE_XLIM[1])  # the plotted range
    # the shifted curve is Fig. 6's moved by -shift (same resampling draws); it differs only by interpolation
    moved = np.interp(t[in_x] + shift, t, annotated[1])

    near = np.abs(np.round(t, 6)) <= TROUGH_S

    def curve_stats(band, n_chases, trials):
        return {"n_chases": int(np.sum(n_chases)), "n_trials": int(len(trials)), **window_means(t, band[1]),
                "peak_per_min": float(band[1][in_x].max()), "peak_s": float(t[in_x][np.argmax(band[1][in_x])]),
                "trough_per_min": float(band[1][near].min()), "trough_s": float(t[near][np.argmin(band[1][near])])}

    if list(rates[f"{end}_shifted_trials"]) != fig6["trial_identifiers"]:
        raise ValueError(f"event_rates.npz ({end}) has other trials than Fig. 6; rerun chase_validation_rates.py")
    return {
        "n": int(np.sum(rates[f"{end}_relabeled_n_chases"])),
        "x_range_s": list(RATE_XLIM), "kernel_sigma_s": cer.KERNEL_SIGMA_S, "shift_s": shift,
        # the shifted chases are Fig. 6's (chase_validation_rates checks them against its JSON)
        "annotated": curve_stats(annotated, rates[f"{end}_shifted_n_chases"], fig6["trial_identifiers"]),
        "shifted": curve_stats(shifted, rates[f"{end}_shifted_n_chases"], rates[f"{end}_shifted_trials"]),
        "relabeled": curve_stats(relabeled, rates[f"{end}_relabeled_n_chases"], rates[f"{end}_relabeled_trials"]),
        "annotated_subset": curve_stats(subset, rates[f"{end}_annotated_subset_n_chases"],
                                        rates[f"{end}_annotated_subset_trials"]),
        "null_per_min": float(null[1][in_x].mean()),
        "null_band_per_min": [float(null[0][in_x].mean()), float(null[2][in_x].mean())],
        "relabeled_null_per_min": float(cer.per_minute_band(rates[f"{end}_relabeled_null_resamples"])[1][in_x].mean()),
        "shifted_minus_moved_annotated_max_abs": float(np.max(np.abs(shifted[1][in_x] - moved))),
    }


def main() -> None:
    logging.disable(logging.INFO)
    set_light_style()
    C.rc_tweaks()
    pairs = pd.read_csv(PROCESSED / "chase_validation" / "pairs.csv")
    mot = np.load(PROCESSED / "chase_validation" / "motion.npz")
    path = PROCESSED / "chase_validation" / "event_rates.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run chase_validation_rates.py (chain validation)")
    rates = np.load(path)
    stats: dict = {}

    fig = plt.figure(figsize=(W_CM * C.CM, H_CM * C.CM))
    ax_a = C.axes_cm(fig, 0.6, 6.55, 4.6, 3.6)
    ax_b = C.axes_cm(fig, 6.85, 6.55, 4.6, 3.6)
    ax_c = C.axes_cm(fig, 12.5, 6.55, 4.6, 3.6)
    ax_d = C.axes_cm(fig, 1.4, 1.2, 7.0, 3.6)
    ax_e = C.axes_cm(fig, 10.1, 1.2, 7.0, 3.6)

    # A: shifts
    stats["shifts"] = shift_ridges(ax_a, pairs)

    # B, C: video motion
    lags = mot["lags_s"]
    sel = (lags >= MOTION_XLIM[0]) & (lags <= MOTION_XLIM[1])
    for ax, end in ((ax_b, "onset"), (ax_c, "offset")):
        motion_curve(ax, lags[sel], mot[f"all_boris_{end}"][:, sel], ALL_C, lw=0.9, ls=(0, (3, 1.5)), band=False)
        mb = motion_curve(ax, lags[sel], mot[f"pair_boris_{end}"][:, sel], ANNOT_C)
        ma = motion_curve(ax, lags[sel], mot[f"pair_author_{end}"][:, sel], RELABEL_C)
        ax.axvline(0, **REF)
        ax.set_xlim(*MOTION_XLIM)
        ax.set_xticks(MOTION_XTICKS)
        ax.set_xlabel(f"Time relative to chase {end} [s]")
        t = lags[sel]
        stats[f"motion_{end}"] = {
            "n_all": int(len(mot[f"all_boris_{end}"])), "n_pairs": int(len(mot[f"pair_author_{end}"])),
            "peak_s": {"boris_all": float(t[np.nanargmax(np.nanmean(mot[f"all_boris_{end}"][:, sel], 0))]),
                       "boris": float(t[np.nanargmax(mb)]), "author": float(t[np.nanargmax(ma)])}}
    ax_b.set_ylabel("Video motion\n" r"[$\times$ chase median]")
    for ax in (ax_b, ax_c):  # one scale for both, with room for the key above the curves
        ax.set_ylim(0.8, 2.45)
    keys = [Line2D([], [], color=ALL_C, lw=0.9, ls=(0, (3, 1.5))), Line2D([], [], color=ANNOT_C, lw=1.0),
            Line2D([], [], color=RELABEL_C, lw=1.0)]
    # white, edgeless box: the key sits across the event line at 0
    ax_c.legend(keys, ["annotated, all chases", "annotated, subset", "re-labeled, subset"], loc="upper right",
                frameon=True, facecolor="white", edgecolor="none", framealpha=1, borderaxespad=0.2)

    # D, E: loser chirp rate, Fig. 6 B1 / C1 zoomed, with the shifted and re-labeled chases
    for ax, end in ((ax_d, "onset"), (ax_e, "offset")):
        stats[f"chirps_{end}"] = rate_panel(ax, end, cer.fig6_json(end), rates)
    ax_d.set_ylabel("Loser chirp rate [1/min]")
    keys = [Line2D([], [], color=LOSER_C, lw=1), Line2D([], [], color=LOSER_C, lw=1, ls=(0, (3, 1.5))),
            Line2D([], [], color=RELABEL_C, lw=1), Patch(color=NULL_C, alpha=0.5, lw=0)]
    ax_d.legend(keys, ["annotated, all chases", "shifted, all chases", "re-labeled, subset", "null"],
                loc="upper left", frameon=False, borderaxespad=0.2)

    for ax in (ax_b, ax_c, ax_d, ax_e):
        sns.despine(ax=ax, offset=4, trim=False)
    sns.despine(ax=ax_a, left=True, offset=4, trim=False)
    C.letters_cm(fig, [(ax_a, "A", 0.3, 0.3), (ax_b, "B", 1.4, 0.3), (ax_c, "C", 0.85, 0.3),
                       (ax_d, "D", 1.1, 0.3), (ax_e, "E", 0.85, 0.3)])
    paths = C.save(fig, NAME)
    print(json.dumps(stats, indent=1))
    print("\n".join(map(str, paths)))
    write_stats("chase_validation_figure", stats)


if __name__ == "__main__":
    main()
