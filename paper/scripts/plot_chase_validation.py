"""Figure (supplement): timing of the BORIS chase annotations against the author's re-labelling.

A   shifts of the author's onsets and offsets relative to BORIS (author - BORIS), one dot per chase,
    median and interquartile range
B   video motion around onsets: BORIS onsets of all analysed chases, and BORIS vs author onsets of the
    re-labelled chases (mean, 95 % bootstrap band over chases)
C   the same around offsets
D   loser chirp rate around offsets, BORIS vs author offsets of the same re-labelled chases
E   loser rise rate around onsets, BORIS vs author onsets of the same re-labelled chases

Rates: Gaussian kernel per event (sigma 0.3 s for chirps, 0.5 s for the sparser rises), averaged over
chases; bands are 95 % bootstrap intervals over chases. Inputs: ``chase_validation/pairs.csv``,
``chase_validation/motion.npz``, the run's chirps (``chirps.load_chirps``) and the loser's rises
(``analyze_event_time_rates_trial.load_data``). The numbers the caption cites go to write_stats
("chase_validation_figure").
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

from analyze_event_time_rates_trial import load_data
from manuscript_stats import write_stats
from paper_config import DATA_ROOT, FIGURES, PROCESSED
from style import letter_subplots, set_light_style

# The figure helpers and palette of v2_fig_common, copied: importing that module needs the run to be one
# of its pipeline runs (v2_fig_common.RUNS), so the figure failed in the sensitivity run at threshold 0.9.
_CM = 1 / 2.54
_ANNOT_FS = "small"


def _rc_tweaks() -> None:
    mpl.rcParams.update({"lines.markeredgewidth": 0.5, "xtick.major.size": 3, "ytick.major.size": 3,
                         "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "axes.linewidth": 0.8, "legend.handlelength": 1.4,
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


C = SimpleNamespace(CM=_CM, INK="k", CORAL="#e2655c", MUTED="#6e6e6e", GREY="#a0a0a0", rc_tweaks=_rc_tweaks,
                    axes_cm=_axes_cm, letters_cm=_letters_cm, save=_save)

NAME = "plot_chase_validation"
BORIS_C, AUTHOR_C, ALL_C = C.INK, C.CORAL, C.MUTED
LAG = np.arange(-6, 6.0001, 0.02)
REF = {"color": C.INK, "lw": 0.5, "ls": ":", "zorder": 0}  # the event time (reference line)
XLIM = (-6, 6)
XTICKS = [-6, -4, -2, 0, 2, 4, 6]  # the same time ticks in B-E
SEED = 42
W_CM, H_CM = 17.4, 10.6


def boot_band(M: np.ndarray, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(M), (n, len(M)))
    bs = np.array([np.nanmean(M[i], 0) for i in idx])
    return np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)


def kernel_rates(events: list[tuple[str, float]], times: dict[str, np.ndarray], sigma: float) -> np.ndarray:
    """Per event: events of `times` around it, kernel-smoothed, per minute, on LAG."""
    out = []
    for rec, t0 in events:
        x = times[rec] - t0
        x = x[np.abs(x) <= LAG[-1] + 4 * sigma]
        out.append(60 * np.exp(-0.5 * ((LAG[:, None] - x[None, :]) / sigma) ** 2).sum(1) / (sigma * np.sqrt(2 * np.pi)))
    return np.array(out)


def curve(ax, x, M, color, label, lw=1.2, ls="-", band=True):
    m = np.nanmean(M, 0)
    if band:
        lo, hi = boot_band(M)
        ax.fill_between(x, lo, hi, color=color, alpha=0.25, lw=0)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    return m


def strip(ax, d: pd.Series, y: float, color, rng):
    d = d.dropna().to_numpy()
    shown = np.clip(d, -6, 3)
    ax.scatter(shown, y + rng.uniform(-0.18, 0.18, len(d)), s=5, color=color, alpha=0.6, lw=0)
    q25, q50, q75 = np.percentile(d, [25, 50, 75])
    ax.plot([q25, q75], [y - 0.32, y - 0.32], color=C.INK, lw=2.2, solid_capstyle="butt")
    ax.plot([q50, q50], [y - 0.42, y - 0.22], color=C.INK, lw=1.2)
    return {"n": int(len(d)), "n_clipped": int(np.sum((d < -6) | (d > 3)))}


def main() -> None:
    logging.disable(logging.INFO)
    set_light_style()
    C.rc_tweaks()
    pairs = pd.read_csv(PROCESSED / "chase_validation" / "pairs.csv")
    mot = np.load(PROCESSED / "chase_validation" / "motion.npz")
    summary = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0).set_index("recording", drop=False)
    chirps, rises = {}, {}
    for rec in pairs["recording"].unique():
        out = load_data(DATA_ROOT / rec, summary.loc[rec], skip_ag_events=True)
        chirps[rec], rises[rec] = np.sort(out[3]), np.sort(out[5])  # loser chirps, loser rises
    stats: dict = {}

    fig = plt.figure(figsize=(W_CM * C.CM, H_CM * C.CM))
    ax_a = C.axes_cm(fig, 1.6, 6.45, 4.1, 3.6)
    ax_b = C.axes_cm(fig, 7.25, 6.45, 4.4, 3.6)
    ax_c = C.axes_cm(fig, 12.7, 6.45, 4.4, 3.6)
    ax_d = C.axes_cm(fig, 1.6, 1.2, 6.8, 3.5)
    ax_e = C.axes_cm(fig, 10.3, 1.2, 6.8, 3.5)

    # A: shifts
    rng = np.random.default_rng(SEED)
    stats["shifts"] = {"onset": strip(ax_a, pairs["d_onset_s"], 1, AUTHOR_C, rng),
                       "offset": strip(ax_a, pairs["d_offset_s"], 0, AUTHOR_C, rng)}
    ax_a.axvline(0, **REF)
    ax_a.set_yticks([1, 0], ["onset", "offset"])
    ax_a.set_xlim(-6.3, 3.3)
    ax_a.set_xticks([-6, -4, -2, 0, 2])
    ax_a.set_ylim(-0.7, 1.5)
    ax_a.set_xlabel("Re-labeled $-$ annotated [s]")

    # B, C: video motion
    lags = mot["lags_s"]
    sel = (lags >= XLIM[0]) & (lags <= XLIM[1])
    for ax, end in ((ax_b, "onset"), (ax_c, "offset")):
        curve(ax, lags[sel], mot[f"all_boris_{end}"][:, sel], ALL_C, f"annotated, all chases (n = {len(mot[f'all_boris_{end}'])})",
              lw=0.9, ls="--", band=False)
        mb = curve(ax, lags[sel], mot[f"pair_boris_{end}"][:, sel], BORIS_C, "annotated, re-labeled chases")
        ma = curve(ax, lags[sel], mot[f"pair_author_{end}"][:, sel], AUTHOR_C, "re-labeled, same chases")
        ax.axvline(0, **REF)
        ax.set_xlim(*XLIM)
        ax.set_xticks(XTICKS)
        ax.set_xlabel(f"Time relative to chase {end} [s]")
        t = lags[sel]
        stats[f"motion_{end}"] = {
            "n_all": int(len(mot[f"all_boris_{end}"])), "n_pairs": int(len(mot[f"pair_author_{end}"])),
            "peak_s": {"boris_all": float(t[np.nanargmax(np.nanmean(mot[f"all_boris_{end}"][:, sel], 0))]),
                       "boris": float(t[np.nanargmax(mb)]), "author": float(t[np.nanargmax(ma)])}}
    ax_b.set_ylabel("Video motion\n" r"[$\times$ chase median]")
    for ax in (ax_b, ax_c):  # one scale for both, with room for the key above the curves
        ax.set_ylim(0.8, 2.45)
    # white, edgeless box: the key sits across the event line at 0
    ax_c.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)

    # D: loser chirps around offsets; E: loser rises around onsets (same re-labelled chases)
    for ax, end, times, sigma, ylab, key in ((ax_d, "offset", chirps, 0.3, "Loser chirp rate [1/min]", "chirps_offset"),
                                             (ax_e, "onset", rises, 0.5, "Loser rise rate [1/min]", "rises_onset")):
        ok = pairs[f"d_{end}_s"].notna()
        ev_b = list(zip(pairs.loc[ok, "recording"], pairs.loc[ok, f"boris_{end}_rec_s"]))
        ev_a = list(zip(pairs.loc[ok, "recording"], pairs.loc[ok, f"{end}_rec_s"]))
        Mb, Ma = kernel_rates(ev_b, times, sigma), kernel_rates(ev_a, times, sigma)
        curve(ax, LAG, Mb, BORIS_C, f"annotated {end}s")
        curve(ax, LAG, Ma, AUTHOR_C, f"re-labeled {end}s")
        ax.axvline(0, **REF)
        ax.set_xlim(*XLIM)
        ax.set_xticks(XTICKS)
        ax.set_xlabel(f"Time relative to chase {end} [s]")
        ax.set_ylabel(ylab)
        # the key in the panel's empty upper corner (D: left, E: right)
        ax.legend(frameon=False, loc="upper left" if ax is ax_d else "upper right")
        pre = (LAG >= -2) & (LAG < 0)
        post = (LAG >= 0) & (LAG < 2)
        stats[key] = {"n": int(ok.sum()), "kernel_sigma_s": sigma,
                      **{who: {"rate_2s_before": float(M[:, pre].mean()), "rate_2s_after": float(M[:, post].mean())}
                         for who, M in (("boris", Mb), ("author", Ma))}}

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        sns.despine(ax=ax, offset=4, trim=False)
    C.letters_cm(fig, [(ax_a, "A", 1.4, 0.3), (ax_b, "B", 1.4, 0.3), (ax_c, "C", 0.85, 0.3),
                       (ax_d, "D", 1.4, 0.3), (ax_e, "E", 1.4, 0.3)])
    paths = C.save(fig, NAME)
    print(json.dumps(stats, indent=1))
    print("\n".join(map(str, paths)))
    write_stats("chase_validation_figure", stats)


if __name__ == "__main__":
    main()
