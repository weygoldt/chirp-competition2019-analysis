"""README figure of chirpdetector: detect, assign and measure chirps.

Left: ten seconds of a real competition recording (two fish) in the
detector's reference spectrogram channel (205 ms window, power summed over
the electrodes), every detected chirp boxed in the colour of the fish it was
assigned to. Right: height against FWHM, from the decoder, of every chirp of
the same recording. Detections come from the full 2019 production run
(released dual-head detector and decoder); chirps at the operating threshold,
assigned, and kept by the post-processing.

    uv run python experiments/readme_figures/chirpdetector_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from chirpdetector.spectrogram import SpectrogramConfig, compute_spectrogram, normalize
from gridtools.io import open_recording
from paths import DATA, REC_2019

RUN = DATA / "detections" / "competition2019_v2_20261113"
OUT = DATA / "readme_figures" / "out"

# Chosen with a search over the run: two fish 118 Hz apart, both chirping,
# no clicks, every chirp inside the window and confidently assigned.
REC, T0, SPAN = "2020-06-08-10_00", 16665.0, 10.0
PAD = 0.6
F_LIM = (625.0, 1175.0)


def kept(chirps: pd.DataFrame) -> pd.DataFrame:
    return chirps[
        chirps.chirp & chirps.track_id.notna() & (chirps.pp_drop_reason == "")
    ]


def load():
    rec = open_recording(REC_2019 / REC)
    config = SpectrogramConfig()
    samples = rec.read_time(T0 - PAD, T0 + SPAN + PAD)
    spec = compute_spectrogram(samples, rec.samplerate_hz, config, t0_s=T0 - PAD)
    img = normalize(spec, config).cpu().numpy()  # (C, F, T) in [0, 1]
    every = kept(pd.read_parquet(RUN / REC / "chirps_rtdetr.parquet"))
    window = every[(every.t_end_s > T0) & (every.t_start_s < T0 + SPAN)].sort_values(
        "t_center_s"
    )
    return spec, img, window, every[every.params_ok.fillna(False).astype(bool)]


def level(v: np.ndarray, vmin: float, vmax: float, gamma: float) -> np.ndarray:
    return np.clip((v - vmin) / (vmax - vmin), 0, 1) ** gamma


def draw(
    theme: style.Theme, spec, img, window: pd.DataFrame, every: pd.DataFrame
) -> None:
    style.apply(theme)
    t, f = spec.times_s, spec.freqs_hz
    tracks = window.groupby("track_id").track_hz.median().sort_values()
    color = {tid: theme.fish[[0, 2][i % 2]] for i, tid in enumerate(tracks.index)}
    name = {tid: f"fish {i + 1}" for i, tid in enumerate(tracks.index)}

    fig = style.figure(6.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    ax.imshow(
        level(img[0], 0.30, 0.82, 1.4),
        extent=(t[0] - T0, t[-1] - T0, f[0], f[-1]),
        aspect="auto",
        cmap=style.ink_cmap(theme, gamma=1.0),
        vmin=0,
        vmax=1,
        interpolation="antialiased",
        rasterized=True,
    )
    for _, r in window.iterrows():
        ax.add_patch(
            Rectangle(
                (r.t_start_s - T0, r.f_low_hz),
                r.t_end_s - r.t_start_s,
                r.f_high_hz - r.f_low_hz,
                fill=False,
                lw=0.75,
                ec=color[r.track_id],
                zorder=3,
            )
        )
    for tid, hz in tracks.items():
        ax.text(
            SPAN + 0.12,
            hz,
            name[tid],
            color=color[tid],
            va="center",
            ha="left",
            fontsize=7.5,
        )
    ax.set_xlim(0, SPAN)
    ax.set_ylim(*F_LIM)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Detect chirps, assign each to its fish", color=theme.ink)

    sc = fig.add_subplot(gs[1])
    order = every.track_id.value_counts().index  # most chirps first, drawn below
    for tid in order:
        g = every[every.track_id == tid]
        if tid not in color:
            continue
        sc.scatter(
            g.fwhm_s * 1e3,
            g.height_hz,
            s=1.6,
            lw=0,
            color=color[tid],
            alpha=0.35 if len(g) > 1000 else 0.7,
            rasterized=True,
        )
    for tid in tracks.index:
        g = every[every.track_id == tid]
        big = np.median(g.height_hz) > 150
        sc.text(
            78,
            560 if big else 150,
            f"{name[tid]}\n{len(g):,} chirps",
            color=color[tid],
            fontsize=7,
            ha="right",
            va="center",
            linespacing=1.2,
        )
    sc.set_xlim(0, 80)
    sc.set_ylim(0, 600)
    sc.set_xlabel("FWHM (ms)")
    sc.set_ylabel("Height (Hz)")
    sc.set_title("Measure them: the whole recording", color=theme.ink)

    style.save(fig, OUT / f"chirpdetector_{theme.name}.png")


def main() -> None:
    spec, img, window, every = load()
    print(
        REC,
        T0,
        SPAN,
        len(window),
        "chirps in the window;",
        window.track_id.value_counts().to_dict(),
    )
    print("whole recording:", every.track_id.value_counts().to_dict())
    for theme in style.THEMES:
        draw(theme, spec, img, window, every)


if __name__ == "__main__":
    main()
