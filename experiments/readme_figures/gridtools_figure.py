"""README figure of gridtools: a simulated grid recording with exact ground truth.

Left: three simulated fish swimming over a 4 x 4 electrode grid for seven
seconds (paths fade into the past; dots are the positions at the end).
Middle: the same seven seconds of the simulated recording (power summed over
the electrodes; the 205 ms reference window of chirpdetector's front end),
every chirp's ground-truth footprint box in its fish's colour. Right: one
chirp up close (51 ms window) with the frequency the simulator generated it
with.

The scene comes from ``readme_scene.toml`` (synthetic noise, chirp templates
of library v4):

    source experiments/env.sh
    uv run gridtools sim run experiments/readme_figures/readme_scene.toml \\
        --seeds 0:8 --out $CHIRP_DATA/readme_figures/scenes
    uv run python experiments/readme_figures/gridtools_figure.py [SEED T0]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mc
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from chirpdetector.spectrogram import SpectrogramConfig, compute_spectrogram, normalize
from gridtools.io import open_recording
from paths import DATA

SCENES = DATA / "readme_figures" / "scenes"
OUT = DATA / "readme_figures" / "out"

# Chosen for the figure: fish 60 Hz or more apart, all three swimming and chirping.
SEED, T0, SPAN = "000007", 10.0, 7.0
if len(sys.argv) == 3:
    SEED, T0 = f"{int(sys.argv[1]):06d}", float(sys.argv[2])
SCENE = SCENES / SEED
F_LIM = (650.0, 1400.0)


def load():
    rec = open_recording(SCENE)
    config = SpectrogramConfig()
    pad = 0.6
    a, b = max(0.0, T0 - pad), min(rec.n_samples / rec.samplerate_hz, T0 + SPAN + pad)
    spec = compute_spectrogram(rec.read_time(a, b), rec.samplerate_hz, config, t0_s=a)
    img = normalize(spec, config).cpu().numpy()
    chirps = pd.read_parquet(SCENE / "chirps.parquet")
    chirps = chirps[(chirps.box_t1_s > T0) & (chirps.box_t0_s < T0 + SPAN)].sort_values(
        "t_center_s"
    )
    paths = dict(np.load(SCENE / "paths.npz"))
    traces = dict(np.load(SCENE / "chirp_traces.npz"))
    tracks = {
        k: np.load(SCENE / f"{k}.npy") for k in ("times", "fund_v", "ident_v", "idx_v")
    }
    assert rec.geometry is not None
    return rec, spec, img, chirps, paths, traces, tracks, rec.geometry.positions_m


def baseline(tracks, fish: int, t: np.ndarray) -> np.ndarray:
    sel = tracks["ident_v"] == fish
    return np.interp(
        t, tracks["times"][tracks["idx_v"][sel].astype(int)], tracks["fund_v"][sel]
    )


def true_frequency(rec, traces, tracks, chirp_id: int):
    k = int(np.flatnonzero(traces["chirp_id"] == chirp_id)[0])
    lo, hi = int(traces["offsets"][k]), int(traces["offsets"][k + 1])
    t = (traces["start"][k] + np.arange(hi - lo)) / rec.samplerate_hz
    fish = int(traces["fish_id"][k])
    return t, baseline(tracks, fish, t) + traces["delta_f_hz"][lo:hi]


def level(v: np.ndarray, vmin: float, vmax: float, gamma: float) -> np.ndarray:
    return np.clip((v - vmin) / (vmax - vmin), 0, 1) ** gamma


def pick_zoom(chirps: pd.DataFrame) -> pd.Series:
    """The tallest short, clearly visible chirp that overlaps no other chirp."""
    c = chirps[
        (chirps.duration_s < 0.03)
        & ~chirps.overlaps_other_chirp
        & (chirps.visible_fraction > 0.8)
        & (chirps.t_center_s > T0 + 0.5)
        & (chirps.t_center_s < T0 + SPAN - 0.5)
    ]
    return c.sort_values("excursion_hz").iloc[-1]


def draw(
    theme: style.Theme, rec, spec, img, chirps, paths, traces, tracks, electrodes, zoom
) -> None:
    style.apply(theme)
    cmap = style.ink_cmap(theme, gamma=1.0)
    fish_ids = sorted(
        int(i) for i in np.unique(tracks["ident_v"][~np.isnan(tracks["ident_v"])])
    )
    order = sorted(
        fish_ids,
        key=lambda i: float(np.nanmedian(tracks["fund_v"][tracks["ident_v"] == i])),
    )
    color = {fid: theme.fish[k] for k, fid in enumerate(order)}

    fig = style.figure(6.0)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 2.1, 0.85], wspace=0.38)

    # --- fish on the grid
    ax = fig.add_subplot(gs[0])
    ax.scatter(
        electrodes[:, 0], electrodes[:, 1], s=7, color=theme.muted, lw=0, zorder=2
    )
    times = paths["times_s"]
    sel = (times >= T0) & (times <= T0 + SPAN)
    for fid in order:
        xy = paths["xy_m"][fid][sel]
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        rgba = np.tile(np.r_[mc.to_rgb(color[fid]), 1.0], (len(segs), 1))
        rgba[:, 3] = np.linspace(0.05, 1.0, len(segs)) ** 1.3
        ax.add_collection(
            LineCollection(segs, colors=rgba, lw=1.4, capstyle="round", zorder=3)
        )
        ax.scatter(*xy[-1], s=16, color=color[fid], lw=0, zorder=4)
    pad = 0.12
    ax.set_xlim(electrodes[:, 0].min() - pad, electrodes[:, 0].max() + pad)
    ax.set_ylim(electrodes[:, 1].min() - pad, electrodes[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    span = electrodes[:, 0].max() - electrodes[:, 0].min()
    x0, y0 = electrodes[:, 0].min(), electrodes[:, 1].min() - 0.085
    ax.plot(
        [x0, x0 + span / 3], [y0, y0], color=theme.muted, lw=0.8, solid_capstyle="butt"
    )
    ax.text(
        x0 + span / 6,
        y0 - 0.02,
        f"{span / 3 * 100:.0f} cm",
        ha="center",
        va="top",
        fontsize=6.5,
        color=theme.muted,
    )
    ax.set_title("Fish swim over a grid", color=theme.ink)

    # --- the recording with ground-truth boxes
    sp = fig.add_subplot(gs[1])
    t, f = spec.times_s, spec.freqs_hz
    sp.imshow(
        level(img[0], 0.30, 0.82, 1.4),
        extent=(t[0] - T0, t[-1] - T0, f[0], f[-1]),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="antialiased",
        rasterized=True,
    )
    for _, r in chirps.iterrows():
        sp.add_patch(
            Rectangle(
                (r.box_t0_s - T0, r.box_f0_hz),
                r.box_t1_s - r.box_t0_s,
                r.box_f1_hz - r.box_f0_hz,
                fill=False,
                lw=0.75,
                ec=color[int(r.emitter_id)],
                zorder=3,
            )
        )
    sp.set_xlim(0, SPAN)
    sp.set_ylim(*F_LIM)
    sp.set_xlabel("Time (s)")
    sp.set_ylabel("Frequency (Hz)")
    sp.set_title("The recording, every chirp labelled", color=theme.ink)

    # --- one chirp with its exact frequency
    zm = fig.add_subplot(gs[2])
    tt, ff = true_frequency(rec, traces, tracks, int(zoom.chirp_id))
    base = float(baseline(tracks, int(zoom.emitter_id), np.array([zoom.t_center_s]))[0])
    zm.imshow(
        level(img[1], 0.42, 0.86, 1.6),
        extent=(
            t[0] - zoom.t_center_s,
            t[-1] - zoom.t_center_s,
            f[0] - base,
            f[-1] - base,
        ),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        rasterized=True,
    )
    zc = color[int(zoom.emitter_id)]
    x = tt - zoom.t_center_s
    zm.plot(
        [-0.06, x[0]],
        [ff[0] - base] * 2,
        color=zc,
        lw=1.1,
        zorder=4,
        solid_capstyle="butt",
    )
    zm.plot(
        [x[-1], 0.06],
        [ff[-1] - base] * 2,
        color=zc,
        lw=1.1,
        zorder=4,
        solid_capstyle="butt",
    )
    zm.plot(x, ff - base, color=zc, lw=1.1, zorder=4)
    zm.set_xlim(-0.06, 0.06)
    zm.set_ylim(-0.45 * zoom.excursion_hz, 1.35 * zoom.excursion_hz)
    zm.set_xticks([-0.04, 0, 0.04])
    zm.set_xticklabels(["-40", "0", "40"])
    zm.set_xlabel("Time (ms)")
    zm.set_ylabel("Above baseline (Hz)")
    zm.set_title("One chirp, exactly", color=theme.ink)

    style.save(fig, OUT / f"gridtools_{theme.name}.png")


def main() -> None:
    data = load()
    chirps = data[3]
    zoom = pick_zoom(chirps)
    print(
        SEED,
        T0,
        len(chirps),
        "chirps;",
        chirps.emitter_id.value_counts().sort_index().to_dict(),
        f"zoom: chirp {zoom.chirp_id} fish {zoom.emitter_id} {zoom.excursion_hz:.0f} Hz",
    )
    for theme in style.THEMES:
        draw(theme, *data, zoom)


if __name__ == "__main__":
    main()
