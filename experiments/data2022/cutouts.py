"""Spectrogram cut-outs of detections, for looking at what the detector found.

Each cut-out has two panels:

- **overview**: 5 s around the detection with the labeller's analysis window
  (204.8 ms, electrode-summed power, the real_v1 session's view), frequency
  range from the lowest live baseline to above the box top;
- **zoom**: 0.8 s around the detection with a shorter window (default 51.2 ms),
  which resolves the time course of brief (10-30 ms) chirps that the long
  window smears into a ~0.25 s blob.

Both use :func:`chirpdetector.labeling.render.band_spectrogram` (the
labeller's own spectrogram code) on the raw recording, and draw the live
tracks (dashed, numbered), the detection box (white) and the predicted
emitter EODf (cyan tick). Colours: magma over the panel's top 50 dB.

Use :func:`contact_sheet` to draw a grid of detections given as rows of a
detection table (``chirps_rtdetr.parquet`` or ``candidates_scored.csv``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

REC_2019 = Path("/mnt/data2/2019_tube_competition")
REC_2022 = Path("/mnt/data2/2022_tube_competition")
TRACKS_GAPFILL = Path("/mnt/data1/chirp_v2/labels/real_v1/tracks_gapfill")


@dataclass(frozen=True)
class CutoutConfig:
    overview_s: float = 5.0
    overview_window_s: float = 0.2048
    zoom_s: float = 0.8
    zoom_window_s: float = 0.0512
    hop_s: float = 0.0016
    zoom_hop_s: float = 0.002
    pad_hz: float = 60.0
    db_range: float = 50.0
    max_gap_s: float = 10.0


def recording_dir(name: str) -> Path:
    """Raw recording directory of a 2019/2020 or 2022 recording."""
    return REC_2022 / "raw" / name if name.startswith("2022") else REC_2019 / name


def tracks_dir(name: str) -> Path:
    """Tracks the real_v1 labeller showed: gap-filled for 2019/2020 trials, else the originals."""
    if name.startswith("2022"):
        return REC_2022 / "interrim" / name
    gap = TRACKS_GAPFILL / name
    return gap if gap.exists() else REC_2019 / name


@lru_cache(maxsize=4)
def open_rec(name: str):
    from gridtools.io.recording import open_recording

    return open_recording(recording_dir(name))


@lru_cache(maxsize=4)
def open_tracks(name: str):
    from gridtools.io.tracks import load_tracks

    return load_tracks(tracks_dir(name))


def live_tracks(tracks, t0: float, t1: float, max_gap_s: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """``(ident, times, freqs)`` of every track with a detection within ``max_gap_s`` of the span."""
    out = []
    for tr in tracks:
        m = (tr.times_s >= t0 - max_gap_s) & (tr.times_s <= t1 + max_gap_s)
        if m.sum() >= 2:
            out.append((int(tr.ident), tr.times_s[m], tr.freqs_hz[m]))
    return out


def _spec(rec, window, window_s, hop_s):
    from chirpdetector.labeling.render import band_spectrogram

    return band_spectrogram(rec, window, window_s=window_s, hop_s=hop_s, fft_pad=4)


def draw(ax_over: Axes, ax_zoom: Axes, row: pd.Series, cfg: CutoutConfig = CutoutConfig(),
         title: str = "") -> None:
    """Draw one detection (a row with recording, t_start_s, t_end_s, f_low_hz, f_high_hz, emitter_hz)."""
    from matplotlib.patches import Rectangle

    rec, tracks = open_rec(row["recording"]), open_tracks(row["recording"])
    tc = 0.5 * (row["t_start_s"] + row["t_end_s"])
    t0 = max(0.0, tc - cfg.overview_s / 2)
    t1 = min(rec.duration_s, tc + cfg.overview_s / 2)
    live = live_tracks(tracks, t0, t1, cfg.max_gap_s)
    base = [float(np.interp(tc, t, f)) for _, t, f in live]
    f0 = max(0.0, min([row["f_low_hz"], *base]) - cfg.pad_hz)
    f1 = min(0.5 * rec.samplerate_hz, max([row["f_high_hz"], *base]) + cfg.pad_hz)
    z0, z1 = max(0.0, tc - cfg.zoom_s / 2), min(rec.duration_s, tc + cfg.zoom_s / 2)
    panels = (
        (ax_over, (t0, t1, f0, f1), cfg.overview_window_s, cfg.hop_s),
        (ax_zoom, (z0, z1, f0, f1), cfg.zoom_window_s, cfg.zoom_hop_s),
    )
    for ax, win, wlen, hop in panels:
        s = _spec(rec, win, wlen, hop)
        vmax = float(s.db.max())
        ax.imshow(s.db, origin="lower", aspect="auto", extent=s.extent, cmap="magma",
                  vmin=vmax - cfg.db_range, vmax=vmax, interpolation="nearest")
        for ident, t, f in live:
            m = (t >= win[0]) & (t <= win[1])
            if m.any():
                ax.plot(t[m], f[m], color="#9ecae1", lw=0.6, ls=(0, (4, 3)), alpha=0.9)
                ax.text(win[0], float(np.interp(win[0], t, f)) + 4, str(ident), color="#9ecae1",
                        fontsize=5, va="bottom")
        ax.add_patch(Rectangle((row["t_start_s"], row["f_low_hz"]), row["t_end_s"] - row["t_start_s"],
                               row["f_high_hz"] - row["f_low_hz"], fill=False, ec="white", lw=0.6,
                               ls=(0, (2, 2))))
        if np.isfinite(row.get("emitter_hz", np.nan)):
            ax.plot([win[0], win[0] + 0.04 * (win[1] - win[0])], [row["emitter_hz"]] * 2, color="cyan", lw=1.5)
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.tick_params(labelsize=5, length=2)
    ax_over.set_title(title, fontsize=6)
    ax_zoom.set_xlabel("time (s)", fontsize=5, labelpad=1)
    ax_over.set_ylabel("Hz", fontsize=5, labelpad=1)


def describe(row: pd.Series) -> str:
    """Short title: recording, time, score, label, height/FWHM, assigned fish."""
    parts = [f"{row['recording'][:10]} t={row['t_center_s']:.1f}s", f"score {row['score']:.2f}"]
    if isinstance(row.get("label"), str):
        parts.append({"chirp": "CHIRP", "chirp_emitter_unclear": "CHIRP(?)", "not_chirp": "NOT"}[row["label"]])
    if np.isfinite(row.get("height_hz", np.nan)):
        parts.append(f"h {row['height_hz']:.0f} Hz, {1e3 * row['fwhm_s']:.0f} ms")
    if isinstance(row.get("status"), str):
        parts.append(row["status"])
    return " | ".join(parts)


def contact_sheet(rows: pd.DataFrame, path: Path, suptitle: str, ncol: int = 4,
                  cfg: CutoutConfig = CutoutConfig()) -> None:
    """A grid of cut-outs (overview above zoom), one per row of ``rows``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 3.6 * nrow + 0.6))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.22)
    for k, (_, row) in enumerate(rows.iterrows()):
        inner = outer[k // ncol, k % ncol].subgridspec(2, 1, height_ratios=[0.5, 0.5], hspace=0.18)
        draw(fig.add_subplot(inner[0]), fig.add_subplot(inner[1]), row, cfg, describe(row))
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
