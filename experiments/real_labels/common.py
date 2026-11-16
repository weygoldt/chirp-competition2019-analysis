"""Shared definitions of the real-data labelling session real_v1 (2026-11-14).

Paths, the labeller view of the blind human benchmark (reused unchanged),
the pixel masks of the contrast check, and the recordings of both datasets.

The view is the one of the blind simulated benchmark, as it was run
(transcript of 2026-11-13 and ``labels/human_benchmark_v1/VIEW_CHANGES.md``):
``label-emitters --blind --span-s 5 --span-hz 600 --serve`` with every other
option at its default (window 204.8 ms, zoom 204.8 ms, hop 1.6 ms, pad 0.25 s
/ 40 Hz, track gap 10 s, snap tolerance 15 Hz, hint box 0.3 s x 200 Hz from
30 Hz below the candidate's lower edge, colours from the 5th percentile of a
panel to its maximum, magma). chirpdetector's view and render code has not
changed since that session (v2 36f3ac7 .. main 33779b3: no diff in
``labeling/view.py`` and ``labeling/render.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2019, REC_2022  # noqa: E402

SESSION = "real_v1"
ROOT = DATA / "labels" / SESSION  # truth, pools, review (never read by the labeller)
CANDIDATES = DATA / "labels" / "candidates" / f"{SESSION}.parquet"  # what the labeller reads
LABELS_OUT = DATA / "labels" / f"{SESSION}_labels"  # label-emitters --out
TRACKS_GAPFILL = ROOT / "tracks_gapfill"  # wavetracker copies of the gap-filled trial tracks

RUN_2019 = DATA / "detections" / "competition2019_v2_t001"
RUN_2022 = DATA / "detections" / "competition2022_v2_t001"
PRODUCTION_2019 = DATA / "detections" / "competition2019_v2_20261113"
GAPFILL_2019 = DATA / "detections" / "competition2019_v2_20261113_gapfill"
BACKGROUND = DATA / "background" / "v2"
REAL_LABELS = DATA / "eval" / "real_labels_365" / "labels"
EMITTER_LABELS = DATA / "labels" / "emitters" / "labels"

DARK_PHASE_S = 10_800.0  # gridtools.background.sources.DARK_PHASE_S: first 3 h dark

# the benchmark's labeller view (see the module docstring)
VIEW = {
    "span_s": 5.0,
    "span_hz": 600.0,
    "pad_s": 0.25,
    "pad_hz": 40.0,
    "window_s": 0.2048,
    "max_gap_s": 10.0,
    "blind_hint": True,
}
LABELLER_ARGS = ["--blind", "--span-s", "5", "--span-hz", "600"]


def view_config():  # noqa: ANN201
    """The benchmark's :class:`chirpdetector.labeling.view.ViewConfig`."""
    from chirpdetector.labeling.view import ViewConfig

    return ViewConfig(**VIEW)


def recording_dir(dataset: str, name: str) -> Path:
    """Raw recording directory of a recording."""
    if dataset == "competition2019":
        return REC_2019 / name
    return REC_2022 / "raw" / name


def tracks_dir(dataset: str, name: str) -> Path:
    """Original wavetracker directory of a recording."""
    if dataset == "competition2019":
        return REC_2019 / name
    return REC_2022 / "interrim" / name


def dataset_of(name: str) -> str:
    """``competition2019`` (2019/2020 recordings) or ``competition2022``."""
    return "competition2022" if name.startswith("2022") else "competition2019"


# pixel masks -----------------------------------------------------------------


def panel_axes(panel) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:  # noqa: ANN001
    """Centre times (columns) and frequencies (rows) of a view panel."""
    t0, t1, f0, f1 = panel.extent
    n_f, n_t = panel.db.shape
    t = t0 + (np.arange(n_t) + 0.5) * (t1 - t0) / n_t
    f = f0 + (np.arange(n_f) + 0.5) * (f1 - f0) / n_f
    return t, f


def track_mask(
    panel,  # noqa: ANN001
    tracks,  # noqa: ANN001
    half_width_hz: float = 12.0,
    harmonics: tuple[int, ...] = (1, 2, 3),
) -> npt.NDArray[np.bool_]:
    """Pixels within ``half_width_hz`` of a live track (or its harmonics)."""
    t, f = panel_axes(panel)
    mask = np.zeros(panel.db.shape, dtype=bool)
    for tr in tracks:
        ft = np.interp(t, tr.times_s, tr.freqs_hz)
        for h in harmonics:
            mask |= np.abs(f[:, None] - h * ft[None, :]) < half_width_hz
    return mask


def box_mask(
    panel,  # noqa: ANN001
    boxes: npt.ArrayLike,
    pad_s: float = 0.0,
    pad_hz: float = 0.0,
) -> npt.NDArray[np.bool_]:
    """Pixels inside any box ``(t0, t1, f0, f1)`` padded by ``pad_s``/``pad_hz``."""
    t, f = panel_axes(panel)
    mask = np.zeros(panel.db.shape, dtype=bool)
    for t0, t1, f0, f1 in np.atleast_2d(np.asarray(boxes, dtype=float)):
        rows = (f >= f0 - pad_hz) & (f <= f1 + pad_hz)
        cols = (t >= t0 - pad_s) & (t <= t1 + pad_s)
        mask |= rows[:, None] & cols[None, :]
    return mask


def background_clim(
    db: npt.NDArray[np.floating],
    background: npt.NDArray[np.bool_],
    floor_percentile: float = 10.0,
    span_db: float = 30.0,
) -> tuple[float, float]:
    """The background-anchored alternative: floor = a low percentile of the
    background pixels (away from the tracks), ceiling = floor + a fixed span."""
    values = db[background] if background.sum() >= 100 else db.ravel()
    floor = float(np.percentile(values, floor_percentile))
    return floor, floor + span_db


def colour_index(db: npt.ArrayLike, clim: tuple[float, float]) -> npt.NDArray[np.float64]:
    """Position in the colour map (0-1) of dB values under colour limits."""
    lo, hi = clim
    return np.clip((np.asarray(db, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
