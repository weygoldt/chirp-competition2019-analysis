"""The one loader of chirp detections used by every ported script.

The old scripts read ``np.load(trial_path / "chirp_times_rcnn.npy")`` and
``np.load(trial_path / "chirp_ids_rcnn.npy")`` directly. They now call
:func:`load_chirps`, which reads the detections named by
``paper_config.DETECTIONS`` (or an explicit ``source``).

Both detection sets have the same legacy form: float64 chirp times (box
centre, seconds from the recording's first sample) and float64 emitter ids
(wavetracker identities, ``ident_v`` values). For ``rtdetr`` the legacy
arrays are the rows of ``chirps_rtdetr.parquet`` that are chirps (score at
or above the operating threshold, not dropped by post-processing) and
assigned to a track; :func:`load_chirp_table` gives the full parquet with
that selection as ``keep``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_config import DETECTIONS


def load_chirps(trial_path: str | Path, source: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Chirp times (s) and emitter ids of one recording, as the old ``*_rcnn.npy`` pair."""
    trial_path = Path(trial_path)
    source = source or DETECTIONS
    times = np.load(trial_path / f"chirp_times_{source}.npy")
    ids = np.load(trial_path / f"chirp_ids_{source}.npy")
    if len(times) != len(ids):
        raise ValueError(f"{trial_path}: {len(times)} chirp times but {len(ids)} ids ({source})")
    return times, ids


def load_chirp_table(trial_path: str | Path, source: str | None = None) -> pd.DataFrame:
    """All v2 detections of one recording with a ``keep`` column (the legacy selection).

    Reads ``chirps_<source>.parquet`` (``source`` defaults to ``DETECTIONS``;
    v2 sources only: ``rtdetr``, ``rtdetr_gapfill``).
    """
    source = source or DETECTIONS
    if not source.startswith("rtdetr"):
        raise ValueError(f"no parquet for detection source {source!r} (v2 sources only)")
    df = pd.read_parquet(Path(trial_path) / f"chirps_{source}.parquet")
    df["keep"] = df["chirp"] & df["track_id"].notna() & (df["pp_drop_reason"] == "")
    return df
