"""The files of each 2022 competition recording (video, SLEAP export, LEDs).

``REC_2022/raw/<rec>/``: video ``*converted.mp4``, ``fishgrid.cfg``,
``led_idxs.csv`` (LED flashes as sample indices of the electric recording).
``REC_2022/interrim/<rec>/``: the SLEAP export ``*.analysis.nix`` and
``LED_frames.npy`` (video frames of the LED flashes).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import REC_2022  # noqa: E402

LIGHT_ON_S = 10800.0  # meta.csv light_sec: dark (IR) for the first 3 h of the recording


@dataclass(frozen=True)
class Recording2022:
    name: str
    raw: Path
    interrim: Path

    @property
    def nix(self) -> Path:
        return _one(self.interrim, "*.analysis.nix")

    @property
    def video(self) -> Path:
        return _one(self.raw, "*converted.mp4")

    def sample_rate(self) -> float:
        """AISampleRate of fishgrid.cfg (Hz)."""
        txt = (self.raw / "fishgrid.cfg").read_text(errors="replace")
        m = re.search(r"AISampleRate\s*:\s*([\d.]+)\s*kHz", txt)
        if not m:
            raise ValueError(f"{self.raw}/fishgrid.cfg: no AISampleRate in kHz")
        return float(m.group(1)) * 1000.0

    def n_channels(self) -> int:
        txt = (self.raw / "fishgrid.cfg").read_text(errors="replace")
        rows = int(re.search(r"Rows1\s*:\s*(\d+)", txt).group(1))
        cols = int(re.search(r"Columns1\s*:\s*(\d+)", txt).group(1))
        return rows * cols

    def recording_duration_s(self) -> float:
        """Length of traces-grid1.raw (float32, n_channels interleaved) in seconds."""
        size = (self.raw / "traces-grid1.raw").stat().st_size
        return size / 4 / self.n_channels() / self.sample_rate()

    def led_frames(self) -> np.ndarray:
        return np.load(self.interrim / "LED_frames.npy")

    def led_idx(self) -> np.ndarray:
        return pd.read_csv(self.raw / "led_idxs.csv", header=None).iloc[:, 0].to_numpy()


def _one(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{folder}: {len(hits)} files match {pattern}, expected 1")
    return hits[0]


def list_recordings(root: Path = REC_2022) -> list[Recording2022]:
    names = sorted(p.name for p in (root / "raw").iterdir() if p.is_dir() and p.name.startswith("2022-"))
    return [Recording2022(n, root / "raw" / n, root / "interrim" / n) for n in names]
