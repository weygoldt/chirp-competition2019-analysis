"""Tests of frames.py on a small synthetic video (each frame shows its own index)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames  # noqa: E402

N = 60


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import cv2

    path = tmp_path_factory.mktemp("v") / "counter.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    for i in range(N):
        img = np.full((48, 64, 3), 4 * i, np.uint8)  # brightness encodes the frame index
        w.write(img)
    w.release()
    return path


def index_of(img: np.ndarray) -> int:
    """Brightness 4 i comes back as about 4 i - 2 after MJPG; take the nearest encoded index."""
    return int(np.argmin(np.abs(4 * np.arange(N) - 2 - float(img.mean()))))


def test_read_sequential_returns_the_wanted_frames(video: Path) -> None:
    got = dict(frames.read_sequential(video, [50, 3, 3, 17]))
    assert sorted(got) == [3, 17, 50]
    assert all(index_of(img) == f for f, img in got.items())


def test_read_sequential_fails_past_the_end(video: Path) -> None:
    with pytest.raises(RuntimeError):
        list(frames.read_sequential(video, [N + 5]))


def test_seek_matches_sequential(video: Path) -> None:
    assert all(frames.check_seek(video, [0, 9, 31, 59]).values())
