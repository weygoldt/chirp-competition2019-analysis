"""Read individual frames of the competition videos by index.

Two ways, which give identical images where seeking is exact (checked with
``check_seek``):

- :func:`read_sequential` decodes the video from its start and keeps the
  wanted frames: exact by construction (the frame index is the count of
  decoded frames), about 1.7 ms per frame for the 2048x1536 2022 videos, i.e.
  ~10 min for a whole 4-h video;
- :func:`read_seek` seeks to each wanted frame (OpenCV/FFmpeg: jump to the
  preceding keyframe, decode forward), ~0.2 s per frame; exact only if the
  container's timestamps are regular.

Frames come back as uint8 grey images (the videos are grey; channel 0).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np


def _open(video: Path):  # noqa: ANN202
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open {video}")
    return cap


def read_sequential(video: Path, frames: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame, image)`` for the wanted frames, in increasing order, decoding from frame 0."""
    wanted = sorted({int(f) for f in frames})
    if not wanted:
        return
    if wanted[0] < 0:
        raise ValueError(f"negative frame index {wanted[0]}")
    cap = _open(video)
    try:
        k = 0
        for f in range(wanted[-1] + 1):
            if f == wanted[k]:
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f"{video}: video ends before frame {f}")
                yield f, img[..., 0].copy()
                k += 1
            elif not cap.grab():
                raise RuntimeError(f"{video}: video ends before frame {f}")
    finally:
        cap.release()


def read_seek(video: Path, frames: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame, image)`` for the wanted frames, in increasing order, by seeking to each."""
    import cv2

    cap = _open(video)
    try:
        for f in sorted({int(x) for x in frames}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"{video}: cannot read frame {f}")
            yield f, img[..., 0].copy()
    finally:
        cap.release()


def check_seek(video: Path, frames: Iterable[int]) -> dict[int, bool]:
    """For each frame: is the seeked image identical to the sequentially decoded one?"""
    frames = sorted({int(f) for f in frames})
    seq = dict(read_sequential(video, frames))
    return {f: bool(np.array_equal(img, seq[f])) for f, img in read_seek(video, frames)}
