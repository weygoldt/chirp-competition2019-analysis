"""Masks of the shelters (the tube and the four blocks) in the 2022 videos, and occluded-point removal.

A fish under a shelter is hidden from the top-down camera. The labelling rule
(author, 2026-11-19) is to label only the visible part of a fish, but in round 1
the author first deleted points under shelters and later placed them where the
hidden body would be. To make all labels follow the rule, points inside a shelter
are removed (author's go, 2026-11-19: "auto delete occlusions").

Mask of a shelter: the dark pixels (grey < ``DARK``) inside its bounding box
(``b1_sleap/tables/arena_objects.csv``, found in one light-phase frame per
recording, ``b1_sleap/background/<rec>.png``), their convex hull, shrunk by
``SHRINK_PX`` so that a point placed at the shelter's edge (where a visible fish
emerges) is kept. The small dark discs next to the tube (cable posts) are not
shelters.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import pose_labels as pl

B1 = Path("/mnt/data1/chirp_v2/video2022/b1_sleap")
DARK = 60
SHRINK_PX = 4
KINDS = ("tube", "block")


@lru_cache(maxsize=32)
def mask(recording: str) -> np.ndarray:
    """Boolean (height, width) mask of the recording's shelters."""
    import cv2

    img = cv2.imread(str(B1 / "background" / f"{recording}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(B1 / "background" / f"{recording}.png")
    objs = pd.read_csv(B1 / "tables" / "arena_objects.csv")
    objs = objs[(objs["recording"] == recording) & objs["kind"].isin(KINDS)]
    if len(objs) != 5:
        raise ValueError(f"{recording}: {len(objs)} shelters found, expected 5 (tube + 4 blocks)")
    out = np.zeros(img.shape, np.uint8)
    for o in objs.itertuples():
        x0, y0, x1, y1 = int(o.x0), int(o.y0), int(o.x1), int(o.y1)
        ys, xs = np.nonzero(img[y0:y1 + 1, x0:x1 + 1] < DARK)
        if len(xs) < 3:
            continue
        hull = cv2.convexHull(np.stack([xs + x0, ys + y0], 1).astype(np.int32))
        cv2.fillConvexPoly(out, hull, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SHRINK_PX + 1, 2 * SHRINK_PX + 1))
    return cv2.erode(out, k).astype(bool)


def remove_occluded(fish: list[pl.Fish], recording: str) -> tuple[list[pl.Fish], int]:
    """The fish without their points inside a shelter (fish left without points are dropped); and how many points were removed."""
    m = mask(recording)
    h, w = m.shape
    out, removed = [], 0
    for f in fish:
        keep = {}
        for n, (x, y) in f.points.items():
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h and m[yi, xi]:
                removed += 1
            else:
                keep[n] = (x, y)
        if keep:
            out.append(pl.Fish(keep, origin=f.origin))
    return out, removed
