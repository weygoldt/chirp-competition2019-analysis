"""Tests of identity_tracks.fill_gaps on a synthetic winner / loser pair."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import identity_tracks as it

BL = 200.0
FRAMES = np.arange(0, 120, 2)  # 60 samples, 0.08 s apart
TIMES = FRAMES / 25.0


def det(frame: int, x: float, y: float, d: int, ident: str | None = None) -> dict:
    r = {"frame": frame, "det": d, "snout_x": x + 40, "snout_y": y, "spine1_x": x, "spine1_y": y,
         "tail_x": x - 160, "tail_y": y}
    if ident:
        r["identity"] = ident
    return r


def scene() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Winner moves along y = 500 (10 px / sample), missing at samples 10-19 (0.8 s) and 30-49 (1.6 s);
    loser sits at (1500, 900) throughout."""
    lab, raw = [], []
    for j, f in enumerate(FRAMES):
        x = 100 + 10 * j
        if not (10 <= j <= 19 or 30 <= j <= 49):
            lab.append(det(f, x, 500, 0, "winner"))
            raw.append(det(f, x, 500, 0))
        lab.append(det(f, 1500, 900, 1, "loser"))
        raw.append(det(f, 1500, 900, 1))
        if 12 <= j <= 15:  # unused detections on the winner's path (unknown identity / low confidence)
            raw.append(det(f, x + 5, 505, 2))
        if j == 16:  # a detection far from the winner's path
            raw.append(det(f, x + 400, 500, 3))
        if j == 17:  # a detection on top of the loser
            raw.append(det(f, 1505, 900, 4))
    kp = [c for c in it.KP]
    lab, raw = pd.DataFrame(lab), pd.DataFrame(raw)
    return lab.astype({c: "float32" for c in kp}), raw.astype({c: "float32" for c in kp})  # as stored


def test_fill_uses_nearby_detections_and_interpolates_short_gaps():
    lab, raw = scene()
    out = it.fill_gaps(lab, raw, FRAMES, TIMES, BL)
    w = out[out["identity"] == "winner"].set_index("frame")
    filled = w[w["source"] == "filled"]
    assert list(filled.index) == list(FRAMES[12:16])
    assert np.allclose(filled["spine1_y"], 505)
    # the rest of the 0.8 s gap (no usable detection) is interpolated along the path
    interp = w[w["source"] == "interp"]
    assert set(interp.index) == set(FRAMES[10:20]) - set(FRAMES[12:16])
    assert interp["spine1_y"].between(500, 505).all()  # along the path, between labelled and filled points
    # the 1.6 s gap without detections is longer than max_interp_s: left empty
    assert not set(FRAMES[30:50]) & set(w.index)
    # the loser is untouched, the far and the loser-overlapping detections unused
    lo = out[out["identity"] == "loser"]
    assert (lo["source"] == "labelled").all()
    assert len(lo) == len(FRAMES)
    assert not ((w["spine1_x"] > 1000) & (w.index == FRAMES[16])).any()
