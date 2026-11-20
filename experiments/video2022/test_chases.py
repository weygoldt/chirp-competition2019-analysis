"""Tests of chases.py on synthetic fish."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chases as ch

BL = 200.0
STRIDE, FPS = 2, 25.0
OFFSETS = {"snout": -0.18, "spine1": 0.0, "spine2": 0.27, "spine3": 0.54, "tail": 0.82}  # BL behind spine1


def fish_rows(path_xy: np.ndarray, heading: np.ndarray, frames: np.ndarray, conf: float = 0.9) -> list[dict]:
    """Detections of one straight fish: spine1 at path_xy, body pointing against its heading (unit vectors)."""
    rows = []
    for f, (x, y), (hx, hy) in zip(frames, path_xy, heading, strict=True):
        r = {"frame": int(f), "t_s": f / FPS, "box_conf": conf}
        for n, o in OFFSETS.items():
            r[f"{n}_x"], r[f"{n}_y"] = x - o * BL * hx, y - o * BL * hy
        rows.append(r)
    return rows


def scene(*fish: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame([r for f in fish for r in f]).sort_values("frame", kind="stable")
    d["det"] = d.groupby("frame").cumcount()
    return d.reset_index(drop=True)


def straight(frames: np.ndarray, x0: float, y0: float, v_bl: float, delay_s: float = 0.0) -> list[dict]:
    t = frames / FPS - delay_s
    xy = np.stack([x0 + v_bl * BL * t, np.full(len(t), y0)], axis=1)
    return fish_rows(xy, np.tile([1.0, 0.0], (len(t), 1)), frames)


FRAMES = np.arange(0, 100, STRIDE)  # 4 s
SEP = ch.Params(sep_grow_bl=0.5)  # the separation rule (off by default)


def test_follower_on_the_leaders_path_is_a_chase():
    d = scene(straight(FRAMES, 1000, 700, 3.0), straight(FRAMES, 1000, 700, 3.0, delay_s=0.4))
    tr, _, _, ev = ch.detect(d, BL, STRIDE, FPS)
    assert tr["track"].nunique() == 2
    assert len(ev) == 1
    at50 = tr[tr["frame"] == 50]
    lead = at50.loc[at50["spine1_x"].idxmax(), "track"]
    assert ev["leader"].iloc[0] == lead
    assert ev["duration_s"].iloc[0] >= 3.0
    assert abs(ev["lag_s"].iloc[0] - 0.4) <= 0.08


def test_side_by_side_is_no_chase():
    d = scene(straight(FRAMES, 1000, 700, 3.0), straight(FRAMES, 1000, 900, 3.0))
    assert ch.detect(d, BL, STRIDE, FPS)[3].empty


def test_slow_following_is_no_chase():
    d = scene(straight(FRAMES, 1000, 700, 0.5), straight(FRAMES, 1000, 700, 0.5, delay_s=1.0))
    assert ch.detect(d, BL, STRIDE, FPS)[3].empty


def test_short_chase_is_dropped():
    f = np.arange(0, 20, STRIDE)  # 0.8 s
    d = scene(straight(f, 1000, 700, 3.0), straight(f, 1000, 700, 3.0, delay_s=0.4))
    assert ch.detect(d, BL, STRIDE, FPS)[3].empty


def test_clean_drops_short_pieces_and_weak_boxes():
    d = scene(straight(FRAMES[:3], 1000, 700, 1.0))
    short = d.copy()
    for n in ("spine2", "spine3", "tail"):
        short[f"{n}_x"], short[f"{n}_y"] = short["spine1_x"], short["spine1_y"]
    weak = d.assign(box_conf=0.3)
    assert len(ch.clean(d, BL)) == 3
    assert ch.clean(short, BL).empty
    assert ch.clean(weak, BL).empty


def test_two_fish_keep_their_tracks_when_passing():
    # two fish swimming towards each other on lines 0.8 BL apart
    a = straight(FRAMES, 600, 700, 3.0)
    b = fish_rows(np.stack([1800 - 3.0 * BL * FRAMES / FPS, np.full(len(FRAMES), 860.0)], axis=1),
                  np.tile([-1.0, 0.0], (len(FRAMES), 1)), FRAMES)
    tr = ch.track(scene(a, b), BL)
    assert tr["track"].nunique() == 2
    for _, g in tr.groupby("track"):
        assert g["spine1_y"].nunique() == 1


def falling_behind() -> pd.DataFrame:
    """Leader at 3 BL/s; follower on its path 0.4 s behind, slowing to 2 BL/s after 2 s (the gap opens)."""
    t = FRAMES / FPS
    lead = np.stack([1000 + 3.0 * BL * t, np.full(len(t), 700.0)], axis=1)
    xf = np.where(t <= 2.0, 1000 + 3.0 * BL * (t - 0.4), 1000 + 3.0 * BL * 1.6 + 2.0 * BL * (t - 2.0))
    foll = np.stack([xf, np.full(len(t), 700.0)], axis=1)
    head = np.tile([1.0, 0.0], (len(t), 1))
    return scene(fish_rows(lead, head, FRAMES), fish_rows(foll, head, FRAMES))


def test_chase_ends_when_the_gap_opens():
    ev = ch.detect(falling_behind(), BL, STRIDE, FPS, SEP)[3]
    assert len(ev) == 1
    assert 2.0 <= ev["t_end"].iloc[0] <= 2.4


def test_without_the_separation_rule_the_chase_runs_on():
    ev = ch.detect(falling_behind(), BL, STRIDE, FPS)[3]
    assert len(ev) == 1
    assert ev["t_end"].iloc[0] > 3.4


def test_separation_pieces():
    d = np.array([1.0, 0.8, 0.6, 0.65, 0.7, 0.9, 1.2, 1.4, 1.0, 0.7, 0.6])
    # minimum 0.6 at 2; 1.2 at 6 is > 0.6 + 0.5 -> piece ends at the last value <= 0.75 before (position 4);
    # the gap grows until 7, the next piece starts there (the follower closes in again)
    assert ch.separation_pieces(d, 0.5, 0.15) == [(0, 4), (7, 10)]
    assert ch.separation_pieces(d, np.inf, 0.15) == [(0, 10)]
    # the gap keeps growing: no further piece
    assert ch.separation_pieces(np.array([0.6, 0.7, 1.2, 1.5, 1.9]), 0.5, 0.15) == [(0, 1)]


def test_short_chase_ending_in_a_separation_keeps_its_onset():
    """A 1.5 s run of following whose gap opens after 0.6 s: the event keeps its onset and ends early."""
    t = FRAMES / FPS
    lead = np.stack([1000 + 3.0 * BL * t, np.full(len(t), 700.0)], axis=1)
    xf = np.where(t <= 1.0, 1000 + 3.0 * BL * (t - 0.4), 1000 + 3.0 * BL * 0.6 + 1.5 * BL * (t - 1.0))
    head = np.tile([1.0, 0.0], (len(t), 1))
    d = scene(fish_rows(lead, head, FRAMES), fish_rows(np.stack([xf, np.full(len(t), 700.0)], axis=1), head, FRAMES))
    plain = ch.detect(d, BL, STRIDE, FPS)[3]
    ev = ch.detect(d, BL, STRIDE, FPS, SEP)[3]
    assert len(plain) == len(ev) == 1
    assert ev["t_start"].iloc[0] == plain["t_start"].iloc[0]
    assert ev["t_end"].iloc[0] < plain["t_end"].iloc[0]
    assert ev["duration_s"].iloc[0] < 1.0
