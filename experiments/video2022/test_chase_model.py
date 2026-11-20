"""Tests of chase_model.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chase_model as cm

RAW = cm.Decoder(smooth_s=cm.DT, hold_s=0.4, max_gap_s=0.4, min_dur_s=0.4)  # no smoothing: 5 samples hold / gap


def test_decode_bridges_short_dips_and_gaps():
    p = np.r_[np.full(10, 0.1), np.full(20, 0.8), np.full(3, 0.2), np.full(10, 0.8), np.full(4, np.nan),
              np.full(10, 0.8), np.full(20, 0.1)]
    assert cm.decode(p, RAW) == [(10, 56)]


def test_decode_splits_at_long_dips_and_gaps():
    p = np.r_[np.full(10, 0.8), np.full(8, 0.2), np.full(10, 0.8), np.full(8, np.nan), np.full(10, 0.8)]
    assert cm.decode(p, RAW) == [(0, 9), (18, 27), (36, 45)]


def test_decode_drops_short_chases():
    p = np.r_[np.full(10, 0.1), np.full(3, 0.9), np.full(10, 0.1)]
    assert cm.decode(p, RAW) == []


def test_logistic_separates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    p = cm.Logistic(l2=0.1).fit(X, y).predict(X)
    assert ((p > 0.5) == y).mean() > 0.9
    X[0, 1] = np.nan
    assert np.isnan(cm.Logistic().fit(X, y).predict(X)[0])


def test_directed_features_chaser_aims_and_chasee_flees():
    t = np.arange(50) * cm.DT
    s = pd.DataFrame({"x_A": 100 + 400 * t, "y_A": 500.0, "x_B": 300 + 400 * t, "y_B": 500.0,  # A behind B, both +x
                      "vx_A": 2.0, "vy_A": 0.0, "vx_B": 2.0, "vy_B": 0.0, "speed_A": 2.0, "speed_B": 2.0,
                      "accel_A": 0.0, "accel_B": 0.0, "dist": 1.0, "ddist": 0.0, "cos": 1.0})
    f = cm.directed_features(s)
    assert np.allclose(f["A"]["aim"], 1)  # A swims at B
    assert np.allclose(f["A"]["flee"], 1)  # B swims away from A
    assert np.allclose(f["B"]["aim"], -1)  # B swims away from A
