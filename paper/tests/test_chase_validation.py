"""The chase timing validation and the chirping-starts-before-offset test (pure parts, synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chase_validation as cv
import stats_bayesian_logreg_loser_chirping_starts_before_chasing_offset as cs


def test_events_code_the_start_of_chirping_and_its_chance_level():
    chirps = {"r": np.array([8.0, 9.5, 10.0, 10.5, 30.0, 31.0, 49.0, 50.0])}
    df = cs.events([("r", 10.0), ("r", 30.0), ("r", 50.0), ("r", 70.0)], chirps, 2.0)
    # offset 10, window [8, 12): 8.0 and 9.5 before; 10.0 (at the offset) and 10.5 after
    first = df.iloc[0]
    assert (first["before"], first["after"], first["k"], first["y"]) == (2, 2, 4, 1)
    assert first["p0"] == pytest.approx(1 - 0.5 ** 4)
    assert first["lp0"] == pytest.approx(np.log(first["p0"] / (1 - first["p0"])))
    second = df.iloc[1]  # chirps only at and after the offset
    assert (second["before"], second["after"], second["y"]) == (0, 2, 0)
    third = df.iloc[2]  # one before, one at the offset (counts as after)
    assert (third["before"], third["after"], third["y"]) == (1, 1, 1)
    assert len(df) == 3  # the offset at 70 s has no chirp within +-2 s and is not an event


def test_symmetric_chirps_match_the_chance_level():
    """Chirp times drawn symmetric around the offsets: the observed share of 'starts before' is the expected one."""
    rng = np.random.default_rng(0)
    offs = [("r", 100.0 * i) for i in range(1, 2001)]
    times = []
    for _, x in offs:
        k = rng.integers(0, 4)
        times += list(x + rng.uniform(-2, 2, k))
    df = cs.events(offs, {"r": np.sort(np.array(times))}, 2.0)
    assert df["y"].mean() == pytest.approx(df["p0"].mean(), abs=0.02)


def test_one_chase_reviews_are_censored_at_the_window_edges():
    session = {"chases": [{"id": "a", "recording": "r", "onset_rec_s": 10.0, "offset_rec_s": 20.0,
                           "onset_frame": 250, "offset_frame": 500}]}
    rev = {"a": {"mode": "target", "not_a_chase": False, "window": {"frame0": 100, "n": 701},
                 "onset_frame": 100, "offset_frame": 480, "onset_rec_s": 4.0, "offset_rec_s": 19.2,
                 "d_onset_s": -6.0, "d_offset_s": -0.8, "blind": True, "saved_at": "t"}}
    (p,) = cv.target_pairs(session, rev)
    assert p["onset_censored"] and not p["offset_censored"]
    assert np.isnan(p["d_onset_s"]) and p["d_offset_s"] == pytest.approx(-0.8)
    rev["a"]["offset_frame"] = 800  # the window's last frame
    (p,) = cv.target_pairs(session, rev)
    assert p["offset_censored"] and np.isnan(p["d_offset_s"])


def test_shift_stats():
    d = pd.Series([-2.0, -1.0, -1.0, 0.5, np.nan])
    s = cv.shift_stats(d, pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), pd.Series(["a", "a", "a", "b", "b"]))
    assert s["n"] == 4 and s["median_s"] == pytest.approx(-1.0)
    assert s["earlier_fraction"] == pytest.approx(0.75)
