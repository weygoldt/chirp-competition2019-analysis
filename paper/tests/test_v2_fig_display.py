"""The power display of the pipeline-v2 figures: dB conversion and display levels."""

from __future__ import annotations

import numpy as np
import pytest

import v2_fig_extract as X


def test_to_db_converts_power_to_decibels():
    power = np.array([1.0, 1e-3, 1e-10, 2.0])
    db = X.to_db(power)
    assert db.dtype == np.float16
    np.testing.assert_allclose(db.astype(float), [0.0, -30.0, -100.0, 3.0103], atol=0.07)


def test_to_db_floors_zeros_like_the_front_end():
    """Zero power maps to the float32 floor (finite), as chirpdetector's normalize."""
    db = X.to_db(np.zeros(3))
    tiny_db = 10 * np.log10(np.finfo(np.float32).tiny)
    assert np.all(np.isfinite(db.astype(float)))
    np.testing.assert_allclose(db.astype(float), tiny_db, atol=0.1)


def test_to_db_keeps_shape():
    power = np.full((3, 4, 5), 1e-6)
    assert X.to_db(power).shape == (3, 4, 5)


@pytest.fixture(scope="module")
def common():
    """v2_fig_common needs the production run (operating threshold at import)."""
    try:
        import v2_fig_common
    except (ValueError, OSError) as err:  # run directory not mounted
        pytest.skip(f"v2_fig_common not importable here: {err}")
    return v2_fig_common


def test_clip_levels_are_symmetric_percentiles(common):
    values = np.arange(101, dtype=float)  # percentile q of 0..100 is q
    assert common.clip_levels(values, q=2) == pytest.approx((2.0, 98.0))
    assert common.clip_levels(values, q=0) == pytest.approx((0.0, 100.0))
    assert common.clip_levels(values) == pytest.approx((common.CLIP_PCT, 100 - common.CLIP_PCT))


def test_clip_levels_fail_loudly(common):
    with pytest.raises(ValueError, match="not in"):
        common.clip_levels(np.arange(10.0), q=50)
    with pytest.raises(ValueError, match="flat"):
        common.clip_levels(np.ones(10), q=2)


# --- snippet windows (v2_fig_extract) ------------------------------------------


def test_snippet_window_with_and_without_band():
    assert X.snippet_window(("rec", 1.0, 2.0)) == ("rec", 1.0, 2.0, None)
    assert X.snippet_window(("rec", 1, 2, (600, 900))) == ("rec", 1.0, 2.0, (600.0, 900.0))


@pytest.mark.parametrize("entry", [
    ("rec", 2.0, 1.0),  # t0 after t1
    ("rec", 1.0, 2.0, (400.0, 900.0)),  # band below the front end's 500 Hz
    ("rec", 1.0, 2.0, (900.0, 600.0)),  # band reversed
    ("rec", 1.0),  # too short
])
def test_snippet_window_fails_loudly(entry):
    with pytest.raises(ValueError):
        X.snippet_window(entry)


def test_every_snippet_entry_is_valid():
    for entry in X.SNIPPETS.values():
        X.snippet_window(entry)


def test_parse_window():
    assert X.parse_window(["f3_long", "rec", "10", "12.5"]) == ("f3_long", ("rec", 10.0, 12.5))
    assert X.parse_window(["f3_long", "rec", "10", "12.5", "600", "1300"]) == (
        "f3_long", ("rec", 10.0, 12.5, (600.0, 1300.0)))
    with pytest.raises(ValueError):
        X.parse_window(["f3_long", "rec", "10"])


# --- tracks: detected (solid) and interpolated (dashed) pieces -------------------


def test_track_segments_bridges_short_gaps_only(common):
    t = np.array([0.0, 0.3, 0.6, 2.0, 2.3, 20.0, 20.3])  # gaps of 1.4 s and 17.7 s
    f = np.arange(len(t), dtype=float)
    fl = np.zeros(len(t), bool)
    ts, _, _, _, pieces = common.track_segments(t, f, fl, bridge_s=10.0)
    assert pieces == [(0.6, 2.0)]  # the 17.7-s gap stays open
    assert np.isnan(ts).sum() == 2  # the solid line breaks at both gaps
    assert common.track_segments(t, f, fl)[4] == []  # no bridges by default


def test_track_segments_joins_filled_samples_to_their_neighbours(common):
    t = np.arange(6, dtype=float) * 0.3
    f = np.arange(6, dtype=float)
    fl = np.array([False, False, True, True, False, False])
    _, fs, _, _, pieces = common.track_segments(t, f, fl, bridge_s=10.0)
    assert pieces == [(0.3, 1.2)]
    assert np.isnan(fs[2]) and np.isnan(fs[3])  # filled samples are not solid
