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
