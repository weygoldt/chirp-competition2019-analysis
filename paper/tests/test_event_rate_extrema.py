"""Extrema of the event-triggered rates (Fig. event_correlations): split at the event time."""

from __future__ import annotations

import numpy as np
import plot_event_time_rates_winner_loser_chirp_src_paper as etr

TIME = np.round(np.arange(-30, 30.05, 0.1), 1)
BASELINE = np.full_like(TIME, 0.67)
LO, HI = BASELINE - 0.1, BASELINE + 0.1


def bump(centre: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((TIME - centre) / width) ** 2)


def winner_like_rate() -> np.ndarray:
    """One stretch above the CI from about -13.7 to +7.7 s, highest at -3 s, a smaller
    hump at +0.6 s, and a separate small blip at +9.9 s (the shape of winner chirps
    around loser chirps)."""
    return (
        BASELINE + bump(-3.0, 5.0, 1.0) + bump(0.6, 0.4, 0.15) + bump(9.9, 0.15, 0.13)
    )


def test_unsplit_gives_the_crossing_stretch_to_one_side():
    rate = winner_like_rate()
    by = {
        (e["side"], e["type"]): e
        for e in etr.find_significant_extrema(TIME, rate, LO, BASELINE, HI)
    }
    assert by[("before", "peak")]["time"] < 0
    assert by[("before", "peak")]["region_end"] > 5  # the stretch crosses the event
    # the after side is left with the blip
    assert abs(by[("after", "peak")]["time"] - 9.9) < 0.15


def test_split_gives_each_side_its_own_part():
    rate = winner_like_rate()
    by = {
        (e["side"], e["type"]): e
        for e in etr.split_extrema(TIME, rate, LO, BASELINE, HI)
    }
    after, before = by[("after", "peak")], by[("before", "peak")]
    assert 0 < after["time"] < 1.5
    assert after["region_start"] == 0.0
    assert after["region_end"] > 5
    assert after["fold_point"] > 2
    assert abs(before["time"] + 3.0) < 0.15
    assert -0.15 < before["region_end"] < 0
    assert before["region_start"] < -10


def test_split_equals_unsplit_when_no_stretch_crosses_the_event():
    rate = BASELINE + bump(-8, 1.0, 0.5) + bump(3, 1.0, 0.8) - bump(12, 1.0, 0.4)
    assert etr.split_extrema(
        TIME, rate, LO, BASELINE, HI
    ) == etr.find_significant_extrema(TIME, rate, LO, BASELINE, HI)


def test_extrema_stats_adds_split_and_keeps_the_rest():
    rate = winner_like_rate()
    extrema = etr.find_significant_extrema(TIME, rate, LO, BASELINE, HI)
    split = etr.split_extrema(TIME, rate, LO, BASELINE, HI)
    out = etr.extrema_stats(extrema, split)
    without = {k: v for k, v in out.items() if k != "split_at_event"}
    assert without == etr.extrema_stats(extrema, [])
    assert 0 < out["split_at_event"]["after"]["peak"]["time_s"] < 1.5
    assert out["split_at_event"]["after"]["peak"]["region_s"][0] == 0.0
