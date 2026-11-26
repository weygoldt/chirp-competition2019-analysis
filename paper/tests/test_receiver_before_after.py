"""plot_receiver_before_after: rows, within-competition weights, medians and per-competition ratios (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plot_receiver_before_after as rba

WINDOW = (-5.0, 11.0)  # width 16 s


def table(rows):
    """Per-chase rows (recording, chirped, previous interval, next interval, window duration)."""
    return pd.DataFrame(rows, columns=[rba.COMP, rba.PREDICTOR, rba.PREV, rba.NEXT, "offset_window_duration"])


def test_subsets_keep_complete_windows_with_both_intervals():
    obs = table([
        ("r1", True, 5.0, 20.0, 16.0),  # complete
        ("r1", False, 8.0, 30.0, 16.0),  # complete
        ("r1", True, 3.0, 4.0, 9.0),  # next chase after 4 s: window clipped
        ("r1", False, np.nan, 40.0, 16.0),  # first chase: no previous interval
        ("r1", True, 6.0, np.nan, 16.0),  # last chase: no next interval
        ("r2", False, 2.0, 12.0, 15.0),  # chase shorter than 5 s: window clipped at its onset
    ])
    complete = rba.analysed_events(obs, WINDOW, "complete")
    assert list(complete.index) == [0, 1]
    assert complete["chirp"].tolist() == [True, False]
    full = rba.analysed_events(obs, WINDOW, "full")
    assert list(full.index) == [0, 1, 2, 5]
    with pytest.raises(ValueError):
        rba.analysed_events(obs, WINDOW, "matched")
    with pytest.raises(KeyError):
        rba.analysed_events(obs.drop(columns=[rba.NEXT]), WINDOW, "full")


def test_a_complete_window_with_a_short_next_interval_is_an_error():
    obs = table([("r1", True, 5.0, 8.0, 16.0)])  # window complete although the next chase came after 8 s
    with pytest.raises(ValueError):
        rba.analysed_events(obs, WINDOW, "complete")


def test_weights_give_each_competition_the_same_share_in_both_groups():
    # r1: 3 with, 1 without; r2: 2 with, 2 without; r3: only without (no within-competition contrast)
    flags = {"r1": [True] * 3 + [False], "r2": [True, True, False, False], "r3": [False, False]}
    ev = rba.analysed_events(table([(r, f, 10.0, 20.0, 16.0) for r, fl in flags.items() for f in fl]),
                             WINDOW, "full")
    w_c = rba.competition_weights(ev)
    assert w_c["r1"] == pytest.approx(3 * 1 / 4) and w_c["r2"] == pytest.approx(2 * 2 / 4) and w_c["r3"] == 0
    w = rba.event_weights(ev)
    for chirp in (True, False):
        g = ev["chirp"] == chirp
        assert w[g].sum() == pytest.approx(1.0)
        share = w[g].groupby(ev.loc[g, rba.COMP]).sum()
        assert share["r1"] == pytest.approx(0.75 / 1.75) and share["r2"] == pytest.approx(1.0 / 1.75)
    assert (w[ev[rba.COMP] == "r3"] == 0).all()


def test_weighted_median_and_ecdf():
    xs, F = rba.weighted_ecdf([3.0, 1.0, 2.0], [1, 1, 2])
    assert xs.tolist() == [1.0, 2.0, 3.0] and F == pytest.approx([0.25, 0.75, 1.0])
    assert rba.weighted_median([3.0, 1.0, 2.0], [1, 1, 2]) == 2.0
    assert rba.weighted_median([1.0, 2.0], [1, 1]) == 1.0  # reaches one half at the first value


def test_trial_composition_cancels_in_the_competition_average():
    """No difference within competitions, but the chirp group comes mostly from a densely chasing competition:
    the pooled medians differ, the competition-averaged ones do not."""
    rng = np.random.default_rng(1)
    rows = []
    for comp, scale, n_chirp, n_none in (("dense", 10.0, 90, 10), ("sparse", 100.0, 10, 90)):
        for chirp, n in ((True, n_chirp), (False, n_none)):
            # identical interval sets for both groups of a competition (quantiles of one distribution)
            q = np.exp(np.linspace(-1, 1, n))
            for prev, nxt in zip(scale * q, scale * rng.permutation(q) * 2):
                rows.append((comp, chirp, prev, max(nxt, 11.0), 16.0))
    ev = rba.analysed_events(table(rows), WINDOW, "complete")
    s = rba.summarise(ev)
    assert s["n_events"] == {"chirp": 100, "no_chirp": 100, "total": 200} and s["n_competitions"] == 2
    assert s["previous"]["pooled_median_ratio"] < 0.3  # composition alone
    assert s["previous"]["median_ratio"] == pytest.approx(1.0, abs=0.1)
    assert s["previous"]["competitions"]["median_ratio"] == pytest.approx(1.0, rel=1e-3)


def test_competition_ratios_and_counts():
    rows = [("a", True, 10.0, 40.0, 16.0), ("a", False, 20.0, 20.0, 16.0),  # previous 0.5, next 2
            ("b", True, 30.0, 15.0, 16.0), ("b", False, 10.0, 30.0, 16.0),  # previous 3, next 0.5
            ("c", True, 5.0, 12.0, 16.0), ("c", False, 20.0, 24.0, 16.0),  # previous 0.25, next 0.5
            ("d", False, 7.0, 30.0, 16.0)]  # one group only: no ratio
    ev = rba.analysed_events(table(rows), WINDOW, "complete")
    prev = rba.competition_ratios(ev, rba.PREV)
    assert prev.to_dict() == pytest.approx({"a": 0.5, "b": 3.0, "c": 0.25})
    s = rba.summarise(ev)
    assert s["previous"]["competitions"] == pytest.approx(
        {"n": 3, "n_below_1": 2, "n_above_1": 1, "median_ratio": 0.5, "min_ratio": 0.25, "max_ratio": 3.0})
    assert (s["next"]["competitions"]["n_below_1"], s["next"]["competitions"]["n_above_1"]) == (2, 1)
    assert s["n_competitions"] == 3 and s["n_competitions_all"] == 4
