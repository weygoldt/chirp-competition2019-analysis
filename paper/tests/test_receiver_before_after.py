"""plot_receiver_before_after: the previous-interval panel (C: rows, within-competition weights, medians,
per-competition ratios), the raster (A) and the inputs of the matched panel (B) on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import plot_receiver_before_after as rba


def table(rows):
    """Per-chase rows (recording, chirped, previous interval)."""
    return pd.DataFrame(rows, columns=[rba.COMP, rba.PREDICTOR, rba.PREV])


def test_events_are_the_chases_with_a_previous_interval():
    obs = table([
        ("r1", True, 5.0),
        ("r1", False, 8.0),
        ("r1", True, np.nan),  # first chase: no previous interval
        ("r2", False, 0.0),  # not positive
        ("r2", True, 2.0),  # the last chase of a recording (no next interval) is kept
    ])
    ev = rba.analysed_events(obs)
    assert list(ev.index) == [0, 1, 4]
    assert ev["chirp"].tolist() == [True, False, True]
    with pytest.raises(KeyError):
        rba.analysed_events(obs.drop(columns=[rba.PREV]))
    with pytest.raises(ValueError):
        rba.analysed_events(obs.assign(**{rba.PREDICTOR: [True, None, True, False, True]}))


def test_weights_give_each_competition_the_same_share_in_both_groups():
    # r1: 3 with, 1 without; r2: 2 with, 2 without; r3: only without (no within-competition contrast)
    flags = {"r1": [True] * 3 + [False], "r2": [True, True, False, False], "r3": [False, False]}
    ev = rba.analysed_events(table([(r, f, 10.0) for r, fl in flags.items() for f in fl]))
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
    rows = []
    for comp, scale, n_chirp, n_none in (("dense", 10.0, 90, 10), ("sparse", 100.0, 10, 90)):
        for chirp, n in ((True, n_chirp), (False, n_none)):
            # identical interval sets for both groups of a competition (quantiles of one distribution)
            for prev in scale * np.exp(np.linspace(-1, 1, n)):
                rows.append((comp, chirp, prev))
    s = rba.summarise_previous(rba.analysed_events(table(rows)))
    assert s["n_events"] == {"chirp": 100, "no_chirp": 100, "total": 200} and s["n_competitions"] == 2
    assert s["pooled_median_ratio"] < 0.3  # composition alone
    assert s["median_ratio"] == pytest.approx(1.0, abs=0.1)
    assert s["competitions"]["median_ratio"] == pytest.approx(1.0, rel=1e-3)


def test_competition_ratios_and_counts():
    rows = [("a", True, 10.0), ("a", False, 20.0),  # 0.5
            ("b", True, 30.0), ("b", False, 10.0),  # 3
            ("c", True, 5.0), ("c", False, 20.0),  # 0.25
            ("d", False, 7.0)]  # one group only: no ratio
    ev = rba.analysed_events(table(rows))
    assert rba.competition_ratios(ev).to_dict() == pytest.approx({"a": 0.5, "b": 3.0, "c": 0.25})
    s = rba.summarise_previous(ev)
    assert s["competitions"] == pytest.approx(
        {"n": 3, "n_below_1": 2, "n_above_1": 1, "median_ratio": 0.5, "min_ratio": 0.25, "max_ratio": 3.0})
    assert s["n_competitions"] == 3 and s["n_competitions_all"] == 4


def intervals():
    """Five intervals of one competition (offset at 100 s each, for readability)."""
    return pd.DataFrame({
        rba.COMP: "r1", "onset": [90.0, 97.0, 95.0, 90.0, 90.0], "offset": 100.0,
        "length": [10.0, 8.0, 80.0, 30.0, 20.0], "event": [1, 1, 1, 0, 1],
        # after the offset; in the tail; none; none; none counted (a chirp at 92 s is before offset - 5 s)
        "first_chirp": [103.0, 98.0, np.nan, np.nan, np.nan],
        "prev_interval": [5.0, 6.0, 7.0, np.nan, 9.0]})


def test_primary_intervals_are_those_with_a_previous_interval():
    iv = rba.primary_intervals(intervals())
    assert list(iv.index) == [0, 1, 2, 4]
    with pytest.raises(KeyError):
        rba.primary_intervals(intervals().drop(columns=["first_chirp"]))


def test_raster_rows_are_a_seeded_sample_sorted_by_length():
    iv = pd.concat([intervals()] * 10, ignore_index=True).assign(length=np.arange(50.0, 0.0, -1.0))
    a, b = rba.raster_rows(iv, 12, seed=3), rba.raster_rows(iv, 12, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 12 and a["length"].is_monotonic_increasing and a.index.is_unique
    assert not rba.raster_rows(iv, 12, seed=4).index.equals(a.index)
    with pytest.raises(ValueError):
        rba.raster_rows(iv, 51)


def test_raster_counts_each_kind_of_interval():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    counts = rba.raster_panel(ax, intervals(), {"r1": np.array([92.0, 98.0, 103.0, 130.0])}, left_edge=-5.0)
    plt.close(fig)
    # 0: switch at 3 s; 1: tail chirp (98) -> at the offset; 2: no counted chirp, 80 s (> 60: clipped);
    # 3: none, censored at 30 s; 4: the chirp at 92 lies before the offset - 5 s: not counted
    assert counts == {"n_rows": 5, "n_clipped": 1, "n_chirped_at_offset": 1, "n_chirped_after_offset": 1,
                      "n_no_chirp": 3, "n_censored": 1}


def test_loser_chirps_by_recording_sorted():
    t = pd.DataFrame({rba.COMP: ["a", "b", "a"], "time_s": [5.0, 1.0, 2.0]})
    out = rba.loser_chirps(t)
    assert out["a"].tolist() == [2.0, 5.0] and out["b"].tolist() == [1.0]


def test_model_hr_reads_the_hazard_stats(tmp_path):
    path = tmp_path / "reattack_hazard.json"
    path.write_text(json.dumps({"values": {"fits": {"primary": {"hr": {"median": 1.1, "hdi": [0.9, 1.3],
                                                                         "p_lt_1": 0.2}}}}}))
    assert rba.model_hr(path) == {"median": 1.1, "hdi": [0.9, 1.3]}
    with pytest.raises(ValueError):
        rba.model_hr(path, "unadjusted")
    with pytest.raises(FileNotFoundError):
        rba.model_hr(tmp_path / "missing.json")


def test_load_matched_needs_both_outputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        rba.load_matched(tmp_path)
    np.savez(tmp_path / "curves.npz", grid=np.arange(3.0))
    (tmp_path / "summary.json").write_text(json.dumps({"n_intervals": 3}))
    curves, summary = rba.load_matched(tmp_path)
    assert curves["grid"].tolist() == [0.0, 1.0, 2.0] and summary == {"n_intervals": 3}
