"""reattack_hazard: intervals, person-period rows (exposure, events, covariate switches, censoring), rates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reattack_hazard as rh

WINDOW = (-5.0, 11.0)
# Chases A (10-20), B (30-32, shorter than 5 s), C (50-60); window ends at 100 s.
CHASES = [(10.0, 20.0), (30.0, 32.0), (50.0, 60.0)]
# 12: in A but before A's offset - 5 (does not count); 25: after A's offset (switches interval 0 on at 5 s);
# 28: before B's onset (does not count for B: its tail starts at the onset); 31.5: in B's tail (interval 1 on
# from its offset); 75: in interval 2, after C's offset (switches it on at 15 s).
CHIRPS = [12.0, 25.0, 28.0, 31.5, 75.0]


def comp(chirps=CHIRPS, chases=CHASES, t_max=100.0, straddling=np.nan, rec="r1"):
    return rh.Competition(rec, np.array(chases), np.array(chirps), t_max, straddling)


def tables(c=None, covariates=rh.TIME_VARYING):
    c = comp() if c is None else c
    iv = rh.intervals_table([c], WINDOW)
    return iv, rh.person_period(iv, {c.recording: c.loser_chirps}, covariates=covariates)


def test_intervals_run_from_each_offset_to_the_next_onset_and_the_last_is_censored():
    iv, _ = tables()
    assert iv["offset"].tolist() == [20.0, 32.0, 60.0]
    assert iv["end"].tolist() == [30.0, 50.0, 100.0]
    assert iv["event"].tolist() == [1, 1, 0]
    assert iv["length"].tolist() == [10.0, 18.0, 40.0]
    assert iv["duration"].tolist() == [10.0, 2.0, 10.0]
    assert np.isnan(iv.loc[0, "prev_interval"]) and iv["prev_interval"].tolist()[1:] == [10.0, 18.0]
    assert iv["first_chirp"].tolist() == [25.0, 31.5, 75.0]


def test_window_predictors_of_this_and_the_next_chase():
    iv, _ = tables()
    # A: [15, min(31, 30)] holds 25 and 28; B: [30, 43] holds 31.5; C: [55, 71] holds none (75 is later)
    assert iv["chirped_window"].tolist() == [1.0, 1.0, 0.0]
    assert iv["next_chirped"].tolist()[:2] == [1.0, 0.0] and np.isnan(iv.loc[2, "next_chirped"])


def test_exposure_adds_up_to_each_interval_and_events_to_intervals_with_a_next_chase():
    iv, pp = tables()
    per = pp.groupby("interval")["exposure"].sum()
    assert per.to_numpy() == pytest.approx(iv["length"].to_numpy())
    assert (pp["t1"] - pp["t0"]).to_numpy() == pytest.approx(pp["exposure"].to_numpy())
    assert pp.groupby("interval")["event"].sum().tolist() == [1, 1, 0]
    # the event sits in the last row of its interval, and rows tile [0, length) without gaps
    for i, g in pp.groupby("interval"):
        assert g["event"].iloc[:-1].sum() == 0
        assert g["t0"].iloc[0] == 0 and g["t1"].iloc[-1] == pytest.approx(iv.loc[i, "length"])
        assert np.allclose(g["t0"].iloc[1:].to_numpy(), g["t1"].iloc[:-1].to_numpy())


def test_rows_split_at_the_pieces_and_carry_their_piece():
    _, pp = tables(covariates=())
    g = pp[pp["interval"] == 2]  # 40 s: pieces 0-2, 2-5, 5-10, 10-20, 20-40
    assert g["t0"].tolist() == [0, 2, 5, 10, 20] and g["t1"].tolist() == [2, 5, 10, 20, 40]
    assert g["piece"].tolist() == [0, 1, 2, 3, 4]
    assert rh.piece_labels()[:3] == ["0-2", "2-5", "5-10"] and rh.piece_labels()[-1] == "500+"


def test_chirped_switches_on_at_the_first_chirp_from_the_chase_tail_on():
    _, pp = tables(covariates=("chirped",))
    for i, switch in ((0, 5.0), (1, 0.0), (2, 15.0)):  # 25 - 20; 31.5 is before B's offset; 75 - 60
        g = pp[pp["interval"] == i]
        assert switch in g["t0"].tolist()
        assert (g["chirped"] == (g["t0"] >= switch).astype(int)).all()


def test_chirps_before_the_tail_or_the_onset_do_not_count():
    # only the chirp at 12 (A, but 8 s before its offset) and 28 (before B's onset): never chirped
    iv, pp = tables(comp(chirps=[12.0, 28.0]), covariates=("chirped",))
    assert iv.loc[0, "first_chirp"] == 28.0 and iv["first_chirp"].iloc[1:].isna().all()
    # 28 lies in interval 0 itself (after A's offset): it switches interval 0 on at 8 s ...
    g0 = pp[pp["interval"] == 0]
    assert (g0["chirped"] == (g0["t0"] >= 8.0).astype(int)).all()
    # ... but not interval 1 (B's tail starts at its onset, 30 s) nor interval 2
    assert pp.loc[pp["interval"] > 0, "chirped"].sum() == 0


def test_a_chirp_after_the_next_onset_does_not_switch_the_interval():
    iv, pp = tables(comp(chirps=[40.0, 55.0]), covariates=("chirped",))
    # 40 switches interval 1 at 8 s; 55 is inside chase C, so interval 0 stays off; C's tail starts at 55
    assert pp.loc[pp["interval"] == 0, "chirped"].sum() == 0
    assert iv.loc[2, "first_chirp"] == 55.0 and (pp.loc[pp["interval"] == 2, "chirped"] == 1).all()


def test_recency_and_guard_indicators():
    _, pp = tables()
    g0 = pp[pp["interval"] == 0]  # chirps 25 and 28: 1-s windows [5, 6) and [8, 9); guard [5.5, 6.5), [8.5, 9.5)
    for col, on in (("recent_1s", [(5, 6), (8, 9)]), ("guard", [(5.5, 6.5), (8.5, 9.5)]),
                    ("recent_5s", [(5, 13)])):
        expect = np.zeros(len(g0), int)
        for a, b in on:
            expect |= ((g0["t0"] >= a) & (g0["t1"] <= b)).to_numpy().astype(int)
        assert g0[col].tolist() == expect.tolist(), col
    g1 = pp[pp["interval"] == 1]  # 31.5 (in chase B): recent_1s on for [0, 0.5), guard for [0, 1)
    assert g1.loc[g1["recent_1s"] == 1, "exposure"].sum() == pytest.approx(0.5)
    assert g1.loc[g1["guard"] == 1, "exposure"].sum() == pytest.approx(1.0)
    assert g1.loc[g1["recent_5s"] == 1, "exposure"].sum() == pytest.approx(4.5)


def test_censoring_and_a_chase_that_straddles_the_window_end():
    iv, pp = tables(comp(straddling=95.0))
    assert iv["end"].iloc[-1] == 95.0 and iv["event"].iloc[-1] == 1
    assert pp["event"].sum() == 3 and pp["exposure"].sum() == pytest.approx(10 + 18 + 35)
    iv, pp = tables(comp())
    assert iv["end"].iloc[-1] == 100.0 and pp.loc[pp["interval"] == 2, "event"].sum() == 0
    with pytest.raises(ValueError):
        comp(straddling=55.0)  # before the last offset (60 s)
    with pytest.raises(ValueError):
        comp(chases=[(10.0, 20.0), (15.0, 25.0)])  # overlapping chases must be merged first
    with pytest.raises(ValueError):
        comp(chirps=[100.0])  # chirps at or after the window end are cut by the loader


def test_aggregation_keeps_events_and_exposure():
    iv, pp = tables()
    pp = rh.add_interval_covariates(pp, iv)
    agg = rh.aggregate(pp, ["recording", "piece", "chirped"])
    assert agg["event"].sum() == pp["event"].sum()
    assert agg["exposure"].sum() == pytest.approx(pp["exposure"].sum())
    assert len(agg) < len(pp)
    assert np.isnan(pp.loc[pp["interval"] == 0, "log_prev"]).all()
    assert pp.loc[pp["interval"] == 1, "log_dur"].iloc[0] == pytest.approx(np.log(2.0))


def test_check_against_obs():
    iv, _ = tables()
    obs = pd.DataFrame({"recording": iv["recording"], "chasing_duration": iv["duration"],
                        "time_since_previous_chasing": iv["prev_interval"],
                        "time_to_next_chasing": [10.0, 18.0, np.nan],
                        "loser_chirped_offset_window": iv["chirped_window"].astype(bool)})
    assert rh.check_against_obs(iv, obs) == {"n_rows": 3, "n_recordings": 1}
    bad = obs.copy()
    bad.loc[1, "time_to_next_chasing"] = 18.5
    with pytest.raises(ValueError):
        rh.check_against_obs(iv, bad)
    with pytest.raises(ValueError):
        rh.check_against_obs(iv, obs.iloc[:2])


def test_mantel_haenszel_rates_and_competition_ratios():
    # two competitions, one piece; c1: rate 0.2 with, 0.1 without; c2: 0.02 with, 0.01 without; c3: no
    # exposure without the covariate (weight 0)
    rows = [("c1", 0, 1, 20, 100.0), ("c1", 0, 0, 10, 100.0), ("c2", 0, 1, 2, 100.0), ("c2", 0, 0, 3, 300.0),
            ("c3", 0, 1, 5, 50.0)]
    pp = pd.DataFrame(rows, columns=["recording", "piece", "chirped", "event", "exposure"])
    cells = rh.cell_table(pp)
    r = rh.mh_rates(cells).loc[0]
    w1, w2 = 100 * 100 / 200, 100 * 300 / 400
    assert r["rate1"] == pytest.approx((w1 * 0.2 + w2 * 0.02) / (w1 + w2))
    assert r["rate0"] == pytest.approx((w1 * 0.1 + w2 * 0.01) / (w1 + w2))
    assert r["ratio"] == pytest.approx(2.0) and r["n_comp"] == 2 and r["E1_all"] == 27
    cr = rh.competition_rate_ratios(cells)
    assert cr.loc["c1", "ratio"] == pytest.approx(2.0) and cr.loc["c2", "ratio"] == pytest.approx(2.0)
    assert np.isnan(cr.loc["c3", "ratio"])
