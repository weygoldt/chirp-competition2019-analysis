"""reattack_matched: chirp triggers, matched controls (no outcome conditioning), estimators and surrogates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reattack_hazard as rh
import reattack_matched as rm

WINDOW = (-5.0, 11.0)


def toy_intervals() -> pd.DataFrame:
    """One competition, five intervals (offset at 0 for readability; time since the offset = absolute time).

    0: first chirp at 3 s after the offset (a trigger at s = 3), re-attack at 10 s
    1: chirp in the chase tail (-2 s): chirped from the offset, no trigger; re-attack at 8 s
    2: no chirp; re-attack at 20 s
    3: first chirp at 6 s (a trigger at s = 6, and a control for s = 3, censored at 6), re-attack at 12 s
    4: no chirp; re-attack at 2 s (not at risk at s = 3)
    """
    return pd.DataFrame({
        rh.COMP: "r1", "onset": [-10.0, -10.0, -10.0, -10.0, -10.0], "offset": 0.0,
        "length": [10.0, 8.0, 20.0, 12.0, 2.0], "event": 1,
        "first_chirp": [3.0, -2.0, np.nan, 6.0, np.nan]})


def test_switch_times_tail_chirps_switch_at_the_offset_and_silence_never():
    sw = rm.switch_times(toy_intervals())
    assert sw.tolist() == [3.0, 0.0, np.inf, 6.0, np.inf]


def test_first_chirp_triggers_are_the_switches_after_the_offset_only():
    trg = rm.first_chirp_triggers(toy_intervals())
    assert trg["interval"].tolist() == [0, 3]
    assert trg["s"].tolist() == [3.0, 6.0]


def test_controls_are_at_risk_and_unchirped_at_s_and_censored_at_their_own_chirp():
    iv = toy_intervals()
    f = rm.matched_followups(iv, rm.first_chirp_triggers(iv))
    s3 = f[f["set"] == 0].set_index("interval")
    # trigger 0 at s = 3: controls 2 (silent) and 3 (chirps later, at 6: still eligible, censored there);
    # 1 is chirped since the offset, 4 was re-attacked at 2 s
    assert sorted(s3.index[s3["group"] == rm.CONTROL]) == [2, 3]
    assert s3.loc[0, "exit"] == 7.0 and s3.loc[0, "event"] == 1 and s3.loc[0, "weight"] == 1.0
    assert s3.loc[2, "exit"] == 17.0 and s3.loc[2, "event"] == 1
    assert s3.loc[3, "exit"] == 3.0 and s3.loc[3, "event"] == 0  # censored at its own chirp (6 - 3)
    assert s3.loc[3, "by_chirp"] == 1 and s3.loc[[0, 2], "by_chirp"].tolist() == [0, 0]
    assert s3.loc[[2, 3], "weight"].sum() == pytest.approx(1.0)
    # trigger 3 at s = 6: only interval 2 (0 is chirped since 3; 1 since 0; 4 ended)
    s6 = f[f["set"] == 1]
    assert s6.loc[s6["group"] == rm.CONTROL, "interval"].tolist() == [2]
    assert f.attrs["n_without_controls"] == 0


def test_the_never_chirped_trap_drops_later_chirpers_and_does_not_censor():
    iv = toy_intervals()
    f = rm.matched_followups(iv, rm.first_chirp_triggers(iv), rule="never_chirped")
    s3 = f[(f["set"] == 0) & (f["group"] == rm.CONTROL)]
    assert s3["interval"].tolist() == [2]


def test_controls_come_from_the_same_competition_and_triggers_without_controls_are_dropped():
    iv = toy_intervals()
    other = toy_intervals().assign(**{rh.COMP: "r2"})
    other.index = other.index + 100
    other.loc[102, "first_chirp"] = 15.0  # trigger at 15 s: no other interval of r2 is at risk any more
    both = pd.concat([iv, other])
    trg = rm.first_chirp_triggers(both)
    assert 102 in set(trg["interval"])
    f = rm.matched_followups(both, trg)
    for _, g in f.groupby("set"):
        assert g[rh.COMP].nunique() == 1
    assert f.attrs["n_without_controls"] == 1
    assert 102 not in set(f["interval"][f["group"] == rm.TRIGGER])
    # r1's interval 2 (silent, 20 s) is a control in r1 only; r2's 102 serves r2's triggers at 3 and 6 s
    assert set(f.loc[f["interval"] == 2, rh.COMP]) == {"r1"}
    assert set(f.loc[f["interval"] == 102, "group"]) == {rm.CONTROL}


def test_sampled_controls_are_reproducible_and_weighted():
    iv = synthetic(n_comp=2, n_int=60, seed=3)
    trg = rm.first_chirp_triggers(iv)
    a = rm.matched_followups(iv, trg, n_controls=3, seed=7)
    b = rm.matched_followups(iv, trg, n_controls=3, seed=7)
    pd.testing.assert_frame_equal(a, b)
    n = a[a["group"] == rm.CONTROL].groupby("set").size()
    assert (n <= 3).all() and (n == 3).mean() > 0.5
    w = a[a["group"] == rm.CONTROL].groupby("set")["weight"].sum()
    assert np.allclose(w, 1.0)


def test_all_chirp_triggers_are_the_chirps_strictly_between_offset_and_next_onset():
    iv = toy_intervals().iloc[:2]
    trg = rm.all_chirp_triggers(iv, {"r1": np.array([-2.0, 0.0, 3.0, 5.0, 9.0, 12.0])})
    # interval 0 (length 10): 3, 5, 9; interval 1 (length 8): 3, 5
    assert trg["interval"].tolist() == [0, 0, 0, 1, 1]
    assert trg["s"].tolist() == [3.0, 5.0, 9.0, 3.0, 5.0]


def test_reflected_kernel_keeps_the_mass_of_events_near_zero():
    grid = np.arange(0, 30.0001, 0.01)
    k = rm.reflected_kernel(grid, np.array([0.05, 10.0]), np.array([1.0, 2.0]))
    assert np.trapezoid(k, grid) == pytest.approx(3.0, rel=1e-3)


# ---------------------------------------------------------------------------
# estimators on synthetic data
# ---------------------------------------------------------------------------
def synthetic(n_comp=4, n_int=400, rate=1 / 20, chirp_rate=1 / 15, hr=1.0, seed=0) -> pd.DataFrame:
    """Intervals with a constant re-attack hazard ``rate`` (per s), multiplied by ``hr`` from the loser's
    first chirp on; first chirps from a Poisson process (``chirp_rate``) starting at the offset."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_comp):
        for i in range(n_int):
            fc = rng.exponential(1 / chirp_rate)
            t0 = rng.exponential(1 / rate)
            if t0 < fc:
                L, first = t0, np.nan
            else:
                L, first = fc + rng.exponential(1 / (rate * hr)), fc
            rows.append({rh.COMP: f"c{c}", "onset": -5.0, "offset": 0.0, "length": L, "event": 1,
                         "first_chirp": first})
    return pd.DataFrame(rows)


def test_kaplan_meier_matches_the_unweighted_product_limit_and_exp_survival():
    iv = synthetic(n_comp=3, n_int=600, seed=1)
    f = rm.matched_followups(iv, rm.first_chirp_triggers(iv))
    sm = rm.summarise(f)
    S = rm.survival(sm, at=np.array([10.0, 20.0]))
    # triggers: unweighted, uncensored -> empirical fraction of exits beyond t
    tr = f[f["group"] == rm.TRIGGER]
    assert S[rm.TRIGGER] == pytest.approx([(tr["exit"] > 10).mean(), (tr["exit"] > 20).mean()])
    # constant hazard 1/20: both close to exp(-t/20)
    for g in rm.GROUPS:
        assert S[g] == pytest.approx(np.exp(-np.array([10.0, 20.0]) / 20), abs=0.06)


def test_window_rates_are_events_over_time_at_risk():
    iv = toy_intervals()
    f = rm.matched_followups(iv, rm.first_chirp_triggers(iv))
    w = rm.window_rates(rm.summarise(f, windows=(5.0,)))["5"]
    # triggers: set 0 exit 7 (no event in 5 s), set 1 exit 6 (none) -> 0 events, 10 s at risk
    assert w["events"][rm.TRIGGER] == 0 and w["exposure_s"][rm.TRIGGER] == pytest.approx(10.0)
    # controls: set 0: 2 (exit 17, w .5) and 3 (exit 3, censored, w .5); set 1: 2 (exit 14, w 1)
    assert w["events"][rm.CONTROL] == 0
    assert w["exposure_s"][rm.CONTROL] == pytest.approx(0.5 * 5 + 0.5 * 3 + 5)


def test_kernel_hazard_recovers_a_constant_rate():
    iv = synthetic(n_comp=4, n_int=1500, rate=1 / 20, seed=2)
    sm = rm.summarise(rm.matched_followups(iv, rm.first_chirp_triggers(iv)), sigma=2.0)
    h = rm.hazard_per_min(sm)
    inner = (sm.grid > 3) & (sm.grid < 20)
    for g in rm.GROUPS:
        assert np.nanmedian(h[g][inner]) == pytest.approx(3.0, rel=0.15)  # 1/20 per s = 3 per min


@pytest.mark.parametrize("hr", [1.0, 0.5, 2.0])
def test_matched_ratio_recovers_the_planted_hazard_ratio(hr):
    iv = synthetic(n_comp=4, n_int=1500, hr=hr, seed=4)
    w = rm.window_rates(rm.summarise(rm.matched_followups(iv, rm.first_chirp_triggers(iv))))
    assert w["30"]["ratio"] == pytest.approx(hr, rel=0.15)


def test_the_never_chirped_trap_is_biased_under_no_effect():
    # chirping and re-attack independent: the trap keeps short intervals as controls -> ratio well below 1
    iv = synthetic(n_comp=4, n_int=1500, hr=1.0, chirp_rate=1 / 8, seed=5)
    trg = rm.first_chirp_triggers(iv)
    good = rm.window_rates(rm.summarise(rm.matched_followups(iv, trg)))["30"]["ratio"]
    trap = rm.window_rates(rm.summarise(rm.matched_followups(iv, trg, rule="never_chirped")))["30"]["ratio"]
    assert good == pytest.approx(1.0, rel=0.12)
    assert trap < 0.85 * good


def test_resample_weights():
    b = rm.resample_weights(15, 50, seed=1)
    assert b.shape == (50, 15) and (b.sum(axis=1) == 15).all()
    lo = rm.resample_weights(15, 50, seed=1, mode="leave_out")
    assert ((lo == 0).sum(axis=1) == 1).all()
    with pytest.raises(ValueError):
        rm.resample_weights(15, 5, mode="jackknife")


def test_resampled_with_unit_weights_reproduces_the_point_estimates():
    iv = synthetic(n_comp=3, n_int=200, seed=6)
    sm = rm.summarise(rm.matched_followups(iv, rm.first_chirp_triggers(iv)))
    r = rm.resampled(sm, np.ones((2, 3)))
    w = rm.window_rates(sm)
    assert r["ratio"][0] == pytest.approx([w["10"]["ratio"], w["30"]["ratio"]])
    assert np.allclose(r["survival"][rm.CONTROL][1], rm.survival(sm)[rm.CONTROL])


# ---------------------------------------------------------------------------
# surrogates
# ---------------------------------------------------------------------------
def test_shuffle_keeps_lengths_and_events_and_counts_chirps_in_the_receiving_range():
    iv = toy_intervals()
    chirps = {"r1": np.array([-2.0, 3.0, 6.0])}  # every interval offset at 0: same train for all
    out = rm.shuffle_first_chirps(iv, chirps, np.random.default_rng(0), left=WINDOW[0])
    assert out["length"].tolist() == iv["length"].tolist() and out["event"].tolist() == iv["event"].tolist()
    # interval 4 (2 s long) can only keep -2; all trains start with -2 -> every first chirp is -2
    assert np.allclose(out["first_chirp"], -2.0)


def test_shuffle_ignores_chirps_before_the_receiving_chase_onset():
    iv = toy_intervals().assign(onset=-1.0, length=10.0)  # chases 1 s long: their tail starts at -1
    out = rm.shuffle_first_chirps(iv, {"r1": np.array([-2.0, 3.0])}, np.random.default_rng(0))
    assert np.allclose(out["first_chirp"], 3.0)  # -2 lies before every receiving chase's onset


@pytest.mark.parametrize("hr", [1.0, 0.5])
def test_simulated_reattacks_carry_the_planted_ratio(hr):
    base = synthetic(n_comp=4, n_int=1500, hr=1.0, seed=8)
    sim = rm.simulate_reattacks(base, hr, np.random.default_rng(9))
    assert (sim["event"] == 1).all() and (sim["length"] > 0).all()
    rel = sim["first_chirp"] - sim["offset"]
    assert ((rel < sim["length"]) | rel.isna()).all()
    w = rm.window_rates(rm.summarise(rm.matched_followups(sim, rm.first_chirp_triggers(sim))))
    assert w["30"]["ratio"] == pytest.approx(hr, rel=0.2)


def test_simulation_keeps_censored_intervals_censored_at_their_real_length():
    base = synthetic(n_comp=1, n_int=50, seed=10)
    base.loc[0, ["length", "event"]] = [1e6, 0]
    sim = rm.simulate_reattacks(base, 1.0, np.random.default_rng(0))
    assert sim.loc[0, "event"] == 1 and sim.loc[0, "length"] < 1e6  # baseline hazard > 0: an onset first
    base.loc[0, "length"] = 0.01
    sim = rm.simulate_reattacks(base, 1.0, np.random.default_rng(0))
    assert sim.loc[0, "event"] == 0 and sim.loc[0, "length"] == 0.01


def test_invalid_inputs_raise():
    iv = toy_intervals()
    with pytest.raises(ValueError):
        rm.matched_followups(iv, rm.first_chirp_triggers(iv), rule="no_chirp_ever")
    bad = iv.copy()
    bad.loc[0, "first_chirp"] = 50.0  # after the interval's end
    with pytest.raises(ValueError):
        rm.first_chirp_triggers(bad)
    with pytest.raises(KeyError):
        rm.first_chirp_triggers(iv.drop(columns="first_chirp"))
