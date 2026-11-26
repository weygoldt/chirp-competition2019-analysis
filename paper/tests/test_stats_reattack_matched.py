"""stats_reattack_matched: the numbers of a design and of the surrogates (write_stats-ready), the trigger table,
the check against the hazard fits' intervals, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import manuscript_stats
import reattack_hazard as rh
import reattack_matched as rm
import stats_reattack_matched as srm
from test_reattack_matched import synthetic, toy_intervals


def test_design_numbers_are_complete_and_valid_stats():
    iv = synthetic(n_comp=3, n_int=150, seed=11)
    res = srm.run_design(iv, rm.first_chirp_triggers(iv), "unchirped_at_s", sigma=1.0, n_boot=20, seed=1)
    d = srm.design_numbers(res)
    manuscript_stats._clean(d, "design")  # plain numbers and word keys only
    assert d["n_triggers"] + d["n_dropped_no_match"] == d["n_candidates"]
    assert d["n_competitions"] == 3
    assert {f"w{T:g}" for T in srm.WINDOWS_S} <= set(d)
    assert set(d["survival"]) == {f"t{T:g}" for T in srm.SURVIVAL_AT_S}
    t30 = d["survival"]["t30"]
    assert t30["if_hr_0p5"] == pytest.approx(t30["matched"] ** 0.5)
    lo, hi = d["w30"]["ratio_ci"]
    assert lo <= hi and d["matched_per_trigger"]["min"] >= 1
    # matched weight stopped by its own chirp: a share
    assert 0 <= d["frac_matched_weight_stopped_by_chirp_30s"] <= 1
    assert d["n_matched_stopped_by_chirp_30s"] <= d["n_matched_stopped_by_chirp"]


def test_the_trap_never_censors_at_a_chirp():
    iv = synthetic(n_comp=2, n_int=150, seed=12)
    res = srm.run_design(iv, rm.first_chirp_triggers(iv), "never_chirped", sigma=1.0, n_boot=5, seed=1)
    assert srm.design_numbers(res)["n_matched_stopped_by_chirp"] == 0


def test_triggers_table_marks_dropped_triggers_with_zero_matches():
    iv = toy_intervals()
    other = toy_intervals().assign(**{rh.COMP: "r2"})
    other.index = other.index + 100
    other.loc[102, "first_chirp"] = 15.0  # no other interval of r2 at risk at 15 s
    both = pd.concat([iv, other])
    res = {"triggers": rm.first_chirp_triggers(both)}
    res["followups"] = rm.matched_followups(both, res["triggers"])
    t = srm.triggers_table(res)
    assert t.loc[t["interval"] == 102, "n_matched"].tolist() == [0]
    assert (t.loc[t["interval"] != 102, "n_matched"] > 0).all()


def test_surrogate_numbers_summarise_and_refuse_non_finite():
    sur = pd.DataFrame({"scenario": ["a"] * 4, "planted": 1.0, "rep": range(4), "n_triggers": [5, 6, 7, 8],
                        **{c: [0.9, 1.0, 1.1, 1.2] for c in ("design_w10", "design_w30", "trap_w10", "trap_w30",
                                                              "mh")}})
    out = srm.surrogate_numbers(sur)
    manuscript_stats._clean(out, "surrogates")
    assert out["a"]["design_w30"]["median"] == pytest.approx(1.05) and out["a"]["n_replicates"] == 4
    sur.loc[0, "trap_w30"] = np.nan
    with pytest.raises(ValueError):
        srm.surrogate_numbers(sur)


def test_the_stated_reference_is_the_one_drawn():
    import plot_receiver_before_after as rba

    assert srm.APPEASE_HR == rba.APPEASE_HR


def test_scenario_names_are_stat_keys():
    for name in srm.SCENARIOS:
        manuscript_stats._clean({name: 1.0}, "scenarios")


def test_check_against_saved_refuses_other_intervals():
    iv = toy_intervals().assign(end=lambda d: d["offset"] + d["length"])
    srm.check_against_saved(iv, iv.copy())
    moved = iv.copy()
    moved.loc[0, "first_chirp"] = 4.0
    with pytest.raises(ValueError):
        srm.check_against_saved(iv, moved)
    with pytest.raises(ValueError):
        srm.check_against_saved(iv, iv.iloc[:-1])
