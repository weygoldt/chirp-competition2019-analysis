"""Tests for the offset-locked chirps of hz_sim and for hz_acausal_sim.

Run: .venv/bin/python src_hazard/test_hz_acausal_sim.py (pytest is not installed in
this venv; the file also works under pytest). No real data needed.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hz_acausal_sim as AS  # noqa: E402
import hz_sim as S  # noqa: E402
from hz_data import Trial  # noqa: E402
from test_hz_sim import BW, _cfg, _templates  # noqa: E402

EDGES = np.array([0.0, 0.25, 0.5, 1.0, 5.0])


def _params() -> S.SimParams:
    return S.estimate_params(_templates(), _cfg())


def _sim(P: S.SimParams, seed: int = 5) -> list[Trial]:
    return S._simulate(P, BW, np.random.SeedSequence(seed), "t")


def _extra(a: list[Trial], b: list[Trial]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per trial: chirps in b that are not in a (b = a + offset-locked chirps), true offsets."""
    out = []
    for x, y in zip(a, b):
        assert np.array_equal(x.chases, y.chases)  # separate stream: chases unchanged (beta 0)
        assert np.all(np.isin(x.chirps_sub, y.chirps_sub))
        out.append((y.chirps_sub[~np.isin(y.chirps_sub, x.chirps_sub)], y.chases[:, 1]))
    return out


def test_defaults_and_zero_sizes_leave_generator_unchanged():
    P = _params()
    a = _sim(P)
    b = _sim(replace(P, resp_edges=EDGES, resp_density=np.array([0, 0, 1.0, 0]),
                     sym_edges=EDGES, sym_density=np.array([0, 1.0, 0, 0])))
    for x, y in zip(a, b):
        assert np.array_equal(x.chases, y.chases)
        assert np.array_equal(x.chirps_sub, y.chirps_sub)
        assert np.array_equal(x.chirps_dom, y.chirps_dom)


def test_response_burst_lands_at_the_given_latencies_and_rate():
    P = _params()
    Pr = replace(P, resp_edges=EDGES, resp_density=np.array([0, 0, 1.0, 0]),
                 trials=[replace(tp, resp_n=2.0) for tp in P.trials])
    n_x = n_off = 0
    for x, off in _extra(_sim(P), _sim(Pr)):
        L = x[:, None] - off[None, :]
        assert np.all(np.any((L >= 0.5 - 1e-9) & (L < 1.0), axis=1))  # 0.5-1 s after an offset
        n_x += len(x)
        n_off += int(np.sum(off < x.max() + 1)) if len(x) else 0
    # Poisson(2) per offset (offsets near the window end lose theirs): within 5 SD
    assert abs(n_x - 2 * n_off) < 5 * np.sqrt(2 * n_off) + 0.05 * n_off


def test_symmetric_cluster_is_symmetric_around_true_offsets():
    P = _params()
    Ps = replace(P, sym_edges=EDGES, sym_density=np.array([0, 1.0, 0, 0]), sym_m=1.0,
                 trials=[replace(tp, sym_n=0.8) for tp in P.trials])
    before = after = 0
    for x, off in _extra(_sim(P, 7), _sim(Ps, 7)):
        L = x[:, None] - off[None, :]
        near = (np.abs(L) >= 0.25 - 1e-9) & (np.abs(L) < 0.5)
        assert np.all(near.any(axis=1))
        before += int(np.sum((near & (L < 0)).any(axis=1)))
        after += int(np.sum((near & (L > 0)).any(axis=1)))
    assert before > 50 and abs(before - after) < 4 * np.sqrt(before + after)


def test_bout_multiplier_raises_response_after_offsets_preceded_by_a_chirp():
    P = _params()
    base = dict(resp_edges=EDGES, resp_density=np.array([0, 0, 1.0, 0]))
    trials = [replace(tp, resp_n=0.5) for tp in P.trials]
    rate = {}
    for mult in (1.0, 6.0):
        Pb = replace(P, **base, resp_bout_mult=mult, trials=trials)
        num = {True: 0, False: 0}
        den = {True: 0, False: 0}
        for (x, off), tr in zip(_extra(_sim(P, 9), _sim(Pb, 9)), _sim(P, 9)):
            for o in off[off < tr.obs_end - 2]:
                pre = bool(np.any((tr.chirps_sub >= o - 1.0) & (tr.chirps_sub < o)))
                num[pre] += int(np.sum((x >= o + 0.5) & (x < o + 1.0)))
                den[pre] += 1
        rate[mult] = (num[True] / max(den[True], 1)) / max(num[False] / max(den[False], 1), 1e-9)
    assert rate[6.0] > 3 * max(rate[1.0], 1e-9) and rate[6.0] > 3


def test_window_stats_counts_like_hz_acausal():
    tr = Trial(recording="w", group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
               t_start=0.0, t_end=100.0, chases=np.array([[10.0, 20.0], [30.0, 40.0]]),
               chirps_sub=np.array([19.5, 18.5, 20.4, 21.5, 39.2, 45.0, 29.9]))
    w = AS.window_stats(tr)
    # chase 1: pre 19.5 (0.5 s), 18.5 (1.5 s); post 20.4 (0.4 s), 21.5 (1.5 s)
    # chase 2: pre 39.2 (0.8 s); post 45.0 (5 s: outside); 29.9 is before chase 2's onset
    assert (w["pre_1s"], w["pre_2s"], w["post_1s"], w["post_2s"]) == (2, 3, 1, 2)
    assert w["both_1s"] == 1 and w["off_pre_1s"] == 2 and w["n_off_win"] == 2
    assert np.isclose(w["both_1s_indep"], 2 * 1 / 2)


def test_window_stats_neighbour_and_far_pairings():
    ch = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
    # pre (last 1 s of the chase) at chases 0, 1, 2; post (first 1 s after) at chases 0, 1, 3;
    # chase 1 also has a chirp 2-3 s after its offset (next onset 10 s later: available)
    x = [19.5, 20.5, 39.5, 40.5, 42.5, 59.5, 80.5]
    tr = Trial(recording="w", group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
               t_start=0.0, t_end=100.0, chases=ch, chirps_sub=np.array(x))
    w = AS.window_stats(tr)
    pre, post = np.array([1, 1, 1, 0], bool), np.array([1, 1, 0, 1], bool)
    assert w["both_1s"] == 2 and np.isclose(w["both_1s_indep"], 3 * 3 / 4)
    # neighbours: (pre i, post i+1) for i = 0..2 and (pre i, post i-1) for i = 1..3
    O = E = 0.0
    for a, b in ((pre[:-1], post[1:]), (pre[1:], post[:-1])):
        O += np.sum(a & b)
        E += a.sum() * b.sum() / len(a)
    assert w["nb_both_1s"] == O and np.isclose(w["nb_both_1s_indep"], E)
    far = np.array([0, 1, 0, 0], bool)
    assert w["far_both"] == 1 and np.isclose(w["far_indep"], 3 * 1 / 4)
    assert np.all(far <= pre)


def test_bout_rate_ratio_neighbour_shift():
    ch = np.array([[10.0 + 20 * i, 15.0 + 20 * i] for i in range(6)])
    # post-offset chirps only after chases preceded by a pre-offset chirp
    x = []
    for i in (0, 2, 4):
        x += [14.5 + 20 * i, 15.5 + 20 * i]
    tr = Trial(recording="b", group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
               t_start=0.0, t_end=200.0, chases=ch, chirps_sub=np.array(x))
    x2 = x + [15.7, 55.7, 95.7, 35.5]  # one post chirp after an offset without a pre chirp
    tr2 = replace(tr, chirps_sub=np.sort(np.array(x2)))
    assert AS.bout_rate_ratio([tr2]) > 1.0
    # neighbouring offsets: the post chirps after offset i+1 never follow a pre chirp at i
    assert AS.bout_rate_ratio([tr2], shifts=(1,)) < 1.0


def test_burst_cross_keeps_the_generator_and_adds_crossing_offspring():
    P = _params()
    P = replace(P, trials=[replace(tp, sub_q=0.6) for tp in P.trials])
    a = _sim(P, 13)
    b = _sim(replace(P, burst_cross=True), 13)
    n_extra = 0
    for x, y in zip(a, b):
        assert np.array_equal(x.chases, y.chases)  # no effect: chases unchanged
        assert np.all(np.isin(x.chirps_sub, y.chirps_sub))  # same draws, more kept
        extra = y.chirps_sub[~np.isin(y.chirps_sub, x.chirps_sub)]
        n_extra += len(extra)
        # every kept extra chirp lies shortly after a chase boundary (burst continuation)
        bnd = y.chases.ravel()
        lag = extra[:, None] - bnd[None, :]
        assert np.all(np.any((lag >= 0) & (lag < 20.0), axis=1))
    assert n_extra > 0


def test_slow_modulation_keeps_chases_and_overdisperses_chirping():
    P = _params()
    Pm = replace(P, sub_mod_frac=0.25, sub_mod_dur=120.0, sub_mod_mult=8.0)
    d = {}
    for lab, PP in (("flat", P), ("mod", Pm)):
        v = []
        for x in _sim(PP, 17):
            c = np.histogram(x.chirps_sub, np.arange(0, x.obs_end + 1e-9, 60.0))[0]
            v.append(c.var() / max(c.mean(), 1e-9))
        d[lab] = float(np.mean(v))
    a, b = _sim(P, 17), _sim(Pm, 17)
    for x, y in zip(a, b):
        assert np.array_equal(x.chases, y.chases)
    na, nb = sum(len(x.chirps_sub) for x in a), sum(len(y.chirps_sub) for y in b)
    assert abs(nb - na) < 0.3 * na  # renormalised: the mean rate is unchanged
    assert d["mod"] > 2 * d["flat"]


def _row(sc, stat, med, lo, hi, obs, frac, side, n=30, **kw):
    k = int(round((1 - frac) * n))  # datasets the observed value lies beyond
    return dict(scenario=sc, stat=stat, median=med, q025=lo, q975=hi, observed=obs, n=n,
                obs_inside=lo <= obs <= hi, exceed_frac=frac, exceed_side=side, exceed_n=k, **kw)


def test_exceedance_and_status_bands():
    x = np.arange(1.0, 31.0)  # 30 datasets: 1, 2, ..., 30
    e = AS.exceedance(x, 30.5)  # above all
    assert (e["exceed_side"], e["exceed_n"], e["exceed_frac"]) == ("above", 30, 0.0)
    assert AS.status_of(e["exceed_frac"]) == "inconsistent"
    e = AS.exceedance(x, 29.5)  # above 29/30: 1 of 30 at or beyond (3.3%)
    assert (e["exceed_side"], e["exceed_n"]) == ("above", 29)
    assert np.isclose(e["exceed_frac"], 1 / 30) and AS.status_of(e["exceed_frac"]) == "borderline"
    e = AS.exceedance(x, 28.5)  # 2 of 30 at or beyond (6.7%)
    assert AS.status_of(e["exceed_frac"]) == "borderline"
    e = AS.exceedance(x, 28.0)  # tie: 3 of 30 at or above (28, 29, 30) = 10%
    assert (e["n_below_obs"], e["n_above_obs"]) == (27, 2) and np.isclose(e["exceed_frac"], 0.1)
    assert AS.status_of(e["exceed_frac"]) == "consistent"  # 10% is outside the borderline band
    e = AS.exceedance(x, 0.5)  # below all
    assert (e["exceed_side"], e["exceed_n"], e["exceed_frac"]) == ("below", 30, 0.0)
    e = AS.exceedance(x, 15.5)  # middle: 15 on either side
    assert np.isclose(e["exceed_frac"], 0.5) and AS.status_of(e["exceed_frac"]) == "consistent"
    y = np.arange(1.0, 101.0)  # 100 datasets
    assert AS.status_of(AS.exceedance(y, 98.5)["exceed_frac"]) == "inconsistent"  # 2%
    assert AS.status_of(AS.exceedance(y, 97.5)["exceed_frac"]) == "borderline"  # 3%
    assert AS.status_of(AS.exceedance(y, 90.5)["exceed_frac"]) == "consistent"  # 10%
    assert AS.status_of(AS.exceedance(y, np.nan)["exceed_frac"]) == ""
    # z criterion: two-sided p
    assert [AS.status_of(p=p) for p in (0.01, 0.0499, 0.05, 0.19, 0.2, 0.7)] == [
        "inconsistent", "inconsistent", "borderline", "borderline", "consistent", "consistent"]
    assert AS._exc(pd.Series(dict(exceed_side="above", exceed_n=29, n=30))) == "above 29/30"


def test_summarize_verdicts_and_borderline_dependence():
    """End to end from estimates: a statistic whose only compatible account is borderline is
    'causal' when borderline counts as compatible and 'above' when it does not."""
    rows = []
    for sc, grd in (("c", np.linspace(1.0, 1.29, 30)), ("n", np.linspace(0.9, 1.19, 30))):
        for rep in range(30):
            rows += [dict(scenario=sc, rep=rep, stat="hr_primary", value=1.5 + rep / 29),
                     dict(scenario=sc, rep=rep, stat="ratio_pf", value=0.8 + 0.6 * rep / 29),
                     dict(scenario=sc, rep=rep, stat="ratio_grd", value=grd[rep])]
    rows += [dict(scenario="observed", rep=0, stat="hr_primary", value=2.0),
             dict(scenario="observed", rep=0, stat="ratio_pf", value=1.1),
             dict(scenario="observed", rep=0, stat="ratio_grd", value=1.285)]
    summ = AS.summarize(pd.DataFrame(rows))
    r = summ[(summ["scenario"] == "c") & (summ["stat"] == "ratio_grd")].iloc[0]
    assert (r["exceed_side"], r["exceed_n"], r["status"], r["criterion"]) == (
        "above", 29, "borderline", "range")
    r = summ[(summ["scenario"] == "n") & (summ["stat"] == "ratio_grd")].iloc[0]
    assert (r["exceed_n"], r["status"]) == (30, "inconsistent")
    scen = pd.DataFrame(dict(scenario=["c", "n"], short=["c", "n"], causal=[True, False]))
    fit, fit_bl, L, v = AS._verdicts(summ, scen, ("hr_primary",),
                                     (("ratio_pf", "pf"), ("ratio_grd", "grd")))
    assert fit == ["c", "n"] and fit_bl == []
    assert v["ratio_pf"]["cat"] == "none" and v["ratio_pf"]["strict"] == "none"
    assert v["ratio_grd"]["cat"] == "causal" and v["ratio_grd"]["strict"] == "above"
    assert "borderline: c (above 29/30)" in L[1] and "inconsistent: n (above 30/30)" in L[1]
    txt = " ".join(AS.reading(summ, scen))
    assert "hinges on borderline accounts" in txt
    # the same observed value one dataset further out: both inconsistent, no dependence
    summ2 = AS.summarize(pd.DataFrame(rows[:-1] + [dict(scenario="observed", rep=0,
                                                        stat="ratio_grd", value=1.30)]))
    _, _, _, v2 = AS._verdicts(summ2, scen, ("hr_primary",), (("ratio_grd", "grd"),))
    assert v2["ratio_grd"]["cat"] == v2["ratio_grd"]["strict"] == "above"
    c = AS.consistency(summ, scen, ("ratio_grd", "ratio_pf"))
    assert c["c"] == {"inconsistent": [], "borderline": ["ratio_grd"]}
    assert c["n"] == {"inconsistent": ["ratio_grd"], "borderline": []}


def test_criterion_and_verdicts():
    se = dict(sim_se_q975=0.33, sim_cr_cover=1.0)
    rows = []
    for sc in ("c", "n", "k"):
        grd = dict(c=(1.1, 0.9, 1.3, 0.0, "above"), n=(1.0, 0.85, 1.2, 0.0, "above"),
                   k=(1.6, 1.3, 2.0, 0.3, "above"))[sc]
        rows.append(_row(sc, "ratio_grd", *grd[:3], 1.66, *grd[3:], obs_se_log=0.30,
                         sim_sd_log=0.12, p_obs=0.5 if sc == "k" else 0.01, **se))
        kk = dict(c=(1.1, 0.9, 1.3, 0.05, "above"), n=(1.0, 0.85, 1.2, 0.0, "above"),
                  k=(1.6, 1.3, 2.0, 0.0, "below"))[sc]
        rows.append(_row(sc, "ratio_k", *kk[:3], 1.25, *kk[3:], obs_se_log=0.13,
                         sim_sd_log=0.06, p_obs=0.5, sim_se_q975=0.17, sim_cr_cover=1.0))
        rows.append(_row(sc, "ratio_pf", 1.1, 0.9, 1.3, 1.1, 0.4, "above", obs_se_log=0.17,
                         sim_sd_log=0.15, p_obs=0.5, sim_se_q975=0.30, sim_cr_cover=1.0))
        rows.append(_row(sc, "hr_primary", 2.0, 1.8, 2.05, 2.11, 0.0, "above", obs_se_log=0.19,
                         sim_sd_log=0.08, p_obs=0.6, sim_se_q975=0.14, sim_cr_cover=1.0))
    summ = pd.DataFrame(rows)
    assert AS.criterion(summ, "ratio_grd")[0] == "range"
    assert AS.criterion(summ, "hr_primary")[0] == "z"  # observed SE above the simulated SEs
    scen = pd.DataFrame(dict(scenario=["c", "n", "k"], short=["c", "n", "k"],
                             causal=[True, False, True]))
    fit, _, L, v = AS._verdicts(summ, scen, ("hr_primary",), (("ratio_pf", "pf"),
                                                               ("ratio_grd", "grd"),
                                                               ("ratio_k", "k")))
    assert fit == ["c", "n", "k"]  # the HR is judged by z (p 0.6), not by its exceedance (0%)
    assert v["ratio_pf"]["cat"] == "none"  # consistent with causal c, k and non-causal n
    assert v["ratio_grd"]["cat"] == "causal"  # consistent with k only
    # ratio_k by the exceedance: only causal c, and c only borderline (5%); strictly, the
    # observed value is above c and n but below k
    assert v["ratio_k"]["cat"] == "causal" and v["ratio_k"]["strict"] == "between"
    assert v["ratio_k"]["alt"] is None  # firm criterion: no criterion dependence reported
    # without a simulated SE and a borderline SD ratio, the verdict is reported for both criteria
    summ2 = summ.drop(columns=["sim_se_q975", "sim_cr_cover"]).assign(
        sim_sd_log=np.where(summ["stat"] == "ratio_k", 0.13 / 1.6, summ["sim_sd_log"]))
    assert AS.criterion(summ2, "ratio_k")[0] == "z" and not AS.criterion(summ2, "ratio_k")[2]
    _, _, L2, v2 = AS._verdicts(summ2, scen, ("hr_primary",), (("ratio_k", "k"),))
    assert v2["ratio_k"]["cat"] == "none" and v2["ratio_k"]["alt"] == "causal"
    assert "by the other criterion" in L2[0]


def test_onset_start_stats_counts_chirps_after_onsets_within_the_chase():
    tr = Trial(recording="o", group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
               t_start=0.0, t_end=100.0, chases=np.array([[10.0, 20.0], [30.0, 30.8]]),
               chirps_sub=np.array([9.5, 10.2, 10.7, 11.2, 11.9, 12.5, 29.2, 30.6, 30.9]))
    s = AS.onset_start_stats(tr)
    # chase 1: 0.2 / 0.7 / 1.2 / 1.9 s (2.5 s outside); chase 2: 0.6 s (30.9 is after its offset)
    assert [s[f"on_start_{i}"] for i in range(4)] == [1, 2, 1, 1]
    assert s["on_pre_1s"] == 2 and s["n_on_win"] == 2


def test_calibrate_start_raises_the_profile_where_the_data_chirp_more():
    cfg = _cfg()
    T = _templates()
    # 'real' data with one extra sub chirp 0.5-1 s after every onset
    T = [replace(tr, chirps_sub=np.sort(np.r_[tr.chirps_sub, tr.chases[:, 0] + 0.75]))
         for tr in T]
    P = replace(S.estimate_params(T, cfg), **AS.LATTICE)
    P2, log, mult = AS.calibrate_start(P, T, cfg, "t", n_iter=2, n_reps=2, workers=1)
    assert mult.shape == (4,) and np.all(np.isfinite(mult)) and mult[1] > 1.5
    assert np.allclose(P2.shape_in[:4], P.shape_in[:4] * mult)
    assert np.allclose(P2.shape_in[4:], P.shape_in[4:])
    first = log[(log["start_round"] == 0)].iloc[-1]["ratio_on_start_1"]
    last = log.iloc[-1]["ratio_on_start_1"]
    assert abs(np.log(last)) < abs(np.log(first))  # the 0.5-1 s bin moves toward the data


def test_locked_update_moves_toward_real_counts_and_corrects_the_cluster_post_side():
    prof = dict(mass2=dict(resp=0.5, sym=1.0, resid=0.5), bout_mult=1.0)
    tps = [S.TrialParams(recording="r", template=None, dur=None, gap=None, sub_base_in=0,
                         sub_base_out=0, dom_rate_in=0, dom_rate_out=0, sub_q=0, dom_q=0,
                         contact_rate=0, n_chases=10, n_sub_in=0, resp_n=0.2, sym_n=0.1)]
    real = pd.DataFrame(dict(n_off_win=[10.0], pre_2s=[5.0], post_2s=[6.0]), index=["r"])
    sim = pd.DataFrame(dict(n_off_win=[10.0], pre_2s=[3.0], post_2s=[6.0], off_pre_1s=[2.0]),
                       index=["r"])
    resp = AS.AcScenario("x", "", {}, "resp", "x")
    up = AS._locked_update(resp, prof)(tps, sim, real)[0]
    assert np.isclose(up.resp_n, 0.2)  # post already matched
    sym = AS.AcScenario("y", "", {}, "sym", "y")
    up = AS._locked_update(sym, prof)(tps, sim, real)[0]
    assert np.isclose(up.sym_n, 0.1 + 0.2)  # (5 - 3) / 10 per offset / mass 1
    # the cluster adds 0.2 per offset after the offset too, so the residual burst shrinks
    assert np.isclose(up.resp_n, max(0.0, 0.2 - 0.2 * 1.0 / 0.5))


def test_acausal_sim_stats_runs_on_synthetic_trials():
    P = _params()
    Pr = replace(P, resp_edges=EDGES, resp_density=np.array([0.2, 0.3, 0.3, 0.2]),
                 beta_off=np.log(3.0), trials=[replace(tp, resp_n=0.5) for tp in P.trials])
    st = AS.acausal_sim_stats(_sim(Pr, 11), _cfg())
    for k in ("hr_primary", "hr_past", "hr_future", "ratio_pf", "ratio_k", "ratio_rm"):
        assert np.isfinite(st[k]) and st[k] > 0, k
    assert np.isclose(st["ratio_pf"], st["hr_past"] / st["hr_future"], rtol=1e-6)
    assert np.isclose(st["ratio_rm"], st["hr_restricted"] / st["hr_mirror"])
    assert st["hr_primary"] > 1.3  # the causal effect is visible


if __name__ == "__main__":  # fallback runner when pytest is not installed
    import traceback

    n_ok, failed = 0, []
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            n_ok += 1
            print(f"ok    {name}")
        except Exception:  # noqa: BLE001
            failed.append(name)
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"{n_ok} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
