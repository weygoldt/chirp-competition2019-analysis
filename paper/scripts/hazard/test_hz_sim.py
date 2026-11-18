"""Tests for hz_sim: generative model, effect covariate, fitting path, CRN.

Run: .venv/bin/python src_hazard/test_hz_sim.py (pytest is not installed in
this venv; the file also works under pytest). No real data needed.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hz_hazard as hz  # noqa: E402
import hz_sim as S  # noqa: E402
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402

BW = 0.05


def _template(rec: str, seed: int, T: float = 3600.0) -> Trial:
    """Synthetic 'real' trial: log-normal chases and gaps, Poisson chirps."""
    rng = np.random.default_rng(seed)
    ch, t = [], 5.0 + rng.exponential(20)
    while True:
        d = 1.0 + rng.lognormal(1.3, 0.6)
        if t + d >= T:
            break
        ch.append((t, t + d))
        t += d + 2.0 + rng.lognormal(3.0, 0.8)
        if t >= T:
            break
    ch = np.array(ch)
    sub = np.sort(rng.uniform(0, T, 300))
    dom = np.sort(rng.uniform(0, T, 40))
    return Trial(
        recording=rec,
        group=5,
        dom_id=1.0,
        sub_id=2.0,
        dom_fish=1,
        sub_fish=2,
        t_start=0.0,
        t_end=T,
        chases=ch,
        contacts=np.sort(rng.uniform(0, T, 20)),
        chirps_sub=sub,
        chirps_dom=dom,
        size_dom=10.0,
        size_sub=9.0 + seed % 3,
    )


def _templates() -> list[Trial]:
    return [_template(f"T{i}", i) for i in range(4)]


def _cfg() -> Config:
    return Config(bin_width=BW, tau_grid=(1.0,), n_sims=2)


def test_surv_table_respects_floor_and_median():
    rng = np.random.default_rng(0)
    d = 1.0 + rng.lognormal(1.5, 0.5, 400)
    tab = S.surv_table(d, floor=float(d.min()))
    x = tab.sample(20_000, rng)
    assert x.min() >= d.min() - 1e-9
    assert abs(np.median(x) / np.median(d) - 1) < 0.05
    assert tab.logsurv(np.array([0.0, d.min()]))[0] == 0.0
    assert np.all(np.diff(tab.logsurv(np.linspace(1, 60, 200))) <= 1e-12)


def test_effect_covariate_matches_design_indicator():
    """x_k of the simulator = hz_hazard's '>= 1 chirp in bins k-J..k-1'."""
    P = S.estimate_params(_templates(), _cfg())
    sim = S._TrialSim(P.trials[0], P, BW, np.random.SeedSequence(1), "x")
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 300, 40))
    sim._add(t, +1.0)
    ka, kb = 0, 6000
    x = sim._x(ka, kb)
    pad = 25
    cnt = hz._binned(t, 0.0, BW, kb, pad)
    ref = hz._any_before(cnt, 20, pad, kb)
    np.testing.assert_array_equal(x.astype(bool), ref)


def test_null_simulation_matches_template_rates():
    temps = _templates()
    cfg = _cfg()
    P = S.estimate_params(temps, cfg)
    real = [S.trial_stats(t) for t in temps]
    sims = [
        S.trial_stats(t)
        for rep in range(6)
        for t in S._simulate(P, BW, np.random.SeedSequence([0, rep]), f"r{rep}")
    ]
    n_real = sum(r["n_chases"] for r in real)
    n_sim = sum(r["n_chases"] for r in sims) / 6
    assert abs(n_sim / n_real - 1) < 0.12
    s_real = sum(r["n_sub"] for r in real)
    s_sim = sum(r["n_sub"] for r in sims) / 6
    assert abs(s_sim / s_real - 1) < 0.12


def test_synthetic_trials_are_valid_and_causal_order():
    P = replace(
        S.estimate_params(_templates(), _cfg()),
        beta_off=S.LOG2,
        boundary_sd=0.3,
        boundary_step=0.5,
        p_detect_in=0.7,
    )
    for tr in S._simulate(P, BW, np.random.SeedSequence(5), "v"):
        ch = tr.chases
        assert np.all(ch[:, 1] > ch[:, 0]) and np.all(ch[1:, 0] >= ch[:-1, 1])
        assert tr.recording.endswith("#v") and tr.meta["synthetic"]


def test_common_random_numbers_keep_chases_across_trigger_strength():
    """With no causal effect, chase times do not depend on the chirp process."""
    P = S.estimate_params(_templates(), _cfg())
    a = S._simulate(P, BW, np.random.SeedSequence(9), "a")
    b = S._simulate(replace(P, trigger_share=0.5), BW, np.random.SeedSequence(9), "b")
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.chases, y.chases)
    assert sum(t.meta["n_trigger_chirps"] for t in b) > 0
    assert sum(t.meta["n_trigger_chirps"] for t in a) == 0


def test_fit_fe_matches_fit_models():
    cfg = _cfg()
    P = replace(S.estimate_params(_templates(), cfg), beta_off=S.LOG2)
    trials = S._simulate(P, BW, np.random.SeedSequence(2), "f")
    d = hz.build_design(trials, "offset", cfg)
    rows, _ = S.fit_fe(d, cfg, "tau_1000ms")
    mine = {r["term"]: r for r in rows}
    ref = hz.fit_models(d, "offset", cfg, models=["tau_1000ms"], backends=["glm_fe"])
    r = ref[ref["term"] == "sub_any_1000ms"].iloc[0]
    m = mine["sub_any_1000ms"]
    np.testing.assert_allclose(
        [m["estimate"], m["se_cluster"], m["lo_cluster"], m["hi_cluster"]],
        [r["estimate"], r["se"], r["ci_low"], r["ci_high"]],
        rtol=1e-8,
    )


def test_strong_offset_effect_is_recovered():
    cfg = _cfg()
    temps = [_template(f"T{i}", i, T=7200.0) for i in range(4)]
    for t in temps:  # more in-chase chirps so the effect is estimable
        t.chirps_sub = np.sort(
            np.concatenate([t.chirps_sub, np.random.default_rng(1).uniform(0, 7200, 1500)])
        )
    P = replace(S.estimate_params(temps, cfg), beta_off=np.log(4.0))
    trials = S._simulate(P, BW, np.random.SeedSequence(4), "s")
    d = hz.build_design(trials, "offset", cfg)
    rows, _ = S.fit_fe(d, cfg, "tau_1000ms")
    est = {r["term"]: r for r in rows}["sub_any_1000ms"]
    assert abs(est["estimate"] - np.log(4.0)) < 4 * est["se_naive"]


def test_truths():
    P = S.estimate_params(_templates(), _cfg())
    lags = np.arange(1, 41) * BW
    t = S._truths(replace(P, beta_on=-S.LOG2), "onset", BW, lags)
    assert t["sub_any_1000ms"] == -S.LOG2 and np.isnan(t["sub_any_500ms"])
    assert abs(t["kernel_sub_mean"] + S.LOG2 / 2) < 1e-12
    assert S._truths(P, "offset", BW, lags)["sub_any_500ms"] == 0.0


def test_heterogeneous_effects_vary_by_trial_and_keep_crn():
    P = S.estimate_params(_templates(), _cfg())
    a = S._simulate(replace(P, beta_off=S.LOG2), BW, np.random.SeedSequence(5), "a")
    b = S._simulate(replace(P, beta_off=S.LOG2, beta_off_sd=0.5), BW,
                    np.random.SeedSequence(5), "b")
    assert {t.meta["beta_off_trial"] for t in a} == {S.LOG2}
    vb = np.array([t.meta["beta_off_trial"] for t in b])
    assert len(set(vb)) == len(vb) and np.all(np.abs(vb - S.LOG2) < 5 * 0.5)
    # the effect draws come from their own stream: no-effect chases are unchanged
    c = S._simulate(replace(P, beta_off_sd=0.5, beta_off=0.0), BW, np.random.SeedSequence(5), "c")
    d = S._simulate(P, BW, np.random.SeedSequence(5), "d")
    assert any(not np.allclose(x.chases, y.chases) for x, y in zip(c, d))  # effects act
    assert S.trigger_p_before(P) > 0.6


def test_offset_lag_delays_only_annotated_offsets():
    P = S.estimate_params(_templates(), _cfg())
    a = S._simulate(P, BW, np.random.SeedSequence(7), "a")
    b = S._simulate(replace(P, offset_lag=0.5), BW, np.random.SeedSequence(7), "b")
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.chases[:, 0], y.chases[:, 0])
        d = y.chases[:, 1] - x.chases[:, 1]
        assert np.all(d > 0) and np.all(d <= 0.5 + 1e-9) and np.mean(np.isclose(d, 0.5)) > 0.9
        np.testing.assert_allclose(x.chirps_sub, y.chirps_sub)


def test_trigger_share_is_realized_before_offset():
    P = S.estimate_params(_templates(), _cfg())
    tr = S._simulate(replace(P, trigger_share=0.3), BW, np.random.SeedSequence(3), "t")
    n_in = sum(int(t.in_chase(t.chirps_sub).sum()) for t in tr)
    n_trig_in = sum(t.meta["n_trigger_in_chase"] for t in tr)
    assert 0.2 < n_trig_in / n_in < 0.4


def test_calibration_matches_counts_under_effect():
    cfg = _cfg()
    temps = _templates()
    P = S.estimate_params(temps, cfg)
    eff = dict(beta_off=np.log(4.0), beta_on=-np.log(4.0))
    tps, log = S.calibrate_params(P, eff, temps, cfg, "box", n_iter=2, n_reps=4, workers=1)
    assert len(tps) == len(temps) and len(log) == 3
    dev0 = abs(log["ratio_n_sub_in"].iloc[0] - 1)
    dev1 = abs(log["ratio_n_sub_in"].iloc[-1] - 1)
    assert dev1 <= max(dev0, 0.05)
    assert any(tp.sub_mult_in != 1.0 for tp in tps)


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
