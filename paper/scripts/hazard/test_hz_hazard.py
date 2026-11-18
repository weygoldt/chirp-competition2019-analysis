"""Tests for hz_hazard: no leakage of same-bin events, risk sets, solvers.

Run: .venv/bin/python -m pytest src_hazard/test_hz_hazard.py -q
pytest is not installed in this venv; without it, run the file directly
(.venv/bin/python src_hazard/test_hz_hazard.py), which uses the small runner
at the bottom.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # minimal stand-in so the file runs without pytest
    import math
    import types

    def _parametrize(names: str, values: list) -> object:
        def deco(fn):
            fn._params = ([n.strip() for n in names.split(",")], values)
            return fn
        return deco

    def _approx(v: float, rel: float = 1e-6, abs: float = 1e-12) -> object:
        class _A:
            def __eq__(self, other: float) -> bool:
                return math.isclose(other, v, rel_tol=rel, abs_tol=abs)
        return _A()

    pytest = types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=_parametrize),
                                   approx=_approx)

sys.path.insert(0, str(Path(__file__).parent))
import hz_hazard as hz  # noqa: E402
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402

BW = 0.05


def _cfg(**kw) -> Config:
    base = dict(bin_width=BW, hist_max=2.0, n_basis=5, tau_grid=(0.25, 1.0), n_jitter=5,
                extra=dict(offset_min_elapsed=0.0, onset_min_elapsed=0.0))
    base.update(kw)
    return Config(**base)


def _trial(chirps_sub, chirps_dom=(), chases=((10.0, 20.02), (30.02, 40.0)), rec="T1") -> Trial:
    """60 s trial; chase 1 ends in bin 400 [20.00, 20.05), chase 2 starts in bin 600."""
    return Trial(recording=rec, group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
                 t_start=0.0, t_end=60.0, chases=np.array(chases), chirps_sub=np.array(chirps_sub),
                 chirps_dom=np.array(chirps_dom), size_dom=10.0, size_sub=9.0)


def _row(design, k):
    r = design[design["k"] == k]
    assert len(r) == 1, f"bin {k} not (uniquely) in the risk set"
    return r.iloc[0]


SUB_HIST = [f"sub_h{i + 1}" for i in range(5)] + ["sub_any_250ms", "sub_any_1000ms"]


@pytest.mark.parametrize("kind,k_event,t_same,t_before", [
    ("offset", 400, 20.01, 19.97),  # offset at 20.02 s
    ("onset", 600, 30.01, 29.97),  # onset at 30.02 s
])
def test_same_bin_chirp_not_in_history(kind, k_event, t_same, t_before):
    cfg = _cfg()
    d = hz.build_design([_trial([t_same])], kind, cfg)
    row = _row(d, k_event)
    assert row["y"] == 1
    for c in SUB_HIST:
        assert row[c] == 0, f"{c}: chirp in the outcome bin leaked into its history"


@pytest.mark.parametrize("kind,k_event,t_before", [("offset", 400, 19.97), ("onset", 600, 29.97)])
def test_previous_bin_chirp_in_history(kind, k_event, t_before):
    cfg = _cfg()
    lags, B = hz.raised_cosine_basis(cfg)
    d = hz.build_design([_trial([t_before])], kind, cfg)
    row = _row(d, k_event)
    assert row["y"] == 1
    assert row["sub_any_250ms"] == 1 and row["sub_any_1000ms"] == 1
    # lag of exactly one bin -> first basis row
    np.testing.assert_allclose([row[f"sub_h{i + 1}"] for i in range(5)], B[0], atol=1e-12)
    assert row["sub_h1"] > 0


def test_same_bin_chirp_enters_next_bin():
    """The chirp excluded from the offset bin is history one bin later (onset risk set)."""
    cfg = _cfg()
    lags, B = hz.raised_cosine_basis(cfg)
    d = hz.build_design([_trial([20.01])], "onset", cfg)
    row = _row(d, 401)
    np.testing.assert_allclose([row[f"sub_h{i + 1}"] for i in range(5)], B[0], atol=1e-12)


def test_tau_window_edges():
    """'>= 1 chirp in [t_k - tau, t_k)': 0.25 s = 5 bins back, not 6."""
    cfg = _cfg()
    k = 400
    inside = hz.build_design([_trial([(k - 5) * BW + 0.01])], "offset", cfg)
    outside = hz.build_design([_trial([(k - 6) * BW + 0.01])], "offset", cfg)
    assert _row(inside, k)["sub_any_250ms"] == 1
    assert _row(outside, k)["sub_any_250ms"] == 0


def test_dom_chirps_do_not_enter_sub_columns():
    d = hz.build_design([_trial([], chirps_dom=[19.97])], "offset", _cfg())
    row = _row(d, 400)
    assert row["dom_any_250ms"] == 1 and all(row[c] == 0 for c in SUB_HIST)


def test_rise_guard_delays_rises():
    """Rise stamps (frame centres) enter history only one frame (rise_guard) later."""
    cfg = _cfg(use_rises=True)
    tr = _trial([])
    tr.rises_sub = np.array([19.97])
    tr.rises_dom = np.zeros(0)
    off = hz.build_design([tr], "offset", cfg)
    assert all(_row(off, 400)[f"sub_rise_h{i + 1}"] == 0 for i in range(5))
    on = hz.build_design([tr], "onset", cfg)
    k_seen = int(np.floor((19.97 + 0.33) / BW)) + 1
    assert _row(on, k_seen)["sub_rise_h1"] > 0
    assert all(_row(on, k_seen - 1)[f"sub_rise_h{i + 1}"] == 0 for i in range(5))


def test_risk_sets_partition_window():
    # event bin = floor((t - t0) / bw); times off bin edges so that the index
    # rule and "state at bin start" (half-open chases) coincide
    cfg = _cfg()
    tr = _trial([5.0, 15.0, 35.0], chases=((10.01, 20.02), (30.02, 40.01)))
    off = hz.build_design([tr], "offset", cfg)
    on = hz.build_design([tr], "onset", cfg)
    K = int(round(60.0 / BW))
    ks = np.sort(np.concatenate([off["k"], on["k"]]))
    np.testing.assert_array_equal(ks, np.arange(K))
    assert off["y"].sum() == 2 and on["y"].sum() == 2
    # offset risk set = bins whose start lies inside a chase
    t = off["t"].to_numpy()
    assert np.all(tr.in_chase(t))
    assert not np.any(tr.in_chase(on["t"].to_numpy()))
    # elapsed is time since onset / since last offset (or window start)
    np.testing.assert_allclose(_row(off, 400)["elapsed"], 20.0 - 10.01)
    assert _row(on, 0)["first_gap"] == 1 and _row(on, 401)["first_gap"] == 0
    np.testing.assert_allclose(_row(on, 401)["elapsed"], 20.05 - 20.02)


def test_structural_zero_cutoff_keeps_events():
    cfg = _cfg(extra={})  # auto min_elapsed
    tr = _trial([], chases=((10.0, 11.02), (13.02, 20.0), (25.0, 30.0)))
    off = hz.build_design([tr], "offset", cfg)
    on = hz.build_design([tr], "onset", cfg)
    assert off["y"].sum() == 3 and on["y"].sum() == 3
    assert off["elapsed"].min() >= 1.02 - 2 * BW - 1e-9
    assert on.loc[on["first_gap"] == 0, "elapsed"].min() >= 2.0 - 2 * BW - 1e-9


def test_raised_cosine_basis():
    lags, B = hz.raised_cosine_basis(_cfg())
    assert B.shape == (40, 5)
    np.testing.assert_allclose(lags[[0, -1]], [0.05, 2.0])
    assert B[0, 0] == pytest.approx(1.0) and np.allclose(B[-1], 0.0)
    assert np.all(B.sum(1)[:-1] > 0)  # no hole in lag coverage


def test_fit_glm_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    n, G = 20_000, 6
    g = rng.integers(0, G, n)
    X = np.column_stack([np.eye(G)[g], rng.normal(size=(n, 3))])
    eta = -4 + 0.3 * g / G + X[:, G:] @ np.array([0.5, -0.3, 0.0])
    for link, sm_link in (("cloglog", sm.families.links.CLogLog()),
                          ("logit", sm.families.links.Logit())):
        mu = -np.expm1(-np.exp(eta)) if link == "cloglog" else 1 / (1 + np.exp(-eta))
        y = (rng.uniform(size=n) < mu).astype(float)
        mine = hz.fit_glm(X, y, link=link, groups=g)
        ref = sm.GLM(y, X, family=sm.families.Binomial(link=sm_link)).fit(
            cov_type="cluster", cov_kwds={"groups": g})
        np.testing.assert_allclose(mine.params, ref.params, atol=1e-6)
        np.testing.assert_allclose(np.sqrt(np.diag(mine.cov_cluster)), ref.bse, rtol=1e-4)


def test_mixed_hessian_matches_finite_differences():
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from statsmodels.tools.numdiff import approx_fprime
    from scipy import sparse

    rng = np.random.default_rng(1)
    n, G = 3000, 5
    g = rng.integers(0, G, n)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = (rng.uniform(size=n) < 0.1).astype(float)
    Z = sparse.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, G))
    m = BinomialBayesMixedGLM(y, X, Z, np.zeros(G, dtype=int), vcp_p=2.0, fe_p=5.0)
    v = np.concatenate([[-2.0, 0.3, -0.1], [np.log(0.7)], rng.normal(0, 0.3, G)])
    num = approx_fprime(v, lambda p: -m.logposterior_grad(p), centered=True)
    np.testing.assert_allclose(hz._neg_logpost_hessian(m, v), num, rtol=1e-4, atol=1e-4)


def test_chord_refit_equals_full_irls():
    rng = np.random.default_rng(2)
    n, G = 30_000, 4
    g = np.sort(rng.integers(0, G, n))
    Xd = rng.normal(size=(n, 3))
    S = (rng.uniform(size=(n, 1)) < 0.05).astype(float)
    X = np.column_stack([np.eye(G)[g], Xd, S])
    y = (rng.uniform(size=n) < 0.01 * np.exp(0.5 * S[:, 0])).astype(float)
    full = hz.fit_glm(X, y, cov=False)
    ch = hz._Chord.at(g, Xd, S, y, "cloglog", full.params, None, None)
    S2 = (rng.uniform(size=(n, 1)) < 0.05).astype(float)  # a "jittered" column
    beta, ok = ch.refit(S2)
    ref = hz.fit_glm(np.column_stack([np.eye(G)[g], Xd, S2]), y, cov=False)
    assert ok
    np.testing.assert_allclose(beta, ref.params, atol=1e-4)


def test_null_trains_preserve_counts_per_state():
    cfg = _cfg()
    tr = _trial([5.0, 12.0, 15.0, 35.0, 50.0, 55.0])
    trains = hz.null_sub_trains([tr], 20, cfg, "shift")["T1"]
    assert trains.shape == (21, 6)
    np.testing.assert_array_equal(trains[0], tr.chirps_sub)
    n_in = tr.in_chase(trains).sum(1)
    assert np.all(n_in == tr.in_chase(tr.chirps_sub).sum())


@pytest.mark.parametrize("t_chirp,on", [(19.49, 1), (19.51, 0), (18.51, 1), (18.49, 0)])
def test_guard_indicator_window_edges(t_chirp, on):
    """guard_500ms: >= 1 chirp in [t_k - 1.5 s, t_k - 0.5 s); bin 400 starts at 20.0 s."""
    cfg = _cfg()
    d = hz.build_design([_trial([t_chirp])], "offset", cfg)
    assert hz.guard_model(cfg) == "guard_500ms"
    assert _row(d, 400)["sub_grd_500ms"] == on


def _busy_trial(rec: str, seed: int, rises: bool = False) -> Trial:
    """1200 s trial with ~60 chases and random chirps (and rises)."""
    rng = np.random.default_rng(seed)
    on = np.cumsum(rng.uniform(12, 25, 60))
    ch = np.c_[on, on + rng.uniform(1.5, 8, 60)]
    ch = ch[ch[:, 1] < 1190]
    tr = Trial(recording=rec, group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
               t_start=0.0, t_end=1200.0, chases=ch, chirps_sub=rng.uniform(0, 1200, 150),
               chirps_dom=rng.uniform(0, 1200, 40), size_dom=10.0, size_sub=9.0 + 0.1 * seed)
    if rises:
        tr.rises_sub, tr.rises_dom = rng.uniform(0, 1200, 30), rng.uniform(0, 1200, 30)
    return tr


def test_use_rises_does_not_leak_into_chirp_statistics():
    """With use_rises the 'sub_rise_h*' columns must not be taken as chirp terms."""
    cfg = _cfg(use_rises=True, extra=dict(offset_min_elapsed=0.0, onset_min_elapsed=0.0,
                                          n_workers=1))
    trials = [_busy_trial(f"T{i}", i, rises=True) for i in range(4)]
    d = hz.build_design(trials, "offset", cfg)
    assert any(c.startswith("sub_rise_h") for c in d.columns)
    lo = hz.leave_one_dyad_out(d, cfg)
    assert len(lo) == 4 and set(lo["stat"]) == {"sub_any_1000ms"}
    j = hz.jitter_test(trials, "offset", cfg, design=d, n_jitter=3, n_workers=1)
    assert set(j["stat"]) == {"tau_1000ms_sub", "kernel_sub_mean"}
    assert np.isfinite(j["value"]).all()
    # the observed tau statistic equals the fitted sub indicator coefficient
    e = hz.fit_models(d, "offset", cfg, models=["tau_1000ms"], backends=["glm_fe"])
    b = e.loc[e["term"] == "sub_any_1000ms", "estimate"].iloc[0]
    obs = j.loc[(j["stat"] == "tau_1000ms_sub") & (j["iter"] == 0), "value"].iloc[0]
    assert obs == pytest.approx(b, rel=1e-6, abs=1e-8)


def test_re_meta_recovers_mean_and_handles_zero_counts():
    rng = np.random.default_rng(3)
    E = rng.uniform(2, 30, 200)
    O = rng.poisson(E * np.exp(np.log(0.5) + rng.normal(0, 0.5, 200)))
    r = hz.re_meta(O, E)
    assert r["mu_low"] < np.log(0.5) < r["mu_high"]
    assert 0.3 < r["tau"] < 0.7 and r["p_G"] < 1e-3
    # homogeneous, with zeros: tau near 0, finite interval
    E2 = np.array([0.5, 1.0, 2.0, 5.0, 8.0, 12.0])
    r2 = hz.re_meta(np.array([0, 0, 1, 2, 4, 7]), E2)
    assert np.isfinite(r2["mu_low"]) and np.isfinite(r2["mu_high"]) and r2["tau"] < 0.3


def test_dyad_oe_counts_indicator_on_events():
    cfg = _cfg(extra=dict(offset_min_elapsed=0.0, onset_min_elapsed=0.0))
    trials = [_busy_trial(f"T{i}", i) for i in range(3)]
    d = hz.build_design(trials, "offset", cfg)
    tab = hz.dyad_oe(d, cfg, "tau_1000ms")
    on = d["sub_any_1000ms"] == 1
    assert tab["observed"].sum() == int(d.loc[on, "y"].sum())
    assert (tab["expected"] > 0).all()


def test_eodf_baseline_is_step_hold():
    """Baseline at grid g uses only raw samples <= g; NaN before the first sample."""
    import hz_data
    t = np.array([0.0, 10.0, 20.0])
    f = np.array([100.0, 100.0, 300.0])
    grid = np.array([-1.0, 5.0, 15.0, 19.99, 20.0])
    b = hz_data._baseline(t, f, 1.0, grid)  # 1 s window: median = the sample itself
    assert np.isnan(b[0])
    np.testing.assert_allclose(b[1:], [100.0, 100.0, 100.0, 300.0])


if __name__ == "__main__":  # fallback runner when pytest is not installed
    import traceback

    n_ok, failed = 0, []
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        cases = [dict(zip(fn._params[0], v if isinstance(v, tuple) else (v,)))
                 for v in fn._params[1]] if hasattr(fn, "_params") else [{}]
        for kw in cases:
            label = f"{name}{kw or ''}"
            try:
                fn(**kw)
                n_ok += 1
                print(f"ok    {label}")
            except Exception:  # noqa: BLE001
                failed.append(label)
                print(f"FAIL  {label}\n{traceback.format_exc()}")
    print(f"{n_ok} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
