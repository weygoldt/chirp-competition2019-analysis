"""Tests for hz_acausal: future-window covariates and surrogate chirp trains.

Run: .venv/bin/python -m pytest src_hazard/test_hz_acausal.py -q
pytest is not installed in this venv; without it, run the file directly
(.venv/bin/python src_hazard/test_hz_acausal.py), which uses the small runner
at the bottom.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # minimal stand-in so the file runs without pytest
    import types

    def _parametrize(names: str, values: list) -> object:
        def deco(fn):
            fn._params = ([n.strip() for n in names.split(",")], values)
            return fn
        return deco

    pytest = types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=_parametrize))

sys.path.insert(0, str(Path(__file__).parent))
import hz_acausal as ac  # noqa: E402
import hz_hazard as hz  # noqa: E402
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402

BW = 0.05


def _cfg(**kw) -> Config:
    base = dict(bin_width=BW, hist_max=2.0, n_basis=5, tau_grid=(0.25, 1.0), n_jitter=5,
                extra=dict(offset_min_elapsed=0.0, onset_min_elapsed=0.0, n_workers=1))
    base.update(kw)
    return Config(**base)


def _trial(chirps_sub, chirps_dom=(), chases=((10.0, 20.02), (30.02, 40.0)), rec="T1",
           t_end=60.0, exclude=()) -> Trial:
    """60 s trial; chase 1 ends in bin 400 [20.00, 20.05), chase 2 starts in bin 600."""
    return Trial(recording=rec, group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
                 t_start=0.0, t_end=t_end, chases=np.array(chases), chirps_sub=np.array(chirps_sub),
                 chirps_dom=np.array(chirps_dom), size_dom=10.0, size_sub=9.0,
                 exclude_intervals=np.array(exclude).reshape(-1, 2))


def _design(tr: Trial, kind: str, cfg: Config):
    d = hz.build_design([tr], kind, cfg)
    return ac.add_future_columns(d, [tr], cfg)


def _row(design, k):
    r = design[design["k"] == k]
    assert len(r) == 1, f"bin {k} not (uniquely) in the risk set"
    return r.iloc[0]


FUT = [f"sub_fh{i + 1}" for i in range(5)] + ["sub_fut_250ms", "sub_fut_1000ms"]
PAST = [f"sub_h{i + 1}" for i in range(5)] + ["sub_any_250ms", "sub_any_1000ms"]


@pytest.mark.parametrize("kind,k_event,t_same,t_next", [
    ("offset", 400, 20.01, 20.07),  # offset at 20.02 s, bin 400 = [20.00, 20.05)
    ("onset", 600, 30.01, 30.07),  # onset at 30.02 s, bin 600 = [30.00, 30.05)
])
def test_future_indicator_excludes_outcome_bin(kind, k_event, t_same, t_next):
    """A chirp in the outcome bin is neither past nor future; one in the next bin is future."""
    cfg = _cfg()
    lags, B = hz.raised_cosine_basis(cfg)
    same = _row(_design(_trial([t_same]), kind, cfg), k_event)
    assert same["y"] == 1
    for c in FUT + PAST:
        assert same[c] == 0, f"{c}: chirp in the outcome bin leaked"
    nxt = _row(_design(_trial([t_next]), kind, cfg), k_event)
    assert nxt["sub_fut_250ms"] == 1 and nxt["sub_fut_1000ms"] == 1
    np.testing.assert_allclose([nxt[f"sub_fh{i + 1}"] for i in range(5)], B[0], atol=1e-12)
    for c in PAST:  # the past covariates do not see a later chirp
        assert nxt[c] == 0


def test_past_indicator_unchanged_by_future_columns():
    """add_future_columns leaves the standard (past) columns untouched."""
    cfg = _cfg()
    tr = _trial([5.0, 12.0, 15.0, 19.97, 20.3, 35.0, 50.0], t_end=70.0)
    tr.obs_end = 60.0  # recording runs on, so no bin is dropped
    for kind in ("offset", "onset"):
        d = hz.build_design([tr], kind, cfg)
        d2 = ac.add_future_columns(d, [tr], cfg)
        assert len(d2) == len(d)
        for c in d.columns:
            np.testing.assert_array_equal(d2[c].to_numpy(), d[c].to_numpy(), err_msg=c)


def test_future_window_edges_mirror_past():
    """'>= 1 chirp in [t_k + bw, t_k + bw + tau)': 0.25 s = 5 bins ahead, not 6; guard mirrored."""
    cfg = _cfg()
    k = 400
    inside = _design(_trial([(k + 5) * BW + 0.01]), "offset", cfg)
    outside = _design(_trial([(k + 6) * BW + 0.01]), "offset", cfg)
    assert _row(inside, k)["sub_fut_250ms"] == 1
    assert _row(outside, k)["sub_fut_250ms"] == 0
    # guard: past = bins k-30..k-11, future = bins k+11..k+30
    for dk, on in ((10, 0), (11, 1), (30, 1), (31, 0)):
        d = _design(_trial([(k + dk) * BW + 0.01]), "offset", cfg)
        assert _row(d, k)["sub_fgd_500ms"] == on, dk
        d = _design(_trial([(k - dk) * BW + 0.01]), "offset", cfg)
        assert _row(d, k)["sub_grd_500ms"] == on, -dk


def test_future_uses_chirps_after_window_and_drops_bins_past_recording_end():
    cfg = _cfg()
    # chase 2 is censored at obs_end = 30.6 s; the chirp at 31.2 s lies after the window
    tr = _trial([31.2], chases=((10.0, 20.02), (29.0, 31.0)), t_end=60.0)
    tr.obs_end = 30.6
    d = _design(tr, "offset", cfg)
    assert _row(d, 611)["sub_fut_1000ms"] == 1  # bin 611 = [30.55, 30.60)
    assert _row(d, 611)["sub_any_1000ms"] == 0
    # recording ends 1 s after the window: bins whose future window passes t_end are dropped
    tr3 = _trial([], t_end=31.6)
    tr3.obs_end = 30.6
    base = hz.build_design([tr3], "onset", cfg)
    d3 = ac.add_future_columns(base, [tr3], cfg)
    ext = ac.future_extent(cfg)
    assert np.all(d3["t"].to_numpy() + BW + ext <= tr3.t_end + 1e-9)
    n_drop = int(d3.attrs["future_bookkeeping"]["n_dropped_future_past_end"].sum())
    assert n_drop == len(base) - len(d3) and n_drop > 0


def test_mirror_construction():
    """Latency reflection, keep rule, next-onset truncation, real in-chase chirps removed."""
    ch = ((10.0, 20.0), (21.5, 22.5), (30.0, 40.0))
    x = [5.0,  # out of chase, before any chase: kept unchanged
         15.0, 19.5,  # real in-chase (chase 0): removed
         20.4,  # post-offset of chase 0, L = 0.4 -> mirrored to 19.6
         21.2,  # L = 1.2 -> 18.8
         21.7,  # in chase 1: removed (real in-chase)
         22.6,  # post-offset of chase 1, L = 0.1 -> 22.4 (inside chase 1)
         24.4,  # post-offset of chase 1, L = 1.9 -> 20.6, before chase 1 onset: dropped
         26.0,  # L = 3.5 >= W: not mirrored
         41.0]  # post-offset of chase 2, L = 1.0 -> 39.0
    tr = _trial(x, chases=ch)
    train, info = ac.surrogate_train(tr, "mirror", W=2.0)
    expect = sorted([5.0, 20.4, 21.2, 22.6, 24.4, 26.0, 41.0,  # real out-of-chase
                     19.6, 18.8, 22.4, 39.0])  # mirrored
    np.testing.assert_allclose(train, expect)
    assert not np.any(np.isin([15.0, 19.5, 21.7], train))
    assert info["n_candidates"] == 5 and info["n_kept"] == 4 and info["n_dropped"] == 1
    # next-onset truncation: the 1.5 s gap (20.0 -> 21.5) limits chase 0's window even at W = 5
    t, i, L = ac.post_offset_chirps(tr, 5.0)
    assert 21.7 not in t and np.all(t[i == 0] < 21.5)
    np.testing.assert_allclose(L[t == 22.6], [0.1])
    assert i[t == 22.6][0] == 1


def test_mirror_stops_at_unknown_state_interval():
    ch = ((10.0, 20.0), (30.0, 40.0))
    tr = _trial([20.3, 20.9, 21.5], chases=ch, exclude=((20.8, 25.0),))
    t, i, L = ac.post_offset_chirps(tr, 2.0)
    np.testing.assert_allclose(t, [20.3])


def test_restricted_real_construction():
    ch = ((10.0, 20.0), (30.0, 40.0))
    x = [5.0, 15.0, 18.5, 19.9, 20.5, 35.0, 39.0]
    tr = _trial(x, chases=ch)
    train, info = ac.surrogate_train(tr, "restricted_real", W=2.0)
    np.testing.assert_allclose(train, [5.0, 18.5, 19.9, 20.5, 39.0])
    assert info["n_in_chase"] == 3 and info["n_real_in_chase"] == 5


def test_restricted_real_matches_the_mirror_support():
    """A real in-chase chirp at latency L is kept only if the mirror could hold a chirp at the
    same latency: off + L before the next onset and outside / before unknown-state intervals."""
    ch = ((10.0, 20.0), (21.0, 30.0), (40.0, 50.0))
    x = [19.7, 19.2, 18.8,  # chase 0: L 0.3, 0.8 kept; L 1.2 -> 21.2 >= next onset: dropped
         20.3, 20.8,  # chase 0 post-offset (L 0.3, 0.8)
         49.5, 49.0,  # chase 2: L 0.5 kept; L 1.0 -> 51.0 inside the unknown state: dropped
         50.5]  # chase 2 post-offset (L 0.5)
    tr = _trial(x, chases=ch, exclude=((50.8, 55.0),))
    train, info = ac.surrogate_train(tr, "restricted_real", W=2.0)
    np.testing.assert_allclose(train, [19.2, 19.7, 20.3, 20.8, 49.5, 50.5])
    assert info["n_candidates"] == 5 and info["n_kept"] == 3 and info["n_dropped"] == 2
    mir, minfo = ac.surrogate_train(tr, "mirror", W=2.0)
    # same latencies on both sides: the arms hold the same in-chase latencies
    _, _, Lr = ac.pre_offset_chirps(tr, 2.0, match_post=True)
    _, _, Lm = ac.post_offset_chirps(tr, 2.0)
    np.testing.assert_allclose(np.sort(Lr), np.sort(Lm))
    assert minfo["n_kept"] == info["n_kept"]


def test_comparator_windows_of_the_shift_arms():
    assert ac.comparator_windows((2.0, 5.0), 2.0, (0.5, 1.0, 2.0)) == [0.5, 1.0]
    assert ac.comparator_windows((2.0,), 2.0, (3.0,)) == []  # Delta > W: window W, already in B
    assert ac.comparator_windows((), 2.0, (1.0,)) == [1.0]


def test_state_rates_count_onsets_inside_the_window():
    ch = ((10.0, 20.0), (40.0, 70.0))  # chase 2 starts inside the 60 s window, ends after it
    tr = _trial([9.5, 10.2, 20.5, 39.6, 40.3, 40.8, 57.0], chases=ch, t_end=80.0)
    tr = replace(tr, obs_end=60.0)
    r = ac.state_rates([tr]).iloc[0]
    assert r["n_chases"] == 1 and r["n_onsets"] == 2
    assert r["first_1s_of_chase"] == 3  # 10.2, 40.3, 40.8
    assert r["last_1s_before_onset"] == 2  # 9.5, 39.6
    assert r["last_1s_of_chase"] == 0 and r["first_1s_after_offset"] == 1  # 20.5


def test_shift_construction():
    ch = ((10.0, 20.0), (30.0, 40.0))
    x = [15.0, 20.3, 20.8, 21.9, 41.0]
    tr = _trial(x, chases=ch)
    train, info = ac.surrogate_train(tr, "shift", W=2.0, delta=1.0)
    # 20.3 -> 19.3 (kept), 20.8 -> 19.8 (kept), 21.9 -> 20.9 (still after offset: dropped),
    # 41.0 -> 40.0 (= offset, not inside: dropped); real in-chase 15.0 removed
    np.testing.assert_allclose(train, sorted([20.3, 20.8, 21.9, 41.0, 19.3, 19.8]))
    assert info["n_kept"] == 2 and info["n_dropped"] == 2
    assert info["n_dropped_still_after_offset"] == 2 and info["n_dropped_before_onset"] == 0
    # a large delta pushes the chirp of a short chase before its onset
    tr2 = _trial([21.0], chases=((19.5, 20.0), (30.0, 40.0)))
    _, info2 = ac.surrogate_train(tr2, "shift", W=2.0, delta=2.0)
    assert info2["n_dropped_before_onset"] == 1 and info2["n_kept"] == 0


def _busy_trial(rec: str, seed: int) -> Trial:
    rng = np.random.default_rng(seed)
    on = np.cumsum(rng.uniform(12, 25, 60))
    ch = np.c_[on, on + rng.uniform(1.5, 8, 60)]
    ch = ch[ch[:, 1] < 1190]
    return Trial(recording=rec, group=5, dom_id=1.0, sub_id=2.0, dom_fish=1, sub_fish=2,
                 t_start=0.0, t_end=1200.0, chases=ch, chirps_sub=rng.uniform(0, 1200, 250),
                 chirps_dom=rng.uniform(0, 1200, 60), size_dom=10.0, size_sub=9.0 + 0.1 * seed)


def test_acausal_stats_runs_and_contrasts_are_consistent():
    """End to end on random trials: the contrasts equal differences of the fitted terms."""
    cfg = _cfg(tau_grid=(1.0,))
    trials = [_busy_trial(f"T{i}", i) for i in range(4)]
    e = ac.acausal_stats(trials, cfg, n_boot=3)
    fe = e[(e["variant"] == "two_sided") & (e["model"] == "pf_1000ms") & (e["backend"] == "glm_fe")]
    p = fe.loc[fe["term"] == "sub_any_1000ms", "estimate"].iloc[0]
    f = fe.loc[fe["term"] == "sub_fut_1000ms", "estimate"].iloc[0]
    c = fe.loc[fe["term"] == "sub_any_1000ms-sub_fut_1000ms", "estimate"].iloc[0]
    assert abs(c - (p - f)) < 1e-10
    d = e[e["variant"] == "restricted_real-mirror"]
    assert set(d["term"]) == {"sub_any_1000ms", "sub_grd_500ms"}
    rr = e[(e["variant"] == "restricted_real") & (e["model"] == "tau_1000ms")
           & (e["term"] == "sub_any_1000ms")]["estimate"].iloc[0]
    mi = e[(e["variant"] == "mirror") & (e["model"] == "tau_1000ms")
           & (e["term"] == "sub_any_1000ms")]["estimate"].iloc[0]
    got = d.loc[d["term"] == "sub_any_1000ms", "estimate"].iloc[0]
    assert abs(got - (rr - mi)) < 1e-6
    k = e[(e["model"] == "kernel_two_sided") & (e["term"] == "kernel_sub_fut")]
    assert len(k) == 40 and (k["lag"] < 0).all()


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
