"""plot_receiver_before_after: rows, within-competition weights, medians, per-competition ratios (A) and the
re-attack rates and rate ratios (B, C) on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plot_receiver_before_after as rba
import reattack_hazard as rh


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


def person_period():
    """Two competitions with rates 2x as high once chirped, in pieces 1 and 2; piece 0 has no chirped event."""
    rows = []
    for comp, base in (("c1", 0.01), ("c2", 0.02)):
        for piece in (0, 1, 2):
            for chirped in (0, 1):
                x = 100.0
                e = 0 if (piece == 0 and chirped) else int(round(base * x * (2 if chirped else 1) * (piece + 1)))
                rows.append((comp, 1, piece, e, x, chirped, 0.0))
    return pd.DataFrame(rows, columns=[rba.COMP, "interval", "piece", "event", "exposure", rba.COVARIATE,
                                       "log_prev"])


def test_rates_per_piece_and_competition():
    rates, comp, stats = rba.summarise_rates(person_period())
    assert len(rates) == len(rh.PIECES_S) - 1  # every piece, also those without rows
    assert rates.loc[0, "rate1"] == 0 and rates.loc[1, "ratio"] == pytest.approx(2.0)
    assert stats["rate_per_min"]["chirp"][0] == 0 and stats["rate_per_min"]["chirp"][3] == 0
    assert stats["rate_per_min"]["no_chirp"][1] == pytest.approx(60 * (0.02 + 0.04) / 2)
    assert len(stats["events"]["chirp"]) == len(rh.PIECES_S) - 1
    assert comp["ratio"].notna().all() and stats["competitions"]["n_above_1"] == 2
    assert all(np.isfinite(v) for v in stats["rate_per_min"]["no_chirp"])


def test_primary_rows_drop_intervals_without_a_previous_one():
    pp = person_period()
    pp.loc[pp[rba.COMP] == "c2", "log_prev"] = np.nan
    assert set(rba.primary_rows(pp)[rba.COMP]) == {"c1"}
    with pytest.raises(KeyError):
        rba.primary_rows(pp.drop(columns=["log_prev"]))


def test_piece_centres_within_the_axis():
    x = rba.piece_centres()
    assert x[0] == pytest.approx(np.sqrt(rba.XLIM_S[0] * 2.0))  # 0-2 s, from the axis' left edge
    assert x[1] == pytest.approx(np.sqrt(2.0 * 5.0))
    assert x[-1] == pytest.approx(np.sqrt(500.0 * rba.XLIM_S[1]))  # the open piece, to the right edge


def test_load_hr(tmp_path, monkeypatch):
    pd.DataFrame([{"fit": "primary", "hr": 1.1, "hdi_lo": 0.9, "hdi_hi": 1.3, "p_lt_1": 0.2}]).to_csv(
        tmp_path / "hr_summary.csv", index=False)
    monkeypatch.setattr(rba, "HAZARD_DIR", tmp_path)
    assert rba.load_hr() == {"median": 1.1, "hdi": [0.9, 1.3], "p_lt_1": 0.2}
    with pytest.raises(ValueError):
        rba.load_hr("unadjusted")


def fake_trace(alpha, u, beta, gamma_prev, n_draws=4):
    """A posterior in bambi's layout (pieces in the order of their labels as strings), every draw alike."""
    import arviz as az

    labels = rh.piece_labels()
    order = sorted(range(len(labels)), key=lambda k: labels[k])
    rep = lambda v: np.broadcast_to(np.asarray(v, float), (1, n_draws, *np.shape(v))).copy()  # noqa: E731
    return az.from_dict(
        posterior={"piece_label": rep([alpha[k] for k in order]), "chirped": rep(beta),
                   "log_prev_c": rep(gamma_prev), "log_dur_c": rep(0.0),
                   f"1|{rba.COMP}": rep(u), f"chirped|{rba.COMP}": rep(np.zeros(len(u)))},
        coords={"piece_label_dim": [labels[k] for k in order], "recording__factor_dim": ["c1", "c2"]},
        dims={"piece_label": ["piece_label_dim"], f"1|{rba.COMP}": ["recording__factor_dim"],
              f"chirped|{rba.COMP}": ["recording__factor_dim"]})


def test_standardized_rates_average_the_model_over_the_observed_time_at_risk():
    pp = person_period().assign(log_prev_c=0.0, log_dur_c=0.0)
    # c1's chirped time at risk follows longer previous intervals (log_prev_c = 1); hazard x2 per unit
    pp.loc[(pp[rba.COMP] == "c1") & (pp[rba.COVARIATE] == 1), "log_prev_c"] = 1.0
    alpha = np.log(np.arange(1, 10) / 600.0)  # 0.1, 0.2, ... per min
    tr = fake_trace(alpha, u=[0.0, np.log(2.0)], beta=np.log(1.5), gamma_prev=np.log(2.0))
    st = rba.standardized_rates(tr, pp).set_index(["chirp", "piece"])
    # equal weights (100 s with and without in every cell); c1's cells: (2 + 1) / 2, c2's: 2 (its u)
    for k in (0, 1, 2):
        base = 60 * np.exp(alpha[k])
        assert st.loc[(0, k), "median"] == pytest.approx(base * (1.5 + 2.0) / 2)
        assert st.loc[(1, k), "median"] == pytest.approx(base * (1.5 + 2.0) / 2 * 1.5)
        assert st.loc[(1, k), "piece_label"] == rh.piece_labels()[k]
    assert st.loc[(0, 0), "lo"] == pytest.approx(st.loc[(0, 0), "median"])  # identical draws
