"""chase_event_rates: Fig. 6's loser chirp rates for other chase times (synthetic data)."""

from __future__ import annotations

import importlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import analyze_event_time_rates_trial as etr
import chase_event_rates as cer
import paper_config

N = 40  # resamples in these tests (1000 in the analysis)


@pytest.fixture(autouse=True)
def few_resamples(monkeypatch):
    monkeypatch.setattr(etr, "N_RESAMPLES", N)


def synthetic_trials(seed: int = 0, n_trials: int = 4) -> cer.ChaseTrials:
    rng = np.random.default_rng(seed)
    t = cer.ChaseTrials([], [], [], [])
    for i in range(n_trials):
        on = np.sort(rng.uniform(100, 3 * 3600 - 100, 5 + 3 * i))
        t.identifiers.append(f"rec{i}")
        t.onsets.append(on)
        t.offsets.append(on + rng.uniform(2, 20, len(on)))
        # chirps clustered after the offsets, plus background
        t.loser_chirps.append(np.sort(np.concatenate([rng.uniform(0, 4 * 3600, 300),
                                                      (t.offsets[-1][:, None] + rng.normal(1, 1, (len(on), 4))).ravel()])))
    t.onsets[0] = np.append(t.onsets[0], 3 * 3600 + 5.0)  # one chase outside the active phase
    t.offsets[0] = np.append(t.offsets[0], 3 * 3600 + 9.0)
    return t


def fig6_like(trials: cer.ChaseTrials, end: str) -> dict:
    """What analyze_event_time_rates_trial.main(--ag-events) writes for the loser around ``end``."""
    by_name = {"agon": trials.onsets, "agoff": trials.offsets}
    np.random.seed(paper_config.SEED)
    for c1, c2 in cer.AG_EVENTS_ORDER:
        centres = [cer.in_active_phase(c) for c in by_name[c1]]
        surrounds = trials.loser_chirps  # the draws do not depend on the surrounds
        centres, surrounds, ids = cer.drop_trials_without_centres(centres, surrounds, trials.identifiers)
        jk, per_trial, lags = etr.observed_rate_bootstrap(centres, surrounds, cer.CONV_TIME,
                                                          n_bootstraps=etr.N_RESAMPLES, leave_out_frac=etr.LEAVE_OUT_FRAC)
        if (c1, c2) == (cer.CENTRING[end], "lchirps"):
            return {"conv_time": cer.CONV_TIME.tolist(), "jackknifed_eventrate": jk.tolist(),
                    "per_trial_rates": [x.tolist() for x in per_trial], "centered_times": [x.tolist() for x in lags],
                    "trial_identifiers": ids}
        etr.null_rate_bootstrap(centres, surrounds, active_phase=etr.ACTIVE_PHASE_S, conv_time=cer.CONV_TIME,
                                n_bootstraps=etr.N_RESAMPLES, leave_out_frac=etr.LEAVE_OUT_FRAC,
                                null_mode="jittered_centers", jitter_amplitude_s=etr.CENTER_JITTER_AMPLITUDE_S)
    raise AssertionError


@pytest.mark.parametrize("end", cer.ENDS)
def test_replayed_draws_reproduce_fig6(end):
    """Unshifted chases after replay_fig6_draws: the resamples of main, number for number."""
    trials = synthetic_trials()
    fig6 = fig6_like(trials, end)
    centres, surrounds, ids = cer.fig6_centres(trials, end)
    assert ids == fig6["trial_identifiers"]
    assert sum(len(c) for c in centres) == sum(len(c) for c in getattr(trials, end + "s")) - 1  # one after 3 h
    cer.replay_fig6_draws(trials, end)
    jk, per_trial, _ = cer.resampled_rate(cer.shifted_centres(centres, 0.0), surrounds)
    assert np.array_equal(jk, np.array(fig6["jackknifed_eventrate"]))
    assert np.array_equal(np.array(per_trial), np.array(fig6["per_trial_rates"]))
    assert cer.check_against_fig6(trials, end, fig6, n_resamples=5)["n_trials"] == len(ids)


def test_check_against_fig6_fails_on_other_chases():
    trials = synthetic_trials()
    fig6 = fig6_like(trials, "offset")
    trials.offsets[1] = trials.offsets[1] + 0.04  # one frame later
    with pytest.raises(AssertionError, match="per_trial_rates|lags"):
        cer.check_against_fig6(trials, "offset", fig6, n_resamples=5)


def test_shifted_rate_is_the_rate_moved_by_the_shift():
    """Same draws: every resample of the shifted chases is the unshifted one moved (1 s = 10 grid steps)."""
    trials = synthetic_trials()
    centres, surrounds, _ = cer.fig6_centres(trials, "offset")
    np.random.seed(1)
    jk0, pt0, _ = cer.resampled_rate(centres, surrounds)
    np.random.seed(1)
    jk1, pt1, _ = cer.resampled_rate(cer.shifted_centres(centres, -1.0), surrounds)
    inner = slice(20, -20)  # away from the ends of the +-30 s grid
    assert np.allclose(jk1[:, 10:][:, inner], jk0[:, :-10][:, inner], rtol=1e-9, atol=1e-12)
    assert np.allclose(np.array(pt1)[:, 10:][:, inner], np.array(pt0)[:, :-10][:, inner], rtol=1e-9, atol=1e-12)


def pairs_for(trials: cer.ChaseTrials) -> pd.DataFrame:
    rows = []
    for rec, on, off in zip(trials.identifiers, trials.onsets, trials.offsets):
        for k in range(0, min(3, len(on))):
            rows.append({"recording": rec, "boris_onset_rec_s": on[k], "boris_offset_rec_s": off[k],
                         "onset_rec_s": on[k] - 1.5, "offset_rec_s": off[k] - 1.0,
                         "d_onset_s": -1.5, "d_offset_s": -1.0})
    return pd.DataFrame(rows)


def test_relabeled_centres_per_trial():
    trials = synthetic_trials()
    centres, surrounds, ids = cer.fig6_centres(trials, "onset")
    pairs = pairs_for(trials)
    pairs = pairs[pairs["recording"] != "rec2"].copy()  # a trial without re-labeled chases
    pairs.loc[pairs.index[0], "d_onset_s"] = np.nan  # a censored onset
    c, s, i = cer.relabeled_centres(pairs, "onset", centres, surrounds, ids)
    assert i == ["rec0", "rec1", "rec3"]  # Fig. 6's order, rec2 dropped
    assert [len(x) for x in c] == [2, 3, 3]
    assert np.allclose(c[0], trials.onsets[0][1:3] - 1.5)
    assert s[2] is surrounds[3]  # each trial keeps its own chirps
    b, _, _ = cer.relabeled_centres(pairs, "onset", centres, surrounds, ids, column="boris_onset_rec_s")
    assert np.allclose(b[0], trials.onsets[0][1:3])


def test_relabeled_centres_must_be_fig6_chases():
    trials = synthetic_trials()
    centres, surrounds, ids = cer.fig6_centres(trials, "offset")
    pairs = pairs_for(trials)
    pairs.loc[pairs.index[0], "boris_offset_rec_s"] += 0.5
    with pytest.raises(ValueError, match="not a centre"):
        cer.relabeled_centres(pairs, "offset", centres, surrounds, ids)
    pairs = pairs_for(trials)
    pairs.loc[pairs.index[0], "recording"] = "elsewhere"
    with pytest.raises(ValueError, match="outside"):
        cer.relabeled_centres(pairs, "offset", centres, surrounds, ids)


def test_band_is_what_fig6_draws(tmp_path, monkeypatch):
    """per_minute_band gives the curves Fig. 6's plot_conv_rates draws from the same JSON."""
    monkeypatch.setattr(paper_config, "FIGURES", tmp_path)  # its import names its figure file
    rng = np.random.default_rng(3)
    data = {"conv_time": cer.CONV_TIME.tolist(), "jackknifed_eventrate": rng.gamma(2, 0.05, (50, 600)),
            "bootstrapped_baseline_jittered": rng.gamma(2, 0.03, (50, 600))}
    with mpl.rc_context():
        fig6 = importlib.import_module("plot_event_time_rates_winner_loser_chirp")
        fig, ax = plt.subplots()
        fig6.plot_conv_rates(ax, data)
    drawn = [np.asarray(line.get_ydata()) for line in ax.lines]
    plt.close(fig)
    for key in ("jackknifed_eventrate", "bootstrapped_baseline_jittered"):
        med = cer.per_minute_band(data[key])[1]
        assert any(np.array_equal(med, y) for y in drawn), key


def test_shift_density_integrates_to_one():
    import plot_chase_validation as pcv

    x = np.linspace(-20, 20, 4001)
    d = pcv.kde(np.array([-1.2, -1.0, 0.3]), x, pcv.KDE_BW_S)
    assert np.trapezoid(d, x) == pytest.approx(1.0, abs=1e-6)
    w = pcv.window_means(np.round(np.arange(-3, 3, 0.1), 6), np.r_[np.zeros(30), np.ones(30)])
    assert (w["rate_2s_before"], w["rate_2s_after"]) == (0.0, 1.0)
