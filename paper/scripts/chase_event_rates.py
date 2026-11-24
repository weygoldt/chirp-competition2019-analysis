"""Loser chirp rates around chase onsets and offsets exactly as in Fig. event_correlations, for other chase times.

Fig. event_correlations (Fig. 6; ``plot_event_time_rates_winner_loser_chirp.py``) draws the JSONs of
``analyze_event_time_rates_trial.py --ag-events``: per trial a Gaussian kernel (sigma 0.3 s) over the loser
chirps relative to each chase onset (``lchirps@agon``) or offset (``lchirps@agoff``), pooled over the trials
weighted by their number of chases; 1000 resamples that leave 10 % of the trials out (``jackknifed_eventrate``)
and a null of chases jittered by +-120 s (``bootstrapped_baseline_jittered``); the figure shows the median and
the 2.5th-97.5th percentiles of both, per minute. This module repeats that computation for other chase times
with the functions of ``analyze_event_time_rates_trial`` (trials, loading, resampling), so that the timing
validation (``chase_validation_rates.py``, ``plot_chase_validation.py``) compares like with like.

The trial selection and the loop over centrings of ``analyze_event_time_rates_trial.main`` are not functions
of that module; ``chase_trials`` and ``fig6_centres`` repeat them, and ``check_against_fig6`` verifies the
result against the JSONs (identical trials, lags and per-trial rates). ``replay_fig6_draws`` puts numpy's
global random state where ``main`` had it when it started resampling a centring, so a rate computed after it
leaves out the same trials in every resample as Fig. 6 did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_event_time_rates_trial as etr
from paper_config import DATA_ROOT, PROCESSED, SEED

ENDS = ("onset", "offset")
CENTRING = {"onset": "agon", "offset": "agoff"}  # the centring names of analyze_event_time_rates_trial
# analyze_event_time_rates_trial.main(ag_events=True): the order in which it resamples the centrings, each
# observed then jittered null; the random draws depend only on the number of trials and of their centres.
AG_EVENTS_ORDER = (("agoff", "lchirps"), ("agoff", "wchirps"), ("agon", "lchirps"), ("agon", "wchirps"),
                   ("contacts", "lchirps"), ("contacts", "wchirps"))
CONV_TIME = np.arange(-etr.MAX_CONV_DT_S, etr.MAX_CONV_DT_S, etr.CONV_T_DT_S)  # as in main
BAND_PERCENTILES = [2.5, 50, 97.5]
KERNEL_SIGMA_S = etr.gaussian_kde_params["sigma"]


def per_minute_band(resamples) -> np.ndarray:
    """(lower, median, upper) per minute over the resamples: the expression of Fig. 6's ``plot_conv_rates``."""
    return np.percentile(resamples, BAND_PERCENTILES, axis=0) * 60


def fig6_json(end: str, processed: Path | None = None) -> dict:
    """The JSON behind Fig. 6's loser panel around chase onsets (B1) or offsets (C1)."""
    path = (PROCESSED if processed is None else processed) / "event_time_rates" / f"lchirps@{CENTRING[end]}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run analyze_event_time_rates_trial.py --ag-events (chain rates)")
    return json.loads(path.read_text())


@dataclass
class ChaseTrials:
    """The trials of Fig. 6's chase panels with their BORIS chases (grid s) and loser chirps."""

    identifiers: list[str]
    onsets: list[np.ndarray]
    offsets: list[np.ndarray]
    loser_chirps: list[np.ndarray]

    def chases(self, end: str) -> list[np.ndarray]:
        return self.onsets if end == "onset" else self.offsets


def chase_trials(data_root: Path = DATA_ROOT) -> ChaseTrials:
    """Trials and times as ``analyze_event_time_rates_trial.main(ag_events=True)`` collects them."""
    summary = pd.read_csv(data_root / "trial_summary.csv")
    out = ChaseTrials([], [], [], [])
    for i in range(len(summary)):
        row = summary.iloc[i]
        path = data_root / row["recording"]
        if not etr.is_good_trial(path, row, skip_ag_events=False):
            continue
        _, ag_on, ag_off, loser_chirps, *_ = etr.load_data(path, row, skip_ag_events=False)
        out.identifiers.append(row["recording"])
        out.onsets.append(np.asarray(ag_on))
        out.offsets.append(np.asarray(ag_off))
        out.loser_chirps.append(loser_chirps)
    return out


def in_active_phase(times) -> np.ndarray:
    times = np.asarray(times)
    start, end = etr.ACTIVE_PHASE_S
    return times[(times >= start) & (times < end)]


def drop_trials_without_centres(centres, surrounds, identifiers):
    """main drops the trials without centres: nothing to centre around."""
    keep = [i for i in range(len(centres)) if len(centres[i]) > 0]
    return [centres[i] for i in keep], [surrounds[i] for i in keep], [identifiers[i] for i in keep]


def fig6_centres(trials: ChaseTrials, end: str):
    """Centres (chases in the active phase), surrounds (loser chirps) and trials of Fig. 6's panel."""
    centres = [in_active_phase(c) for c in trials.chases(end)]
    return drop_trials_without_centres(centres, list(trials.loser_chirps), list(trials.identifiers))


def shifted_centres(centres: list[np.ndarray], shift_s: float) -> list[np.ndarray]:
    """Every centre moved by ``shift_s``; the set of chases stays Fig. 6's (selected on the annotated times)."""
    return [np.asarray(c) + shift_s for c in centres]


def relabeled_centres(pairs: pd.DataFrame, end: str, centres, surrounds, identifiers, column: str | None = None):
    """The re-labeled times of the matched chases whose ``end`` is not censored, per trial of Fig. 6's panel.

    ``column`` is the time column of ``pairs.csv`` (default the author's, ``<end>_rec_s``; the BORIS times of
    the same chases: ``boris_<end>_rec_s``). Every pair's BORIS time must be one of the panel's centres. Times
    outside the active phase are dropped as in main, then the trials without re-labeled chases. Returns
    centres, surrounds and identifiers in the order of Fig. 6's trials.
    """
    column = f"{end}_rec_s" if column is None else column
    ok = pairs[pairs[f"d_{end}_s"].notna()]
    unknown = sorted(set(ok["recording"]) - set(identifiers))
    if unknown:
        raise ValueError(f"re-labeled chases in trials outside Fig. 6's {end} panel: {unknown}")
    out = []
    for c, rec in zip(centres, identifiers):
        mine = ok[ok["recording"] == rec]
        boris = mine[f"boris_{end}_rec_s"].to_numpy(float)
        found = np.isclose(boris[:, None], np.asarray(c)[None, :], rtol=0, atol=1e-9).any(axis=1)
        if not found.all():
            raise ValueError(f"{rec}: {int((~found).sum())} re-labeled chases whose BORIS {end} is not a centre "
                             "of Fig. 6 (pairs.csv out of date?)")
        out.append(np.sort(in_active_phase(mine[column].to_numpy(float))))
    return drop_trials_without_centres(out, list(surrounds), list(identifiers))


def resampled_rate(centres, surrounds):
    """Fig. 6's observed rate: (resamples, per-trial rates, per-trial lags) in chirps/s per centre.

    ``analyze_event_time_rates_trial.observed_rate_bootstrap`` with main's grid, number of resamples and
    leave-out fraction; it draws from numpy's global random state (seed it, or ``replay_fig6_draws``).
    """
    return etr.observed_rate_bootstrap(centres, surrounds, CONV_TIME, n_bootstraps=etr.N_RESAMPLES,
                                       leave_out_frac=etr.LEAVE_OUT_FRAC)


def jittered_null(centres, surrounds) -> np.ndarray:
    """Fig. 6's null (centres jittered by +-120 s): resamples in chirps/s per centre."""
    return etr.null_rate_bootstrap(centres, surrounds, active_phase=etr.ACTIVE_PHASE_S, conv_time=CONV_TIME,
                                   n_bootstraps=etr.N_RESAMPLES, leave_out_frac=etr.LEAVE_OUT_FRAC,
                                   null_mode="jittered_centers", jitter_amplitude_s=etr.CENTER_JITTER_AMPLITUDE_S)


def _replay_observed(n_trials: int) -> None:
    for _ in range(etr.N_RESAMPLES):
        etr._leave_out_indices(n_trials, etr.LEAVE_OUT_FRAC)


def _replay_null(n_centres: list[int]) -> None:
    for _ in range(etr.N_RESAMPLES):
        for i in etr._leave_out_indices(len(n_centres), etr.LEAVE_OUT_FRAC):
            if n_centres[i]:
                np.random.uniform(-etr.CENTER_JITTER_AMPLITUDE_S, etr.CENTER_JITTER_AMPLITUDE_S, n_centres[i])


def replay_fig6_draws(trials: ChaseTrials, end: str) -> None:
    """Seed numpy's global random state as main does and replay its draws up to the observed resampling of
    the loser chirps around ``end`` (the draws need only the number of trials and of their centres)."""
    target = (CENTRING[end], "lchirps")
    np.random.seed(SEED)
    for centring in AG_EVENTS_ORDER:
        if centring == target:
            return
        src = trials.onsets if centring[0] == "agon" else trials.offsets if centring[0] == "agoff" else None
        if src is None:
            raise ValueError(f"cannot replay the draws of {centring}")
        n = [len(c) for c in (in_active_phase(x) for x in src) if len(c)]
        _replay_observed(len(n))
        _replay_null(n)
    raise ValueError(f"{target} is not resampled by analyze_event_time_rates_trial --ag-events")


def check_against_fig6(trials: ChaseTrials, end: str, fig6: dict, n_resamples: int = 20) -> dict:
    """Raise unless the chases and chirps give Fig. 6's panel: trials, all lags and per-trial rates
    identical, and the first ``n_resamples`` resamples identical after ``replay_fig6_draws``."""
    centres, surrounds, ids = fig6_centres(trials, end)
    if ids != fig6["trial_identifiers"]:
        raise AssertionError(f"{end}: trials differ from Fig. 6 ({ids} vs {fig6['trial_identifiers']})")
    replay_fig6_draws(trials, end)
    jk, per_trial, lags = etr.observed_rate_bootstrap(centres, surrounds, CONV_TIME, n_bootstraps=n_resamples,
                                                      leave_out_frac=etr.LEAVE_OUT_FRAC)
    checks = {
        "lags": all(np.array_equal(a, np.asarray(b)) for a, b in zip(lags, fig6["centered_times"])),
        "per_trial_rates": np.array_equal(np.array(per_trial), np.array(fig6["per_trial_rates"])),
        "resamples": np.array_equal(jk, np.array(fig6["jackknifed_eventrate"][:n_resamples])),
        "conv_time": np.array_equal(CONV_TIME, np.array(fig6["conv_time"])),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"{end}: not Fig. 6's computation ({', '.join(failed)} differ from lchirps@"
                             f"{CENTRING[end]}.json; rerun chain rates?)")
    return {"n_trials": len(ids), "n_chases": int(sum(len(c) for c in centres)), "n_resamples_checked": n_resamples}
