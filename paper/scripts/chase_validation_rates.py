"""Loser chirp rates around chase onsets and offsets for the timing-validation figure: Fig. 6's rates
(panels B1, C1) for two other sets of chase times.

New step. Fig. event_correlations (Fig. 6) shows the loser chirp rate around the BORIS onsets and offsets of
every chase in the first 3 h (``analyze_event_time_rates_trial.py --ag-events``). The same computation
(``chase_event_rates``: kernel, trials, weighting by chases, 1000 resamples leaving 10 % of the trials out)
for

``shifted``           every chase of Fig. 6 moved by the median re-labeling shift (``chase_validation``
                      stats: ``onset.median_s``, ``offset.median_s``). The resamples replay Fig. 6's random
                      draws (the same trials left out), so the curve is Fig. 6's curve moved by the shift.
                      No chases are merged, as in Fig. 6.
``relabeled``         the author's onsets / offsets of the matched chases with that end uncensored
                      (``pairs.csv``), in the trials that have such chases (the others are dropped, as Fig. 6
                      drops trials without chases); each trial weighted by its number of re-labeled chases.
``annotated_subset``  the BORIS times of the same matched chases (same trials and resampling draws as
                      ``relabeled``; for the stats).

plus the jittered null of the re-labeled subset (for the stats; the figure shows Fig. 6's null). Before
that, ``chase_event_rates.check_against_fig6`` verifies that the chases and chirps give Fig. 6's panels
(identical trials, lags, per-trial rates and first resamples).

Inputs: ``event_time_rates/lchirps@agon.json``, ``lchirps@agoff.json`` (chain rates),
``chase_validation/pairs.csv`` and the stats ``chase_validation`` (chain validation). Output:
``chase_validation/event_rates.npz`` (resamples in chirps/s per chase on ``conv_time``). ~8 min.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

import chase_event_rates as cer
from paper_config import OUT, PROCESSED, SEED, processed_dir

CURVES = ("shifted", "relabeled", "annotated_subset")


def median_shifts(pairs: pd.DataFrame) -> dict[str, float]:
    """The median shifts of the stats ``chase_validation``, checked against ``pairs.csv``."""
    path = OUT / "stats" / "chase_validation.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run chase_validation.py (chain validation)")
    values = json.loads(path.read_text())["values"]
    shifts = {end: float(values[end]["median_s"]) for end in cer.ENDS}
    for end, s in shifts.items():
        if not np.isclose(s, pairs[f"d_{end}_s"].median(), rtol=0, atol=1e-12):
            raise ValueError(f"{path}: {end}.median_s {s} is not the median of pairs.csv; rerun chase_validation.py")
    return shifts


def main() -> None:
    logging.disable(logging.INFO)  # analyze_event_time_rates_trial logs every resample
    pairs = pd.read_csv(PROCESSED / "chase_validation" / "pairs.csv")
    shifts = median_shifts(pairs)
    trials = cer.chase_trials()
    out: dict[str, np.ndarray] = {"conv_time": cer.CONV_TIME}
    for end in cer.ENDS:
        print(end, cer.check_against_fig6(trials, end, cer.fig6_json(end)), "= Fig. 6", flush=True)
        centres, surrounds, ids = cer.fig6_centres(trials, end)
        sets = {
            "shifted": (cer.shifted_centres(centres, shifts[end]), surrounds, ids),
            "relabeled": cer.relabeled_centres(pairs, end, centres, surrounds, ids),
            "annotated_subset": cer.relabeled_centres(pairs, end, centres, surrounds, ids,
                                                      column=f"boris_{end}_rec_s"),
        }
        for name in CURVES:
            c, s, i = sets[name]
            if name == "shifted":
                cer.replay_fig6_draws(trials, end)  # Fig. 6's draws: its curve, moved
            else:
                np.random.seed(SEED)  # the same draws for both subsets (same trials)
            jk, per_trial, _ = cer.resampled_rate(c, s)
            out[f"{end}_{name}_resamples"] = jk
            out[f"{end}_{name}_per_trial_rates"] = np.array(per_trial)
            out[f"{end}_{name}_n_chases"] = np.array([len(x) for x in c])
            out[f"{end}_{name}_trials"] = np.array(i)
            print(f"{end} {name}: {sum(len(x) for x in c)} chases in {len(i)} trials", flush=True)
        out[f"{end}_shift_s"] = np.array(shifts[end])
        np.random.seed(SEED)
        out[f"{end}_relabeled_null_resamples"] = cer.jittered_null(*sets["relabeled"][:2])
    path = processed_dir("chase_validation") / "event_rates.npz"
    np.savez_compressed(path, **out)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
