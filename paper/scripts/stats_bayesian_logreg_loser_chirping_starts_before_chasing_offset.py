"""Does loser chirping start before chasing offset more (or less) often than its timing alone predicts?

New step; replaces ``stats_bayesian_logreg_loser_chirps_before_vs_after_chasing_offset.py`` (namespace
``before_offset_logreg``). That test coded an offset as "pre-offset" when any loser chirp fell in the
10 s before it and compared the share with 50 %. But with k chirps placed at random around the offset,
at least one falls before it with probability 1 - 0.5^k (75 % for k = 2), so 50 % is no reference, and
10 s before an offset is mostly chase.

Test: for each chasing offset x with at least one loser chirp in [x - W, x + W), the loser's chirping
"starts before the offset" (y = 1) when a chirp falls in [x - W, x), also when chirps follow (the
first chirp is before x); y = 0 when all chirps are in [x, x + W). With k chirps in the window and their
times symmetric around the offset, P(y = 1) = 1 - 0.5^k. Bernoulli GLMM with that expectation as an
offset term and a random intercept per recording:

    logit P(y_i = 1) = logit(1 - 0.5^k_i) + beta_0 + u_rec[i],

so exp(beta_0) is the odds ratio of starting before the offset relative to symmetric timing, for a
typical recording: > 1 chirping starts before the offset more often than its timing predicts (as if it
could end the chase), < 1 it starts after it more often.

Three timings of the offsets: ``boris`` (the BORIS offsets, all analysed chases), ``author`` (the
author's re-labelled offsets, ``chase_validation.py``), ``shifted`` (the BORIS offsets shifted by the
median author - BORIS offset shift, all chases); windows W = 1, 2, 3, 5, 10 s (2 s primary).

Chases as in the other offset analyses (``make_chirps_around_chasing_obs.window_counts``): trials that
pass ``is_bad_trial``, chases and chirps before 3 h, overlapping chases merged. The author's offsets:
matched pairs with an uncensored offset whose BORIS chase is in that set.

Outputs: ``$PAPER_OUT/data/processed/offset_chirp_start/`` (events per timing and window, traces of the
fits) and the stats namespace ``offset_chirp_start``. Run after ``chase_validation.py``.
"""

from __future__ import annotations

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from analyze_event_time_rates_trial import load_and_convert_boris_events
from chirps import load_chirps
from manuscript_stats import write_stats
from paper_config import DATA_ROOT, PROCESSED, processed_dir
from summarize_submissive_chirp_effect_on_dominant import _merge_overlapping_chases, is_bad_trial

T_MAX_S = 3 * 60 * 60
WINDOWS_S = (1, 2, 3, 5, 10)
PRIMARY_W = 2
HDI_PROB = 0.95
SEED = 20250101
SAMPLER = dict(tune=6000, draws=1000, chains=10, target_accept=0.99)  # as the replaced model


def analysed_offsets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Merged BORIS offsets and loser chirps (both < 3 h) per recording."""
    summary = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    offsets, chirps = {}, {}
    for _, trial in summary.iterrows():
        if is_bad_trial(trial, DATA_ROOT):
            continue
        rec = trial["recording"]
        folder = DATA_ROOT / rec
        _, on_off, _, _ = load_and_convert_boris_events(folder, rec, sr=20_000)
        on, off = on_off[:, 0], on_off[:, 1]
        keep = (on < T_MAX_S) & (off < T_MAX_S)
        _, merged_off = _merge_overlapping_chases(on[keep], off[keep], rec)
        times, ids = load_chirps(folder)
        offsets[rec] = np.asarray(merged_off, float)
        chirps[rec] = np.sort(times[(ids == trial["lose_ID"]) & (times < T_MAX_S)])
    return offsets, chirps


def events(offs: list[tuple[str, float]], chirps: dict[str, np.ndarray], w: float) -> pd.DataFrame:
    """Offsets with a loser chirp in [x - w, x + w): chirp counts before / after and the outcome."""
    rows = []
    for rec, x in offs:
        c = chirps[rec]
        before = int(np.sum((c >= x - w) & (c < x)))
        after = int(np.sum((c >= x) & (c < x + w)))
        k = before + after
        if k:
            p0 = 1 - 0.5 ** k
            rows.append({"recording": rec, "offset_s": x, "before": before, "after": after, "k": k,
                         "y": int(before > 0), "p0": p0, "lp0": float(np.log(p0 / (1 - p0)))})
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame):
    model = bmb.Model("y ~ 1 + offset(lp0) + (1|recording)", df, family="bernoulli")
    return model.fit(random_seed=SEED, progressbar=False, **SAMPLER)


def summarize(df: pd.DataFrame, trace, n_offsets: int) -> dict:
    """Event counts and the odds ratio exp(beta_0): median, 95 % HDI, P(> 1).

    The HDI is taken on the odds-ratio scale (of exp(beta_0)), the convention of every other
    multiplier in the manuscript (``summarize_reported_posteriors``, ``stats_chirp_coupling``,
    ``stats_reattack_hazard``); HDIs are not invariant under exp, and exp of the log-odds HDI made
    the re-labelled offsets' interval include 1 where the odds-ratio HDI excludes it (2026-11-27).
    """
    b0 = trace.posterior["Intercept"].to_numpy().ravel()
    lo, hi = az.hdi(np.exp(b0), hdi_prob=HDI_PROB).tolist()
    div = int(trace.sample_stats["diverging"].to_numpy().sum())
    return {"n_offsets": int(n_offsets), "n_events": int(len(df)), "n_recordings": int(df["recording"].nunique()),
            "observed_fraction": float(df["y"].mean()), "expected_fraction": float(df["p0"].mean()),
            "n_chirps_before": int(df["before"].sum()), "n_chirps_after": int(df["after"].sum()),
            "odds_ratio": {"median": float(np.exp(np.median(b0))), "hdi": [float(lo), float(hi)],
                           "p_gt_1": float(np.mean(b0 > 0))},
            "divergences": div}


def main() -> None:
    offsets, chirps = analysed_offsets()
    pairs = pd.read_csv(PROCESSED / "chase_validation" / "pairs.csv")
    shift = float(pairs["d_offset_s"].median())
    author = pairs[pairs["d_offset_s"].notna() & pairs["in_analysis_window"]]
    missing = set(author["recording"]) - set(offsets)
    if missing:
        raise ValueError(f"re-labelled chases in recordings outside the analysis: {sorted(missing)}")
    timings = {
        "boris": [(r, x) for r, xs in offsets.items() for x in xs],
        "author": list(zip(author["recording"], author["offset_rec_s"])),
        "shifted": [(r, x + shift) for r, xs in offsets.items() for x in xs],
    }
    out = processed_dir("offset_chirp_start")
    values = {"windows_s": list(WINDOWS_S), "primary_window_s": PRIMARY_W, "offset_shift_s": shift,
              "n_author_offsets": len(timings["author"])}
    for name, offs in timings.items():
        values[name] = {}
        for w in WINDOWS_S:
            df = events(offs, chirps, w)
            df.to_csv(out / f"{name}_w{w}_events.csv", index=False)
            trace = fit(df)
            trace.to_netcdf(out / f"{name}_w{w}_trace.nc")
            s = summarize(df, trace, len(offs))
            values[name][f"w{w}"] = s
            print(f"{name:8s} +-{w:2d} s: {s['n_events']:4d} events in {s['n_recordings']} recordings; starts before "
                  f"{s['observed_fraction']:.3f} (expected {s['expected_fraction']:.3f}); odds ratio "
                  f"{s['odds_ratio']['median']:.2f} [{s['odds_ratio']['hdi'][0]:.2f}, {s['odds_ratio']['hdi'][1]:.2f}], "
                  f"P(>1) {s['odds_ratio']['p_gt_1']:.3f}, divergences {s['divergences']}", flush=True)
    write_stats("offset_chirp_start", values)


if __name__ == "__main__":
    main()
