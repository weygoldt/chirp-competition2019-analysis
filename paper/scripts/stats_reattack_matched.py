"""Matched chirp-triggered comparison of the winner's re-attack (Fig. 8 B; supplement Fig. receiver_matched_checks).

The re-attack hazard (``stats_reattack_hazard.py``) compares the winner's rate of re-attack per second at risk
before and after the loser's first chirp. This step makes the same comparison as an event-triggered average
(``reattack_matched.py``), so that it can be drawn as curves against time since the chirp:

Triggers
    the loser's first chirp after a chase offset in an interval in which the loser had not chirped since
    max(offset - 5 s, chase onset): exactly the moments at which the hazard model's ``chirped`` switches from 0
    to 1 after the offset. At most one per interval. (An interval whose first counted chirp falls in the
    chase's last 5 s is chirped from the offset on and has no trigger.)
Matched moments (controls)
    for a trigger s seconds after its offset in competition c: the same time s after the offset in every
    other interval of c in which, at s, the winner had not yet attacked again and the loser had not yet
    chirped (since max(offset - 5 s, onset)). Only what is known at s decides; nothing that happens later
    (the loser may chirp later, the winner may attack at once). A matched moment leaves the risk set when its
    own loser chirps (censored, as the hazard model moves that time to ``chirped`` = 1). Every eligible
    moment is used, with weight 1 / (number eligible) per trigger; no sampling. A trigger without any eligible
    moment (late in a long interval) is dropped.
Follow-up
    from the trigger (or matched moment) to the next chase onset (the event) or the censoring (end of the
    3-h window; for matched moments also their own first chirp).

Estimators: weighted Kaplan-Meier "not yet re-attacked"; the kernel-smoothed re-attack rate (Fig. 6's
Gaussian kernel shape, SD ``SIGMA_S``, reflected at 0) per interval at risk; events per minute at risk in
[0, T] after the trigger and their ratio (triggers / matched) for T = 1, 10, 30 s; intervals: 2.5-97.5 % of
``N_BOOT`` resamples of the competitions with replacement (bootstrap).

Checks (same estimators): every loser chirp between the offset and the next onset as a trigger; the
selection trap (matched intervals required to stay silent until their re-attack: conditions on the future);
surrogates: loser chirp trains shuffled among the intervals of a competition, keeping their time since the
offset (no effect possible), and re-attacks simulated with a planted hazard ratio of 1, 0.5 and 2
(``N_SURROGATES`` replicates each).

Intervals: those of the primary hazard model (every inter-chase interval with a previous one; checked
against ``reattack_hazard/intervals.csv`` and the windowed models' obs.csv).

Outputs under ``$PAPER_OUT/data/processed/reattack_matched/``: ``curves.npz`` (grid, curves and bands),
``summary.json`` (the numbers), ``triggers_<design>.csv`` (every trigger, its time since the offset and its
number of matched moments; dropped ones with 0), ``surrogates.csv`` (one row per replicate),
``loser_chirps.csv`` (for the raster of Fig. 8 A); the numbers to ``write_stats("reattack_matched")`` only
with the default settings.

    paper/run.sh stats_reattack_matched.py          # chain reattack, after stats_reattack_hazard.py (~2-3 min)
    paper/run.sh stats_reattack_matched.py --n-boot 200 --n-surrogates 10    # quick look, no stats
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import reattack_hazard as rh
import reattack_matched as rm
from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import DATA_ROOT, PROCESSED, SEED, processed_dir

NAMESPACE = "reattack_matched"
HAZARD_DIR = PROCESSED / "reattack_hazard"
OBS_DATA = PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data" / "chirp_effect_on_dominant_obs.csv"

# Kernel SD of the smoothed rate (s). Fig. 6 uses 0.3 s, but it smooths thousands of chirps; there are ~280
# re-attacks in the 30 s after the first-chirp triggers, and at 0.3 s the curve is mostly noise.
SIGMA_S = 1.0
N_BOOT = 1000
N_SURROGATES = 100
DEFAULTS = {"n_boot": N_BOOT, "n_surrogates": N_SURROGATES, "sigma": SIGMA_S, "seed": SEED}
WINDOWS_S = (1.0, 10.0, 30.0)  # rate ratios in [0, T] after the trigger
SURVIVAL_AT_S = (10.0, 30.0)
BAND = (2.5, 97.5)
APPEASE_HR = 0.5  # Fig. 8 B's dotted reference (plot_receiver_before_after.APPEASE_HR)
DESIGNS = {  # name: (triggers, control rule)
    "first_chirp": ("first", "unchirped_at_s"),
    "all_chirps": ("all", "unchirped_at_s"),
    "trap": ("first", "never_chirped"),
}
SCENARIOS = {"shuffled": None, "sim_hr_1": 1.0, "sim_hr_0p5": 0.5, "sim_hr_2": 2.0}  # planted HR (None: shuffle)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def check_against_saved(iv: pd.DataFrame, saved: pd.DataFrame, atol: float = 1e-6) -> None:
    """Raise unless the intervals are those of the hazard fits (``reattack_hazard/intervals.csv``)."""
    if len(iv) != len(saved) or not (iv[rh.COMP].astype(str).to_numpy()
                                     == saved[rh.COMP].astype(str).to_numpy()).all():
        raise ValueError("intervals differ from the hazard fits' intervals.csv (rerun stats_reattack_hazard.py)")
    for col in ("onset", "offset", "end", "length", "first_chirp"):
        a, b = iv[col].to_numpy(float), saved[col].to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)) or np.nanmax(np.abs(a - b)) > atol:
            raise ValueError(f"{col} differs from the hazard fits' intervals.csv (rerun stats_reattack_hazard.py)")
    if not (iv["event"].to_numpy(int) == saved["event"].to_numpy(int)).all():
        raise ValueError("event differs from the hazard fits' intervals.csv (rerun stats_reattack_hazard.py)")


def load():
    """Window, the primary model's intervals and the loser chirps (recording -> times)."""
    window = offset_window()
    comps = rh.load_competitions(DATA_ROOT)
    iv = rh.intervals_table(comps, window)
    check_against_saved(iv, pd.read_csv(HAZARD_DIR / "intervals.csv"))
    rh.check_against_obs(iv, pd.read_csv(OBS_DATA))
    chirps = {c.recording: c.loser_chirps for c in comps}
    return window, iv[iv["prev_interval"].notna()], chirps


# ---------------------------------------------------------------------------
# one design: follow-ups, estimators, resamples
# ---------------------------------------------------------------------------
def run_design(iv: pd.DataFrame, triggers: pd.DataFrame, rule: str, sigma: float, n_boot: int,
               seed: int) -> dict:
    f = rm.matched_followups(iv, triggers, rule=rule)
    sm = rm.summarise(f, competitions=sorted(iv[rh.COMP].unique()), sigma=sigma, windows=WINDOWS_S)
    at = np.asarray(SURVIVAL_AT_S)
    W = rm.resample_weights(len(sm.competitions), n_boot, seed=seed, mode="bootstrap")
    boot = rm.resampled(sm, W, survival_at=at)
    boot_curves = rm.resampled(sm, W)
    return {"triggers": triggers, "followups": f, "summary": sm, "windows": rm.window_rates(sm),
            "survival_at": rm.survival(sm, at=at), "boot": boot,
            "curves": {"hazard": rm.hazard_per_min(sm), "survival": rm.survival(sm)},
            "bands": {k: {g: np.nanpercentile(boot_curves[k][g], BAND, axis=0) for g in rm.GROUPS}
                      for k in ("hazard", "survival")}}


def _q(x) -> list[float]:
    return [float(v) for v in np.nanpercentile(np.asarray(x, float), BAND)]


def _iqr(x) -> list[float]:
    return [float(v) for v in np.percentile(np.asarray(x, float), [25, 75])]


def design_numbers(res: dict) -> dict:
    """The numbers of one design (write_stats-ready)."""
    f, trg = res["followups"], res["triggers"]
    t, c = f[f["group"] == rm.TRIGGER], f[f["group"] == rm.CONTROL]
    used = set(t["set"])
    dropped = trg.loc[[i for i in trg.index if i not in used], "s"]
    per = c.groupby("set").size()
    out = {
        "n_candidates": int(len(trg)), "n_triggers": int(len(t)), "n_dropped_no_match": int(len(dropped)),
        "dropped_s_min": float(dropped.min()) if len(dropped) else 0.0,
        "n_competitions": int(t[rh.COMP].nunique()),
        "n_trigger_intervals": int(t["interval"].nunique()),
        "n_matched_moments": int(len(c)), "n_matched_intervals": int(c["interval"].nunique()),
        "matched_per_trigger": {"median": float(per.median()), "iqr": _iqr(per), "min": int(per.min()),
                                "max": int(per.max())},
        "trigger_s": {"median": float(t["s"].median()), "iqr": _iqr(t["s"])},
        # matched moments that stopped counting because their own loser chirped (all; within the 30 s shown),
        # and their share of the matched weight
        "n_matched_stopped_by_chirp": int(c["by_chirp"].sum()),
        "n_matched_stopped_by_chirp_30s": int(((c["by_chirp"] == 1) & (c["exit"] <= rm.HORIZON_S)).sum()),
        "frac_matched_weight_stopped_by_chirp_30s": float(
            (c["weight"] * ((c["by_chirp"] == 1) & (c["exit"] <= rm.HORIZON_S))).sum() / c["weight"].sum()),
    }
    for k, T in enumerate(WINDOWS_S):
        w = res["windows"][f"{T:g}"]
        out[f"w{T:g}"] = {"ratio": float(w["ratio"]), "ratio_ci": _q(res["boot"]["ratio"][:, k]),
                          "rate_trigger_per_min": float(w["rate_per_min"][rm.TRIGGER]),
                          "rate_matched_per_min": float(w["rate_per_min"][rm.CONTROL]),
                          "events_trigger": float(w["events"][rm.TRIGGER]),
                          "events_matched": float(w["events"][rm.CONTROL])}
    out["survival"] = {}
    for k, T in enumerate(SURVIVAL_AT_S):
        out["survival"][f"t{T:g}"] = {
            "trigger": float(res["survival_at"][rm.TRIGGER][k]),
            "trigger_ci": _q(res["boot"]["survival"][rm.TRIGGER][:, k]),
            "matched": float(res["survival_at"][rm.CONTROL][k]),
            "matched_ci": _q(res["boot"]["survival"][rm.CONTROL][:, k]),
            # the reference curve of Fig. 8 B: what a chirp that halved the re-attack rate would give
            "if_hr_0p5": float(res["survival_at"][rm.CONTROL][k] ** APPEASE_HR)}
    return out


def triggers_table(res: dict) -> pd.DataFrame:
    """Every trigger with its time since the offset and its number of matched moments (0: dropped)."""
    f, trg = res["followups"], res["triggers"]
    n = f[f["group"] == rm.CONTROL].groupby("set").size()
    return trg.assign(n_matched=n.reindex(trg.index, fill_value=0).astype(int).to_numpy())


# ---------------------------------------------------------------------------
# surrogates
# ---------------------------------------------------------------------------
def mh_ratio(iv: pd.DataFrame, chirps: dict) -> float:
    """Mantel-Haenszel rate ratio of ``chirped`` over the person-period table (competitions x pieces): the
    frequentist analogue of the unadjusted hazard model."""
    pp = rh.person_period(iv, chirps, covariates=("chirped",))
    c = rh.competition_rate_ratios(rh.cell_table(pp, "chirped"))
    return float(c["num"].sum() / c["den"].sum())


def surrogates(iv: pd.DataFrame, chirps: dict, n: int, seed: int, left: float) -> pd.DataFrame:
    """Per replicate of every scenario: the matched ratios of the design and the trap in [0, 10] and [0, 30] s,
    the person-period Mantel-Haenszel ratio and the number of triggers."""
    rng = np.random.default_rng(seed)
    tiny = np.array([0.0, 1.0])  # window ratios only: no curves needed
    rows = []
    for name, hr in SCENARIOS.items():
        for rep in range(n):
            x = (rm.shuffle_first_chirps(iv, chirps, rng, left=left) if hr is None
                 else rm.simulate_reattacks(iv, hr, rng))
            trg = rm.first_chirp_triggers(x)
            row = {"scenario": name, "planted": 1.0 if hr is None else hr, "rep": rep, "n_triggers": len(trg)}
            for label, rule in (("design", "unchirped_at_s"), ("trap", "never_chirped")):
                w = rm.window_rates(rm.summarise(rm.matched_followups(x, trg, rule=rule), grid=tiny))
                row[f"{label}_w10"], row[f"{label}_w30"] = w["10"]["ratio"], w["30"]["ratio"]
            row["mh"] = mh_ratio(x, chirps)
            rows.append(row)
    return pd.DataFrame(rows)


def surrogate_numbers(sur: pd.DataFrame) -> dict:
    out = {}
    for name, g in sur.groupby("scenario", sort=False):
        d = {"planted": float(g["planted"].iloc[0]), "n_replicates": int(len(g)),
             "n_triggers_median": float(g["n_triggers"].median())}
        for col in ("design_w10", "design_w30", "trap_w10", "trap_w30", "mh"):
            v = g[col].to_numpy(float)
            if not np.isfinite(v).all():
                raise ValueError(f"{name}: non-finite {col} in a replicate")
            d[col] = {"median": float(np.median(v)), "q": _q(v)}
        out[name] = d
    return out


# ---------------------------------------------------------------------------
def save_curves(path, results: dict, grid: np.ndarray) -> None:
    arrays = {"grid": grid}
    for name in ("first_chirp", "all_chirps"):
        res = results[name]
        for est in ("hazard", "survival"):
            for g in rm.GROUPS:
                arrays[f"{name}_{est}_{g}"] = res["curves"][est][g]
                arrays[f"{name}_{est}_{g}_lo"], arrays[f"{name}_{est}_{g}_hi"] = res["bands"][est][g]
    np.savez(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-surrogates", type=int, default=N_SURROGATES)
    ap.add_argument("--sigma", type=float, default=SIGMA_S, help="kernel SD of the smoothed rate (s)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    settings = {"n_boot": args.n_boot, "n_surrogates": args.n_surrogates, "sigma": args.sigma, "seed": args.seed}

    t0 = time.time()
    window, iv, chirps = load()
    trig = {"first": rm.first_chirp_triggers(iv), "all": rm.all_chirp_triggers(iv, chirps)}
    results = {name: run_design(iv, trig[t], rule, args.sigma, args.n_boot, args.seed)
               for name, (t, rule) in DESIGNS.items()}
    print(f"matched designs: {time.time() - t0:.0f} s", flush=True)
    sur = surrogates(iv, chirps, args.n_surrogates, args.seed, window[0])
    print(f"surrogates: {time.time() - t0:.0f} s", flush=True)

    sw = rm.switch_times(iv)
    values = {
        "window_s": list(window),
        "settings": {"kernel_sigma_s": args.sigma, "n_boot": args.n_boot, "n_surrogates": args.n_surrogates,
                     "seed": args.seed, "horizon_s": rm.HORIZON_S, "band_percentiles": list(BAND)},
        "n_intervals": int(len(iv)), "n_competitions": int(iv[rh.COMP].nunique()),
        "n_chirped_at_offset": int((sw == 0).sum()),
        "n_chirped_after_offset": int((np.isfinite(sw) & (sw > 0)).sum()),
        "n_never_chirped": int((~np.isfinite(sw)).sum()),
        "n_loser_chirps_between_chases": int(len(trig["all"])),
        **{name: design_numbers(res) for name, res in results.items()},
        "mh_ratio_person_period": mh_ratio(iv, chirps),
        "surrogates": surrogate_numbers(sur),
    }

    out = processed_dir("reattack_matched")
    save_curves(out / "curves.npz", results, results["first_chirp"]["summary"].grid)
    (out / "summary.json").write_text(json.dumps(values, indent=1))
    for name in ("first_chirp", "all_chirps"):
        triggers_table(results[name]).to_csv(out / f"triggers_{name}.csv", index=False)
    sur.to_csv(out / "surrogates.csv", index=False)
    pd.DataFrame([(rec, t) for rec, ts in chirps.items() for t in ts], columns=[rh.COMP, "time_s"]).to_csv(
        out / "loser_chirps.csv", index=False)
    print(json.dumps({k: values[k] for k in ("first_chirp", "trap")}, indent=1))
    if settings == DEFAULTS:
        write_stats(NAMESPACE, values)
    else:
        print(f"settings {settings} differ from the defaults {DEFAULTS}: stats not written")
    print(f"outputs in {out}; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
