"""Bayesian re-attack hazard: does a loser chirp change the hazard of the winner's next chase onset?

Piecewise-exponential survival model of every inter-chase interval (chase offset -> next onset, the event; the
last interval of a trial censored at the end of the 3-h window), fitted as a Poisson GLMM with log link and
``offset(log(exposure))`` on the person-period table of ``reattack_hazard.py`` (rows split at the baseline
pieces of time since the offset and wherever the chirp covariate changes):

    event_r ~ Poisson(exposure_r * lambda_r)
    log lambda_r = alpha_piece[r] + (beta + b_c) * chirp_r + gamma_1 x_i + gamma_2 d_i + u_c

``chirp_r`` is time-varying (``chirped``: 1 once the loser has chirped, counting from max(offset - 5 s, onset);
see ``reattack_hazard``), x_i the centred log previous inter-chase interval (chasing density), d_i the centred
log duration of the chase that just ended, u_c and b_c a random intercept and a random chirp slope per
competition (recording). exp(beta) is the hazard ratio (HR) of a re-attack once the loser has chirped, in a
typical competition; an appeasement signal predicts HR < 1.

Fits (as the windowed Gamma GLMMs M1-M4 of ``stats_chirp_affects_chasing.py``, there for the next interval):

  unadjusted              chirped                         all intervals              (M1)
  density                 chirped + x                     intervals with a previous  (M3)
  primary                 chirped + x + d                 intervals with a previous  (M4)
  primary_intercept_only  as primary, random intercept only (no random slope)
  recent_1s, recent_5s    >= 1 loser chirp in the preceding 1 s / 5 s, + x + d
  guard                   >= 1 loser chirp 0.5-1.5 s before (the hazard analysis' lag guard), + x + d
  placebo_unadjusted      next_chirped (the windowed predictor of the NEXT chase: measured after the event,
                          so it cannot cause it; shows the density confound, the hazard analogue of M2)
  placebo                 next_chirped + x + d

Priors as the paper's GLMMs: slopes Normal(0, 2); the baseline log hazard (per s) of each piece Normal(0, 5)
(the GLMMs' intercept prior; cell-means coding, so no separate intercept); group-level SDs bambi's defaults
(HalfNormal, printed in the log); group-level effects centred (``NONCENTERED``). Sampler as the paper's
GLMMs (tune 3000, draws 1000, 8 chains, target_accept 0.99, seed 42) on 4 cores (they used 8). Rows that
share every model column are summed first
(``reattack_hazard.aggregate``; the Poisson posterior depends only on these sums).

Outputs under ``$PAPER_OUT/data/processed/reattack_hazard/``: ``intervals.csv`` (one row per interval),
``person_period.parquet`` (every covariate), ``<fit>.nc`` (posteriors), ``hr_summary.csv``; the numbers to
``write_stats("reattack_hazard")`` (only for a full run with the default sampler settings).

    paper/run.sh stats_reattack_hazard.py                      # all fits (chain reattack)
    paper/run.sh stats_reattack_hazard.py --fits primary --tune 300 --draws 200 --chains 4   # timing only
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import typer

import paper_config  # noqa: F401  (fork start method before pymc samples)
import reattack_hazard as rh
from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import DATA_ROOT, PROCESSED, processed_dir

app = typer.Typer(pretty_exceptions_show_locals=False)

NAMESPACE = "reattack_hazard"
OBS_DATA = PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data" / "chirp_effect_on_dominant_obs.csv"

SAMPLER = dict(tune=3000, draws=1000, chains=8, cores=4, target_accept=0.99, random_seed=42)
PRIOR_PIECE = (0.0, 5.0)  # Normal(mu, sigma) on each piece's log hazard per s
PRIOR_SLOPE = (0.0, 2.0)  # Normal on every slope (the paper's "common")
HDI_PROB = 0.95
OK_RHAT, OK_ESS = 1.01, 400  # the windowed fits' flags
# Centred group-level effects: every competition contributes many events, so the likelihood pins each
# competition's effects and the noncentred form (bambi's default) mixes poorly (short chains of the primary:
# half the speed, step sizes differing 4-fold between chains). A fit can override it with "noncentered".
NONCENTERED = False

X, D = "log_prev_c", "log_dur_c"
FITS: dict[str, dict] = {
    "unadjusted": {"chirp": "chirped", "adjust": [], "slope": True, "as": "M1"},
    "density": {"chirp": "chirped", "adjust": [X], "slope": True, "as": "M3"},
    "primary": {"chirp": "chirped", "adjust": [X, D], "slope": True, "as": "M4"},
    "primary_intercept_only": {"chirp": "chirped", "adjust": [X, D], "slope": False, "as": "M4"},
    "recent_1s": {"chirp": "recent_1s", "adjust": [X, D], "slope": True, "as": "M4"},
    "recent_5s": {"chirp": "recent_5s", "adjust": [X, D], "slope": True, "as": "M4"},
    "guard": {"chirp": "guard", "adjust": [X, D], "slope": True, "as": "M4"},
    # noncentred: the placebo's slope SD is near 0, where the centred form has a funnel (short chains: R-hat
    # 1.37 and bulk ESS 9 for the SD, centred; 1.03 and 157, noncentred)
    "placebo_unadjusted": {"chirp": "next_chirped", "adjust": [], "slope": True, "as": "M2", "noncentered": True},
    "placebo": {"chirp": "next_chirped", "adjust": [X, D], "slope": True, "as": "M2", "noncentered": True},
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_tables(window: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Intervals (checked against the windowed models' obs.csv) and the person-period table with every
    covariate, centred interval covariates and the piece labels."""
    comps = rh.load_competitions(DATA_ROOT)
    iv = rh.intervals_table(comps, window)
    check = rh.check_against_obs(iv, pd.read_csv(OBS_DATA))
    pp = rh.person_period(iv, {c.recording: c.loser_chirps for c in comps})
    pp = rh.add_interval_covariates(pp, iv)
    # centred over intervals, as the windowed models centre over chases
    pp[X] = pp["log_prev"] - np.log(iv["prev_interval"]).mean()
    pp[D] = pp["log_dur"] - np.log(iv["duration"]).mean()
    labels = rh.piece_labels()
    pp["piece_label"] = pd.Categorical([labels[i] for i in pp["piece"]], categories=labels)
    if not np.isclose(pp["exposure"].sum(), iv["length"].sum()) or pp["event"].sum() != iv["event"].sum():
        raise AssertionError("person-period exposure or events do not add up to the intervals")
    info = {"obs_check": check, "n_straddling": int(sum(np.isfinite(c.straddling_onset) for c in comps)),
            "centring": {"log_prev_mean": float(np.log(iv["prev_interval"]).mean()),
                         "log_dur_mean": float(np.log(iv["duration"]).mean())}}
    return iv, pp, info


def fit_rows(pp: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """The aggregated rows of one fit: rows with every model column present, summed over equal columns."""
    cols = [spec["chirp"], *spec["adjust"]]
    keep = pp[cols].notna().all(axis=1)
    keys = [rh.COMP, "piece_label", *cols]
    agg = rh.aggregate(pp.loc[keep], keys)
    agg[spec["chirp"]] = agg[spec["chirp"]].astype(int)
    agg["log_exposure"] = np.log(agg["exposure"])
    return agg


def formula(spec: dict) -> str:
    chirp = spec["chirp"]
    fixed = " + ".join(["0 + piece_label", chirp, *spec["adjust"]])
    re = f"(1 + {chirp} | {rh.COMP})" if spec["slope"] else f"(1 | {rh.COMP})"
    return f"event ~ {fixed} + {re} + offset(log_exposure)"


# ---------------------------------------------------------------------------
# fitting and summaries
# ---------------------------------------------------------------------------
def fit(name: str, spec: dict, rows: pd.DataFrame, sampler: dict):
    import bambi as bmb

    priors = {"piece_label": bmb.Prior("Normal", mu=PRIOR_PIECE[0], sigma=PRIOR_PIECE[1]),
              "common": bmb.Prior("Normal", mu=PRIOR_SLOPE[0], sigma=PRIOR_SLOPE[1])}
    model = bmb.Model(formula(spec), data=rows, family="poisson", priors=priors,
                      noncentered=spec.get("noncentered", NONCENTERED))
    print(f"\n=== {name}: {formula(spec)}\n{model}", flush=True)
    t0 = time.time()
    trace = model.fit(**sampler)
    return model, trace, time.time() - t0


def _flat(trace, var: str) -> np.ndarray:
    return np.asarray(trace.posterior[var].values).reshape(-1)


def hr_summary(trace, spec: dict) -> dict:
    """exp(beta): posterior median, 95 % HDI, P(HR < 1); beta mean and SD; SD of the random slope."""
    beta = _flat(trace, spec["chirp"])
    hr = np.exp(beta)
    hdi = az.hdi(hr, hdi_prob=HDI_PROB)
    out = {"median": float(np.median(hr)), "hdi": [float(hdi[0]), float(hdi[1])],
           "p_lt_1": float((beta < 0).mean()), "beta_mean": float(beta.mean()), "beta_sd": float(beta.std())}
    for cov in spec["adjust"]:
        b = _flat(trace, cov)
        h = az.hdi(b, hdi_prob=HDI_PROB)
        out[f"{cov}_beta"] = {"median": float(np.median(b)), "hdi": [float(h[0]), float(h[1])]}
    sd_var = f"{spec['chirp']}|{rh.COMP}_sigma"
    if sd_var in trace.posterior:
        s = _flat(trace, sd_var)
        h = az.hdi(s, hdi_prob=HDI_PROB)
        out["slope_sd"] = {"median": float(np.median(s)), "hdi": [float(h[0]), float(h[1])]}
    s = _flat(trace, f"1|{rh.COMP}_sigma")
    h = az.hdi(s, hdi_prob=HDI_PROB)
    out["intercept_sd"] = {"median": float(np.median(s)), "hdi": [float(h[0]), float(h[1])]}
    return out


def diagnostics(trace) -> dict:
    """Divergences; max R-hat and min bulk / tail ESS over the population-level terms and the group SDs
    (and, separately, over every variable including the per-competition effects)."""
    names = list(trace.posterior.data_vars)
    main = [v for v in names if "|" not in v or v.endswith("_sigma")]
    s_main = az.summary(trace, var_names=main, kind="diagnostics")
    s_all = az.summary(trace, kind="diagnostics")
    out = {"divergences": int(trace.sample_stats["diverging"].sum()),
           "max_rhat": float(s_main["r_hat"].max()), "min_ess_bulk": float(s_main["ess_bulk"].min()),
           "min_ess_tail": float(s_main["ess_tail"].min()),
           "max_rhat_all": float(s_all["r_hat"].max()), "min_ess_bulk_all": float(s_all["ess_bulk"].min())}
    out["ok"] = int(out["divergences"] == 0 and out["max_rhat_all"] <= OK_RHAT
                    and out["min_ess_bulk_all"] >= OK_ESS and out["min_ess_tail"] >= OK_ESS)
    return out


def model_rates(trace, spec: dict) -> pd.DataFrame:
    """Baseline-piece hazards (events per min) of a typical competition (random effects 0, centred
    covariates at 0 = their geometric means), without and with the chirp covariate: median and 95 % HDI."""
    a = trace.posterior["piece_label"].stack(s=("chain", "draw")).values  # (piece, sample)
    b = _flat(trace, spec["chirp"])
    labels = list(trace.posterior["piece_label"].coords["piece_label_dim"].values)
    rows = []
    for k, lab in enumerate(labels):
        for g, eta in ((0, a[k]), (1, a[k] + b)):
            r = 60 * np.exp(eta)
            h = az.hdi(r, hdi_prob=HDI_PROB)
            rows.append({"piece_label": lab, "chirp": g, "median": float(np.median(r)),
                         "lo": float(h[0]), "hi": float(h[1])})
    return pd.DataFrame(rows)


def interval_counts(iv: pd.DataFrame, pp: pd.DataFrame, spec: dict) -> dict:
    keep = pp[[spec["chirp"], *spec["adjust"]]].notna().all(axis=1)
    used = iv.index.isin(pp.loc[keep, "interval"].unique())
    return {"n_intervals": int(used.sum()), "n_events": int(iv.loc[used, "event"].sum()),
            "n_competitions": int(iv.loc[used, rh.COMP].nunique())}


def piece_table(pp: pd.DataFrame) -> dict:
    labels = rh.piece_labels()
    g = pp.groupby("piece")[["event", "exposure"]].sum()
    return {"edges_s": [float(x) for x in rh.PIECES_S[:-1]],
            "labels": list(range(len(labels))),
            "events": [int(g.loc[k, "event"]) if k in g.index else 0 for k in range(len(labels))],
            "exposure_s": [float(g.loc[k, "exposure"]) if k in g.index else 0.0 for k in range(len(labels))]}


def covariate_table(pp: pd.DataFrame) -> dict:
    out = {}
    for c in rh.TIME_VARYING:
        g = pp.groupby(c)[["event", "exposure"]].sum()
        out[c] = {"events_on": int(g.loc[1, "event"]), "events_off": int(g.loc[0, "event"]),
                  "exposure_on_s": float(g.loc[1, "exposure"]), "exposure_off_s": float(g.loc[0, "exposure"])}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
@app.command()
def main(
    fits: str = typer.Option(",".join(FITS), help="comma-separated fits (default: all)"),
    tune: int = SAMPLER["tune"], draws: int = SAMPLER["draws"], chains: int = SAMPLER["chains"],
    cores: int = SAMPLER["cores"], target_accept: float = SAMPLER["target_accept"],
    seed: int = SAMPLER["random_seed"],
) -> None:
    names = [f.strip() for f in fits.split(",") if f.strip()]
    unknown = sorted(set(names) - set(FITS))
    if unknown:
        raise typer.BadParameter(f"unknown fits {unknown} (known: {list(FITS)})")
    sampler = dict(tune=tune, draws=draws, chains=chains, cores=cores, target_accept=target_accept,
                   random_seed=seed)
    full_run = sampler == SAMPLER and set(names) == set(FITS)
    out_dir = processed_dir(NAMESPACE)
    tag = "" if sampler == SAMPLER else "__test"  # never overwrite the real posteriors with a timing run

    window = offset_window()
    print(f"offset window {window}; pieces {rh.piece_labels()}; sampler {sampler}")
    iv, pp, info = build_tables(window)
    print(f"intervals {len(iv)} (events {int(iv['event'].sum())}, censored {int((iv['event'] == 0).sum())}), "
          f"competitions {iv[rh.COMP].nunique()}, person-period rows {len(pp)}; obs.csv check {info['obs_check']}")
    iv.to_csv(out_dir / "intervals.csv", index=False)
    pp.to_parquet(out_dir / "person_period.parquet", index=False)

    results: dict[str, dict] = {}
    rates = []
    for name in names:
        spec = FITS[name]
        rows = fit_rows(pp, spec)
        _, trace, secs = fit(name, spec, rows, sampler)
        az.to_netcdf(trace, out_dir / f"{name}{tag}.nc")
        r = {**interval_counts(iv, pp, spec), "n_rows": int(len(rows)), "hr": hr_summary(trace, spec),
             "diagnostics": diagnostics(trace), "seconds": float(secs),
             "random_slope": int(spec["slope"]), "noncentered": int(spec.get("noncentered", NONCENTERED))}
        results[name] = r
        print(f"--- {name}: HR {r['hr']['median']:.3f} [{r['hr']['hdi'][0]:.3f}, {r['hr']['hdi'][1]:.3f}], "
              f"P(HR<1) {r['hr']['p_lt_1']:.3f}; {json.dumps(r['diagnostics'])}; {secs:.0f} s", flush=True)
        if spec["chirp"] == "chirped":
            rates.append(model_rates(trace, spec).assign(fit=name))
    if rates:
        pd.concat(rates).to_csv(out_dir / f"model_rates{tag}.csv", index=False)
    summary = pd.DataFrame([{"fit": k, "as": FITS[k]["as"], "covariate": FITS[k]["chirp"],
                             "adjust": "+".join(FITS[k]["adjust"]) or "-", "random_slope": FITS[k]["slope"],
                             **{f: v[f] for f in ("n_intervals", "n_events", "n_rows", "seconds")},
                             "hr": v["hr"]["median"], "hdi_lo": v["hr"]["hdi"][0], "hdi_hi": v["hr"]["hdi"][1],
                             "p_lt_1": v["hr"]["p_lt_1"], **v["diagnostics"]} for k, v in results.items()])
    summary.to_csv(out_dir / f"hr_summary{tag}.csv", index=False)
    print("\n" + summary.to_string(index=False))

    if not full_run:
        print("\nnot a full run with the default sampler settings: no stats written")
        return
    stats = {
        "window_s": list(window),
        "n_intervals": int(len(iv)), "n_events": int(iv["event"].sum()),
        "n_censored": int((iv["event"] == 0).sum()), "n_competitions": int(iv[rh.COMP].nunique()),
        "n_rows": int(len(pp)), "n_straddling": info["n_straddling"],
        "n_intervals_with_previous": int(iv["prev_interval"].notna().sum()),
        "n_chirped_at_offset": int((iv["first_chirp"] <= iv["offset"]).sum()),
        "n_chirped_later": int((iv["first_chirp"] > iv["offset"]).sum()),
        "n_never_chirped": int(iv["first_chirp"].isna().sum()),
        "exposure_s": float(pp["exposure"].sum()),
        "pieces": piece_table(pp), "covariates": covariate_table(pp),
        "sampler": {k: SAMPLER[k] for k in ("tune", "draws", "chains", "target_accept", "random_seed")},
        "fits": results,
        "n_fits_with_divergences": int(sum(r["diagnostics"]["divergences"] > 0 for r in results.values())),
    }
    write_stats(NAMESPACE, stats)


if __name__ == "__main__":
    app()
