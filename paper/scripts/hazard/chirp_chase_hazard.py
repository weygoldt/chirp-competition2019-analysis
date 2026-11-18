# Ported from chirp_competition b65108e (src_hazard/chirp_chase_hazard.py).
# Changes: data root (optional, default paper_config.DATA_ROOT), output directory
# $PAPER_OUT/data/processed/chirp_chase_hazard. Models, windows, seeds and simulations unchanged.
"""Does subordinate chirping change the dominant's chasing? Event-level pipeline.

Dyadic competition trials of Apteronotus leptorhynchus (2019/2020 tube
competition). Two questions, answered at the event level and conditional on
chase timing (session-level chirp and chase counts are confounded by dyad):

  (a) Does a subordinate chirp increase the hazard that an ongoing chase ends?
  (b) Does a subordinate chirp decrease the hazard that a new chase starts?

Steps (one module each; all times in s, frequencies in Hz):

  1. hz_data    load + validate, flags table, tidy event table  -> validation/
  2. hz_desc    chirp/chase alignments vs within-trial nulls      -> descriptives/, figures/desc_*
  3. hz_hazard  discrete-time hazard models (offset, onset),
                kernel + tau scan + lag-guarded indicator, GLMM + fixed-effects GLM,
                per-dyad O/E + random-effects meta-analysis,
                jitter tests, leave-one-dyad-out                   -> hazard/, figures/hazard_*
  4. (here)     sensitivity refits: without flagged trials, without ambiguous chirps,
                with shifted chase boundaries                     -> hazard/hazard_sensitivity.csv
  4b. hz_acausal acausal control (uses chirps AFTER the outcome bin; never part of the
                primary models): past vs future windows, two-sided kernel, mirror
                surrogate and constant backward shift             -> acausal/, figures/acausal_*
  5. hz_sim     generative-model checks                           -> simulation/, figures/sim_*
  5b. hz_acausal_sim  simulation calibration of the acausal control (what each account of the
                offset association produces for the step-4b statistics; runs unless
                --skip-acausal or --skip-sim)   -> simulation/acausal_sim_*, figures/acausal_sim_*
  6. (here)     effects.csv (one tidy table), summary.txt, timings.csv, config.json, run.log

Usage (paper environment; detections from PAPER_DETECTIONS, see paper/README.md):

  PAPER_DETECTIONS=rtdetr_gapfill_t080 paper/run.sh hazard/chirp_chase_hazard.py
  PAPER_DETECTIONS=rtdetr_gapfill_t080 paper/run.sh hazard/chirp_chase_hazard.py --quick --out /tmp/q

See src_hazard/README.md for assumptions, the backend choice and open issues.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/scripts

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import fields, replace  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any, Iterator, TextIO  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import hz_acausal  # noqa: E402
import hz_acausal_sim  # noqa: E402
import hz_data  # noqa: E402
import hz_desc  # noqa: E402
import hz_hazard  # noqa: E402
import hz_sim  # noqa: E402
import paper_config  # noqa: E402
from hz_config import Config  # noqa: E402

PRIMARY, CHECK, VB, META = "mixed_laplace", "glm_fe", "mixed_vb", "re_meta"
TAU_MODEL, TAU_TERM = "tau_1000ms", "sub_any_1000ms"  # >= 1 sub chirp in the preceding 1 s
HYPOTHESIS = {"offset": 1, "onset": -1}  # expected sign of the log ratio
QUESTION = {
    "offset": "(a) Does a subordinate chirp increase the hazard that an ongoing chase ends?",
    "onset": "(b) Does a subordinate chirp decrease the hazard that a new chase starts?",
}
EVENT_NAME = {"offset": "chase offsets", "onset": "chase onsets"}
RISK_NAME = {"offset": "in-chase", "onset": "out-of-chase"}
DEFAULT_OUT = paper_config.PROCESSED / "chirp_chase_hazard"
# boundary-shift sensitivity: annotated offsets / onsets moved by these amounts (s)
SHIFTS = {"offset": (-0.75, -0.5, -0.25, 0.5), "onset": (-0.5, 0.5)}


# ---------------------------------------------------------------------------
# CLI, logging, timing
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Event-level analysis: do subordinate chirps change dominant chasing?")
    ap.add_argument("data_root", type=Path, nargs="?", default=paper_config.DATA_ROOT,
                    help=f"dataset root (read only; default {paper_config.DATA_ROOT})")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--bin-width", type=float, default=None, help="s (default 0.05)")
    ap.add_argument("--hist-max", type=float, default=None,
                    help="s, extent of the raised-cosine history basis (default 2)")
    ap.add_argument("--n-basis", type=int, default=None, help="raised-cosine bumps (default 5)")
    ap.add_argument("--n-jitter", type=int, default=None,
                    help="null replicates for descriptives and jitter tests (default 200)")
    ap.add_argument("--n-sims", type=int, default=None,
                    help="replicates of the main simulation scenarios (default 50)")
    ap.add_argument("--max-time", type=float, default=None,
                    help="analysis window end in s (default 10800 = first 3 h, when chasing "
                         "happens); <= 0: none")
    ap.add_argument("--use-rises", action="store_true",
                    help="add dom/sub rise history covariates (analysis/rise_idx.npy)")
    ap.add_argument("--include-bad", action="store_true",
                    help="keep the 'EODfs too close' recordings")
    ap.add_argument("--include-draws", action="store_true", help="keep draws")
    ap.add_argument("--seed", type=int, default=None, help="master seed (default 42)")
    ap.add_argument("--workers", type=int, default=None,
                    help="forked workers for jitter tests and simulation (default 8)")
    ap.add_argument("--nulls", default="shift,matched",
                    help="comma list of jitter-test nulls (shift, matched), '' for none")
    ap.add_argument("--no-vb", action="store_true",
                    help="skip the mean-field VB cross-check (offset, tau 1 s)")
    ap.add_argument("--skip-descriptives", action="store_true")
    ap.add_argument("--skip-acausal", action="store_true",
                    help="skip the acausal control (hz_acausal; ~25 min) and its simulation "
                         "calibration (hz_acausal_sim)")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="dyad bootstrap resamples of the acausal mirror contrast (default 500)")
    ap.add_argument("--skip-sim", action="store_true",
                    help="skip the simulation checks (hz_sim) and the acausal simulation "
                         "calibration (hz_acausal_sim)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: few jitters and simulation replicates (~10 min)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild effects.csv and summary.txt from the tables in --out")
    return ap.parse_args(argv)


def build_config(a: argparse.Namespace) -> Config:
    """Config from CLI flags; unset flags keep the Config defaults. The default
    output directory is resolved relative to the repository, not the cwd."""
    out = a.out or paper_config.processed_dir("chirp_chase_hazard")
    cfg = Config(data_root=a.data_root, out_dir=out).updated(
        bin_width=a.bin_width, hist_max=a.hist_max, n_basis=a.n_basis,
        n_jitter=a.n_jitter, n_sims=a.n_sims, max_time=a.max_time, seed=a.seed,
        use_rises=a.use_rises or None, quick=a.quick or None,
        exclude_bad_recordings=False if a.include_bad else None,
        exclude_draws=False if a.include_draws else None,
    )
    if a.workers is not None:
        cfg.extra["n_workers"] = a.workers
        cfg.extra["sim_workers"] = a.workers
    if getattr(a, "n_boot", None) is not None:
        cfg.extra["acausal_n_boot"] = a.n_boot
    return cfg


def effective_settings(cfg: Config, nulls: list[str], vb: bool) -> dict[str, Any]:
    """Replicate counts actually used (quick mode caps them)."""
    q = cfg.quick
    n_ac = int(hz_acausal_sim._opt(cfg, "acsim_n_reps"))
    if q:
        n_ac = min(n_ac, int(hz_acausal_sim._opt(cfg, "acsim_quick_reps")))
    return dict(
        n_jitter_hazard=min(cfg.n_jitter, 20) if q else cfg.n_jitter,
        n_jitter_descriptives=min(cfg.n_jitter, 50) if q else cfg.n_jitter,
        n_sims_main_scenarios=min(cfg.n_sims, 8) if q else cfg.n_sims,
        n_reps_acausal_sim=n_ac,
        n_reps_acausal_sim_onset=min(n_ac, int(hz_acausal_sim._opt(cfg, "acsim_onset_reps"))),
        jitter_nulls=nulls, vb_check=bool(vb and not q),
    )


class Tee:
    """Write to several text streams (console + run.log)."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    def isatty(self) -> bool:
        return False


class Timer:
    """Wall-clock timing of pipeline steps."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        print(f"\n=== {name} ({datetime.now():%H:%M:%S}) ===", flush=True)
        t = time.perf_counter()
        yield
        dt = time.perf_counter() - t
        self.rows.append(dict(step=name, seconds=round(dt, 2)))
        print(f"=== {name}: {dt:.1f} s ===", flush=True)

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        total = pd.DataFrame([dict(step="total", seconds=round(time.perf_counter() - self.t0, 2))])
        return pd.concat([df, total], ignore_index=True)


# ---------------------------------------------------------------------------
# Sensitivity refits
# ---------------------------------------------------------------------------
SENS_FLAGS = {
    "drop_close_eodf": "close_eodfs_not_in_bad_list",
    "drop_led_corrected": "led_reference_misaligned",
    "drop_sender_ambiguous": "sender_ambiguous_bbox",
}
SENS_LABEL = {
    "drop_close_eodf": "without the close-EODf dyads",
    "drop_led_corrected": "without the two LED-repaired dyads",
    "drop_sender_ambiguous": "without dyads where most chirp boxes contain both EODfs",
    "unambiguous_chirps": "only chirps whose box contains one fish's EODf",
}


def sensitivity_sets(trials: list[hz_data.Trial], flags: pd.DataFrame) -> dict[str, list[str]]:
    """Recordings to drop per sensitivity refit, taken from the flags table."""
    inc = {tr.recording for tr in trials}
    sets = {}
    for name, cat in SENS_FLAGS.items():
        r = flags.loc[(flags["category"] == cat) & (flags["severity"] != "info"), "recording"]
        sets[name] = sorted(set(r) & inc)
    return {k: v for k, v in sets.items() if v}


def unambiguous_trials(trials: list[hz_data.Trial]) -> list[hz_data.Trial]:
    """Trials with chirps whose detector box contains both fish's EODf removed."""
    out = []
    for tr in trials:
        kw = {}
        for who in ("dom", "sub"):
            amb = tr.meta.get(f"chirps_{who}_ambiguous")
            x = getattr(tr, f"chirps_{who}")
            kw[f"chirps_{who}"] = x[~amb] if amb is not None and len(amb) == len(x) else x
        out.append(replace(tr, **kw))
    return out


def shift_boundaries(trials: list[hz_data.Trial], kind: str, delta: float
                     ) -> list[hz_data.Trial] | None:
    """Trials with every annotated offset (kind 'offset') or onset moved by delta s;
    None if that makes a chase zero-length or overlapping."""
    col = 1 if kind == "offset" else 0
    out = []
    for tr in trials:
        ch = tr.chases.copy()
        ch[:, col] += delta
        try:
            out.append(replace(tr, chases=ch))
        except ValueError:
            return None
    return out


def _sens_fit(sub: list[hz_data.Trial], kind: str, cfg: Config, subset: str, note: str
              ) -> pd.DataFrame:
    d = hz_hazard.build_design(sub, kind, cfg)
    e = hz_hazard.fit_models(d, kind, cfg, models=[TAU_MODEL], backends=[PRIMARY, CHECK])
    e = e[e["term"].isin([TAU_TERM, "dom_any_1000ms"])]
    _, het = hz_hazard.heterogeneity(d, cfg, models=[TAU_MODEL])
    e = pd.concat([e, het[het["term"].isin([TAU_TERM, "heterogeneity_G",
                                            "dyad_sd_sub_effect"])]], ignore_index=True)
    e["analysis"] = "sensitivity"
    e["subset"] = subset
    e["note"] = f"{subset}: {note}; " + e["note"].astype(str)
    return e


def sensitivity_fits(trials: list[hz_data.Trial], flags: pd.DataFrame, cfg: Config
                     ) -> pd.DataFrame:
    """GLMM, FE and random-effects refits of the tau = 1 s model under
    (1) dropped flagged dyads, (2) only unambiguous chirps, (3) chase
    boundaries shifted by SHIFTS (a uniform annotation lag)."""
    rows = []

    def show(e: pd.DataFrame, subset: str, kind: str) -> None:
        parts = []
        for b, lab in ((PRIMARY, "GLMM OR"), (CHECK, "FE HR"), (META, "RE HR")):
            r = e[(e["backend"] == b) & (e["term"] == TAU_TERM)]
            if len(r):
                r = r.iloc[0]
                parts.append(f"{lab} {r['ratio']:.2f} [{r['ratio_low']:.2f}, "
                             f"{r['ratio_high']:.2f}]")
        print(f"  {subset:22s} {kind:6s} " + "; ".join(parts), flush=True)

    sets = sensitivity_sets(trials, flags)
    for name, drop in sets.items():
        sub = [tr for tr in trials if tr.recording not in drop]
        for kind in hz_hazard.KINDS:
            e = _sens_fit(sub, kind, cfg, name,
                          f"without {', '.join(r[:10] for r in drop)}")
            rows.append(e)
            show(e, name, kind)
    ua = unambiguous_trials(trials)
    n_rm = sum(len(a.chirps_sub) - len(b.chirps_sub) for a, b in zip(trials, ua))
    for kind in hz_hazard.KINDS:
        e = _sens_fit(ua, kind, cfg, "unambiguous_chirps",
                      f"{n_rm} sub chirps (and the dom chirps) whose box contains both "
                      "EODfs removed")
        rows.append(e)
        show(e, "unambiguous_chirps", kind)
    for kind, deltas in SHIFTS.items():
        for dlt in deltas:
            sh = shift_boundaries(trials, kind, dlt)
            name = f"shift_{kind}s_{dlt:+g}s"
            if sh is None:
                print(f"  {name}: skipped (zero-length chase)", flush=True)
                continue
            e = _sens_fit(sh, kind, cfg, name,
                          f"every annotated {kind} moved by {dlt:+g} s")
            rows.append(e)
            show(e, name, kind)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# One tidy effects table
# ---------------------------------------------------------------------------
EFFECT_COLS = [
    "analysis", "kind", "sender", "model", "backend", "link", "scale", "term", "term_type",
    "lag", "subset", "left_out", "null", "window_lo", "window_hi", "estimate", "se", "ci_low",
    "ci_high", "p", "ratio", "ratio_low", "ratio_high", "ci_method", "n_bins", "n_events",
    "n_recordings", "n_jitter", "null_mean", "null_sd", "null_q025", "null_q975", "primary",
    "converged", "fit_time_s", "note", "variant",
]
CI_METHOD = {
    PRIMARY: "Laplace posterior mean +- 1.96 SD (model-based; assumes one effect for all dyads)",
    VB: "mean-field VB mean +- 1.96 marginal SD",
    CHECK: "cluster-robust (recording) sandwich, t(G-1)",
    META: "profile likelihood of the Poisson-normal random-effects model, t(k-1)^2 cut-off",
}


def _sender(term: str) -> str:
    if term.startswith(("sub_", "kernel_sub", "rise_sub")) or term.endswith(("_sub", ":sub")):
        return "sub"
    if term.startswith(("dom_", "kernel_dom", "rise_dom")) or term.endswith(("_dom", ":dom")):
        return "dom"
    return ""


def _scale(r: pd.Series) -> str:
    tt = str(r.get("term_type", ""))
    if r["analysis"] == "descriptive":
        return "log rate ratio, observed / null mean"
    if tt == "test":
        return "test statistic (chi2, F or G)"
    if tt == "re_sd":
        return ("SD of the sub-chirp log HR between dyads"
                if str(r["term"]).startswith("dyad_sd_sub_effect")
                else "SD of dyad intercept (log-odds)")
    if tt == "contrast":  # acausal control: past - future, or restricted real - mirror
        return {"logit": "difference of log odds ratios",
                "cloglog": "difference of log hazard ratios"}.get(str(r["link"]), "")
    return {"logit": "log odds ratio", "cloglog": "log hazard ratio"}.get(str(r["link"]), "")


def effects_table(haz: dict[str, Any], desc: dict[str, pd.DataFrame] | None,
                  sens: pd.DataFrame | None,
                  acausal: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """All fitted effects with CIs: hazard fits (incl. random-effects meta rows),
    sensitivity and leave-one-dyad-out refits, jitter-test summaries,
    descriptive window tests and, last, the acausal control (analysis
    'acausal', labelled by 'variant')."""
    parts = [haz["effects"].assign(analysis="hazard")]
    if sens is not None and len(sens):
        parts.append(sens)
    lodo = haz.get("leave_one_dyad_out")
    if lodo is not None and len(lodo):
        lo = lodo.rename(columns={"stat": "term"})
        nan = pd.Series(np.nan, index=lo.index)
        on, rr = lo.get("dyad_events_on", nan), lo.get("dyad_crude_rr", nan)
        lo = lo.assign(
            analysis="leave_one_dyad_out", backend=CHECK,
            link=str(hz_hazard.DEFAULTS["glm_link"]),
            ci_low=np.log(lo["ratio_low"]), ci_high=np.log(lo["ratio_high"]),
            note=[(f"refit without {rec}; that dyad: {int(k)} events with indicator on, "
                   f"crude rate ratio {q:.2f}" if np.isfinite(k) else f"refit without {rec}")
                  for rec, k, q in zip(lo["left_out"], on, rr, strict=True)],
        )
        parts.append(lo)
    js = haz.get("jitter_summary")
    if js is not None and len(js):
        parts.append(pd.DataFrame(dict(
            analysis="jitter_test", kind=js["kind"], null=js["null"], term=js["stat"],
            model=["kernel" if s == "kernel_sub_mean" else s.removesuffix("_sub")
                   for s in js["stat"]],
            backend=CHECK, link=str(hz_hazard.DEFAULTS["glm_link"]), estimate=js["observed"],
            ratio=np.exp(js["observed"]), p=js["p_two_sided"], n_jitter=js["n_jitter"],
            null_mean=js["null_mean"], null_sd=js["null_sd"], null_q025=js["null_q025"],
            null_q975=js["null_q975"], ci_method="permutation (no CI); null quantiles given",
            note=[f"two-sided rank p vs {int(n)} null refits; min attainable "
                  f"{2 / (n + 1):.3g}" + ("; matched null does not keep inter-chirp intervals "
                                           "(mildly anti-conservative)" if nl == "matched" else
                                           "; keeps inter-chirp intervals")
                  for n, nl in zip(js["n_jitter"], js["null"], strict=True)],
        )))
    if desc is not None and len(desc.get("tests", [])):
        t = desc["tests"]
        parts.append(pd.DataFrame(dict(
            analysis="descriptive", kind=t["event"], sender=t["sender"], model=t["family"],
            backend="descriptive", term=t["analysis"], null=t["null"], window_lo=t["window_lo"],
            window_hi=t["window_hi"], estimate=np.log(t["ratio"].where(t["ratio"] > 0)),
            ratio=t["ratio"], ratio_low=t["ratio_ci_low"], ratio_high=t["ratio_ci_high"],
            ci_low=np.log(t["ratio_ci_low"].where(t["ratio_ci_low"] > 0)),
            ci_high=np.log(t["ratio_ci_high"].where(t["ratio_ci_high"] > 0)),
            p=t["p_window_two_sided"], n_events=t["n_triggers"], n_recordings=t["n_trials"],
            n_jitter=t["n_jitter"], primary=t["primary"],
            ci_method="trial (dyad) bootstrap of observed / null-mean ratio", note=t["note"],
        )))
    if acausal is not None and len(acausal.get("effects", [])):
        # control / spline / intercept rows of the acausal refits stay in
        # acausal/acausal_effects.csv only
        a = acausal["effects"]
        parts.append(a[~a["term_type"].isin(["control", "spline", "intercept"])])
    out = pd.concat([p.reset_index(drop=True) for p in parts], ignore_index=True)
    miss = out["sender"].isna() if "sender" in out else pd.Series(True, index=out.index)
    out.loc[miss, "sender"] = out.loc[miss, "term"].astype(str).map(_sender)
    out["scale"] = out.apply(_scale, axis=1)
    cm = out["backend"].map(CI_METHOD)
    out["ci_method"] = out["ci_method"].fillna(cm) if "ci_method" in out else cm
    for c in EFFECT_COLS:
        if c not in out:
            out[c] = np.nan
    return out[EFFECT_COLS]


# ---------------------------------------------------------------------------
# Text answer
# ---------------------------------------------------------------------------
def _pick(df: pd.DataFrame | None, **kw: Any) -> pd.Series | None:
    if df is None or df.empty:
        return None
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        if k not in df:
            return None
        m &= (df[k] == v).to_numpy()
    return df[m].iloc[0] if m.any() else None


def _p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"{p:.0e}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _ci(r: pd.Series | None, p: bool = False) -> str:
    if r is None or not np.isfinite(r["ratio"]):
        return "n/a"
    s = f"{r['ratio']:.2f} [{r['ratio_low']:.2f}, {r['ratio_high']:.2f}]"
    return s + (f", p = {_p(r['p'])}" if p else "")


def _side(r: pd.Series | None) -> int:
    """+1 / -1 if the ratio CI lies entirely above / below 1, else 0."""
    if r is None or not np.isfinite(r["ratio_low"]) or not np.isfinite(r["ratio_high"]):
        return 0
    return 1 if r["ratio_low"] > 1 else (-1 if r["ratio_high"] < 1 else 0)


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%" if np.isfinite(v) else "n/a"


def _rej(r: pd.Series | None, t: str = "naive") -> str:
    if r is None or not np.isfinite(r.get(f"reject_{t}", np.nan)):
        return "n/a"
    return f"{_pct(r[f'reject_{t}'])} [{_pct(r[f'reject_{t}_lo'])}-{_pct(r[f'reject_{t}_hi'])}]"


def _short(rec: str) -> str:
    return rec[5:10]


def _jitter_ok(kind: str, js: pd.DataFrame) -> bool | None:
    """True if every jitter test rejects in the hypothesized direction; None if not run."""
    if js is None or not len(js):
        return None
    s = HYPOTHESIS[kind]
    # with few jitters the smallest attainable p (2 / (n + 1)) exceeds 0.05; the
    # test then 'rejects' when the observed value is beyond every null refit
    alpha = np.maximum(0.05, 2.0 / (js["n_jitter"] + 1)) + 1e-12
    return bool(np.all((js["p_two_sided"] <= alpha)
                       & (np.sign(js["observed"] - js["null_mean"]) == s)))


def _verdict(kind: str, g: pd.Series | None, f: pd.Series | None, re_: pd.Series | None,
             oe: pd.DataFrame, js: pd.DataFrame, lk: pd.DataFrame, het_p: float) -> str:
    """Verdict from the common-effect GLMM (g), the dyad-clustered FE GLM (f), the
    random-effects mean over dyads (re_), the per-dyad O/E table, jitter tests and
    leave-one-dyad-out refits."""
    s = HYPOTHESIS[kind]
    same_w, opp_w = ("higher", "lower") if s > 0 else ("lower", "higher")
    inf = oe[oe["informative"].astype(bool)] if len(oe) else oe
    n_same = int((inf["direction"] == same_w).sum()) if len(inf) else 0
    opp = inf[inf["direction"] == opp_w] if len(inf) else inf
    n_unc = len(inf) - n_same - len(opp)
    jit = _jitter_ok(kind, js)
    lodo_ok = bool(len(lk)) and bool(np.all(np.sign(np.log(lk["ratio"])) == s))
    avg, pooled, clus = _side(re_) == s, _side(g) == s, _side(f) == s
    word = "increases" if s > 0 else "decreases"
    dyads = (f"Informative dyads (>= 2 expected events): {len(inf)}; O/E above 1 in "
             f"{int((inf['direction'] == 'higher').sum()) if len(inf) else 0}, below 1 in "
             f"{int((inf['direction'] == 'lower').sum()) if len(inf) else 0}, unclear in "
             f"{n_unc}.")
    jtxt = "" if jit is not None else " (jitter test not run)"
    if avg and clus and not len(opp) and lodo_ok and jit is not False:
        v = ("YES, as an association conditional on chase timing, on average across dyads: "
             "the random-effects mean over dyads and the dyad-clustered interval exclude 1, "
             "no informative dyad runs the other way, and every leave-one-dyad-out refit "
             "keeps the direction" + ("" if jit is None else "; the within-dyad jitter test "
                                     "rejects") + jtxt + ".")
        if np.isfinite(het_p) and het_p < 0.05:
            v += f" The size of the effect differs between dyads (heterogeneity p = {_p(het_p)})."
        return v + " " + dyads
    if (avg or pooled) and not (_side(re_) == -s or _side(f) == -s):
        why = []
        if not clus:
            why.append("the dyad-clustered FE interval includes 1")
        if not avg:
            why.append("the random-effects mean over dyads includes 1")
        if len(opp):
            why.append(f"{len(opp)} informative dyad(s) run significantly the other way "
                       f"({', '.join(_short(r) for r in opp['recording'])})")
        if not lodo_ok:
            why.append("a leave-one-dyad-out refit changes the sign")
        if jit is False:
            why.append("the within-dyad jitter test does not reject")
        which = " and ".join(x for x, ok in (("random-effects mean over dyads", avg),
                                             ("common-effect GLMM", pooled)) if ok)
        v = (f"NOT ESTABLISHED as a consistent effect; the dyads disagree. On average a sub "
             f"chirp is followed by a lower hazard ({which} exclude 1), but "
             if s < 0 else
             f"NOT ESTABLISHED as a consistent effect; the dyads disagree. On average a sub "
             f"chirp is followed by a higher hazard ({which} exclude 1), but ")
        v += "; ".join(why) + (f" (heterogeneity p = {_p(het_p)})" if np.isfinite(het_p) else "")
        return v + jtxt + ". " + dyads
    if _side(g) == -s or _side(f) == -s or _side(re_) == -s:
        return (f"NO: the association runs opposite to the hypothesis (that a sub chirp "
                f"{word} it)" + jtxt + ". " + dyads)
    return "NO clear evidence: the intervals include 1" + jtxt + ". " + dyads


def _dyad_lines(oe: pd.DataFrame, kind: str, model: str) -> list[str]:
    t = oe[(oe["kind"] == kind) & (oe["model"] == model)] if len(oe) else oe
    if t is None or not len(t):
        return []
    inf = t[t["informative"].astype(bool)].sort_values("expected", ascending=False)
    rest = t[~t["informative"].astype(bool)]
    mark = {"higher": "+", "lower": "-", "unclear": "", "no bins": ""}
    parts = [f"{_short(r.recording)} {int(r.observed)}/{r.expected:.1f} = {r.oe:.2f} "
             f"[{r.oe_low:.2f}, {r.oe_high:.2f}]{mark.get(r.direction, '')}"
             for r in inf.itertuples()]
    lines = ["  Per dyad, observed / expected events in bins with a sub chirp in the preceding "
             "1 s (expected: no chirp effect; exact 95% CI; +/- = CI above/below 1):"]
    for i in range(0, len(parts), 3):
        lines.append("    " + "; ".join(parts[i:i + 3]))
    if len(rest):
        lines.append(f"    uninformative (< 2 expected): {int(rest['observed'].sum())} observed / "
                     f"{rest['expected'].sum():.1f} expected over {len(rest)} dyads")
    return lines


def _desc_dyads(tpt: pd.DataFrame | None, kind: str) -> str:
    if tpt is None or not len(tpt):
        return ""
    t = tpt[(tpt["analysis"] == f"post_chirp_hazard:{kind}:sub") & (tpt["null"] == "matched")]
    if not len(t):
        return ""
    sig = t[t["p_two_sided"] <= 0.05 + 1e-9]
    hi = sig[sig["ratio"] > 1]["recording"].map(_short).tolist()
    lo = sig[sig["ratio"] < 1]["recording"].map(_short).tolist()
    return (f"per dyad (p <= 0.05 vs its own matched null): higher in {len(hi)}"
            f"{' (' + ', '.join(hi) + ')' if hi else ''}, lower in {len(lo)}"
            f"{' (' + ', '.join(lo) + ')' if lo else ''}, not significant in "
            f"{len(t) - len(sig)}")


def _lag_prediction(oe: pd.DataFrame, tpt: pd.DataFrame | None) -> str:
    """Annotation-lag prediction: if (a) came from late-marked offsets, dyads whose
    subordinate chirps most right after an offset should show the largest
    pre-offset O/E. Spearman correlation over informative dyads."""
    from scipy import stats

    if tpt is None or not len(tpt) or not len(oe):
        return ""
    t = tpt[(tpt["analysis"] == "event_to_chirp:offset:sub") & (tpt["window"] == "post")]
    if not len(t):
        return ""
    t = t[t["null"] == ("matched" if (t["null"] == "matched").any() else t["null"].iloc[0])]
    o = oe[(oe["kind"] == "offset") & (oe["model"] == TAU_MODEL) & oe["informative"].astype(bool)]
    m = o.merge(t[["recording", "ratio"]], on="recording").dropna(subset=["ratio", "oe"])
    if len(m) < 5:
        return ""
    r, pv = stats.spearmanr(m["ratio"], m["oe"])
    return (f"  Annotation-lag prediction (late-marked offsets would make dyads with a larger "
            f"post-offset chirp burst show a larger pre-offset O/E): Spearman rho = {r:.2f} "
            f"(p = {_p(pv)}, {len(m)} informative dyads; low power).")


def _scope(trials: list[hz_data.Trial], cfg: Config) -> str:
    sexes = pd.Series([f"{tr.sex_dom.upper()}-dom/{tr.sex_sub.upper()}-sub" for tr in trials])
    comp = ", ".join(f"{n} {k}" for k, n in sexes.value_counts().items())
    mixed = int(sum(tr.sex_dom != tr.sex_sub for tr in trials))
    fish = {(tr.group, tr.dom_fish) for tr in trials} | {(tr.group, tr.sub_fish) for tr in trials}
    win = f"the first {cfg.max_time / 3600:g} h of each recording" if cfg.max_time else \
        "whole recordings"
    return (f"Scope: Apteronotus leptorhynchus, lab tube competitions; {len(trials)} dyads "
            f"({mixed} mixed-sex: {comp}) from {len(fish)} fish that recur across dyads; "
            f"{win}; all RCNN-detected chirps pooled regardless of chirp type (no small/big "
            "or type split). Sex-specific effects were not tested; with mostly mixed-sex dyads, "
            "courtship-context chirping is a live alternative reading of 'subordinate chirp'.")


def answer_text(trials: list[hz_data.Trial], flags: pd.DataFrame, haz: dict[str, Any],
                desc: dict[str, pd.DataFrame] | None, sens: pd.DataFrame | None,
                sim: dict[str, Any] | None, cfg: Config, wall_s: float,
                run_time: str | None = None,
                acausal: dict[str, pd.DataFrame] | None = None,
                acausal_sim: dict[str, pd.DataFrame] | None = None) -> str:
    E = haz["effects"]
    jsum = haz.get("jitter_summary", pd.DataFrame())
    lodo = haz.get("leave_one_dyad_out", pd.DataFrame())
    oe = haz.get("dyad_oe", pd.DataFrame())
    tpt = desc.get("tests_per_trial") if desc is not None else None
    guard = hz_hazard.guard_model(cfg)
    g_term = str(hz_hazard.indicator_column(guard, "sub"))
    g_lag, g_win = float(hz_hazard._opt(cfg, "guard_lag")), float(hz_hazard._opt(cfg,
                                                                                "guard_window"))
    G = len(trials)
    fish = {(tr.group, tr.dom_fish) for tr in trials} | {(tr.group, tr.sub_fish) for tr in trials}
    n_sub = sum(int(tr.valid(tr.chirps_sub).sum()) for tr in trials)
    n_sub_in = sum(int((tr.valid(tr.chirps_sub) & tr.in_chase(tr.chirps_sub)).sum())
                   for tr in trials)
    n_dom = sum(int(tr.valid(tr.chirps_dom).sum()) for tr in trials)
    n_ch = sum(int((tr.chases[:, 0] < tr.obs_end).sum()) for tr in trials)
    groups = sorted({tr.group for tr in trials})
    win = f"0-{cfg.max_time / 3600:g} h" if cfg.max_time else "whole recording"
    sets = sensitivity_sets(trials, flags)
    L = [
        "Do subordinate chirps change the dominant's chasing? Event-level answer",
        "=" * 72,
        _scope(trials, cfg),
        (f"Data: {G} dyads ({len(fish)} fish; groups {', '.join(map(str, groups))}), analysis "
         f"window {win}; {n_ch:,} chases, {n_sub:,} subordinate chirps ({n_sub_in:,} inside "
         f"chases), {n_dom:,} dominant chirps. Bins {cfg.bin_width:g} s."),
        ("Models: discrete-time hazard of the chase event in each bin, conditional on elapsed "
         "chase / inter-chase time (spline), dominant chirp history, time since contact, size "
         "difference, dEODf (trailing median), time in trial and dyad"
         + ("; rise histories" if cfg.use_rises else "") + ". Chirp history uses only chirps "
         "strictly before the bin; no covariate uses future information."),
        ("Three estimates of the same ratio: GLMM = logit, dyad random intercept, one chirp "
         "effect shared by all dyads (odds ratio; its interval is conditional on that "
         "assumption); FE = cloglog GLM, dyad fixed intercepts, dyad-clustered t(G-1) interval "
         "(hazard ratio); RE = random-effects meta-analysis of per-dyad observed/expected "
         "events (mean HR over dyads; allows the effect to differ between dyads). Per-bin "
         "event probabilities are < 1%, so OR ~ HR."),
        "",
    ]
    for kind in hz_hazard.KINDS:
        if not (E["kind"] == kind).any():
            continue
        s = HYPOTHESIS[kind]
        g = _pick(E, kind=kind, backend=PRIMARY, term=TAU_TERM)
        f = _pick(E, kind=kind, backend=CHECK, term=TAU_TERM)
        v = _pick(E, kind=kind, backend=VB, term=TAU_TERM)
        re_ = _pick(E, kind=kind, backend=META, term=TAU_TERM)
        tau = _pick(E, kind=kind, backend=META, model=TAU_MODEL, term="dyad_sd_sub_effect")
        hg = _pick(E, kind=kind, backend=META, model=TAU_MODEL, term="heterogeneity_G")
        het_p = float(hg["p"]) if hg is not None else np.nan
        js = jsum[jsum["kind"] == kind] if len(jsum) else pd.DataFrame()
        lk = lodo[lodo["kind"] == kind] if len(lodo) else pd.DataFrame()
        oe_k = oe[(oe["kind"] == kind) & (oe["model"] == TAU_MODEL)] if len(oe) else oe
        L.append(QUESTION[kind])
        L.append("  Answer: " + _verdict(kind, g, f, re_, oe_k, js, lk, het_p))
        if g is not None:
            L.append(f"  Risk set: {int(g['n_events']):,} {EVENT_NAME[kind]} in "
                     f"{int(g['n_bins']):,} {RISK_NAME[kind]} bins, "
                     f"{int(g['n_recordings'])} dyads.")
        L.append(f"  >= 1 sub chirp in the preceding 1 s: GLMM OR {_ci(g)}; FE HR {_ci(f, p=True)}"
                 + (f"; VB OR {_ci(v)}" if v is not None else "")
                 + (f"; RE mean HR {_ci(re_, p=True)}" if re_ is not None else "") + ".")
        if re_ is not None and tau is not None:
            note = str(re_["note"])
            pr = note[note.find("(95% prediction"):note.find(")", note.find("(95% prediction")) + 1]
            L.append(f"  Between-dyad heterogeneity: SD of the dyad log HR {tau['estimate']:.2f} "
                     f"(LR test p = {_p(tau['p'])}); homogeneity G test p = {_p(het_p)} {pr}. "
                     "The GLMM interval assumes one common effect, which these tests reject."
                     if np.isfinite(het_p) and het_p < 0.05 else
                     f"  Between-dyad heterogeneity: SD {tau['estimate']:.2f}, G test p = "
                     f"{_p(het_p)} {pr}.")
        L += _dyad_lines(oe, kind, TAU_MODEL)
        gg = _pick(E, kind=kind, backend=PRIMARY, term=g_term)
        gf = _pick(E, kind=kind, backend=CHECK, term=g_term)
        gr = _pick(E, kind=kind, backend=META, term=g_term)
        if gg is not None or gf is not None:
            oe_g = oe[(oe["kind"] == kind) & (oe["model"] == guard)] if len(oe) else oe
            inf = oe_g[oe_g["informative"].astype(bool)] if len(oe_g) else oe_g
            L.append(f"  Lag-guarded (sub chirp {g_lag:g}-{g_lag + g_win:g} s before the bin; "
                     f"leaves out chirps within the ~{g_lag:g} s annotation precision): GLMM OR "
                     f"{_ci(gg)}; FE HR {_ci(gf, p=True)}; RE mean HR {_ci(gr, p=True)}; "
                     f"informative dyads above 1: {int((inf['direction'] == 'higher').sum())}, "
                     f"below 1: {int((inf['direction'] == 'lower').sum())}.")
        if sens is not None and len(sens):
            sh = sens[(sens["kind"] == kind) & sens["subset"].astype(str).str.startswith("shift_")
                      & (sens["term"] == TAU_TERM)]
            if len(sh):
                parts = []
                for sub_, grp in sh.groupby("subset", sort=False):
                    fe = _pick(grp, backend=CHECK)
                    rr = _pick(grp, backend=META)
                    dlt = sub_.split("_")[-1]
                    parts.append(f"{dlt} FE {_ci(fe)}, RE {_ci(rr)}")
                L.append(f"  Annotated {kind}s moved by a fixed amount (negative = earlier, "
                         f"undoes late marking): " + "; ".join(parts) + ".")
        km = _pick(E, kind=kind, backend=PRIMARY, term="kernel_sub_mean")
        kl = _pick(E, kind=kind, backend=PRIMARY, term="kernel_sub_mean_late")
        kj = _pick(E, kind=kind, backend=PRIMARY, term="kernel_sub_joint")
        curve = E[(E["kind"] == kind) & (E["backend"] == PRIMARY) & (E["term"] == "kernel_sub")]
        if km is not None:
            txt = f"  Per-chirp kernel (GLMM): mean OR over 0-{cfg.hist_max:g} s {_ci(km)}"
            if kl is not None:
                txt += f", over {g_lag:g}-{cfg.hist_max:g} s {_ci(kl)}"
            if len(curve):
                pk = curve.loc[(s * curve["estimate"]).idxmax()]
                txt += f"; peak OR {pk['ratio']:.2f} at lag {pk['lag']:.2f} s"
            if kj is not None:
                txt += f"; joint test p = {_p(kj['p'])}"
            L.append(txt + ".")
        tau_scan = E[(E["kind"] == kind) & (E["backend"] == PRIMARY)
                     & E["term"].str.match(r"sub_any_\d+ms$")].copy()
        if len(tau_scan):
            tau_scan["tau"] = tau_scan["term"].str.extract(r"(\d+)ms")[0].astype(float) / 1000
            tau_scan = tau_scan.sort_values("tau")
            L.append("  tau scan (GLMM OR, >= 1 sub chirp in preceding tau): " + ", ".join(
                f"{t:g} s {r:.2f}" for t, r in zip(tau_scan["tau"], tau_scan["ratio"],
                                                   strict=True)) + ".")
        if len(js):
            z = (js["observed"] - js["null_mean"]) / js["null_sd"]
            parts = [f"{r['null']} {'1 s' if r['stat'].startswith('tau_') else 'kernel'} "
                     f"p = {_p(r['p_two_sided'])}" for _, r in js.iterrows()]
            n = int(js["n_jitter"].min())
            L.append(f"  Jitter test (within dyads; FE refits under {n} null sub chirp trains per "
                     f"null; min attainable p {2 / (n + 1):.3f}): {', '.join(parts)}; observed "
                     f"{np.abs(z).min():.1f}-{np.abs(z).max():.1f} null SD from the null mean. "
                     "It tests a pooled within-dyad association, not consistency across dyads; "
                     "the shift null keeps inter-chirp intervals, the matched null does not.")
        else:
            L.append("  Jitter test: not run.")
        if len(lk):
            i_lo, i_hi = lk["ratio"].idxmin(), lk["ratio"].idxmax()
            L.append(f"  Leave one dyad out (FE HR, 1 s): {lk.loc[i_lo, 'ratio']:.2f} (without "
                     f"{_short(lk.loc[i_lo, 'left_out'])}) to {lk.loc[i_hi, 'ratio']:.2f} (without "
                     f"{_short(lk.loc[i_hi, 'left_out'])}).")
        if kind == "offset":
            lp = _lag_prediction(oe, tpt)
            if lp:
                L.append(lp)
        led = sets.get("drop_led_corrected", [])
        if led and len(oe_k):
            n_led = int(oe_k[oe_k["recording"].isin(led)]["observed"].sum())
            L.append(f"  Support: {n_led} of {int(oe_k['observed'].sum())} {EVENT_NAME[kind]} "
                     f"preceded by a sub chirp (1 s) come from the {len(led)} dyads whose "
                     f"video-to-grid alignment had to be repaired "
                     f"({', '.join(_short(r) for r in led)}).")
        if sens is not None and len(sens):
            parts = []
            for name in [*sets, "unambiguous_chirps"]:
                ss = sens[(sens["kind"] == kind) & (sens["subset"] == name)
                          & (sens["term"] == TAU_TERM)]
                if not len(ss):
                    continue
                gg_, ff_, rr_ = (_pick(ss, backend=b) for b in (PRIMARY, CHECK, META))
                nrec = int(ff_["n_recordings"]) if ff_ is not None else 0
                parts.append(f"{name} ({SENS_LABEL[name]}; {nrec} dyads): GLMM {_ci(gg_)}, FE "
                             f"{_ci(ff_, p=True)}, RE {_ci(rr_)}")
            if parts:
                L.append("  Sensitivity (1 s): " + "; ".join(parts) + ".")
        if desc is not None and len(desc.get("tests", [])):
            t = desc["tests"]
            d = t[(t["analysis"] == f"post_chirp_hazard:{kind}:sub") & (t["null"] == "matched")]
            if len(d) and np.isfinite(d["ratio"].iloc[0]):
                r = d.iloc[0]
                L.append(f"  Descriptive: hazard in the {r['window_hi']:g} s after an "
                         f"{RISK_NAME[kind]} sub chirp / elapsed-matched null = {r['ratio']:.2f} "
                         f"[{r['ratio_ci_low']:.2f}, {r['ratio_ci_high']:.2f}] (dyad bootstrap); "
                         + _desc_dyads(tpt, kind) + ".")
        dm = _pick(E, kind=kind, backend=PRIMARY, term="dom_any_1000ms")
        if dm is not None:
            L.append(f"  Comparison, dominant chirp in the preceding 1 s: GLMM OR {_ci(dm)} "
                     f"({dm['note'].split(';')[0]}); not interpretable (caveat 5).")
        if sim is not None:
            L += _sim_lines(sim, kind, f, cfg)
        if acausal is not None:
            L += hz_acausal.summary_lines(acausal["effects"], acausal.get("counts"),
                                          acausal.get("lodo"), kind, acausal.get("rates"))
            L += hz_acausal_sim.summary_lines(acausal_sim, kind)
        L.append("")

    n_jit = int(jsum["n_jitter"].min()) if len(jsum) else 0
    n_rep = "none"
    if sim is not None and len(sim.get("scenarios", [])):
        sc = sim["scenarios"]
        n_rep = str(int(sc.loc[sc["scenario"] == "no_effect", "n_reps"].max()))
    L.append("Caveats that matter")
    L += _caveats(trials, flags, haz, sens, sim, cfg, sets)
    L += ["", (f"Run {run_time or datetime.now().strftime('%Y-%m-%d %H:%M')}, wall time "
               f"{wall_s / 60:.1f} min, seed {cfg.seed}, jitter refits per null {n_jit}, "
               f"simulation replicates {n_rep}"
               f"{' (QUICK MODE: too few replicates for inference)' if cfg.quick else ''}. "
               "All effects with CIs: effects.csv; per-dyad O/E: hazard/hazard_dyad_oe.csv; "
               "details: hazard/hazard_summary.txt, simulation/sim_summary.txt, "
               "validation/data_report.md.")]
    return "\n".join(L)


def _sim_lines(sim: dict[str, Any], kind: str, f: pd.Series | None, cfg: Config) -> list[str]:
    summ, est = sim["summary"], sim["estimates"]
    meta = sim.get("trigger_meta", pd.DataFrame())
    L: list[str] = []

    def row(sc: str, term: str = TAU_TERM, backend: str = CHECK) -> pd.Series | None:
        return _pick(summ, scenario=sc, kind=kind, term=term, backend=backend)

    rec, nul = row("recovery"), row("no_effect")
    if rec is not None and nul is not None:
        rg, rr = row("recovery", backend=PRIMARY), row("recovery", backend=META)
        L.append(
            f"  Simulation, recovery (rates calibrated to these data under the effect; true HR "
            f"{np.exp(rec['truth']):.2f}): FE estimate {rec['ratio_median']:.2f} "
            f"[{rec['ratio_q025']:.2f}, {rec['ratio_q975']:.2f}], power {_pct(rec['reject_cluster'])}"
            + (f"; GLMM coverage {_pct(rg['coverage_naive'])}" if rg is not None else "")
            + (f"; RE coverage {_pct(rr['coverage_naive'])}, power {_pct(rr['reject_naive'])}"
               if rr is not None else "") + ".")
        for sc, lab in (("no_effect", "one common zero effect"),
                        ("het_no_effect", "dyad effects vary around zero (SD fitted to the data)")):
            a, b = row(sc), row(sc, backend=PRIMARY)
            c = row(sc, backend=META)
            if a is None:
                continue
            sd = row(sc, term="dyad_sd_sub_effect", backend=META)
            L.append(f"  Simulation, false positives at nominal 5%, {lab}: FE model-based Wald "
                     f"{_rej(a, 'naive')}, FE cluster-robust {_rej(a, 'cluster')}, GLMM "
                     f"{_rej(b)}, RE {_rej(c)}"
                     + (f" (true dyad SD {sd['truth']:.2f})" if sd is not None
                        and np.isfinite(sd.get('truth', np.nan)) and sd['truth'] > 0 else "")
                     + ".")
        ann, det = row("annotation"), row("detection")
        het = row("het_recovery", backend=META)
        L.append("  Simulation, observation error: "
                 + (f"0.3 s SD boundary error on a 0.5 s lattice gives {ann['ratio_median']:.2f} "
                    f"(true {np.exp(ann['truth']):.2f}; cluster coverage "
                    f"{_pct(ann['coverage_cluster'])})" if ann is not None else "")
                 + (f"; missed chirps {det['ratio_median']:.2f}" if det is not None else "")
                 + (f"; heterogeneous effect (mean HR {np.exp(het['truth']):.2f}): RE mean "
                    f"{het['ratio_median']:.2f}, coverage {_pct(het['coverage_naive'])}"
                    if het is not None and np.isfinite(het.get("truth", np.nan)) else "") + ".")
    if kind == "offset" and f is not None:
        phi, frac = hz_sim.trigger_match(summ, meta, float(np.log(f["ratio"])))
        if np.isfinite(phi):
            L.append(f"  Common trigger: with NO causal chirp effect, a hidden event that ends "
                     f"the chase and elicits a sub chirp reproduces the observed FE HR if its "
                     f"chirps make up ~{phi:.0%} of in-chase sub chirps (realized share; ~"
                     f"{frac:.0%} of offsets preceded by one).")
        lags = []
        for lag in (0.25, 0.5, 0.75):
            r = row(f"lag_{lag:g}")
            if r is not None:
                pk = hz_sim.kernel_peak(sim.get("kernels", pd.DataFrame()), f"lag_{lag:g}",
                                        "offset")
                lags.append(f"{lag:g} s: HR {r['ratio_median']:.2f} [{r['ratio_q025']:.2f}, "
                            f"{r['ratio_q975']:.2f}]"
                            + (f", kernel peak {pk:.2f} s" if np.isfinite(pk) else ""))
        if lags:
            L.append(f"  Late offset annotation, NO causal effect (observed FE HR "
                     f"{f['ratio']:.2f}): " + "; ".join(lags) + ".")
        prof = {sc: hz_sim.lag_profile(est, sc, "offset") for sc in
                ("observed", "recovery", "annotation", "trigger_0.25", "trigger_0.25_annotation",
                 "trigger_0.5_slow", "lag_0.5")}
        o = prof["observed"]
        if o["n"]:
            parts = [f"{sc} {p['short']:.2f} [{p['short_lo']:.2f}, {p['short_hi']:.2f}] / "
                     f"{p['long']:.2f} [{p['long_lo']:.2f}, {p['long_hi']:.2f}]"
                     for sc, p in prof.items() if sc != "observed" and p["n"]]
            c = prof["recovery"]
            longer = np.isfinite(c["long_hi"]) and o["long"] > c["long_hi"]
            L.append(f"  Lag profile, short = log HR(0.25 s)/log HR(1 s) / long = log HR(5 s)/"
                     f"log HR(1 s): observed {o['short']:.2f} / {o['long']:.2f}; simulated "
                     "(95% range) " + "; ".join(parts) + ". "
                     + ("The observed association lasts longer than a simulated 0-1 s causal "
                        "effect. " if longer else "")
                     + "The lag profile does not separate a causal effect from a common trigger "
                     "or an annotation lag with 0.5-1 s delays.")
    if kind == "onset":
        parts, reach = [], []
        o_hr = float(f["ratio"]) if f is not None else np.nan
        for sc, lab in (("hidden_state", "slow (30 s episodes, 20% of time)"),
                        ("hidden_state_short", "short (3 s episodes, 10% of time)")):
            hs = row(sc)
            if hs is None:
                continue
            lp = hz_sim.lag_profile(est, sc, "onset")
            weak = abs(np.log(hs["ratio_median"])) < 0.05
            parts.append(f"{lab} spurious HR {hs['ratio_median']:.2f} [{hs['ratio_q025']:.2f}, "
                         f"{hs['ratio_q975']:.2f}], long ratio "
                         + ("n/a (no association)" if weak else f"{lp['long']:.2f}"))
            reach.append(bool(np.isfinite(o_hr) and hs["ratio_median"] <= o_hr))
        o = hz_sim.lag_profile(est, "observed", "onset")
        if parts:
            L.append("  Latent 'subordinate away' state (chirps up, chasing down), NO causal "
                     "effect: " + "; ".join(parts) + f"; observed long ratio {o['long']:.2f}. A "
                     "slow state gives an association that persists over tau, unlike the data. A "
                     "short state gives a local association that timing cannot tell from a "
                     "causal effect"
                     + ("" if any(reach) else "; at the simulated strengths it is weaker than "
                        f"the observed FE HR {o_hr:.2f}, but a stronger one was not ruled out")
                     + ".")
    return L


def _caveats(trials: list[hz_data.Trial], flags: pd.DataFrame, haz: dict[str, Any],
             sens: pd.DataFrame | None, sim: dict[str, Any] | None, cfg: Config,
             sets: dict[str, list[str]]) -> list[str]:
    E = haz["effects"]
    G = len(trials)
    fish = {(tr.group, tr.dom_fish) for tr in trials} | {(tr.group, tr.sub_fish) for tr in trials}
    win = f"0-{cfg.max_time / 3600:g} h" if cfg.max_time else "whole recording"
    # annotation-lag numbers
    lag_txt = ""
    if sens is not None and len(sens):
        sh = sens[(sens["kind"] == "offset") & sens["subset"].astype(str).str.startswith(
            "shift_offsets_-") & (sens["term"] == TAU_TERM) & (sens["backend"] == CHECK)]
        if len(sh):
            lows = sh["ratio_low"].to_numpy(float)
            lag_txt = (f" Moving every annotated offset {min(-float(x.split('_')[-1][:-1]) for x in sh['subset']):g}-"
                       f"{max(-float(x.split('_')[-1][:-1]) for x in sh['subset']):g} s earlier "
                       f"(undoing a uniform late-marking lag) leaves FE HR "
                       f"{sh['ratio'].min():.2f}-{sh['ratio'].max():.2f}"
                       + (", every CI above 1" if np.all(lows > 1) else ", not every CI above 1")
                       + ", so a uniform lag of that size does not explain (a) in these data.")
    if sim is not None:
        r = _pick(sim["summary"], scenario="lag_0.5", kind="offset", term=TAU_TERM, backend=CHECK)
        f = _pick(E, kind="offset", backend=CHECK, term=TAU_TERM)
        if r is not None and f is not None:
            lag_txt += (f" In simulation, offsets marked 0.5 s late with no chirp effect give HR "
                        f"{r['ratio_median']:.2f} (observed {f['ratio']:.2f}).")
    fl = flags[flags["recording"].isin({tr.recording for tr in trials})]
    un = fl[fl["category"] == "chirp_unassigned_upstream"]
    n_un = int(un["value"].sum()) if len(un) else 0
    amb = sets.get("drop_sender_ambiguous", [])
    close = sets.get("drop_close_eodf", [])
    la = np.array([float(tr.meta.get("last_boris_event", np.nan)) for tr in trials]) / 3600
    lo_a, hi_a = (float(np.nanmin(la)), float(np.nanmax(la))) if np.isfinite(la).any() else (
        np.nan, np.nan)
    past = [f"{_short(tr.recording)}: {tr.obs_end - tr.meta['last_boris_event']:.0f} s"
            for tr in trials if np.isfinite(tr.meta.get("last_boris_event", np.nan))
            and tr.meta["last_boris_event"] < tr.obs_end]
    het_lines = []
    for kind in hz_hazard.KINDS:
        t = _pick(E, kind=kind, backend=META, model=TAU_MODEL, term="dyad_sd_sub_effect")
        if t is not None:
            het_lines.append(f"{kind} {t['estimate']:.2f}")
    return [
        ("1. Association, not causation. Sub chirps PREDICT chase events beyond elapsed time "
         "and the controls. A hidden common trigger (a change in pursuit or escape that both "
         "elicits a chirp and ends the chase) gives the same pattern; timing alone cannot "
         "separate the two. Annotation lag is a special case: subordinates chirp most just "
         "after chases end, so offsets marked late would put those chirps before the "
         "annotated offset." + lag_txt + " A variable (non-uniform) lag or a common trigger "
         "remains possible; separating them needs chirp playback during chases or an "
         "independent measure of chase end (video tracking)."),
        ("2. Chase boundaries were annotated to ~0.5 s (durations cluster at 0.5 s steps), so "
         "kernel shape below ~0.5 s is not interpretable (e.g. the positive onset kernel at "
         "the shortest lags), and boundary error attenuates effects, most for onsets (see the "
         "simulated boundary-error estimate). The lag-guarded model is the pre-specified check; "
         "for (b) the across-dyad verdict depends on how lags < 0.5 s are handled."),
        (f"3. Chirp detection and sender assignment are automatic (RCNN + EODf tracks). The "
         f"detector dropped {n_un} distinct detections without a sender inside the analysis "
         f"window before writing chirp_times_rcnn.npy (mostly 2020-06-02; flag "
         f"chirp_unassigned_upstream), so the npy files only look fully assigned. Sender "
         f"assignment is geometrically ambiguous (both EODfs inside the chirp box) for most "
         f"chirps in {len(amb)} dyads ({', '.join(_short(r) for r in amb)}), which includes the "
         f"two with large boxes (06-02, 06-11); close EODfs ({len(close)} dyads) are one cause, "
         "large boxes another. See the drop_sender_ambiguous, unambiguous_chirps and "
         "drop_close_eodf sensitivity rows. Missed chirps were simulated; misassigned chirps "
         "were not."),
        (f"4. Few dyads that disagree: {G} dyads from {len(fish)} fish; fish recur across dyads, "
         "which no model here accounts for. The GLMM assumes one chirp effect shared by all "
         "dyads, so its interval (and its very small p) holds only under that assumption; the "
         f"data reject it (between-dyad SD of the log HR: {', '.join(het_lines)}). The FE "
         "cluster-robust interval and the random-effects (RE) mean reflect between-dyad "
         "differences; see the heterogeneous-null simulation for how each backend is "
         "calibrated when dyads differ."),
        ("5. Chase direction is not annotated (BORIS Subject 'No focal subject'; the BORIS "
         "project file configures no subjects); dominant -> subordinate is assumed, not "
         "verified. Dominant-chirp terms are sparse (few dyads and events) and their "
         "cluster-robust joint tests are degenerate; the GLMM's dyad-level controls (size "
         "difference, dEODf, intercept, dyad SD) trade off against the random intercepts and "
         "are not interpretable. Do not interpret these rows in effects.csv."),
        (f"6. Window {win}: chase onsets drop to 0-6 per dyad in 3.0-3.5 h and the last "
         f"annotated BORIS event falls at {lo_a:.2f}-{hi_a:.2f} h in every included dyad, while "
         "some subordinates keep chirping; including later time would bias the onset model. "
         "The draw 2020-06-08 is annotated to the recording end, so chasing most likely "
         "stopped, but a shorter annotation effort cannot be excluded; the light schedule is "
         "not recorded in the dataset."
         + (f" The window extends past the last annotation in {len(past)} dyad(s) "
            f"({', '.join(past)})." if past else "")),
        ("7. The tau = 1 s indicator is where the tau scan peaks, so its ratio is an upper-end "
         "summary; the kernel mean is the unselected per-chirp summary."),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    cfg = build_config(a)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a" if a.report_only else "w")
    stdout = sys.stdout
    sys.stdout = Tee(stdout, log)  # type: ignore[assignment]
    try:
        _run(a, cfg, out)
    finally:
        sys.stdout = stdout
        log.close()


def _read(f: Path) -> pd.DataFrame:
    # keep_default_na=False: pandas would read the scenario name 'null' etc. as NaN;
    # round_trip: --report-only rewrites every float bit-identically
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, keep_default_na=False, na_values=[""], float_precision="round_trip")


def _report_only(cfg: Config, out: Path) -> None:
    """Rebuild effects.csv and summary.txt from saved tables (no refitting).

    The configuration of the original run (config.json) replaces the CLI one,
    so trial selection and window match the saved tables; the run time stamp
    is the original start time.
    """
    f = out / "config.json"
    started = None
    if f.exists():
        js = json.loads(f.read_text())
        saved = js["config"]
        names = {fl.name for fl in fields(Config)} - {"data_root", "out_dir"}
        cfg = replace(cfg, **{k: v for k, v in saved.items() if k in names})
        started = js.get("started")
    trials, flags, _ = hz_data.load_trials(cfg)
    h, sdir = out / "hazard", out / "simulation"
    haz = dict(effects=_read(h / "hazard_effects.csv"),
               jitter_summary=_read(h / "hazard_jitter_summary.csv"),
               leave_one_dyad_out=_read(h / "hazard_leave_one_dyad_out.csv"),
               dyad_oe=_read(h / "hazard_dyad_oe.csv"))
    if haz["effects"].empty:
        raise SystemExit(f"no {h / 'hazard_effects.csv'}; run the pipeline first")
    tests = _read(out / "descriptives" / "tests.csv")
    desc = dict(tests=tests, tests_per_trial=_read(out / "descriptives" / "tests_per_trial.csv")
                ) if len(tests) else None
    sens = _read(h / "hazard_sensitivity.csv")
    sim = None
    if (sdir / "sim_summary.csv").exists():
        sim = dict(summary=_read(sdir / "sim_summary.csv"),
                   estimates=_read(sdir / "sim_estimates.csv"),
                   trigger_meta=_read(sdir / "sim_trigger_meta.csv"),
                   jitter_fpr=_read(sdir / "sim_jitter_fpr.csv"),
                   scenarios=_read(sdir / "sim_scenarios.csv"),
                   kernels=_read(sdir / "sim_kernels.csv"))
    tim = _read(out / "timings.csv")
    wall = float(tim["seconds"].iloc[-1]) if len(tim) else np.nan
    acausal = hz_acausal.load_tables(out)
    acausal_sim = hz_acausal_sim.load_tables(out)
    if acausal is not None:  # acausal_summary.txt carries the simulation block too
        hz_acausal.rewrite_summary(out)
    eff = effects_table(haz, desc, sens, acausal)
    eff.to_csv(out / "effects.csv", index=False)
    run_time = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M") if started else None
    text = answer_text(trials, flags, haz, desc, sens, sim, cfg, wall, run_time, acausal,
                       acausal_sim)
    (out / "summary.txt").write_text(text + "\n")
    print(text)


def _run(a: argparse.Namespace, cfg: Config, out: Path) -> None:
    if a.report_only:
        _report_only(cfg, out)
        return
    timer = Timer()
    started = datetime.now()
    nulls = [x for x in a.nulls.split(",") if x]
    print(f"chirp_chase_hazard: data {cfg.data_root} -> {out}  "
          f"(python {platform.python_version()}, numpy {np.__version__}, "
          f"pandas {pd.__version__})", flush=True)

    def write_config() -> None:
        (out / "config.json").write_text(json.dumps(
            dict(config=_jsonable(cfg.as_dict()), argv=sys.argv, started=started.isoformat(),
                 effective=effective_settings(cfg, nulls, not a.no_vb),
                 hazard_defaults=_jsonable(hz_hazard.DEFAULTS),
                 acausal_defaults=_jsonable(hz_acausal.ACAUSAL_DEFAULTS),
                 acausal_sim_defaults=_jsonable(hz_acausal_sim.ACSIM_DEFAULTS),
                 sim_defaults=_jsonable(hz_sim.SIM_DEFAULTS)),
            indent=2))

    write_config()
    with timer.step("1 load + validate"):
        trials, flags, summary = hz_data.load_trials(cfg)
        rep = hz_data.write_validation_report(trials, flags, summary, out / "validation", cfg)
        ev = hz_data.events_table(trials)
        ev.to_csv(out / "validation" / "events.csv", index=False)
        n_warn = int((flags["severity"] != "info").sum())
        print(f"{len(trials)} of {len(summary)} trials included; {len(ev):,} events; "
              f"{n_warn} warning/error flags -> {rep}")
        if not trials:
            raise SystemExit("no trials included; see validation/data_report.md")

    desc = None
    if not a.skip_descriptives:
        with timer.step("2 descriptives"):
            desc = hz_desc.run_descriptives(trials, cfg)

    with timer.step("3 hazard models + jitter tests"):
        haz = hz_hazard.run_hazard(trials, cfg, jitter_nulls=nulls,
                                   vb_check=not (a.no_vb or cfg.quick))
        print(haz["timings"].to_string(index=False))

    with timer.step("4 sensitivity refits"):
        sens = sensitivity_fits(trials, flags, cfg)
        if len(sens):
            sens.to_csv(out / "hazard" / "hazard_sensitivity.csv", index=False)

    acausal = None
    if not a.skip_acausal:
        with timer.step("4b acausal control"):
            ac = hz_acausal.run_acausal(trials, cfg, verbose=True)
            print(ac["timings"].to_string(index=False))
            acausal = {k: ac[k] for k in ("effects", "counts", "lodo", "rates")}

    sim = None
    if not a.skip_sim:
        with timer.step("5 simulation checks"):
            # heterogeneous scenarios use the between-dyad SD fitted to these data
            for kind in hz_hazard.KINDS:
                t = _pick(haz["effects"], kind=kind, backend=META, model=TAU_MODEL,
                          term="dyad_sd_sub_effect")
                if t is not None and np.isfinite(t["estimate"]):
                    cfg.extra[f"sim_het_sd_{kind}"] = round(float(t["estimate"]), 3)
            write_config()
            sim = hz_sim.run_simulation_checks(trials, cfg)

    acausal_sim = None
    if not (a.skip_acausal or a.skip_sim):
        with timer.step("5b acausal simulation"):
            hz_acausal_sim.run_acausal_sim(trials, cfg, verbose=True)
            acausal_sim = hz_acausal_sim.load_tables(out)  # also rewrites acausal_summary.txt

    with timer.step("6 effects table + summary"):
        eff = effects_table(haz, desc, sens, acausal)
        eff.to_csv(out / "effects.csv", index=False)
        print(f"effects.csv: {len(eff):,} rows "
              f"({', '.join(f'{k} {v}' for k, v in eff['analysis'].value_counts().items())})")
        bad = eff[(eff["converged"] == False)  # noqa: E712
                  & eff["analysis"].isin(["hazard", "sensitivity", "acausal"])]
        if len(bad):
            print("WARNING: non-converged fits: " + ", ".join(
                sorted({f"{r.kind}/{r.model}/{r.backend}" for r in bad.itertuples()})))

    tim = timer.frame()
    tim.to_csv(out / "timings.csv", index=False)
    text = answer_text(trials, flags, haz, desc, sens, sim, cfg, float(tim["seconds"].iloc[-1]),
                       started.strftime("%Y-%m-%d %H:%M"), acausal, acausal_sim)
    (out / "summary.txt").write_text(text + "\n")
    print("\n" + text)
    print("\nTimings (s):\n" + tim.to_string(index=False))


if __name__ == "__main__":
    main()
