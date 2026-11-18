"""Acausal control: are sub chirps specific to the time BEFORE a chase event?

The hazard models (hz_hazard) find that a subordinate chirp in the preceding
1 s goes with a higher hazard that the chase ends (a) and a lower hazard that
a new chase starts (b). Around chase offsets, however, subordinate chirps are
almost symmetric in raw counts: about as many fall in the seconds after the
annotated offset as in the seconds before it. The association could then be one
chirp burst straddling an imprecisely annotated chase end (a response to the
chase end, a common trigger, or annotation lag) rather than something specific
to chirps that precede the event. This module tests that. Every analysis here
DELIBERATELY uses chirps from after the outcome bin; the results are an
"acausal control" and never enter the primary models.

Units: seconds and Hz. Bin k covers [t_k, t_k + bw).

A. Past vs future window (Granger-style; offset and onset). The standard
   design (same risk sets and controls, incl. the dominant past indicator)
   plus future-window columns, mirrored around the outcome bin, which is
   excluded on both sides:
     past    sub_any_<tau>  >= 1 sub chirp in bins k - n .. k - 1  = [t_k - tau, t_k)
     future  sub_fut_<tau>  >= 1 sub chirp in bins k + 1 .. k + n  = [t_k + bw, t_k + bw + tau)
     guard   sub_grd_500ms  bins k - 30 .. k - 11 = [t_k - 1.5, t_k - 0.5)
             sub_fgd_500ms  bins k + 11 .. k + 30 = [t_k + bw + 0.5, t_k + bw + 1.5)
     kernel  sub_h1..5 (lag j = bin k - j) and sub_fh1..5 (bin k + j), same
             raised-cosine basis, signed lag +-j * bw
   and the same for dominant chirps (dom_fut_*, dom_fgd_*, dom_fh*) as
   controls. Models per tau: past only ('past_<ms>', the standard model),
   future only ('fut_<ms>'), joint ('pf_<ms>'); the same for the guard
   ('*_guard_500ms'); the joint two-sided kernel ('kernel_two_sided').
   Backends: glm_fe (cloglog, dyad FE, cluster-robust t(G-1); primary for the
   contrast), mixed_laplace (GLMM), and per-term per-dyad O/E with the
   random-effects meta-analysis (re_meta) for the 1 s and guarded windows.
   Contrasts: log HR_past - log HR_future of the joint model (each adjusted
   for the other), kernel mean before vs after the bin, and a joint Wald test
   of the kernel shape difference; leave-one-dyad-out range of the contrasts.
   The future window uses the full chirp train (chirps after the 3 h window
   or inside unknown-state intervals exist and are used). Bins whose longest
   future window (max of tau_grid, hist_max, guard) runs past the recording
   end (Trial.t_end) are dropped from the acausal designs only and counted.

B. Mirror surrogate (offset; the user's literal request). For each chase i,
   sub chirps after its offset, t in (off_i, min(off_i + W, next onset, next
   unknown-state interval)), latency L = t - off_i, are placed at off_i - L
   and kept only if off_i - L > on_i (inside chase i; dropped ones counted).
   Mirror train = real out-of-chase sub chirps (unchanged) + mirrored chirps;
   ALL real in-chase sub chirps removed. Fair counterpart 'restricted_real' =
   real out-of-chase sub chirps + real in-chase sub chirps less than W before
   their own chase's offset, kept only where the mirror could have a chirp at
   the same latency (off_i + L before the next onset and unknown-state
   interval; the mirror's support). Both arms are selected by offset timing in
   the same way; the models (tau scan, guard, kernel; FE, GLMM, RE) are the
   standard offset models on the replaced train (build_design chirp_override).
   'real_full' (the unchanged train) is the reference. Paired contrast
   log HR(restricted_real) - log HR(mirror) with a dyad bootstrap (dyads
   resampled with replacement, both arms refitted with FE on the same
   resample; percentile CI) for tau = 1 s and the guarded indicator.

C. Constant backward shift (offset; secondary). Post-offset sub chirps (same
   window, W = 2 s) moved to t - Delta, Delta in {0.5, 1, 2} s, kept if inside
   chase i, replacing the real in-chase chirps as in B: what the post-offset
   chirps alone would look like if every offset were annotated Delta late. A
   shifted train keeps only chirps with L < Delta and TRANSLATES them (to
   off - Delta + L; the mirror reflects them to off - L), so it is compared
   with the restricted-real and mirror arms of W = Delta (FE only, no
   bootstrap; built for every Delta whose W is not already in B).

Selected trains (B, C) put every in-chase chirp into the last W seconds of a
chase, so their hazard ratios are inflated by that selection: compare arms
with each other (restricted_real vs mirror / shift), never with the primary
HR 2.11.

Interpretation rule. No acausal statistic has a null of 1: the risk sets are
not time-symmetric (offset: in-chase bins only; the after-window of an offset
bin lies outside the chase and that of a bin just before an offset holds the
post-offset chirp burst). Each statistic is therefore read against its
simulated distribution under each account (hz_acausal_sim), not against 1.
A value compatible with causal and non-causal accounts alike does not
discriminate; in particular, with the observed post-offset chirp response a
causal effect also gives past ~ future, so past ~ future is not evidence
against a causal effect. A value inconsistent with every non-causal account
(fewer than 2.5% of its simulated datasets at or beyond the observed value, or
p < 0.05 by z) indicates direction-specific timing those accounts do not
produce, not causality.
Onset (b): the after-window of an onset bin lies inside the chase the outcome
starts, i.e. it is raised by the outcome itself. The past / future contrast is
then below 1 without any chirp effect, and HR_past of the joint model
(adjusted for a consequence of the outcome) is biased toward < 1, the
direction of (b). Neither tests (b); the (b) estimate stays the standard
past-only HR. A contrast beyond a simulated no-effect account shows an
asymmetry beyond chase-start chirping only if that account reproduces the
observed chase-start chirping.

Run: <venv>/bin/python src_hazard/hz_acausal.py DATA --out DIR [--quick] [--n-boot N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import hz_hazard as hz  # noqa: E402
from hz_config import Config  # noqa: E402
from hz_data import Trial, _in_intervals  # noqa: E402

KINDS = hz.KINDS
_ms = hz._ms

ACAUSAL_DEFAULTS: dict[str, Any] = dict(
    acausal_windows=(2.0, 5.0),  # s, W of the mirror / restricted-real arms
    acausal_report_window=2.0,  # s, W reported first (= hist_max)
    acausal_shift_window=2.0,  # s, W of the constant-shift arms
    acausal_shift_deltas=(0.5, 1.0, 2.0),  # s
    acausal_n_boot=500,  # dyad bootstrap replicates of restricted_real - mirror
    acausal_boot_models=("tau_1000ms", "guard_500ms"),
    # onset leave-one-dyad-out models (offset: every joint window and the kernel)
    acausal_lodo_models=("pf_1000ms", "pf_guard_500ms", "kernel_two_sided"),
    acausal_re_models=("tau_1000ms", "guard_500ms"),  # windows with per-dyad O/E + RE
)

INTERPRETATION = (
    "Reading rule: no acausal statistic has a null of 1 (the risk sets are not time-symmetric "
    "and the post-offset chirp burst enters the after-windows), so each is read against its "
    "simulated distribution under each account (below), not against 1. A value compatible with "
    "causal and non-causal accounts alike does not discriminate; with the observed post-offset "
    "chirp response a causal effect also gives past ~ future, so past ~ future is not evidence "
    "against a causal effect. A value inconsistent with every non-causal account (fewer than "
    "2.5% of its simulated datasets at or beyond the observed value, or p < 0.05 by z where the "
    "dyads differ more than simulated) indicates direction-specific timing that those accounts "
    "do not produce, not causality; verdicts that rest on borderline accounts (2.5-10%, or p "
    "0.05-0.2) are flagged, and accounts that were not simulated remain possible.")
INTERPRETATION_ONSET = (
    "Reading rule (b): the onset acausal statistics are not a test of (b) and are not used as "
    "evidence for or against it. Their no-effect values are shifted (simulation below); a "
    "contrast beyond a simulated no-effect account says only that the before / after asymmetry "
    "exceeds what that account produces, and shows an asymmetry beyond chase-start chirping "
    "only if the account also reproduces the observed chase-start chirping (HR_future, chirps "
    "in the first 1 s of chases).")

VARIANT_NOTE = {
    "two_sided": "acausal control A: past and future chirp windows around the outcome bin",
    "real_full": "reference: unchanged sub chirp train (= primary model)",
    "restricted_real": ("real out-of-chase sub chirps + real in-chase sub chirps < W before "
                        "their chase's offset where the mirror could have one (off + L before "
                        "the next onset / unknown state); all other in-chase chirps removed"),
    "mirror": ("real out-of-chase sub chirps + post-offset sub chirps (latency L < W, before "
               "the next onset) reflected to off - L inside the same chase; all real "
               "in-chase chirps removed"),
    "shift": ("real out-of-chase sub chirps + post-offset sub chirps (L < W) moved to t - "
              "Delta, kept if inside the same chase; all real in-chase chirps removed"),
}
SELECTION_NOTE = ("selected train: every in-chase chirp lies in the last W s of a chase, so "
                  "the HR is inflated by selection; compare arms, not with the primary HR")


def _opt(cfg: Config, name: str) -> Any:
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return cfg.extra.get(name, ACAUSAL_DEFAULTS[name])


def _gname(cfg: Config) -> str:
    return _ms(hz._guard(cfg)[0])


# ---------------------------------------------------------------------------
# A. Future-window covariates
# ---------------------------------------------------------------------------
def future_extent(cfg: Config) -> float:
    """Longest future window (s) after the outcome bin: max(tau_grid, hist_max, guard)."""
    g, w = hz._guard(cfg)
    return float(max(cfg.hist_max, *cfg.tau_grid, g + w))


def future_features(
    t: np.ndarray, prefix: str, t0: float, K: int, k: np.ndarray, cfg: Config, B: np.ndarray,
    taus: Sequence[float] | None = None, guard: bool = True,
) -> dict[str, np.ndarray]:
    """Future-window covariates of one event train at bins ``k`` (mirror of
    hz_hazard._train_features around the outcome bin, which is excluded).

    <prefix>_fh<i>     sum_{j=1..J} B[j-1, i] * n(bin k + j)
    <prefix>_fut_<ms>  >= 1 event in bins k + 1 .. k + tau / bw
    <prefix>_fgd_<ms>  >= 1 event in bins k + g + 1 .. k + g + w (guard lag g, window w, in bins)
    Events are counted from the full train, beyond the analysis window too.
    """
    bw = cfg.bin_width
    J = B.shape[0]
    n_fut = int(round(future_extent(cfg) / bw)) + 2
    L = K + n_fut + 1
    m = np.floor((np.asarray(t, float) - t0) / bw).astype(np.int64)
    cnt = np.bincount(m[(m >= 0) & (m < L)], minlength=L).astype(float)
    C = np.concatenate([[0.0], np.cumsum(cnt)])  # C[m] = events in bins < m
    k = np.asarray(k, np.int64)
    H = np.zeros((len(k), B.shape[1]))
    for j in range(1, J + 1):
        H += cnt[k + j][:, None] * B[j - 1][None, :]
    out: dict[str, np.ndarray] = {f"{prefix}_fh{i + 1}": H[:, i] for i in range(B.shape[1])}
    for tau in (cfg.tau_grid if taus is None else taus):
        n = max(1, int(round(tau / bw)))
        out[f"{prefix}_fut_{_ms(tau)}"] = ((C[k + 1 + n] - C[k + 1]) > 0).astype(np.int8)
    if guard:
        g, w = hz._guard(cfg)
        ng, nw = int(round(g / bw)), max(1, int(round(w / bw)))
        out[f"{prefix}_fgd_{_ms(g)}"] = ((C[k + ng + nw + 1] - C[k + ng + 1]) > 0).astype(np.int8)
    return out


def add_future_columns(
    design: pd.DataFrame, trials: Sequence[Trial], cfg: Config,
    chirp_override: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Copy of ``design`` with sub/dom future-window columns; bins whose longest
    future window runs past the recording end are dropped (counted in
    ``attrs['future_bookkeeping']``)."""
    bw = cfg.bin_width
    B = design.attrs["basis"]
    ext = future_extent(cfg)
    recs = design["recording"].astype(str).to_numpy()
    kk = design["k"].to_numpy()
    n = len(design)
    cols: dict[str, np.ndarray] = {}
    keep = np.ones(n, bool)
    book = []
    by_rec = {tr.recording: tr for tr in trials}
    for rec in pd.unique(recs):
        tr = by_rec[rec]
        m = np.flatnonzero(recs == rec)
        k = kk[m]
        t0, K = float(tr.obs_start), hz._n_bins(tr, bw)
        sub = tr.chirps_sub if chirp_override is None else chirp_override.get(rec, tr.chirps_sub)
        for pre, x in (("sub", sub), ("dom", tr.chirps_dom)):
            for c, v in future_features(x, pre, t0, K, k, cfg, B).items():
                cols.setdefault(c, np.zeros(n, v.dtype))[m] = v
        past_end = t0 + (k + 1) * bw + ext > float(tr.t_end) + 1e-9
        keep[m[past_end]] = False
        y = design["y"].to_numpy()[m]
        book.append(dict(recording=rec, n_rows=len(m),
                         n_dropped_future_past_end=int(past_end.sum()),
                         n_events_dropped=int(y[past_end].sum()), t_end=float(tr.t_end),
                         last_sub_chirp=float(sub.max()) if len(sub) else np.nan))
    extra = pd.DataFrame(cols, index=design.index)
    out = pd.concat([design, extra], axis=1)
    if not keep.all():
        out = out.loc[keep].reset_index(drop=True)
    out.attrs = dict(design.attrs)
    out.attrs["future_bookkeeping"] = pd.DataFrame(book).assign(kind=design.attrs["kind"],
                                                              future_extent=ext)
    return out


def two_sided_specs(cfg: Config, taus: Sequence[float] | None = None
                    ) -> dict[str, dict[str, Any]]:
    """Model name -> chirp columns and the (past, future) sub terms of the contrast."""
    specs: dict[str, dict[str, Any]] = {}
    for tau in (cfg.tau_grid if taus is None else taus):
        ms = _ms(tau)
        pa, pd_, fa, fd = f"sub_any_{ms}", f"dom_any_{ms}", f"sub_fut_{ms}", f"dom_fut_{ms}"
        specs[f"past_{ms}"] = dict(cols=[pa, pd_], past=pa, fut=None, win=f"tau_{ms}")
        specs[f"fut_{ms}"] = dict(cols=[pd_, fa, fd], past=None, fut=fa, win=f"tau_{ms}")
        specs[f"pf_{ms}"] = dict(cols=[pa, pd_, fa, fd], past=pa, fut=fa, win=f"tau_{ms}")
    g = _gname(cfg)
    pa, pd_, fa, fd = f"sub_grd_{g}", f"dom_grd_{g}", f"sub_fgd_{g}", f"dom_fgd_{g}"
    specs[f"past_guard_{g}"] = dict(cols=[pa, pd_], past=pa, fut=None, win=f"guard_{g}")
    specs[f"fut_guard_{g}"] = dict(cols=[pd_, fa, fd], past=None, fut=fa, win=f"guard_{g}")
    specs[f"pf_guard_{g}"] = dict(cols=[pa, pd_, fa, fd], past=pa, fut=fa, win=f"guard_{g}")
    nb = int(cfg.n_basis)
    kc = [f"{s}_{h}{i + 1}" for s in hz.SENDERS for h in ("h", "fh") for i in range(nb)]
    specs["kernel_two_sided"] = dict(cols=kc, past="kernel", fut="kernel", win="kernel")
    return specs


# ---------------------------------------------------------------------------
# Effect rows for linear combinations of fitted coefficients
# ---------------------------------------------------------------------------
def _base(f: hz._FitOut) -> dict[str, Any]:
    return dict(model=f.model, kind=f.kind, backend=f.backend, link=f.link, n_bins=f.n_bins,
                n_events=f.n_events, n_recordings=f.n_rec, fit_time_s=round(f.fit_time, 2),
                converged=f.converged)


def _lin(f: hz._FitOut, weights: Mapping[str, np.ndarray | float]) -> tuple[float, float]:
    """Estimate and SE of sum_name w_name * beta_name (natural scale)."""
    idx = {nm: i for i, nm in enumerate(f.names)}
    c = np.zeros(len(f.names))
    for nm, w in weights.items():
        if nm not in idx:
            return np.nan, np.nan
        c[idx[nm]] += float(w)
    return float(c @ f.beta), float(np.sqrt(max(c @ f.cov @ c, 0.0)))


def _row(f: hz._FitOut, term: str, term_type: str, est: float, se: float, note: str,
         lag: float = np.nan) -> dict[str, Any]:
    crit = hz._crit(f.df_t)
    p = float(hz._pval(np.array([est / se]), f.df_t)[0]) if se > 0 and np.isfinite(se) else np.nan
    return dict(_base(f), term=term, term_type=term_type, lag=lag, estimate=est, se=se,
                ci_low=est - crit * se, ci_high=est + crit * se, p=p, note=note)


def _wald_row(f: hz._FitOut, term: str, b: np.ndarray, C: np.ndarray, note: str) -> dict[str, Any]:
    try:
        wald = float(b @ np.linalg.solve(C, b))
    except np.linalg.LinAlgError:
        wald = np.nan
    q = len(b)
    if f.df_t:
        pj, stat, dist = float(stats.f.sf(wald / q, q, f.df_t)), wald / q, f"F({q},{f.df_t})"
    else:
        pj, stat, dist = float(stats.chi2.sf(wald, q)), wald, f"chi2({q})"
    return dict(_base(f), term=term, term_type="test", lag=np.nan, estimate=stat, se=np.nan,
                ci_low=np.nan, ci_high=np.nan, p=pj, note=f"{note}, {dist}")


def two_sided_rows(f: hz._FitOut, spec: dict[str, Any], lags: np.ndarray, B: np.ndarray,
                   late_from: float) -> list[dict[str, Any]]:
    """Standard effect rows plus future-kernel rows and past - future contrasts."""
    rows = hz._effect_rows(f, lags, B)
    for r in rows:
        if r["term_type"].startswith("chirp_") and ("_fut_" in r["term"] or "_fgd_" in r["term"]
                                                     or "_fh" in r["term"]):
            r["note"] = "ACAUSAL future-window term (chirps after the outcome bin); " + r["note"]
    idx = {nm: i for i, nm in enumerate(f.names)}
    if spec["win"] == "kernel":
        nb = B.shape[1]
        h = [idx.get(f"sub_h{i + 1}") for i in range(nb)]
        fh = [idx.get(f"sub_fh{i + 1}") for i in range(nb)]
        if any(i is None for i in h + fh):
            return rows
        crit = hz._crit(f.df_t)
        bf, Cf = f.beta[fh], f.cov[np.ix_(fh, fh)]
        k = B @ bf
        ks = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", B, Cf, B), 0, None))
        for lag, kv, sv in zip(lags, k, ks):
            rows.append(dict(_base(f), term="kernel_sub_fut", term_type="kernel", lag=-lag,
                             estimate=kv, se=sv, ci_low=kv - crit * sv, ci_high=kv + crit * sv,
                             p=float(hz._pval(np.array([kv / sv]), f.df_t)[0]) if sv > 0
                             else np.nan,
                             note="ACAUSAL: change per sub chirp |lag| s AFTER the outcome bin "
                                  "(signed lag: negative = chirp after the bin; pointwise CI)"))
        late = lags >= late_from - 1e-9
        for suf, c, lab in (("mean", B.mean(0), f"{lags[0]:g}-{lags[-1]:g} s"),
                            ("mean_late", B[late].mean(0), f"{lags[late][0]:g}-{lags[-1]:g} s")):
            wf = {f"sub_fh{i + 1}": c[i] for i in range(nb)}
            wp = {f"sub_h{i + 1}": c[i] for i in range(nb)}
            est, se = _lin(f, wf)
            rows.append(_row(f, f"kernel_sub_fut_{suf}", "kernel_summary", est, se,
                             f"ACAUSAL: mean kernel over {lab} AFTER the outcome bin"))
            est, se = _lin(f, {**wp, **{k_: -v for k_, v in wf.items()}})
            rows.append(_row(f, f"kernel_sub_{suf}-kernel_sub_fut_{suf}", "contrast", est, se,
                             f"acausal contrast: mean kernel over {lab} before minus after the "
                             "outcome bin (log scale; ratio = before / after)"))
        rows.append(_wald_row(f, "kernel_sub_fut_joint", bf, Cf,
                              "ACAUSAL: joint Wald test of all future sub basis terms"))
        D = np.hstack([np.eye(nb), -np.eye(nb)])
        ii = h + fh
        d = D @ f.beta[ii]
        V = D @ f.cov[np.ix_(ii, ii)] @ D.T
        rows.append(_wald_row(f, "kernel_sub_shape_diff_joint", d, V,
                              "acausal contrast: joint Wald test that the kernels before and "
                              "after the outcome bin are equal (all basis coefficients)"))
    elif spec["past"] and spec["fut"]:
        est, se = _lin(f, {spec["past"]: 1.0, spec["fut"]: -1.0})
        rows.append(_row(f, f"{spec['past']}-{spec['fut']}", "contrast", est, se,
                         "acausal contrast: log HR of the past window minus log HR of the future "
                         "window, each adjusted for the other (ratio = HR_past / HR_future)"))
    for r in rows:
        exp_ok = r["term_type"] not in ("test", "re_sd")
        for c, src in (("ratio", "estimate"), ("ratio_low", "ci_low"), ("ratio_high", "ci_high")):
            r[c] = float(np.exp(r[src])) if exp_ok and np.isfinite(r[src]) else np.nan
    return rows


def _contrast_weights(spec: dict[str, Any], B: np.ndarray, late: np.ndarray | None = None
                      ) -> dict[str, float]:
    if spec["win"] == "kernel":
        c = B.mean(0) if late is None else B[late].mean(0)
        w = {f"sub_h{i + 1}": c[i] for i in range(B.shape[1])}
        w.update({f"sub_fh{i + 1}": -c[i] for i in range(B.shape[1])})
        return w
    return {spec["past"]: 1.0, spec["fut"]: -1.0}


def lodo_contrast(design: pd.DataFrame, cfg: Config, model: str, spec: dict[str, Any],
                  M: hz._Matrix, beta0: np.ndarray) -> pd.DataFrame:
    """glm_fe refits of a joint model with each dyad left out; past - future contrast,
    past and future terms (kernel: kernel means). Warm-started at the full fit."""
    link = str(hz._opt(cfg, "glm_link"))
    y = design["y"].to_numpy().astype(float)
    codes, recs = hz._groups(design)
    B = design.attrs["basis"]
    nb = B.shape[1]
    if spec["win"] == "kernel":
        c = B.mean(0)
        parts = {"past": {f"sub_h{i + 1}": c[i] for i in range(nb)},
                 "fut": {f"sub_fh{i + 1}": c[i] for i in range(nb)}}
    else:
        parts = {"past": {spec["past"]: 1.0}, "fut": {spec["fut"]: 1.0}}
    parts["contrast"] = _contrast_weights(spec, B)
    rows = []
    for g, rec in enumerate(recs):
        keep = codes != g
        cols = [j for j in range(M.X.shape[1]) if j != g]
        res = hz.fit_glm(M.X[keep][:, cols], y[keep], link=link,
                         groups=codes[keep] - (codes[keep] > g), start=beta0[cols])
        names = [M.names[j] for j in cols]
        b = res.params / M.sd[cols]
        C = res.cov_cluster / np.outer(M.sd[cols], M.sd[cols])
        crit = float(stats.t.ppf(0.975, max(res.n_groups - 1, 1)))
        r = dict(kind=design.attrs["kind"], model=model, left_out=rec, converged=res.converged)
        for key, w in parts.items():
            cv = np.zeros(len(names))
            for nm, v in w.items():
                cv[names.index(nm)] = v
            est, se = float(cv @ b), float(np.sqrt(max(cv @ C @ cv, 0.0)))
            r.update({f"{key}": est, f"{key}_low": est - crit * se, f"{key}_high": est + crit * se})
        rows.append(r)
    return pd.DataFrame(rows)


def fit_two_sided(
    design: pd.DataFrame, cfg: Config, specs: Mapping[str, dict[str, Any]],
    backends: Sequence[str] = ("glm_fe", "mixed_laplace"),
    lodo_models: Sequence[str] = (), re_windows: Sequence[str] = (), verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fit the acausal models of A on a design with future columns."""
    kind = design.attrs["kind"]
    lags, B = design.attrs["lags"], design.attrs["basis"]
    late_from = float(hz._opt(cfg, "guard_lag"))
    ctrl: dict[bool, hz._Block] = {}
    rows, lodos, oes = [], [], []
    for model, spec in specs.items():
        for backend in backends:
            fe = backend == "glm_fe"
            if fe not in ctrl:
                ctrl[fe] = hz._control_block(design, cfg, fe)
            f, fit_obj, M = hz._fit_one(design, cfg, model, backend, ctrl[fe], cols=spec["cols"])
            if verbose:
                print(f"  acausal {kind:6s} {model:17s} {backend:13s} {f.fit_time:6.1f} s "
                      f"converged={f.converged}", flush=True)
            rows.extend(two_sided_rows(f, spec, lags, B, late_from))
            if fe and model in lodo_models:
                t0 = time.perf_counter()
                lodos.append(lodo_contrast(design, cfg, model, spec, M, fit_obj.params))
                if verbose:
                    print(f"  acausal {kind:6s} {model:17s} leave-one-dyad-out "
                          f"{time.perf_counter() - t0:6.1f} s", flush=True)
            del M
        if spec["win"] in re_windows:
            for term in (spec["past"], spec["fut"]):
                if not term:
                    continue
                tab, het = hz.heterogeneity(design, cfg, models=[model], term=term,
                                            cols=spec["cols"])
                oes.append(tab)
                # one set of SD / G rows per term: name them by the term
                sdg = het["term"].isin(["dyad_sd_sub_effect", "heterogeneity_G"])
                het.loc[sdg, "term"] = het.loc[sdg, "term"] + ":" + term
                het["note"] = het["note"].astype(str) + f"; term {term} of model {model}"
                rows.extend(het.to_dict("records"))
    eff = pd.DataFrame(rows).reindex(columns=hz.EFFECT_COLUMNS)
    return dict(effects=eff,
                lodo=pd.concat(lodos, ignore_index=True) if lodos else pd.DataFrame(),
                dyad_oe=pd.concat(oes, ignore_index=True) if oes else pd.DataFrame())


# ---------------------------------------------------------------------------
# B / C. Surrogate chirp trains around chase offsets
# ---------------------------------------------------------------------------
def post_offset_chirps(tr: Trial, W: float, x: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-chase sub chirps t with off_i < t < min(off_i + W, next onset, start of the
    next unknown-state interval), for the last offset off_i before t.
    Returns (t, chase index i, latency L = t - off_i)."""
    x = tr.chirps_sub if x is None else np.asarray(x, float)
    ch = tr.chases
    if not len(ch) or not len(x):
        return np.zeros(0), np.zeros(0, np.int64), np.zeros(0)
    x = x[~tr.in_chase(x)]
    i = np.searchsorted(ch[:, 1], x, side="left") - 1  # last offset strictly before t
    ok = i >= 0
    x, i = x[ok], i[ok]
    L = x - ch[i, 1]
    nxt_on = np.r_[ch[1:, 0], np.inf][i]
    ex = np.sort(tr.exclude_intervals[:, 0]) if len(tr.exclude_intervals) else np.zeros(0)
    j = np.searchsorted(ex, ch[i, 1], side="right")
    nxt_ex = np.r_[ex, np.inf][j]
    keep = (L < W) & (x < nxt_on) & (x < nxt_ex) & ~_in_intervals(x, tr.exclude_intervals)
    return x[keep], i[keep], L[keep]


def pre_offset_chirps(tr: Trial, W: float, match_post: bool = False
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-chase sub chirps less than W before their own chase's offset: (t, i, L = off_i - t).

    match_post: keep a chirp only where the mirror arm could have one at the same latency,
    i.e. off_i + L before the next onset and the next unknown-state interval and outside
    unknown-state intervals (the support of post_offset_chirps; restricted-real arm)."""
    x, ch = tr.chirps_sub, tr.chases
    if not len(ch) or not len(x):
        return np.zeros(0), np.zeros(0, np.int64), np.zeros(0)
    inc = tr.in_chase(x)
    x = x[inc]
    i = np.searchsorted(ch[:, 0], x, side="right") - 1
    L = ch[i, 1] - x
    keep = L < W
    if match_post:
        refl = ch[i, 1] + L
        nxt_on = np.r_[ch[1:, 0], np.inf][i]
        ex = np.sort(tr.exclude_intervals[:, 0]) if len(tr.exclude_intervals) else np.zeros(0)
        nxt_ex = np.r_[ex, np.inf][np.searchsorted(ex, ch[i, 1], side="right")]
        keep &= (refl < nxt_on) & (refl < nxt_ex) & ~_in_intervals(refl, tr.exclude_intervals)
    return x[keep], i[keep], L[keep]


def surrogate_train(tr: Trial, variant: str, W: float, delta: float = 0.0
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Sub chirp train of one arm and its bookkeeping.

    variant 'restricted_real', 'mirror' or 'shift' (by ``delta``). Real
    out-of-chase chirps are always kept unchanged; every real in-chase chirp is
    removed and replaced by the arm's in-chase chirps.
    """
    x, ch = tr.chirps_sub, tr.chases
    out_real = x[~tr.in_chase(x)]
    win = (ch[:, 1] >= tr.obs_start) & (ch[:, 1] < tr.obs_end) if len(ch) else np.zeros(0, bool)
    info: dict[str, Any] = dict(recording=tr.recording, variant=variant, window=W, delta=delta)
    if variant == "restricted_real":
        _, i0, _ = pre_offset_chirps(tr, W)
        t, i, L = pre_offset_chirps(tr, W, match_post=True)
        new = t
        info.update(n_candidates=int(win[i0].sum()), n_kept=int(win[i].sum()),
                    n_dropped=int(win[i0].sum() - win[i].sum()))
    elif variant in ("mirror", "shift"):
        t, i, L = post_offset_chirps(tr, W)
        pos = ch[i, 1] - L if variant == "mirror" else t - delta
        inside = (pos > ch[i, 0]) & (pos < ch[i, 1])
        new = pos[inside]
        info.update(n_candidates=int(win[i].sum()), n_kept=int((win[i] & inside).sum()),
                    n_dropped=int((win[i] & ~inside).sum()),
                    n_dropped_still_after_offset=int((win[i] & (pos >= ch[i, 1])).sum()),
                    n_dropped_before_onset=int((win[i] & (pos <= ch[i, 0])).sum()))
    else:
        raise ValueError(f"unknown variant {variant}")
    train = np.sort(np.concatenate([out_real, new]))
    v = tr.valid(train)
    info.update(n_in_chase=int((v & tr.in_chase(train)).sum()),
                n_out_chase=int((v & ~tr.in_chase(train)).sum()),
                n_real_in_chase=int((tr.valid(x) & tr.in_chase(x)).sum()))
    return train, info


def arm_label(variant: str, delta: float = 0.0) -> str:
    return f"shift_{_ms(delta)}" if variant == "shift" else variant


def comparator_windows(windows: Sequence[float], shift_window: float,
                       deltas: Sequence[float]) -> list[float]:
    """Windows W = min(Delta, shift window) of the shift arms not already in ``windows``."""
    have = {round(float(w), 6) for w in windows}
    out = []
    for d in deltas:
        w = round(min(float(d), float(shift_window)), 6)
        if w not in have and w not in out:
            out.append(w)
    return sorted(out)


def build_arms(trials: Sequence[Trial], cfg: Config, windows: Sequence[float],
               shift_window: float, deltas: Sequence[float]
               ) -> tuple[dict[tuple[str, float], dict[str, np.ndarray]], pd.DataFrame,
                          pd.DataFrame]:
    """(arm label, W) -> {recording: train}; per-dyad counts; per-chirp latencies.

    A Delta-shifted train keeps only post-offset chirps with latency < min(Delta, W) (the
    others land after the offset), so its fair comparators are the restricted-real and
    mirror arms of W = min(Delta, W): built here too ('comparator' windows, see
    comparator_windows) for every Delta whose window is not in ``windows``."""
    arms: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    counts, lat = [], []
    allw = list(windows) + comparator_windows(windows, shift_window, deltas)
    todo = [(v, W, 0.0) for W in allw for v in ("restricted_real", "mirror")]
    todo += [("shift", shift_window, d) for d in deltas]
    for v, W, d in todo:
        lab = arm_label(v, d)
        arms[(lab, W)] = {}
        for tr in trials:
            train, info = surrogate_train(tr, v, W, d)
            arms[(lab, W)][tr.recording] = train
            info["variant"] = lab
            counts.append(info)
    for W in windows:
        for tr in trials:
            ch = tr.chases
            win = (ch[:, 1] >= tr.obs_start) & (ch[:, 1] < tr.obs_end)
            for side, fn in (("before_offset", pre_offset_chirps), ("after_offset",
                                                                    post_offset_chirps)):
                t, i, L = fn(tr, W)
                m = win[i]
                lat.append(pd.DataFrame(dict(recording=tr.recording, window=W, side=side,
                                             t=t[m], chase=i[m], latency=L[m],
                                             chase_duration=ch[i[m], 1] - ch[i[m], 0])))
    cnt = pd.DataFrame(counts)
    tot = cnt.groupby(["variant", "window", "delta"], as_index=False, sort=False).sum(
        numeric_only=True).assign(recording="ALL")
    cnt = pd.concat([cnt, tot], ignore_index=True)
    return arms, cnt, pd.concat(lat, ignore_index=True)


def state_rates(trials: Sequence[Trial]) -> pd.DataFrame:
    """Sub chirp rates inside and outside chases (valid window) and around chase
    boundaries, per dyad and pooled ('ALL'). The future window of an onset bin
    lies inside the new chase and that of an offset bin outside the chase, so
    these base rates shift HR_future even without any timing effect."""
    rows = []
    for tr in trials:
        ch = tr.chases[tr.chases[:, 0] < tr.obs_end]
        t_in = float(np.clip(np.minimum(ch[:, 1], tr.obs_end) - ch[:, 0], 0, None).sum())
        t_ex = float((tr.exclude_intervals[:, 1] - tr.exclude_intervals[:, 0]).sum())
        x = tr.chirps_sub
        v, ic = tr.valid(x), tr.in_chase(x)
        # offsets inside the window (n_chases, offset counts); onsets inside the window
        # (n_onsets, onset counts); the first 1 s of a chase stops at its offset
        done = ch[ch[:, 1] < tr.obs_end]
        ons = ch[ch[:, 0] >= tr.obs_start]
        near = {}
        for lab, src, a, b in (("first_1s_of_chase", ons, 0, 0.0),
                               ("last_1s_of_chase", done, 1, -1.0),
                               ("first_1s_after_offset", done, 1, 0.0),
                               ("last_1s_before_onset", ons, 0, -1.0)):
            lo = src[:, a] + b
            hi = np.minimum(lo + 1.0, src[:, 1]) if lab == "first_1s_of_chase" else lo + 1.0
            near[lab] = int(sum(np.sum((x >= l0) & (x < h0)) for l0, h0 in zip(lo, hi)))
        # post-offset chirps 2-5 s after an offset (before the next onset / unknown state) and
        # the exposure of that window: is it still a low-rate gap?
        t25, i25, L25 = post_offset_chirps(tr, 5.0)
        win = (tr.chases[:, 1] >= tr.obs_start) & (tr.chases[:, 1] < tr.obs_end)
        n25 = int(np.sum(win[i25] & (L25 >= 2.0)))
        nxt_on = np.r_[tr.chases[1:, 0], np.inf]
        ex = np.sort(tr.exclude_intervals[:, 0]) if len(tr.exclude_intervals) else np.zeros(0)
        offs = tr.chases[win, 1]
        nxt_ex = np.r_[ex, np.inf][np.searchsorted(ex, offs, side="right")]
        e25 = float(np.clip(np.minimum.reduce([offs + 5.0, nxt_on[win], nxt_ex]) - (offs + 2.0),
                            0, None).sum())
        rows.append(dict(recording=tr.recording, n_chases=len(done), n_onsets=len(ons),
                         time_in_chase_s=t_in,
                         time_out_chase_s=tr.obs_end - tr.obs_start - t_in - t_ex,
                         n_sub_in_chase=int((v & ic).sum()), n_sub_out_chase=int((v & ~ic).sum()),
                         **near, n_post_2_5s=n25, exposure_post_2_5s=e25))
    df = pd.DataFrame(rows)
    tot = df.sum(numeric_only=True).to_dict()
    df = pd.concat([df, pd.DataFrame([dict(recording="ALL", **tot)])], ignore_index=True)
    df["rate_in_per_min"] = 60 * df["n_sub_in_chase"] / df["time_in_chase_s"].where(
        df["time_in_chase_s"] > 0)
    df["rate_out_per_min"] = 60 * df["n_sub_out_chase"] / df["time_out_chase_s"].where(
        df["time_out_chase_s"] > 0)
    df["rate_ratio_in_out"] = df["rate_in_per_min"] / df["rate_out_per_min"]
    df["rate_post_2_5s_per_min"] = 60 * df["n_post_2_5s"] / df["exposure_post_2_5s"].where(
        df["exposure_post_2_5s"] > 0)
    return df


def fit_arm(trials: Sequence[Trial], cfg: Config, override: Mapping[str, np.ndarray] | None,
            backends: Sequence[str] = ("mixed_laplace", "glm_fe"),
            models: Sequence[str] | None = None, re_models: Sequence[str] | None = None,
            verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standard offset models on one arm's train: (effects, per-dyad O/E, design)."""
    d = hz.build_design(trials, "offset", cfg, chirp_override=override)
    e = hz.fit_models(d, "offset", cfg, models=models, backends=backends, verbose=verbose)
    oe = pd.DataFrame()
    if re_models:
        oe, het = hz.heterogeneity(d, cfg, models=list(re_models))
        e = pd.concat([e, het], ignore_index=True)
    return e, oe, d


# ---------------------------------------------------------------------------
# Dyad bootstrap of restricted_real - mirror
# ---------------------------------------------------------------------------
_BOOT: dict[str, Any] = {}


def _boot_worker(reps: Sequence[int]) -> list[dict[str, Any]]:
    S = _BOOT
    out = []
    for r in reps:
        mult = np.bincount(S["draws"][r], minlength=S["G"]).astype(float)
        rows = mult[S["codes"]] > 0
        used = np.flatnonzero(mult > 0)
        w = mult[S["codes"][rows]]
        rec = dict(rep=r)
        for key, st in S["setups"].items():
            cols = list(used) + list(range(S["G"], st["X"].shape[1]))
            res = hz.fit_glm(st["X"][rows][:, cols], S["y"][rows], link=S["link"], weights=w,
                             start=st["beta0"][cols], cov=False)
            b = res.params[len(used) + st["j"] - S["G"]] / st["sd"] if res.converged else np.nan
            rec[key] = float(b)
        out.append(rec)
    return out


def dyad_bootstrap(designs: Mapping[tuple[str, float], pd.DataFrame], cfg: Config,
                   models: Sequence[str], n_boot: int, n_workers: int = 8,
                   verbose: bool = False) -> pd.DataFrame:
    """Dyad-resampling bootstrap of the glm_fe sub-indicator coefficient of every
    (arm, W) design and model; the same resamples for all, so arm differences
    are paired. A resampled dyad counted m times enters with frequency weight m
    (identical to m copies with their own intercepts). Returns one row per
    replicate (rep 0 = full data) with columns '<arm>|<W>|<model>'."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    link = str(hz._opt(cfg, "glm_link"))
    setups: dict[str, dict[str, Any]] = {}
    y = codes = None
    G = 0
    full = dict(rep=0)
    for (arm, W), d in designs.items():
        ctrl = hz._control_block(d, cfg, fe=True)
        c_, recs = hz._groups(d)
        y_ = d["y"].to_numpy().astype(float)
        if y is not None and not (np.array_equal(y, y_) and np.array_equal(codes, c_)):
            raise ValueError("bootstrap arms must share the risk set")
        codes, y, G = c_, y_, len(recs)
        for model in models:
            M = hz._assemble(d, ctrl, hz._chirp_block(d, model), fe=True)
            ind = str(hz.indicator_column(model, "sub"))
            j = M.names.index(ind)
            res = hz.fit_glm(M.X, y, link=link, cov=False)
            key = f"{arm}|{W:g}|{model}"
            setups[key] = dict(X=M.X, j=j, sd=float(M.sd[j]), beta0=res.params)
            full[key] = float(res.params[j] / M.sd[j])
    rng = np.random.default_rng([int(cfg.seed), 20260929])
    draws = rng.integers(0, G, size=(n_boot + 1, G))
    draws[0] = np.arange(G)
    _BOOT.clear()
    _BOOT.update(setups=setups, y=y, codes=codes, G=G, link=link, draws=draws)
    reps = list(range(1, n_boot + 1))
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = [full]
    if n_workers > 1 and n_boot > 1:
        chunks = [reps[i::n_workers * 4] for i in range(min(n_boot, n_workers * 4))]
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork"),
                                 initializer=hz._worker_init) as ex:
            for done, part in enumerate(ex.map(_boot_worker, chunks), 1):
                rows.extend(part)
                if verbose and (done % max(1, len(chunks) // 4) == 0 or done == len(chunks)):
                    print(f"  bootstrap: {done}/{len(chunks)} chunks "
                          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    else:
        rows.extend(_boot_worker(reps))
    _BOOT.clear()
    out = pd.DataFrame(rows).sort_values("rep", ignore_index=True)
    out.attrs["elapsed_s"] = time.perf_counter() - t0
    return out


def bootstrap_rows(boot: pd.DataFrame, windows: Sequence[float], models: Sequence[str],
                   n_rec: int, a: str = "restricted_real", b: str = "mirror"
                   ) -> list[dict[str, Any]]:
    """Effect rows: paired log HR difference a - b with percentile CI (glm_fe)."""
    rows = []
    for W in windows:
        for model in models:
            ka, kb = f"{a}|{W:g}|{model}", f"{b}|{W:g}|{model}"
            if ka not in boot or kb not in boot:
                continue
            d = (boot[ka] - boot[kb]).to_numpy()
            est = float(d[0])
            v = d[1:][np.isfinite(d[1:])]
            n = len(v)
            lo, hi = (np.quantile(v, [0.025, 0.975]) if n else (np.nan, np.nan))
            p = min(1.0, 2 * min((1 + np.sum(v <= 0)) / (n + 1), (1 + np.sum(v >= 0)) / (n + 1))) \
                if n else np.nan
            term = str(hz.indicator_column(model, "sub"))
            rows.append(dict(
                analysis="acausal", variant=f"{a}-{b}", subset=f"w_{_ms(W)}", kind="offset",
                model=model, backend="glm_fe", link=str(hz.DEFAULTS["glm_link"]), term=term,
                term_type="contrast", estimate=est,
                se=float(np.std(v, ddof=1)) if n > 1 else np.nan,
                ci_low=float(lo), ci_high=float(hi), p=float(p), ratio=float(np.exp(est)),
                ratio_low=float(np.exp(lo)), ratio_high=float(np.exp(hi)), n_recordings=n_rec,
                n_jitter=n, converged=bool(n == len(d) - 1),
                ci_method=f"dyad bootstrap (percentile, {n} resamples; both arms refitted on "
                          "the same resample)",
                note=(f"acausal control B: log HR({a}) - log HR({b}), W = {W:g} s, glm_fe; ratio "
                      f"= HR({a}) / HR({b}); p = two-sided bootstrap sign test; "
                      f"{len(d) - 1 - n} non-converged resamples dropped; the n_jitter column "
                      "holds the number of resamples")))
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _tag(e: pd.DataFrame, variant: str, subset: str = "", note: str = "") -> pd.DataFrame:
    e = e.copy()
    e["analysis"] = "acausal"
    e["variant"] = variant
    e["subset"] = subset
    if note:
        e["note"] = note + "; " + e["note"].astype(str)
    return e


def _two_sided_block(trials: Sequence[Trial], kind: str, cfg: Config, specs: dict[str, Any],
                     backends: Sequence[str], lodo_models: Sequence[str],
                     re_windows: Sequence[str], verbose: bool) -> dict[str, pd.DataFrame]:
    d0 = hz.build_design(trials, kind, cfg)
    d = add_future_columns(d0, trials, cfg)
    del d0
    if verbose:
        fb = d.attrs["future_bookkeeping"]
        print(f"[acausal {kind}] {len(d):,} bins, {int(d['y'].sum())} events; "
              f"{int(fb['n_dropped_future_past_end'].sum())} bins dropped (future window past "
              "the recording end)", flush=True)
    res = fit_two_sided(d, cfg, specs, backends=backends, lodo_models=lodo_models,
                        re_windows=re_windows, verbose=verbose)
    res["bookkeeping"] = d.attrs["future_bookkeeping"]
    return res


def acausal_stats(
    trials: Sequence[Trial], cfg: Config, kinds: Sequence[str] = ("offset",),
    parts: Sequence[str] = ("two_sided", "mirror"), backends: Sequence[str] = ("glm_fe",),
    taus: Sequence[float] = (1.0,), windows: Sequence[float] = (2.0,),
    deltas: Sequence[float] = (), n_boot: int = 0, n_workers: int = 1, verbose: bool = False,
) -> pd.DataFrame:
    """The acausal statistics on any list of Trials (real or simulated), no files.

    Defaults are the fast set for simulation replicates: glm_fe only, offset,
    A with tau = 1 s + guard + two-sided kernel, B with W = 2 s (tau = 1 s,
    guard, kernel on restricted_real and mirror), the paired difference as a
    point estimate (bootstrap CI if n_boot > 0). Returns effect rows
    (hz_hazard.EFFECT_COLUMNS + analysis, variant, subset). Key rows:
      variant two_sided, model pf_1000ms, term 'sub_any_1000ms-sub_fut_1000ms' (contrast),
        terms sub_any_1000ms / sub_fut_1000ms (HR_past / HR_future);
      model pf_guard_500ms, term 'sub_grd_500ms-sub_fgd_500ms';
      model kernel_two_sided, term 'kernel_sub_mean-kernel_sub_fut_mean';
      variant restricted_real / mirror (subset w_2000ms), model tau_1000ms, term sub_any_1000ms;
      variant 'restricted_real-mirror', term sub_any_1000ms / sub_grd_500ms (difference).
    """
    out = []
    specs_all = two_sided_specs(cfg, taus)
    g = _gname(cfg)
    specs = {m: s for m, s in specs_all.items() if not m.startswith("past_")}
    if "two_sided" in parts:
        for kind in kinds:
            r = _two_sided_block(trials, kind, cfg, specs, backends, (), (), verbose)
            out.append(_tag(r["effects"], "two_sided"))
    if "mirror" in parts or deltas:
        models = ["kernel"] + [f"tau_{_ms(t)}" for t in taus] + [f"guard_{g}"]
        arms, _, _ = build_arms(trials, cfg, windows if "mirror" in parts else (),
                                windows[0] if windows else 2.0, deltas)
        designs = {}
        for (lab, W), ov in arms.items():
            e, _, d = fit_arm(trials, cfg, ov, backends=backends, models=models)
            out.append(_tag(e, lab, f"w_{_ms(W)}"))
            if lab in ("restricted_real", "mirror") and W in windows:
                designs[(lab, W)] = d
        bm = [m for m in (f"tau_{_ms(1.0)}", f"guard_{g}") if m in models]
        if designs:
            boot = dyad_bootstrap(designs, cfg, bm, n_boot, n_workers=n_workers)
            out.append(pd.DataFrame(bootstrap_rows(boot, [W for W in windows], bm,
                                                   len(trials))))
    eff = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return eff


def run_acausal(trials: Sequence[Trial], cfg: Config, verbose: bool = True,
                n_boot: int | None = None, n_workers: int | None = None) -> dict[str, Any]:
    """A, B and C on the real trials; writes cfg.out_dir/'acausal' and figures.

    Returns dict(effects, dyad_oe, lodo, counts, latencies, bootstrap,
    bookkeeping, timings, text, figures).
    """
    out = Path(cfg.out_dir) / "acausal"
    fig_dir = Path(cfg.out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    quick = bool(cfg.quick)
    n_boot = int(_opt(cfg, "acausal_n_boot") if n_boot is None else n_boot)
    if quick:
        n_boot = min(n_boot, 40)
    n_workers = int(hz._opt(cfg, "n_workers") if n_workers is None else n_workers)
    taus = (1.0,) if quick else tuple(cfg.tau_grid)
    windows = tuple(_opt(cfg, "acausal_windows"))[:1] if quick else tuple(
        _opt(cfg, "acausal_windows"))
    deltas = tuple(_opt(cfg, "acausal_shift_deltas"))[1:2] if quick else tuple(
        _opt(cfg, "acausal_shift_deltas"))
    shift_w = float(_opt(cfg, "acausal_shift_window"))
    g = _gname(cfg)
    boot_models = [m.replace("500ms", g) for m in _opt(cfg, "acausal_boot_models")]
    re_models = [m.replace("500ms", g) for m in _opt(cfg, "acausal_re_models")]
    lodo_models = [m.replace("500ms", g) for m in _opt(cfg, "acausal_lodo_models")]
    timings: list[dict[str, Any]] = []
    effs, oes, lodos, books = [], [], [], []
    import json

    (out / "acausal_config.json").write_text(json.dumps(dict(
        started=time.strftime("%Y-%m-%dT%H:%M:%S"), quick=quick, n_boot=n_boot,
        n_workers=n_workers, seed=int(cfg.seed), taus=list(taus), windows=list(windows),
        shift_window=shift_w, shift_deltas=list(deltas), boot_models=boot_models,
        re_models=re_models, lodo_models=lodo_models, bin_width=cfg.bin_width,
        hist_max=cfg.hist_max, n_basis=int(cfg.n_basis), max_time=cfg.max_time,
        future_extent=future_extent(cfg)), indent=2))

    # --- A --------------------------------------------------------------------
    specs = two_sided_specs(cfg, taus)
    for kind in KINDS:
        t0 = time.perf_counter()
        # offset refits are cheap (~4 s per model): leave-one-dyad-out for every joint
        # window; onset (~2.9e6 bins) only for the configured models
        if kind == "offset":
            lm = [m for m in specs if m.startswith("pf_")] + ["kernel_two_sided"]
        else:
            lm = [] if quick else lodo_models
        r = _two_sided_block(trials, kind, cfg, specs, ("mixed_laplace", "glm_fe"), lm,
                             re_models, verbose)
        effs.append(_tag(r["effects"], "two_sided", note=VARIANT_NOTE["two_sided"]))
        if len(r["dyad_oe"]):
            oes.append(r["dyad_oe"].assign(variant="two_sided", subset=""))
        if len(r["lodo"]):
            lodos.append(r["lodo"])
        books.append(r["bookkeeping"])
        timings.append(dict(step=f"A two_sided {kind}", seconds=time.perf_counter() - t0))

    # --- B, C -----------------------------------------------------------------
    rates = state_rates(trials)
    t0 = time.perf_counter()
    arms, counts, lat = build_arms(trials, cfg, windows, shift_w, deltas)
    timings.append(dict(step="B/C trains", seconds=time.perf_counter() - t0))
    t0 = time.perf_counter()
    e, oe, _ = fit_arm(trials, cfg, None, re_models=re_models, verbose=False)
    effs.append(_tag(e, "real_full", note=VARIANT_NOTE["real_full"]))
    oes.append(oe.assign(variant="real_full", subset=""))
    timings.append(dict(step="B real_full", seconds=time.perf_counter() - t0))
    designs = {}
    for (lab, W), ov in arms.items():
        t0 = time.perf_counter()
        comp = lab in ("restricted_real", "mirror") and W not in windows
        if comp:  # comparator of a shift arm (W = Delta): FE only, no bootstrap
            e, oe, d = fit_arm(trials, cfg, ov, backends=("glm_fe",), verbose=False)
        else:
            e, oe, d = fit_arm(trials, cfg, ov, re_models=re_models, verbose=False)
        base = "shift" if lab.startswith("shift_") else lab
        nt = f"{VARIANT_NOTE[base]} (W = {W:g} s" + (
            f", Delta = {lab.split('_')[1]})" if base == "shift" else ")") + "; " + SELECTION_NOTE
        if comp:
            nt = "comparator of the shift arm with Delta = W; " + nt
        effs.append(_tag(e, lab, f"w_{_ms(W)}", note=nt))
        if len(oe):
            oes.append(oe.assign(variant=lab, subset=f"w_{_ms(W)}"))
        if lab in ("restricted_real", "mirror") and not comp:
            designs[(lab, W)] = d
        else:
            del d
        if verbose:
            r1 = e[(e["backend"] == "glm_fe") & (e["term"] == "sub_any_1000ms")]
            txt = (f"FE 1 s HR {r1['ratio'].iloc[0]:.2f} [{r1['ratio_low'].iloc[0]:.2f}, "
                   f"{r1['ratio_high'].iloc[0]:.2f}]" if len(r1) else "")
            print(f"  acausal arm {lab:16s} W={W:g} s  {txt}  ({time.perf_counter() - t0:.1f} s)",
                  flush=True)
        timings.append(dict(step=f"B/C {lab} W={W:g}", seconds=time.perf_counter() - t0))
    t0 = time.perf_counter()
    boot = dyad_bootstrap(designs, cfg, boot_models, n_boot, n_workers=n_workers,
                          verbose=verbose)
    del designs
    effs.append(pd.DataFrame(bootstrap_rows(boot, windows, boot_models, len(trials))))
    timings.append(dict(step=f"B bootstrap ({n_boot} reps)", seconds=time.perf_counter() - t0))

    cols = ["analysis", "variant", "subset"] + hz.EFFECT_COLUMNS + ["ci_method", "n_jitter"]
    eff = pd.concat([x.reindex(columns=cols) for x in effs], ignore_index=True)
    cm = {"glm_fe": "cluster-robust (recording) sandwich, t(G-1)",
          "mixed_laplace": "Laplace posterior mean +- 1.96 SD (model-based; assumes one effect "
                           "for all dyads)",
          "re_meta": "profile likelihood of the Poisson-normal random-effects model, t(k-1)^2 "
                     "cut-off"}
    eff["ci_method"] = eff["ci_method"].fillna(eff["backend"].map(cm))
    dyad_tab = pd.concat([x for x in oes if len(x)], ignore_index=True)
    lodo = pd.concat(lodos, ignore_index=True) if lodos else pd.DataFrame()
    book = pd.concat(books, ignore_index=True)
    tim = pd.DataFrame(timings)
    tim["seconds"] = tim["seconds"].round(2)
    eff.to_csv(out / "acausal_effects.csv", index=False)
    dyad_tab.to_csv(out / "acausal_dyad_oe.csv", index=False)
    lodo.to_csv(out / "acausal_leave_one_dyad_out.csv", index=False)
    counts.to_csv(out / "acausal_chirp_counts.csv", index=False)
    rates.to_csv(out / "acausal_state_rates.csv", index=False)
    lat.to_csv(out / "acausal_latencies.csv", index=False)
    boot.to_csv(out / "acausal_bootstrap.csv", index=False)
    book.to_csv(out / "acausal_design_summary.csv", index=False)
    tim.to_csv(out / "acausal_timings.csv", index=False)
    text = summary_text(eff, counts, lodo, rates, Path(cfg.out_dir))
    (out / "acausal_summary.txt").write_text(text + "\n")
    figs = plot_two_sided_kernel(eff, fig_dir) + plot_tau_past_future(eff, fig_dir)
    figs += plot_mirror(eff, counts, lat, fig_dir)
    return dict(effects=eff, dyad_oe=dyad_tab, lodo=lodo, counts=counts, latencies=lat,
                bootstrap=boot, bookkeeping=book, rates=rates, timings=tim, text=text,
                figures=figs)


def load_tables(out_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Saved acausal tables of a run (for --report-only), or None."""
    a = Path(out_dir) / "acausal"
    f = a / "acausal_effects.csv"
    if not f.exists():
        return None

    def rd(name: str) -> pd.DataFrame:
        p = a / name
        return pd.read_csv(p, keep_default_na=False, na_values=[""],
                           float_precision="round_trip") if p.exists() else pd.DataFrame()

    return dict(effects=rd("acausal_effects.csv"), counts=rd("acausal_chirp_counts.csv"),
                lodo=rd("acausal_leave_one_dyad_out.csv"),
                bookkeeping=rd("acausal_design_summary.csv"),
                rates=rd("acausal_state_rates.csv"))


def summary_text(E: pd.DataFrame, counts: pd.DataFrame | None, lodo: pd.DataFrame | None,
                 rates: pd.DataFrame | None, out_dir: Path | None) -> str:
    """acausal_summary.txt: the acausal blocks of both kinds, each followed by the simulation
    block of hz_acausal_sim if its tables exist in ``out_dir`` (else a note)."""
    sim = None
    if out_dir is not None:
        import hz_acausal_sim  # lazy: hz_acausal_sim imports this module

        sim = hz_acausal_sim.load_tables(Path(out_dir))
    L = ["Acausal control of the chirp -> chase hazard association", "=" * 56]
    for kind, title in (("offset", "(a) Chase offset hazard"), ("onset", "(b) Chase onset hazard")):
        L += ["", title]
        L += summary_lines(E, counts, lodo, kind, rates, where="acausal")
        if sim is not None:
            import hz_acausal_sim

            L += hz_acausal_sim.summary_lines(sim, kind)
        else:
            L.append("    Simulation of this control: not run yet (hz_acausal_sim.py writes "
                     "simulation/acausal_sim_*; rerun hz_acausal.py --report-only afterwards).")
    return "\n".join(L)


def rewrite_summary(out_dir: Path) -> str | None:
    """Rebuild acausal_summary.txt from the saved tables (no figures); None without tables."""
    t = load_tables(out_dir)
    if t is None:
        return None
    rates = t["rates"] if len(t["rates"]) else None
    text = summary_text(t["effects"], t["counts"], t["lodo"], rates, Path(out_dir))
    (Path(out_dir) / "acausal" / "acausal_summary.txt").write_text(text + "\n")
    return text


def report_from_tables(out_dir: Path) -> str:
    """Redraw the acausal figures and acausal_summary.txt from the saved tables."""
    t = load_tables(out_dir)
    if t is None:
        raise SystemExit(f"no {Path(out_dir) / 'acausal' / 'acausal_effects.csv'}")
    a = Path(out_dir) / "acausal"
    lat = pd.read_csv(a / "acausal_latencies.csv") if (a / "acausal_latencies.csv").exists() \
        else pd.DataFrame(columns=["window", "side", "latency"])
    E, rates = t["effects"], t["rates"] if len(t["rates"]) else None
    text = summary_text(E, t["counts"], t["lodo"], rates, Path(out_dir))
    (a / "acausal_summary.txt").write_text(text + "\n")
    fig_dir = Path(out_dir) / "figures"
    plot_two_sided_kernel(E, fig_dir)
    plot_tau_past_future(E, fig_dir)
    plot_mirror(E, t["counts"], lat, fig_dir)
    return text


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def _pick(E: pd.DataFrame, **kw: Any) -> pd.Series | None:
    m = np.ones(len(E), bool)
    for k, v in kw.items():
        if k not in E:
            return None
        m &= (E[k].astype(str) == str(v)).to_numpy() if isinstance(v, str) else (
            E[k] == v).to_numpy()
    return E[m].iloc[0] if m.any() else None


def _ci(r: pd.Series | None, p: bool = False) -> str:
    if r is None or not np.isfinite(r["ratio"]):
        return "n/a"
    s = f"{r['ratio']:.2f} [{r['ratio_low']:.2f}, {r['ratio_high']:.2f}]"
    if p:
        pv = float(r["p"])
        s += f", p = {pv:.0e}" if pv < 1e-3 else f", p = {pv:.3f}" if pv < 0.1 else \
            f", p = {pv:.2f}"
    return s


def _pv(p: float) -> str:
    return "n/a" if not np.isfinite(p) else (f"{p:.0e}" if p < 1e-3 else f"{p:.3f}" if p < 0.1
                                             else f"{p:.2f}")


def _lodo_txt(lodo: pd.DataFrame, kind: str, model: str) -> str:
    if lodo is None or not len(lodo):
        return ""
    lk = lodo[(lodo["kind"] == kind) & (lodo["model"] == model)]
    if not len(lk):
        return ""
    lo, hi = lk["contrast"].idxmin(), lk["contrast"].idxmax()
    n_pos = int((lk["contrast"] > 0).sum())
    return (f"; leave one dyad out: ratio {np.exp(lk.loc[lo, 'contrast']):.2f} (without "
            f"{str(lk.loc[lo, 'left_out'])[5:10]}) to {np.exp(lk.loc[hi, 'contrast']):.2f} "
            f"(without {str(lk.loc[hi, 'left_out'])[5:10]}), > 1 in {n_pos}/{len(lk)}")


def _reading(c: pd.Series | None, what: str, a: pd.Series | None = None,
             b: pd.Series | None = None, labels: tuple[str, str] = ("HR_past", "HR_future"),
             g: pd.Series | None = None) -> str:
    """Neutral description of a contrast (ratio scale) relative to 1, with the two HRs and the
    GLMM ratio. Whether a value is informative is decided by the simulation, not here."""
    if c is None or not (np.isfinite(c["ci_low"]) and np.isfinite(c["ci_high"])):
        return f"{what}: n/a"
    lo, hi = float(c["ci_low"]), float(c["ci_high"])
    where = ("CI above 1" if lo > 0 else "CI below 1" if hi < 0
             else "CI includes 1: not distinguishable from 1 at the dyad level")
    nums = (f"; {labels[0]} {a['ratio']:.2f}, {labels[1]} {b['ratio']:.2f}"
            if a is not None and b is not None else "")
    gl = f"; GLMM {_ci(g)}" if g is not None else ""
    return f"{what} {_ci(c)} ({where}{nums}{gl})"


def _glmm_txt(r: pd.Series | None) -> str:
    return (f" (GLMM {_ci(r)}, one common effect for all dyads, caveat 4)" if r is not None
            else "")


def _rate(r0: pd.Series, col: str, n: str) -> str:
    nn = max(int(r0[n]), 1) if n in r0 and np.isfinite(r0[n]) else 1
    return f"{60 * float(r0[col]) / nn:.1f} per min, {int(r0[col])} chirps"


def summary_lines(E: pd.DataFrame, counts: pd.DataFrame | None, lodo: pd.DataFrame | None,
                  kind: str, rates: pd.DataFrame | None = None, where: str = "main"
                  ) -> list[str]:
    """'Acausal control' block of summary.txt (where='main') or acausal_summary.txt
    (where='acausal') for one kind. The simulation block (hz_acausal_sim.summary_lines)
    follows it in both files."""
    if E is None or not len(E) or not (E["kind"] == kind).any():
        return []
    g = "500ms"
    gm = E["model"].astype(str).str.extract(r"^pf_guard_(\d+ms)$")[0].dropna()
    if len(gm):
        g = gm.iloc[0]
    A = E[(E["variant"] == "two_sided") & (E["kind"] == kind)]
    fe, re_, gl = "glm_fe", "re_meta", "mixed_laplace"
    src = ("details in acausal/acausal_summary.txt and acausal_effects.csv, simulation in "
           "simulation/acausal_sim_summary.txt" if where == "main" else
           "tables in acausal_effects.csv, acausal_dyad_oe.csv, acausal_leave_one_dyad_out.csv")
    L = [f"  Acausal control (deliberately uses sub chirps AFTER the outcome bin; not part of "
         f"the models above; {src}):"]
    pairs = {}
    for model, pt, ft, lab in (("pf_1000ms", "sub_any_1000ms", "sub_fut_1000ms",
                                "0-1 s before vs 0-1 s after the bin"),
                               (f"pf_guard_{g}", f"sub_grd_{g}", f"sub_fgd_{g}",
                                "0.5-1.5 s before vs 0.5-1.5 s after")):
        p_ = _pick(A, model=model, backend=fe, term=pt)
        f_ = _pick(A, model=model, backend=fe, term=ft)
        c_ = _pick(A, model=model, backend=fe, term=f"{pt}-{ft}")
        cg_ = _pick(A, model=model, backend=gl, term=f"{pt}-{ft}")
        pairs[model] = (c_, p_, f_, cg_)
        if p_ is None:
            continue
        pr = _pick(A, model=model, backend=re_, term=pt)
        fr = _pick(A, model=model, backend=re_, term=ft)
        fm = _pick(A, model=model.replace("pf_", "fut_"), backend=fe, term=ft)
        L.append(f"    Past vs future window ({lab}), joint FE model: HR_past {_ci(p_)}, "
                 f"HR_future {_ci(f_)}; HR_past / HR_future {_ci(c_, p=True)}{_glmm_txt(cg_)}"
                 + _lodo_txt(lodo, kind, model) + f". RE mean: past {_ci(pr)}, future {_ci(fr)}; "
                 f"future window alone (FE) {_ci(fm)}.")
    scan = A[(A["backend"] == fe) & (A["term_type"] == "contrast")
             & A["model"].astype(str).str.match(r"^pf_\d+ms$")].copy()
    r0 = None
    if rates is not None and len(rates) and (rates["recording"] == "ALL").any():
        r0 = rates[rates["recording"] == "ALL"].iloc[0]
    if len(scan) > 1:
        scan["tau"] = scan["model"].str.extract(r"(\d+)ms")[0].astype(float) / 1000
        scan = scan.sort_values("tau")
        txt = ("    HR_past / HR_future by window tau (joint FE model): " + ", ".join(
            f"{t:g} s {r:.2f} [{lo:.2f}, {hi:.2f}]" for t, r, lo, hi in zip(
                scan["tau"], scan["ratio"], scan["ratio_low"], scan["ratio_high"], strict=True))
            + ". Not calibrated: the simulation covers the 1 s and guarded windows only")
        if kind == "offset":
            post = ""
            if r0 is not None and "rate_post_2_5s_per_min" in r0 and np.isfinite(
                    r0["rate_post_2_5s_per_min"]):
                post = (f"sub chirps stay frequent 2-5 s after an offset "
                        f"({float(r0['rate_post_2_5s_per_min']):.1f} per min, "
                        f"{int(r0['n_post_2_5s'])} chirps; {float(r0['rate_in_per_min']):.1f} "
                        f"inside and {float(r0['rate_out_per_min']):.1f} outside chases on "
                        "average), so longer after-windows do not reach a low-rate gap; ")
            tail = (f"{post}risk-set bins within tau before an offset carry the post-offset "
                    "burst in their own after-window, so how the ratio should change with tau "
                    "under each account is unknown.")
            txt += ". " + tail[:1].upper() + tail[1:]
        else:
            txt += (" (and no onset account has a longer window); longer after-windows of an "
                    "onset bin reach further into the new chase.")
        L.append(txt)
    km = _pick(A, model="kernel_two_sided", backend=fe, term="kernel_sub_mean")
    kf = _pick(A, model="kernel_two_sided", backend=fe, term="kernel_sub_fut_mean")
    kc = _pick(A, model="kernel_two_sided", backend=fe,
               term="kernel_sub_mean-kernel_sub_fut_mean")
    kcg = _pick(A, model="kernel_two_sided", backend=gl,
                term="kernel_sub_mean-kernel_sub_fut_mean")
    kj = _pick(A, model="kernel_two_sided", backend=fe, term="kernel_sub_shape_diff_joint")
    if km is not None:
        L.append(f"    Two-sided kernel (FE, per chirp, mean over 0.05-2 s): before the bin "
                 f"{_ci(km)}, after {_ci(kf)}; before / after {_ci(kc, p=True)}"
                 f"{_glmm_txt(kcg)}; shape difference joint p = "
                 f"{_pv(float(kj['p'])) if kj is not None else 'n/a'}"
                 + _lodo_txt(lodo, kind, "kernel_two_sided") + ".")
    c1, p1, f1, c1g = pairs.get("pf_1000ms", (None,) * 4)
    cg, pg, fg, cgg = pairs.get(f"pf_guard_{g}", (None,) * 4)
    if kind == "offset":
        B = E[E["kind"] == "offset"]
        boot_w = sorted({str(s) for s in B.loc[B["variant"] == "restricted_real-mirror",
                                                "subset"]},
                        key=lambda s: int(s.split("_")[1].removesuffix("ms")))
        lag_rows = []  # (label, ratio row) of every contrast at 0.5-1.5 s lags
        if cg is not None:
            lag_rows.append(("A guarded (FE)", cg))
        if kc is not None:
            lag_rows.append(("kernel (FE)", kc))
        for W in boot_w:
            wv = int(W.split("_")[1].removesuffix("ms")) / 1000
            rr = _pick(B, variant="restricted_real", subset=W, backend=fe, term="sub_any_1000ms")
            mi = _pick(B, variant="mirror", subset=W, backend=fe, term="sub_any_1000ms")
            d1 = _pick(B, variant="restricted_real-mirror", subset=W, term="sub_any_1000ms")
            dg = _pick(B, variant="restricted_real-mirror", subset=W, term=f"sub_grd_{g}")
            rrg = _pick(B, variant="restricted_real", subset=W, backend=fe, term=f"sub_grd_{g}")
            mig = _pick(B, variant="mirror", subset=W, backend=fe, term=f"sub_grd_{g}")
            if dg is not None:
                lag_rows.append((f"B guarded W = {wv:g} s", dg))
            cnt = ""
            if counts is not None and len(counts):
                cc = counts[(counts["recording"] == "ALL") & (counts["window"] == wv)]
                a = cc[cc["variant"] == "restricted_real"]
                b = cc[cc["variant"] == "mirror"]
                if len(a) and len(b):
                    na = int(a["n_dropped"].iloc[0]) if "n_dropped" in a else 0
                    cnt = (f"; in-chase sub chirps: restricted real {int(a['n_in_chase'].iloc[0])}"
                           f" ({na} with no mirror counterpart dropped: reflected time at or "
                           f"after the next onset / unknown state), mirror "
                           f"{int(b['n_in_chase'].iloc[0])} (of {int(b['n_candidates'].iloc[0])} "
                           f"post-offset chirps; {int(b['n_dropped'].iloc[0])} dropped, "
                           "reflection before onset)")
            L.append(f"    Mirror surrogate, W = {wv:g} s (post-offset chirps reflected into the "
                     f"chase; all real in-chase chirps replaced{cnt}): FE 1 s restricted real "
                     f"{_ci(rr)} vs mirror {_ci(mi)}, restricted / mirror {_ci(d1, p=True)} "
                     f"(dyad bootstrap); guarded restricted real {_ci(rrg)} vs mirror {_ci(mig)}, "
                     f"ratio {_ci(dg, p=True)}.")
        sh = B[B["variant"].astype(str).str.startswith("shift_") & (B["backend"] == fe)
               & (B["term"] == "sub_any_1000ms")]
        if len(sh):
            parts = []
            for _, r in sh.iterrows():
                dl = int(str(r["variant"]).split("_")[1].removesuffix("ms")) / 1000
                sw = str(r["subset"])
                wsh = int(sw.split("_")[1].removesuffix("ms")) / 1000 if sw.startswith("w_") \
                    else 2.0
                wc = f"w_{_ms(min(dl, wsh))}"
                n = ""
                if counts is not None and len(counts):
                    cc = counts[(counts["recording"] == "ALL")
                                & (counts["variant"] == r["variant"])]
                    if len(cc):
                        n = f", {int(cc['n_in_chase'].iloc[0])} in-chase chirps"
                rrc = _pick(B, variant="restricted_real", subset=wc, backend=fe,
                            term="sub_any_1000ms")
                mic = _pick(B, variant="mirror", subset=wc, backend=fe, term="sub_any_1000ms")
                parts.append(f"Delta {dl:g} s: shift {_ci(r)}{n} vs restricted real {_ci(rrc)} "
                             f"and mirror {_ci(mic)} (W = {min(dl, wsh):g} s)")
            L.append("    Constant backward shift of the post-offset chirps (as if every offset "
                     "were annotated Delta late; a shifted train keeps only chirps with latency "
                     "< Delta and translates them to off - Delta + L, the mirror reflects them to "
                     "off - L), FE 1 s, each with the arms of the matching window: "
                     + "; ".join(parts) + ". The translation puts the post-offset chirps closest "
                     "to the offset ~Delta s before it, so a long shift leaves fewer chirps in "
                     "the last 1 s of the chase than either arm of its window.")
        L.append("    Selected trains (restricted real, mirror, shift) put every in-chase chirp in "
                 "the last W s of a chase, which inflates their HRs: compare arms with the same W, "
                 "not with the primary HR.")
        if lag_rows:
            above = [lab for lab, r in lag_rows if float(r["ratio"]) > 1]
            excl = [lab for lab, r in lag_rows if float(r["ci_low"]) > 0]
            gtxt = ""
            if cgg is not None and float(cgg["ci_low"]) > 0:
                gtxt += f"; GLMM (common effect) A guarded {_ci(cgg)}"
            if kcg is not None and float(kcg["ci_low"]) > 0:
                gtxt += f", kernel {_ci(kcg)}"
            L.append("    Lags 0.5-1.5 s: " + "; ".join(f"{lab} {_ci(r)}" for lab, r in lag_rows)
                     + f". Before > after in {len(above)} of {len(lag_rows)} contrasts; CI above "
                     f"1 in {len(excl)} ({', '.join(excl) if excl else 'none'}; uncorrected for "
                     f"the {len(lag_rows)} contrasts){gtxt}. Whether this lean is more than the "
                     "accounts produce is judged against the simulation below.")
        d2 = _pick(B, variant="restricted_real-mirror", subset="w_2000ms", term="sub_any_1000ms")
        rr1 = _pick(B, variant="restricted_real", subset="w_2000ms", backend=fe,
                    term="sub_any_1000ms")
        mi1 = _pick(B, variant="mirror", subset="w_2000ms", backend=fe, term="sub_any_1000ms")
        if r0 is not None:
            L.append(
                f"    Base rates ({float(r0['rate_in_per_min']):.1f} vs "
                f"{float(r0['rate_out_per_min']):.1f} per min inside vs outside chases; last 1 s "
                f"of a chase {_rate(r0, 'last_1s_of_chase', 'n_chases')}, first 1 s after its "
                f"offset {_rate(r0, 'first_1s_after_offset', 'n_chases')}): the after-window of "
                "an offset bin lies outside the chase, those of the other risk-set bins mostly "
                "inside, and bins just before an offset see the post-offset burst in their "
                "after-window. Neither A nor B has a null of 1: what each account produces "
                "comes from the simulation below.")
        L.append("    Reading (each contrast against 1 only; what the values mean comes from the "
                 "simulation below): " + "; ".join([
                     _reading(c1, "HR_past / HR_future, 1 s:", p1, f1, g=c1g),
                     _reading(cg, "guarded:", pg, fg, g=cgg),
                     _reading(d2, "restricted real / mirror (W = 2 s, 1 s):", rr1, mi1,
                              labels=("restricted", "mirror"))]) + ".")
        L.append("    " + INTERPRETATION)
    else:
        sp = _pick(A, model="past_1000ms", backend=fe, term="sub_any_1000ms")
        sg = _pick(A, model=f"past_guard_{g}", backend=fe, term=f"sub_grd_{g}")
        if r0 is not None:
            no = "n_onsets" if "n_onsets" in r0 else "n_chases"
            rfirst = (60 * float(r0["first_1s_of_chase"]) / max(float(r0[no]), 1.0))
            L.append(
                f"    Base rates ({float(r0['rate_in_per_min']):.1f} vs "
                f"{float(r0['rate_out_per_min']):.1f} per min inside vs outside chases; last 1 s "
                f"before an onset {_rate(r0, 'last_1s_before_onset', no)}, first 1 s of the chase "
                f"{_rate(r0, 'first_1s_of_chase', no)}, "
                f"{rfirst / max(float(r0['rate_out_per_min']), 1e-9):.1f}x the out-of-chase "
                "rate): the after-window of an onset bin lies inside the chase that the outcome "
                "starts, so the after-window indicator is itself raised by the outcome. "
                "HR_future > 1 and a past / future contrast below 1 then follow without any chirp "
                "effect, and HR_past of the joint model, adjusted for that after-window, is "
                "biased toward < 1 (it conditions on a consequence of the outcome), the direction "
                "of (b). Neither the contrast nor the joint HR_past tests (b); the (b) estimate "
                f"stays the standard past-only HR (FE 1 s {_ci(sp)}; guarded {_ci(sg)}).")
        L.append("    " + INTERPRETATION_ONSET)
    return L


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
KIND_TITLE = {"offset": "(a) chase offset hazard", "onset": "(b) chase onset hazard"}
C_PAST, C_FUT = "#2a78d6", "#c2452d"


def plot_two_sided_kernel(E: pd.DataFrame, out_dir: Path, name: str = "acausal_two_sided_kernel"
                          ) -> list[Path]:
    """HR per sub chirp vs signed lag (+ = chirp before the bin, - = after), both kinds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from hz_desc import INK2, _save, _style

    _style()
    A = E[(E["variant"] == "two_sided") & (E["model"] == "kernel_two_sided")]
    kinds = [k for k in KINDS if k in set(A["kind"])]
    if not kinds:
        return []
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.3 * len(kinds) + 0.3, 3.4), squeeze=False,
                             constrained_layout=True)
    for j, kind in enumerate(kinds):
        ax = axes[0, j]
        a = A[A["kind"] == kind]
        vals = []
        for backend, ls, lw, band in (("glm_fe", "-", 1.8, True),
                                      ("mixed_laplace", (0, (3, 2)), 0.8, False)):
            for term, col in (("kernel_sub", C_PAST), ("kernel_sub_fut", C_FUT)):
                k = a[(a["backend"] == backend) & (a["term"] == term)].sort_values("lag")
                if k.empty:
                    continue
                x = k["lag"].to_numpy(float)
                if band:
                    ax.fill_between(x, k["ratio_low"].clip(0.02, 50),
                                    k["ratio_high"].clip(0.02, 50), color=col, alpha=0.16, lw=0,
                                    zorder=2)
                    vals += [k["ratio_low"].to_numpy(), k["ratio_high"].to_numpy()]
                # the GLMM curve lies within ~1% of the FE curve: thin dark dashes on top
                ax.plot(x, k["ratio"], color=col if band else INK2, lw=lw, ls=ls,
                        zorder=4 if band else 5)
                vals.append(k["ratio"].to_numpy())
        v = np.concatenate(vals) if vals else np.array([1.0])
        v = v[np.isfinite(v) & (v > 0)]
        lo = max(0.05, min(0.8, float(v.min()) / 1.15)) if len(v) else 0.5
        hi = min(20.0, max(1.25, float(v.max()) * 1.15)) if len(v) else 2.0
        hz._ratio_axis(ax, lo, hi)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvspan(-0.025, 0.025, color="#dcdbd5", zorder=0)
        ax.set_xlim(-2.05, 2.05)
        ax.set_xlabel("signed lag (s): bin time - chirp time\n(< 0: chirp AFTER the bin, acausal;"
                      " > 0: chirp before the bin)")
        if j == 0:
            ax.set_ylabel("hazard ratio per sub chirp (log scale)")
        km = _pick(a, backend="glm_fe", term="kernel_sub_mean")
        kf = _pick(a, backend="glm_fe", term="kernel_sub_fut_mean")
        kc = _pick(a, backend="glm_fe", term="kernel_sub_mean-kernel_sub_fut_mean")
        kj = _pick(a, backend="glm_fe", term="kernel_sub_shape_diff_joint")
        txt = ""
        if km is not None and kf is not None and kc is not None:
            txt = (f"FE mean HR: before {km['ratio']:.2f}, after {kf['ratio']:.2f}; "
                   f"ratio {kc['ratio']:.2f} [{kc['ratio_low']:.2f}, {kc['ratio_high']:.2f}], "
                   f"p={_pv(float(kc['p']))}"
                   + (f"\nshape difference joint p={_pv(float(kj['p']))}" if kj is not None
                      else ""))
        ax.set_title(f"{KIND_TITLE[kind]}\n{txt}", loc="left", fontsize=7.6)
    handles = [Line2D([], [], color=C_PAST, lw=1.8, label="chirp before the bin (causal side)"),
               Line2D([], [], color=C_FUT, lw=1.8, label="chirp after the bin (acausal side)"),
               Patch(color=INK2, alpha=0.2, label="cloglog GLM, dyad FE, cluster-robust 95% band"),
               Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)),
                      label="logit GLMM, dyad random intercept (point estimate)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Acausal control: two-sided sub-chirp kernel, fitted jointly (outcome bin "
                 "excluded on both sides; pointwise 95% CI)", fontsize=8.6, x=0.01, ha="left")
    return _save(fig, Path(out_dir), name)


def plot_tau_past_future(E: pd.DataFrame, out_dir: Path,
                         name: str = "acausal_tau_past_vs_future") -> list[Path]:
    """HR_past and HR_future vs tau (joint and marginal models, FE) and their ratio."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from hz_desc import INK2, _save, _style

    _style()
    A = E[(E["variant"] == "two_sided") & (E["backend"] == "glm_fe")].copy()
    kinds = [k for k in KINDS if k in set(A["kind"])]
    if not kinds:
        return []
    A["win"] = A["model"].astype(str).str.replace(r"^(pf|past|fut)_", "", regex=True)
    wins = [w for w in A["win"].unique() if w.endswith("ms") and not w.startswith("guard")]
    wins = sorted(wins, key=lambda w: int(w.removesuffix("ms"))) + sorted(
        w for w in A["win"].unique() if w.startswith("guard"))
    labels = [f"{int(w.removesuffix('ms')) / 1000:g}" if not w.startswith("guard") else
              "guard\n0.5-1.5" for w in wins]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4.4 * len(kinds) + 0.3, 5.6), squeeze=False,
                             constrained_layout=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    series = [("past_", "sub_{a}_{w}", C_PAST, "o", "white", -0.21, "past only"),
              ("pf_", "sub_{a}_{w}", C_PAST, "o", C_PAST, -0.07, "past, joint"),
              ("pf_", "sub_{f}_{w}", C_FUT, "s", C_FUT, 0.07, "future, joint"),
              ("fut_", "sub_{f}_{w}", C_FUT, "s", "white", 0.21, "future only")]
    for j, kind in enumerate(kinds):
        ax, bx = axes[0, j], axes[1, j]
        a = A[A["kind"] == kind]
        vals = []
        for pre, tmpl, col, mk, mfc, dx, _ in series:
            for i, w in enumerate(wins):
                g_ = w.startswith("guard")
                ww = w.removeprefix("guard_")
                term = tmpl.format(a="grd" if g_ else "any", f="fgd" if g_ else "fut", w=ww)
                r = _pick(a, model=f"{pre}{w}", term=term)
                if r is None:
                    continue
                ax.errorbar(i + dx, r["ratio"], yerr=[[r["ratio"] - r["ratio_low"]],
                                                      [r["ratio_high"] - r["ratio"]]],
                            fmt=mk, ms=4.5, color=col, mfc=mfc, mec=col, elinewidth=1.0, lw=0)
                vals += [r["ratio"], r["ratio_low"], r["ratio_high"]]
        for i, w in enumerate(wins):
            ww = w.removeprefix("guard_")
            term = (f"sub_grd_{ww}-sub_fgd_{ww}" if w.startswith("guard")
                    else f"sub_any_{ww}-sub_fut_{ww}")
            c = _pick(a, model=f"pf_{w}", term=term)
            if c is None:
                continue
            bx.errorbar(i, c["ratio"], yerr=[[c["ratio"] - c["ratio_low"]],
                                             [c["ratio_high"] - c["ratio"]]],
                        fmt="D", ms=4.5, color=INK2, elinewidth=1.0, lw=0)
            bx.text(i + 0.12, c["ratio"], f"p={_pv(float(c['p']))}", fontsize=6.3,
                    color=INK2, va="center")
        v = np.array([x for x in vals if np.isfinite(x) and x > 0] or [1.0])
        hz._ratio_axis(ax, *hz._lims(v, v, v))
        cv = a[a["term_type"] == "contrast"]
        cvv = np.r_[cv["ratio"], cv["ratio_low"], cv["ratio_high"]].astype(float)
        cvv = cvv[np.isfinite(cvv) & (cvv > 0)]
        hz._ratio_axis(bx, *hz._lims(cvv, cvv, cvv) if len(cvv) else (0.5, 2))
        for x_ in (ax, bx):
            x_.set_xticks(np.arange(len(wins)))
            x_.set_xticklabels(labels)
            x_.set_xlim(-0.5, len(wins) - 0.3)
        ax.set_title(f"{KIND_TITLE[kind]}: >= 1 sub chirp in the window before / after the bin",
                     loc="left", fontsize=7.8)
        want = "> 1" if kind == "offset" else "< 1"
        bx.set_title(f"HR_past / HR_future, joint model ({want}: before beyond after)",
                     loc="left", fontsize=7.8)
        bx.set_xlabel("window tau (s) before / after the outcome bin")
        if j == 0:
            ax.set_ylabel("hazard ratio (log scale)")
            bx.set_ylabel("ratio of HRs (log scale)")
    handles = [Line2D([], [], color=col, marker=mk, mfc=mfc, mec=col, lw=0, ms=4.5, label=lab)
               for _, _, col, mk, mfc, _, lab in series]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Acausal control: past vs future sub-chirp window (cloglog GLM, dyad FE, "
                 "cluster-robust 95% CI; one model per window)", fontsize=8.6, x=0.01, ha="left")
    return _save(fig, Path(out_dir), name)


def plot_mirror(E: pd.DataFrame, counts: pd.DataFrame, lat: pd.DataFrame, out_dir: Path,
                name: str = "acausal_mirror") -> list[Path]:
    """Mirror surrogate (offset): latency histograms, tau scans and kernels per arm, the shift
    arms next to the restricted-real and mirror arms of the same window, and the paired
    restricted-real - mirror contrast."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator

    from hz_desc import INK2, MUTED, _save, _style

    _style()
    B = E[(E["kind"] == "offset") & (E["backend"] == "glm_fe")]
    if B.empty or not (B["variant"] == "mirror").any():
        return []
    W0 = "w_2000ms" if (B["subset"] == "w_2000ms").any() else str(
        B.loc[B["variant"] == "mirror", "subset"].iloc[0])
    w0 = int(W0.split("_")[1].removesuffix("ms")) / 1000
    C_REAL, C_SHIFT = "#1b1b1b", "#7a4fb5"
    arms = [("real_full", "", C_REAL, "o", "real, full train (primary)"),
            ("restricted_real", W0, C_PAST, "o",
             "restricted real (in-chase chirps < W s before the offset)"),
            ("mirror", W0, C_FUT, "o", "mirror (post-offset reflected)")]
    fig = plt.figure(figsize=(12.4, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = {"hist": fig.add_subplot(gs[0, 0]), "tau": fig.add_subplot(gs[0, 1]),
            "shift": fig.add_subplot(gs[0, 2]), "kern": fig.add_subplot(gs[1, 0]),
            "pair": fig.add_subplot(gs[1, 1:])}
    # (1) latency histograms
    ax = axes["hist"]
    lw = lat[lat["window"] == w0] if len(lat) else lat
    bins = np.arange(0, w0 + 1e-9, 0.1)
    for side, col, lab in (("before_offset", C_PAST, "in-chase, before offset"),
                           ("after_offset", C_FUT, "out of chase, after offset")):
        x = lw.loc[lw["side"] == side, "latency"].to_numpy(float)
        ax.hist(x, bins=bins, histtype="step", color=col, lw=1.5, label=f"{lab} (n={len(x)})")
    ax.set_xlabel("|time from annotated offset| (s)")
    ax.set_ylabel("sub chirps per 0.1 s (all dyads)")
    ax.set_title(f"Sub chirps within {w0:g} s of a chase offset (0-3 h, 15 dyads)", loc="left",
                 fontsize=7.8)
    ax.legend(fontsize=6.8, loc="upper right")
    # (2) tau scans per arm (W = w0)
    ax = axes["tau"]
    tt = B[B["term"].astype(str).str.match(r"^sub_any_\d+ms$")].copy()
    tt["tau"] = tt["term"].str.extract(r"(\d+)ms")[0].astype(float) / 1000
    taus = np.sort(tt["tau"].unique())
    offs = np.linspace(-0.2, 0.2, len(arms))
    vals = []
    for (v, sub, col, mk, lab), dx in zip(arms, offs):
        r = tt[(tt["variant"] == v)
               & (tt["subset"].fillna("").astype(str) == sub)].sort_values("tau")
        if r.empty:
            continue
        x = np.searchsorted(taus, r["tau"].to_numpy()) + dx
        ax.errorbar(x, r["ratio"], yerr=[r["ratio"] - r["ratio_low"], r["ratio_high"] - r["ratio"]],
                    fmt=mk, ms=3.8, color=col, elinewidth=0.9, lw=0)
        vals += list(r["ratio"]) + list(r["ratio_low"]) + list(r["ratio_high"])
    v = np.array([x for x in vals if np.isfinite(x) and x > 0] or [1.0])
    hz._ratio_axis(ax, *hz._lims(v, v, v))
    ax.set_xticks(np.arange(len(taus)))
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_xlabel("tau: >= 1 sub chirp in the preceding tau (s)")
    ax.set_ylabel("chase-offset HR (log scale)")
    ax.set_title(f"Standard offset model on each train (W = {w0:g} s; FE, 95% CI)",
                 loc="left", fontsize=7.8)
    # (3) shift arms vs the restricted-real and mirror arms of W = Delta (tau 1 s)
    ax = axes["shift"]
    sh = sorted({str(x) for x in B["variant"] if str(x).startswith("shift_")},
                key=lambda s_: int(s_.split("_")[1].removesuffix("ms")))
    vals, xt = [], []
    for i, s_ in enumerate(sh):
        dl = int(s_.split("_")[1].removesuffix("ms")) / 1000
        wc = f"w_{_ms(min(dl, 2.0))}"
        xt.append(f"{dl:g}")
        for (v_, sub, col, mk), dx in zip(((("restricted_real", wc, C_PAST, "o"),
                                            ("mirror", wc, C_FUT, "o"),
                                            (s_, None, C_SHIFT, "D"))), (-0.18, 0.0, 0.18)):
            r = _pick(B, variant=v_, term="sub_any_1000ms", **({"subset": sub} if sub else {}))
            if r is None:
                continue
            ax.errorbar(i + dx, r["ratio"], yerr=[[r["ratio"] - r["ratio_low"]],
                                                  [r["ratio_high"] - r["ratio"]]],
                        fmt=mk, ms=4.2, color=col, elinewidth=0.9, lw=0)
            vals += [r["ratio"], r["ratio_low"], r["ratio_high"]]
    if vals:
        v = np.array([x for x in vals if np.isfinite(x) and x > 0] or [1.0])
        hz._ratio_axis(ax, *hz._lims(v, v, v))
        ax.set_xticks(np.arange(len(sh)))
        ax.set_xticklabels(xt)
        ax.set_xlim(-0.6, len(sh) - 0.4)
    ax.set_xlabel("Delta = W (s): shift vs the arms of the same window")
    ax.set_ylabel("chase-offset HR, tau = 1 s (log scale)")
    ax.set_title("Constant shift (C) vs restricted real and mirror of W = Delta", loc="left",
                 fontsize=7.8)
    # (4) kernels
    ax = axes["kern"]
    vals = []
    for v_, sub, col, _, lab in arms:
        k = B[(B["variant"] == v_) & (B["subset"].fillna("").astype(str) == sub)
              & (B["term"] == "kernel_sub")].sort_values("lag")
        if k.empty:
            continue
        ax.fill_between(k["lag"], k["ratio_low"].clip(0.02, 50), k["ratio_high"].clip(0.02, 50),
                        color=col, alpha=0.12, lw=0)
        ax.plot(k["lag"], k["ratio"], color=col, lw=1.6)
        vals += list(k["ratio"]) + list(k["ratio_low"]) + list(k["ratio_high"])
    v = np.array([x for x in vals if np.isfinite(x) and x > 0] or [1.0])
    hz._ratio_axis(ax, *hz._lims(v, v, v))
    ax.set_xlim(0, 2)
    ax.set_xlabel("time since sub chirp (s)")
    ax.set_ylabel("HR per chirp (log scale)")
    ax.set_title(f"Per-chirp kernel on each train (W = {w0:g} s; FE, pointwise 95% band)",
                 loc="left", fontsize=7.8)
    # (5) paired contrasts
    ax = axes["pair"]
    C = E[(E["variant"] == "restricted_real-mirror")].copy()
    ylab, ys = [], []
    for i, r in enumerate(C.itertuples()):
        y = len(C) - i
        ax.errorbar(r.ratio, y, xerr=[[r.ratio - r.ratio_low], [r.ratio_high - r.ratio]],
                    fmt="D", ms=4.5, color=INK2, elinewidth=1.1)
        ax.text(r.ratio_high * 1.06, y, f"{r.ratio:.2f} [{r.ratio_low:.2f}, {r.ratio_high:.2f}]"
                f" p={_pv(r.p)}", fontsize=6.6, va="center", color=INK2)
        w = int(str(r.subset).split("_")[1].removesuffix("ms")) / 1000
        win = "1 s" if str(r.term).startswith("sub_any") else "0.5-1.5 s"
        ylab.append(f"W = {w:g} s, {win}")
        ys.append(y)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale("log")
    lo_x, hi_x = 0.5, 2.0
    if len(C):
        lo_x = min(0.8, float(np.nanmin(C["ratio_low"])) / 1.2)
        hi_x = max(1.5, float(np.nanmax(C["ratio_high"])) * 1.8)  # room for the labels
    ax.set_xlim(lo_x, hi_x)
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6) if lo_x <= t <= hi_x]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks(ys)
    ax.set_yticklabels(ylab, fontsize=7)
    ax.set_ylim(0.3, len(C) + 0.7)
    ax.set_xlabel("HR(restricted real) / HR(mirror), log scale (no null of 1: see the simulation)")
    n = int(C["n_jitter"].max()) if len(C) and np.isfinite(C["n_jitter"].max()) else 0
    ax.set_title(f"Paired contrast, dyad bootstrap ({n} resamples, percentile 95% CI)",
                 loc="left", fontsize=7.8)
    handles = [Line2D([], [], color=col, marker=mk, lw=0, ms=4.5, label=lab)
               for _, _, col, mk, lab in arms]
    handles.append(Line2D([], [], color=C_SHIFT, marker="D", lw=0, ms=4.5,
                          label="shift by Delta (post-offset chirps with latency < Delta)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Acausal control B/C (offset): post-offset sub chirps reflected (mirror) or "
                 "shifted into the chase, vs real in-chase chirps from the same window. Selected "
                 "trains inflate HRs: compare arms with the same window.", fontsize=8.2, x=0.01,
                 ha="left")
    return _save(fig, Path(out_dir), name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from hz_data import load_trials

    ap = argparse.ArgumentParser(description="Acausal control of the chirp -> chase hazard "
                                             "association (future windows, mirror surrogate).")
    ap.add_argument("data_root", type=Path, nargs="?", default=Config().data_root)
    ap.add_argument("--out", type=Path, required=True, help="output directory of the pipeline")
    ap.add_argument("--report-only", action="store_true",
                    help="redraw figures and acausal_summary.txt from the tables in --out")
    ap.add_argument("--quick", action="store_true", help="tau 1 s only, W = 2 s, 40 resamples")
    ap.add_argument("--n-boot", type=int, default=None, help="dyad bootstrap resamples (500)")
    ap.add_argument("--workers", type=int, default=None, help="forked bootstrap workers (8)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if a.report_only:
        print(report_from_tables(a.out))
        return
    cfg = Config(data_root=a.data_root, out_dir=a.out).updated(quick=a.quick or None, seed=a.seed)
    t0 = time.perf_counter()
    trials, _, _ = load_trials(cfg)
    print(f"{len(trials)} trials loaded ({time.perf_counter() - t0:.1f} s)", flush=True)
    res = run_acausal(trials, cfg, n_boot=a.n_boot, n_workers=a.workers)
    print("\n" + res["text"])
    print("\ntimings (s):\n" + res["timings"].to_string(index=False))
    print(f"\ntotal {time.perf_counter() - t0:.0f} s; tables in {Path(cfg.out_dir) / 'acausal'}")


if __name__ == "__main__":
    main()
