"""Discrete-time hazard models: does subordinate chirping change dominant chasing?

Units: seconds and Hz. Each trial's analysis window [obs_start, obs_end) is cut
into bins of ``cfg.bin_width`` (default 0.05 s); bin k covers [t_k, t_k + bw).

Risk sets (defined by the chase state at the bin start, so they partition time)
--------------------------------------------------------------------------------
offset  Bins whose start lies inside a chase: k in (k_on, k_off], with k_on and
        k_off the bins containing onset and offset. y = 1 in k_off. Chases that
        run past the window end are right-censored.
onset   All other valid bins: k in (k_off_prev, k_on_next]; y = 1 in k_on_next.
        Before the first chase (and after an unknown-state interval) the clock
        runs from the window start and ``first_gap`` = 1.
Bins whose start is not ``Trial.valid`` are dropped. Structural zeros: every
annotated chase lasts >= ~0.97 s and every inter-chase gap is >= ~1.97 s
(BORIS annotation floor), so bins closer than ``min_elapsed`` to the last state
change cannot hold an event. They are dropped (auto: shortest observed
duration/gap minus two bins; an event bin is never dropped), which spares the
elapsed-time spline from modelling a hard zero.

History covariates (no leakage)
-------------------------------
Event trains are binned and convolved with kernels that are zero at lag 0, so
bin k only sees events in bins < k, i.e. strictly before t_k. A chirp in the
outcome bin never enters that bin's covariates (see test_hz_hazard.py).
- kernel models: log-time raised-cosine basis (Pillow et al. 2008) on lags
  j * bw, j = 1..hist_max / bw; ``n_basis`` bumps, the first peaking at one bin,
  the last reaching zero at ``hist_max``. The fitted kernel
  k(lag) = sum_i beta_i b_i(lag) is the change in log-odds (log-hazard for
  cloglog) per chirp at that lag. Summaries: ``kernel_<s>_mean`` (mean of k
  over lags bw..hist_max, the integrated effect divided by hist_max) and a
  joint test of all basis coefficients of sender s.
- tau models: indicator ">= 1 chirp in [t_k - tau, t_k)" for the subordinate,
  with the same indicator for dominant chirps as a control; one model per tau.
- guard model (pre-specified sensitivity): ">= 1 chirp in
  [t_k - guard_lag - guard_window, t_k - guard_lag)", default 0.5-1.5 s before
  the bin. Chase boundaries are annotated to ~0.5 s, so chirps closer than
  that to a boundary may sit on the wrong side of it; this model leaves them
  out. ``kernel_<s>_mean_late`` is the kernel mean over lags >= guard_lag.

Controls: natural cubic spline of log1p(elapsed) (time since chase onset, or
since the last offset), ``first_gap`` (onset), log1p(time since last contact)
plus a no-contact-yet indicator, size difference dom - sub (cm), signed and
absolute baseline dEODf dom - sub (Hz; the sign is constant within every
included trial; trailing 60 s median read at t_k - 1 s, so no future EODf), a spline of time in trial (h), rise histories if
``cfg.use_rises`` (same basis; rise stamps are spectrogram frame centres, so
they are delayed by ``rise_guard`` = one 0.33 s frame to keep them causal),
and the dyad (recording). All non-dummy columns are
standardized for fitting; estimates are reported on the natural scale.

Backends
--------
mixed_laplace  PRIMARY. statsmodels BinomialBayesMixedGLM: logit link,
    Gaussian random intercept per recording, fixed effects N(0, fe_p^2) on
    standardized columns (fe_p = 5, weak), log dyad SD N(0, vcp_p^2). Fitted
    by its Laplace approximation (posterior mode + inverse Hessian, as in
    ``fit_map``), using the class's own log posterior and gradient plus an
    analytic Hessian (checked against finite differences in the tests) so a
    trust-region Newton solver converges fast on ~3e6 bins. We use Laplace
    rather than the mean-field ``fit_vb`` because the kernel band and the joint
    test need the full covariance of the overlapping (strongly correlated)
    basis coefficients; mean-field VB returns only marginal SDs, which it
    tends to understate, and its 20-point quadrature per bin and gradient step
    is slow (offset: ~70-95 s per model vs 0.5 s for Laplace; onset: minutes).
    ``mixed_vb`` (fit_vb started at the Laplace mode) can be added for
    comparison. No lme4/glmmTMB is available, and with one random intercept
    and >1e5 bins the Laplace posterior is close to Gaussian, so MCMC is not
    needed. Logit vs cloglog: per-bin event probabilities are < 1%, where the
    two links agree to first order. Posterior tail "p" values are reported for
    convenience (2 * normal tail of mean / SD).
glm_fe  CROSS-CHECK 1. Binomial GLM, cloglog link (grouped-time analogue of
    the Cox model), fixed dyad intercepts, cluster-robust (recording) sandwich
    covariance (observed-information bread, as statsmodels) with t(G - 1)
    intervals and F(q, G - 1) joint tests. With ~15 clusters this is still
    anti-conservative, and joint tests of terms carried by few recordings
    (dominant chirps inside chases) are degenerate. Dyad-level covariates
    (size difference) are absorbed by the dyad intercepts. Fitted by the IRLS
    in ``fit_glm``, which matches statsmodels GLM (params 1e-6, cluster SEs
    1e-4 relative; tests) and accumulates X'WX in row chunks.
re_meta  HETEROGENEITY (two-stage). ``dyad_oe`` compares each dyad's
    observed events in indicator-on bins with the number expected under no
    chirp effect (glm_fe baseline); ``re_meta`` pools the per-dyad O/E in a
    Poisson-normal random-effects model (mean log HR across dyads, between-
    dyad SD tau, profile-likelihood CI with a t(k-1)^2 cut-off) and tests
    homogeneity (G test, LR test of tau = 0). The GLMM above assumes one
    chirp effect for all dyads; this does not.
jitter  CROSS-CHECK 2, a within-dyad permutation test. ``jitter_test`` refits glm_fe under null subordinate chirp
    trains from hz_desc.null_replicates (same code and seeds as the
    descriptives): 'shift' (primary; circular shift on each state clock) and
    'matched' (each chirp moved independently to another chase / gap of the
    same trial that lasted longer than its elapsed time, keeping its elapsed
    time since onset/offset, so chirp-vs-elapsed-time structure survives;
    hz_desc.swap_matched). The matched null does NOT keep inter-chirp
    intervals (bursts are broken up), which narrows its null SD by ~10% and
    makes it mildly anti-conservative; 'shift' is the ICI-preserving null.
    Only the
    subordinate columns are recomputed; refits use a chord-Newton iteration
    (_Chord) in forked workers. Statistics: tau = 1 s indicator coefficient
    and kernel mean; p values are two-sided relative to the null distribution
    (with +1 correction, so min p = 2 / (n_jitter + 1)).

Runtime on the 15 real trials (bin 0.05 s, 32-thread Ryzen): designs ~1 s;
offset fits ~0.5 s each, onset (2.9e6 bins, no subsampling) ~5-10 s each;
jitter refits ~0.25 s (offset) and ~2.5 s (onset) per null draw with 8
workers, i.e. ~1 min and ~8 min per null at n_jitter = 200.

Onset controls can be subsampled (``cfg.onset_control_fraction`` f < 1): all
event bins and a fraction f of non-event bins per recording are kept. Under
the logit link this is case-control sampling: slopes are unchanged and every
kept row's log-odds is shifted by +log(1/f), so the fitted intercept (or each
dyad intercept) minus log(1/f) is the full-data one. The cloglog GLM instead
uses inverse-probability weights 1/f on kept controls (consistent pseudo-
likelihood; the sandwich covariance accounts for the weights but not for the
subsampling noise).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import patsy
from scipy import optimize, sparse, stats
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402
# Null chirp trains are shared with the descriptives (same code, same seeds).
from hz_desc import DescParams, _in_iv, null_replicates, trial_states  # noqa: E402

KINDS = ("offset", "onset")
SENDERS = ("sub", "dom")

# Hazard-specific options. Read from a Config attribute of the same name if it
# exists, else from cfg.extra, else these defaults.
DEFAULTS: dict[str, Any] = dict(
    basis_log_offset=0.2,  # s, c in log(lag + c) of the raised-cosine basis
    elapsed_df=5,  # natural-spline df for log1p(elapsed)
    hours_df=4,  # natural-spline df for time in trial
    offset_min_elapsed=None,  # s; None = auto (shortest chase - 2 bins), 0 = keep all
    onset_min_elapsed=None,  # s; None = auto (shortest gap - 2 bins), 0 = keep all
    fe_p=5.0,  # prior SD of standardized fixed effects (mixed)
    vcp_p=2.0,  # prior SD of the log dyad-intercept SD (mixed)
    hazard_backends=("mixed_laplace", "glm_fe"),
    jitter_tau=1.0,  # s, tau of the indicator statistic in jitter_test
    glm_link="cloglog",  # link of glm_fe and the jitter refits
    n_workers=8,  # forked processes for jitter_test (1 = serial)
    rise_guard=0.33,  # s, rise times (spectrogram frame centres) delayed by one frame
    # Lag-guarded indicator (pre-specified sensitivity): >= 1 chirp in
    # [t - guard_lag - guard_window, t - guard_lag). guard_lag = the ~0.5 s
    # precision of the BORIS chase boundaries, so chirps that could sit on the
    # wrong side of a mis-annotated boundary do not enter the statistic.
    guard_lag=0.5,  # s
    guard_window=1.0,  # s
    mixed_sd_starts=(0.5, 3.0),  # initial dyad SDs of the GLMM multi-start
    meta_min_expected=2.0,  # dyads with >= this many expected events are 'informative'
)

EFFECT_COLUMNS = [
    "model", "kind", "backend", "link", "term", "term_type", "lag", "estimate", "se",
    "ci_low", "ci_high", "p", "ratio", "ratio_low", "ratio_high", "n_bins", "n_events",
    "n_recordings", "fit_time_s", "converged", "note",
]


def _opt(cfg: Config, name: str) -> Any:
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return cfg.extra.get(name, DEFAULTS[name])


def _ms(tau: float) -> str:
    return f"{int(round(tau * 1000))}ms"


def guard_model(cfg: Config) -> str:
    """Name of the lag-guarded indicator model, 'guard_<guard_lag ms>'."""
    return f"guard_{_ms(float(_opt(cfg, 'guard_lag')))}"


def model_names(cfg: Config) -> list[str]:
    """'kernel', one 'tau_<ms>' model per cfg.tau_grid entry, and the guard model."""
    return ["kernel"] + [f"tau_{_ms(t)}" for t in cfg.tau_grid] + [guard_model(cfg)]


def _tau_of(model: str) -> float:
    return float(model.split("_")[1].removesuffix("ms")) / 1000.0


def indicator_column(model: str, sender: str) -> str | None:
    """Chirp indicator column of a tau or guard model (None for 'kernel')."""
    if model == "kernel":
        return None
    fam, ms = model.split("_")[:2]
    return f"{sender}_{'grd' if fam == 'guard' else 'any'}_{ms}"


# ---------------------------------------------------------------------------
# History basis and features
# ---------------------------------------------------------------------------
def raised_cosine_basis(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Log-time raised cosines evaluated at lags j * bw, j = 1..hist_max / bw.

    b_i(lag) = (1 + cos(clip((log(lag + c) - m_i) * pi / (2 d), -pi, pi))) / 2
    with centres m_i spaced d apart, the first at log(bw + c) and the last
    ending (support m + 2d) at log(hist_max + c). Returns (lags (J,), B (J, n)).
    """
    bw, n = cfg.bin_width, int(cfg.n_basis)
    J = int(round(cfg.hist_max / bw))
    lags = np.arange(1, J + 1) * bw
    c = float(_opt(cfg, "basis_log_offset"))
    y0, y1 = np.log(bw + c), np.log(cfg.hist_max + c)
    d = (y1 - y0) / (n + 1)
    ctr = y0 + d * np.arange(n)
    u = (np.log(lags + c)[:, None] - ctr[None, :]) * np.pi / (2 * d)
    B = 0.5 * (np.cos(np.clip(u, -np.pi, np.pi)) + 1.0)
    return lags, B


def _binned(t: np.ndarray, t0: float, bw: float, K: int, pad: int) -> np.ndarray:
    """Counts over [t0 - pad * bw, t0 + K * bw); index pad + k is bin k."""
    k = np.floor((np.asarray(t, float) - t0) / bw).astype(np.int64) + pad
    k = k[(k >= 0) & (k < pad + K)]
    return np.bincount(k, minlength=pad + K).astype(float)


def _conv_history(cnt: np.ndarray, B: np.ndarray, pad: int, K: int) -> np.ndarray:
    """H[k, i] = sum_{j>=1} B[j-1, i] * cnt[pad + k - j]: events strictly before bin k."""
    out = np.empty((K, B.shape[1]))
    for i in range(B.shape[1]):
        h = np.concatenate([[0.0], B[:, i]])  # zero at lag 0: no same-bin events
        out[:, i] = np.convolve(cnt, h)[pad : pad + K]
    return out


def _any_before(cnt: np.ndarray, n_lag: int, pad: int, K: int) -> np.ndarray:
    """True if any event in bins k - n_lag .. k - 1."""
    C = np.concatenate([[0.0], np.cumsum(cnt)])  # C[m] = sum(cnt[:m])
    m = pad + np.arange(K)
    return (C[m] - C[np.maximum(m - n_lag, 0)]) > 0


def _any_between(cnt: np.ndarray, n_guard: int, n_win: int, pad: int, K: int) -> np.ndarray:
    """True if any event in bins k - n_guard - n_win .. k - n_guard - 1."""
    C = np.concatenate([[0.0], np.cumsum(cnt)])
    m = pad + np.arange(K) - n_guard
    return (C[np.maximum(m, 0)] - C[np.maximum(m - n_win, 0)]) > 0


def _guard(cfg: Config) -> tuple[float, float]:
    return float(_opt(cfg, "guard_lag")), float(_opt(cfg, "guard_window"))


def _pad_bins(cfg: Config) -> int:
    lags = [cfg.hist_max, *cfg.tau_grid, sum(_guard(cfg))]
    return int(max(round(x / cfg.bin_width) for x in lags)) + 1


def _train_features(
    t: np.ndarray, prefix: str, t0: float, K: int, k: np.ndarray, cfg: Config, B: np.ndarray,
    taus: Sequence[float] | None, guard: bool = False,
) -> dict[str, np.ndarray]:
    """Basis history (plus tau and lag-guarded indicators) of one event train at bins k."""
    pad = _pad_bins(cfg)
    cnt = _binned(t, t0, cfg.bin_width, K, pad)
    H = _conv_history(cnt, B, pad, K)
    out = {f"{prefix}_h{i + 1}": H[k, i] for i in range(B.shape[1])}
    for tau in taus or ():
        n = max(1, int(round(tau / cfg.bin_width)))
        out[f"{prefix}_any_{_ms(tau)}"] = _any_before(cnt, n, pad, K)[k].astype(np.int8)
    if guard:
        g, w = _guard(cfg)
        ng, nw = int(round(g / cfg.bin_width)), max(1, int(round(w / cfg.bin_width)))
        out[f"{prefix}_grd_{_ms(g)}"] = _any_between(cnt, ng, nw, pad, K)[k].astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Risk sets
# ---------------------------------------------------------------------------
def _n_bins(tr: Trial, bw: float) -> int:
    return int(np.floor((tr.obs_end - tr.obs_start) / bw + 1e-9))


def _min_elapsed(trials: Sequence[Trial], kind: str, cfg: Config) -> float:
    """Structural-zero cut-off (s): configured, or shortest duration/gap - 2 bins."""
    v = _opt(cfg, f"{kind}_min_elapsed")
    if v is not None:
        return max(0.0, float(v))
    vals: list[np.ndarray] = []
    for tr in trials:
        ch = tr.chases[tr.chases[:, 1] <= tr.obs_end]
        vals.append(ch[:, 1] - ch[:, 0] if kind == "offset" else ch[1:, 0] - ch[:-1, 1])
    v_all = np.concatenate(vals) if vals else np.zeros(0)
    return max(0.0, float(v_all.min()) - 2 * cfg.bin_width) if len(v_all) else 0.0


def _risk_rows(tr: Trial, kind: str, bw: float, min_el: float) -> dict[str, Any]:
    """Bin indices, outcome, elapsed time and episode id of one trial's risk set."""
    t0 = float(tr.obs_start)
    K = _n_bins(tr, bw)
    starts = t0 + np.arange(K) * bw
    ch = tr.chases[(tr.chases[:, 0] < t0 + K * bw) & (tr.chases[:, 1] > t0)]
    kon = np.floor((ch[:, 0] - t0) / bw).astype(np.int64)
    koff = np.floor((ch[:, 1] - t0) / bw).astype(np.int64)
    d = np.zeros(K + 1, np.int64)
    np.add.at(d, np.clip(kon + 1, 0, K), 1)
    np.add.at(d, np.clip(koff + 1, 0, K), -1)
    inside = np.cumsum(d)[:K] > 0
    valid = tr.valid(starts)
    info: dict[str, Any] = dict(K=K, n_events_possible=0, n_events_lost=0, n_struct_dropped=0)

    if kind == "offset":
        k = np.flatnonzero(inside & valid)
        ci = np.searchsorted(kon, k, side="left") - 1  # chase with k_on < k <= k_off
        el = starts[k] - ch[ci, 0]
        y = k == koff[ci]
        fg = np.zeros(len(k), bool)
        ep = ci
        possible = (koff < K) & (koff > kon)
        info["n_events_possible"] = int(possible.sum())
    else:
        k = np.flatnonzero(~inside & valid)
        y = np.isin(k, kon)
        reset_t = np.full(K, np.nan)
        first = np.zeros(K, bool)
        reset_t[0], first[0] = t0, True
        for _, hi in tr.exclude_intervals:
            kh = int(np.ceil((hi - t0) / bw - 1e-9))
            if 0 <= kh < K:
                reset_t[kh], first[kh] = hi, True
        kk = koff + 1
        m = (kk >= 0) & (kk < K)
        reset_t[kk[m]], first[kk[m]] = ch[m, 1], False
        last = np.maximum.accumulate(np.where(np.isfinite(reset_t), np.arange(K), 0))
        el = starts[k] - reset_t[last[k]]
        fg = first[last[k]]
        ep = last[k]
        info["n_events_possible"] = int(np.sum((kon >= 0) & (kon < K)))
    keep = fg | (el >= min_el)
    info["n_struct_dropped"] = int(np.sum(~keep))
    ev_dropped = int(np.sum(y & ~keep))
    k, y, el, fg, ep = k[keep], y[keep], el[keep], fg[keep], ep[keep]
    info["n_events_lost"] = info["n_events_possible"] - int(y.sum())
    info["n_events_dropped_min_elapsed"] = ev_dropped
    return dict(k=k, y=y, elapsed=el, first_gap=fg, episode=ep, info=info)


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
def build_design(
    trials: Sequence[Trial],
    kind: str,
    cfg: Config,
    chirp_override: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """One row per at-risk bin with outcome ``y``, ``recording`` and covariates.

    ``chirp_override`` maps recording -> replacement subordinate chirp times
    (e.g. a jittered train). The frame's ``attrs`` hold the basis, cut-offs,
    subsampling fraction and per-trial bookkeeping.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    bw = cfg.bin_width
    lags, B = raised_cosine_basis(cfg)
    min_el = _min_elapsed(trials, kind, cfg)
    frac = float(cfg.onset_control_fraction) if kind == "onset" else 1.0
    if not 0 < frac <= 1:
        raise ValueError("onset_control_fraction must be in (0, 1]")
    parts: list[pd.DataFrame] = []
    book: list[dict[str, Any]] = []
    for tr in trials:
        rr = _risk_rows(tr, kind, bw, min_el)
        k, y = rr["k"], rr["y"]
        sel = np.arange(len(k))
        if frac < 1:
            rng = np.random.default_rng([cfg.seed, zlib.crc32(tr.recording.encode())])
            ctrl = np.flatnonzero(~y)
            keep_c = rng.choice(ctrl, size=int(round(frac * len(ctrl))), replace=False)
            sel = np.sort(np.concatenate([np.flatnonzero(y), keep_c]))
        k = k[sel]
        t0, K = float(tr.obs_start), rr["info"]["K"]
        st = t0 + k * bw
        # baseline EODf is a step-hold trailing median on a cfg.eodf_grid_dt grid;
        # reading it one grid step before the bin start keeps interpolation between
        # grid points from reaching past t_k (no future EODf in the covariate).
        # Before a fish's first tracked sample there is no past EODf: NaN here,
        # mean-filled in _standardize and flagged by deodf_missing.
        de = tr.delta_eodf(st - float(cfg.eodf_grid_dt))
        ci = np.searchsorted(tr.contacts, st, side="left") - 1  # contacts strictly before t_k
        no_c = ci < 0
        last_c = tr.contacts[np.maximum(ci, 0)] if len(tr.contacts) else st
        tsc = np.where(no_c, 0.0, st - last_c)
        cols: dict[str, Any] = dict(
            recording=tr.recording, k=k.astype(np.int64), t=st, y=y[sel].astype(np.int8),
            elapsed=rr["elapsed"][sel], first_gap=rr["first_gap"][sel].astype(np.int8),
            episode=rr["episode"][sel].astype(np.int64),
            hours=(st - tr.t_start) / 3600.0, deodf=de, abs_deodf=np.abs(de),
            deodf_missing=np.isnan(de).astype(np.int8),
            dsize=np.full(len(k), tr.size_dom - tr.size_sub),
            log1p_tsc=np.where(no_c, 0.0, np.log1p(tsc)), no_contact=no_c.astype(np.int8),
            w=np.where(y[sel], 1.0, 1.0 / frac),
        )
        sub_t = tr.chirps_sub if chirp_override is None else chirp_override.get(
            tr.recording, tr.chirps_sub)
        cols.update(_train_features(sub_t, "sub", t0, K, k, cfg, B, cfg.tau_grid, guard=True))
        cols.update(_train_features(tr.chirps_dom, "dom", t0, K, k, cfg, B, cfg.tau_grid,
                                    guard=True))
        if cfg.use_rises:  # rise stamps are frame centres: delay by rise_guard (no leakage)
            guard = float(_opt(cfg, "rise_guard"))
            for who in SENDERS:
                r = getattr(tr, f"rises_{who}")
                r = np.zeros(0) if r is None else np.asarray(r) + guard
                cols.update(_train_features(r, f"{who}_rise", t0, K, k, cfg, B, None))
        parts.append(pd.DataFrame(cols))
        book.append(dict(recording=tr.recording, n_rows=len(k), n_events=int(y[sel].sum()),
                         **rr["info"]))
    design = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(design):
        design["recording"] = pd.Categorical(
            design["recording"], categories=[b["recording"] for b in book if b["n_rows"]])
    design.attrs.update(
        kind=kind, bin_width=bw, lags=lags, basis=B, min_elapsed=min_el,
        control_fraction=frac, bookkeeping=pd.DataFrame(book), use_rises=bool(cfg.use_rises),
    )
    return design


# ---------------------------------------------------------------------------
# Model matrices
# ---------------------------------------------------------------------------
@dataclass
class _Block:
    """Standardized columns with their natural-scale mean and SD."""

    X: np.ndarray
    names: list[str]
    mean: np.ndarray
    sd: np.ndarray
    notes: list[str] = field(default_factory=list)


def _standardize(cols: dict[str, np.ndarray], notes: list[str]) -> _Block:
    names, arrs, mu, sd = [], [], [], []
    for name, v in cols.items():
        v = np.asarray(v, float)
        if np.isnan(v).all():
            notes.append(f"{name} all NaN: dropped")
            continue
        if np.isnan(v).any():
            notes.append(f"{name}: {np.isnan(v).sum()} NaN filled with mean")
            v = np.where(np.isnan(v), np.nanmean(v), v)
        m, s = float(v.mean()), float(v.std())
        if s < 1e-12:
            notes.append(f"{name} constant: dropped")
            continue
        names.append(name)
        arrs.append((v - m) / s)
        mu.append(m)
        sd.append(s)
    X = np.column_stack(arrs) if arrs else np.zeros((len(next(iter(cols.values()), [])), 0))
    return _Block(X, names, np.array(mu), np.array(sd), notes)


def _spline(x: np.ndarray, df: int, prefix: str) -> dict[str, np.ndarray]:
    m = np.asarray(patsy.dmatrix(f"cr(x, df={df}, constraints='center') - 1", {"x": x}))
    return {f"{prefix}{i + 1}": m[:, i] for i in range(m.shape[1])}


def _control_block(design: pd.DataFrame, cfg: Config, fe: bool) -> _Block:
    """Standardized control columns (dyad-level ones only when not fe)."""
    kind = design.attrs["kind"]
    cols: dict[str, np.ndarray] = {}
    el = np.log1p(design["elapsed"].to_numpy())
    cols.update(_spline(el, int(_opt(cfg, "elapsed_df")), "el_s"))
    if kind == "onset":
        cols["first_gap"] = design["first_gap"].to_numpy()
    cols.update(_spline(design["hours"].to_numpy(), int(_opt(cfg, "hours_df")), "hr_s"))
    for c in ("log1p_tsc", "no_contact", "deodf", "abs_deodf", "deodf_missing"):
        cols[c] = design[c].to_numpy()
    if not fe:
        cols["dsize"] = design["dsize"].to_numpy()
    if design.attrs.get("use_rises"):
        cols.update({c: design[c].to_numpy() for c in design.columns if "_rise_h" in c})
    return _standardize(cols, [])


def _chirp_columns(model: str, n_basis: int) -> list[str]:
    if model == "kernel":
        return [f"{s}_h{i + 1}" for s in SENDERS for i in range(n_basis)]
    return [str(indicator_column(model, s)) for s in SENDERS]


def _sub_chirp_idx(names: Sequence[str], model: str, n_basis: int,
                   cols: Sequence[str] | None = None) -> list[int]:
    """Positions of the subordinate chirp columns of ``model`` (or of the
    explicit chirp column list ``cols``) in ``names``.

    Selected by exact name: a prefix match on 'sub_' would also catch the
    subordinate rise columns ('sub_rise_h*') when cfg.use_rises is on.
    """
    want = [c for c in (cols or _chirp_columns(model, n_basis)) if c.startswith("sub_")]
    return [names.index(c) for c in want if c in names]


def _chirp_block(design: pd.DataFrame, model: str, cols: Sequence[str] | None = None) -> _Block:
    """Chirp columns of ``model``, or the explicit list ``cols`` (hz_acausal adds
    columns that no standard model has, e.g. future-window indicators)."""
    B = design.attrs["basis"]
    cols = list(cols) if cols else _chirp_columns(model, B.shape[1])
    return _standardize({c: design[c].to_numpy() for c in cols}, [])


def _groups(design: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    g = design["recording"].cat.remove_unused_categories()
    return g.cat.codes.to_numpy().astype(np.int64), list(g.cat.categories)


@dataclass
class _Matrix:
    X: np.ndarray
    names: list[str]  # all columns, dummies included
    mean: np.ndarray  # natural-scale mean per column (0 for intercept/dummies)
    sd: np.ndarray  # natural-scale SD per column (1 for intercept/dummies)
    n_lead: int  # number of intercept/dummy columns at the front
    notes: list[str]


def _assemble(design: pd.DataFrame, ctrl: _Block, chirp: _Block, fe: bool) -> _Matrix:
    n = len(design)
    codes, recs = _groups(design)
    lead = [f"dyad[{r}]" for r in recs] if fe else ["Intercept"]
    p = len(lead) + ctrl.X.shape[1] + chirp.X.shape[1]
    X = np.empty((n, p))
    if fe:
        X[:, : len(lead)] = 0.0
        X[np.arange(n), codes] = 1.0
    else:
        X[:, 0] = 1.0
    a = len(lead)
    X[:, a : a + ctrl.X.shape[1]] = ctrl.X
    X[:, a + ctrl.X.shape[1] :] = chirp.X
    return _Matrix(
        X=X, names=lead + ctrl.names + chirp.names,
        mean=np.concatenate([np.zeros(a), ctrl.mean, chirp.mean]),
        sd=np.concatenate([np.ones(a), ctrl.sd, chirp.sd]), n_lead=a,
        notes=ctrl.notes + chirp.notes,
    )


# ---------------------------------------------------------------------------
# GLM (IRLS) with cluster-robust covariance
# ---------------------------------------------------------------------------
def _inv_link(eta: np.ndarray, link: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and d(mean)/d(eta) for logit or cloglog."""
    if link == "logit":
        mu = expit(eta)
        dmu = mu * (1.0 - mu)
    elif link == "cloglog":
        e = np.exp(np.clip(eta, -40.0, 3.5))
        mu = -np.expm1(-e)
        dmu = e * (1.0 - mu)  # = e * exp(-e)
    else:
        raise ValueError(f"unknown link {link}")
    return np.clip(mu, 1e-15, 1 - 1e-15), np.maximum(dmu, 1e-300)


def _link(mu: np.ndarray, link: str) -> np.ndarray:
    return np.log(mu / (1 - mu)) if link == "logit" else np.log(-np.log1p(-mu))


def _xtwx(X: np.ndarray, W: np.ndarray, z: np.ndarray | None = None, chunk: int = 200_000
          ) -> tuple[np.ndarray, np.ndarray | None]:
    p = X.shape[1]
    H = np.zeros((p, p))
    g = np.zeros(p) if z is not None else None
    for s in range(0, X.shape[0], chunk):
        Xc = X[s : s + chunk]
        XW = Xc * W[s : s + chunk, None]
        H += XW.T @ Xc
        if z is not None:
            g += XW.T @ z[s : s + chunk]
    return H, g


def _observed_weights(y: np.ndarray, mu: np.ndarray, dmu: np.ndarray, eta: np.ndarray,
                      w: np.ndarray, link: str) -> np.ndarray:
    """Per-row weights of the observed information X' diag(.) X.

    With a = dmu / (mu (1 - mu)), -d2 loglik / d eta2 = w (dmu a - (y - mu) a').
    Logit: a = 1, so observed = expected. Cloglog (e = exp(eta)): a = e / mu,
    a' = (e / mu) (1 - e (1 - mu) / mu). statsmodels uses the observed Hessian
    as the bread of its cluster sandwich, so we do too.
    """
    exp_w = w * dmu**2 / (mu * (1 - mu))
    if link == "logit":
        return exp_w
    e = np.exp(np.clip(eta, -40.0, 3.5))
    a = e / mu
    return exp_w - w * (y - mu) * a * (1 - e * (1 - mu) / mu)


def _llf(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * (y * np.log(mu) + (1 - y) * np.log1p(-mu))))


@dataclass
class GLMResult:
    """IRLS fit: params, model-based and cluster-robust covariance."""

    params: np.ndarray
    cov: np.ndarray
    cov_cluster: np.ndarray | None
    converged: bool
    n_iter: int
    llf: float
    n_groups: int


def fit_glm(
    X: np.ndarray, y: np.ndarray, link: str = "cloglog", offset: np.ndarray | None = None,
    weights: np.ndarray | None = None, groups: np.ndarray | None = None,
    start: np.ndarray | None = None, maxiter: int = 100, tol: float = 1e-9, cov: bool = True,
) -> GLMResult:
    """Binomial GLM by IRLS (Fisher scoring) with step halving.

    ``weights`` are frequency/IPW weights. With ``groups`` the cluster-robust
    sandwich uses the statsmodels small-sample factor G/(G-1) * (n-1)/(n-p).
    """
    n, p = X.shape
    y = np.asarray(y, float)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    ridge = 1e-10 * np.eye(p)
    if start is None:
        mu0 = (y + 0.5) / 2.0 if link == "logit" else np.clip((y + y.mean()) / 2, 1e-4, 0.5)
        eta0 = _link(mu0, link)
        _, d0 = _inv_link(eta0, link)
        W0 = w * d0**2 / (mu0 * (1 - mu0))
        H, g = _xtwx(X, W0, eta0 - off)
        beta = np.linalg.solve(H + ridge, g)
    else:
        beta = np.asarray(start, float).copy()
    eta = X @ beta + off
    mu, dmu = _inv_link(eta, link)
    llf = _llf(y, mu, w)
    converged, it = False, 0
    for it in range(1, maxiter + 1):
        var = mu * (1 - mu)
        W = w * dmu**2 / var
        z = eta - off + (y - mu) / dmu
        H, g = _xtwx(X, W, z)
        step = np.linalg.solve(H + ridge, g) - beta
        for _ in range(40):
            b_new = beta + step
            eta_new = X @ b_new + off
            mu_new, dmu_new = _inv_link(eta_new, link)
            llf_new = _llf(y, mu_new, w)
            if llf_new >= llf - 1e-12 * abs(llf):
                break
            step *= 0.5
        done = np.max(np.abs(b_new - beta)) < tol * (1 + np.max(np.abs(beta))) or \
            abs(llf_new - llf) < 1e-13 * abs(llf)
        beta, eta, mu, dmu, llf = b_new, eta_new, mu_new, dmu_new, llf_new
        if done:
            converged = True
            break
    if not cov:
        return GLMResult(beta, np.full((p, p), np.nan), None, converged, it, llf, 0)
    var = mu * (1 - mu)
    H, _ = _xtwx(X, _observed_weights(y, mu, dmu, eta, w, link))
    bread = np.linalg.pinv(H)
    cov_cl, G = None, 0
    if groups is not None:
        groups = np.asarray(groups)
        G = int(groups.max()) + 1
        r = w * (y - mu) * dmu / var
        Z = sparse.csr_matrix((np.ones(n), (np.arange(n), groups)), shape=(n, G))
        S = np.zeros((G, p))
        for s in range(0, n, 200_000):
            S += Z[s : s + 200_000].T @ (X[s : s + 200_000] * r[s : s + 200_000, None])
        meat = S.T @ S
        c = G / max(G - 1, 1) * (n - 1) / max(n - p, 1)
        cov_cl = c * bread @ meat @ bread
    return GLMResult(beta, bread, cov_cl, converged, it, llf, G)


@dataclass
class _Chord:
    """Fixed-information (chord) Newton refits when a few columns are replaced.

    Parameters are ordered [dyad intercepts (G), dense base columns, S]. Dyad
    intercepts enter through group codes (indexing / bincount), and the fixed
    dense base block is kept read-only in float32, so forked workers share it
    and stream half the bytes. H0 is the float64 Fisher information at the
    observed fit; for new S only its S rows/columns are recomputed (O(n q p)).
    Iterating beta += H^-1 score(beta) converges to the new MLE because W
    barely moves when only a chirp train is permuted. The score is exact up to
    float32 rounding of the dense GEMV (~1e-6 of a coefficient, verified
    against full IRLS), so the fixed point is the MLE.
    """

    codes: np.ndarray
    G: int
    Xd: np.ndarray  # dense base columns, float32, C-contiguous
    y: np.ndarray
    link: str
    off: np.ndarray
    w: np.ndarray
    W0: np.ndarray
    Hbb: np.ndarray
    beta0: np.ndarray

    @classmethod
    def at(cls, codes: np.ndarray, Xd: np.ndarray, S: np.ndarray, y: np.ndarray, link: str,
           beta: np.ndarray, off: np.ndarray | None, w: np.ndarray | None) -> _Chord:
        n, G, nd = len(y), int(codes.max()) + 1, Xd.shape[1]
        off = np.zeros(n) if off is None else off
        w = np.ones(n) if w is None else w
        eta = beta[:G][codes] + Xd @ beta[G : G + nd] + S @ beta[G + nd :] + off
        mu, dmu = _inv_link(eta, link)
        W0 = w * dmu**2 / (mu * (1 - mu))
        Zw = sparse.csr_matrix((W0, (codes, np.arange(n))), shape=(G, n))
        Hbb = np.zeros((G + nd, G + nd))
        Hbb[:G, :G] = np.diag(np.bincount(codes, W0, minlength=G))
        Hbb[:G, G:] = Zw @ Xd
        Hbb[G:, :G] = Hbb[:G, G:].T
        Hbb[G:, G:] = _xtwx(Xd, W0)[0]
        Xd32 = np.ascontiguousarray(Xd, dtype=np.float32)
        return cls(codes, G, Xd32, y, link, off, w, W0, Hbb, beta.copy())

    def refit(self, S: np.ndarray, maxiter: int = 80, tol: float = 1e-6) -> tuple[np.ndarray, bool]:
        G, Xd, c = self.G, self.Xd, self.codes
        nb, q = len(self.Hbb), S.shape[1]
        SW = S * self.W0[:, None]
        Hbs = np.vstack([
            np.column_stack([np.bincount(c, SW[:, j], minlength=G) for j in range(q)]),
            (SW.astype(np.float32).T @ Xd).T.astype(np.float64),
        ])
        H = np.zeros((nb + q, nb + q))
        H[:nb, :nb] = self.Hbb
        H[:nb, nb:] = Hbs
        H[nb:, :nb] = Hbs.T
        H[nb:, nb:] = S.T @ SW
        Hinv = np.linalg.inv(H + 1e-10 * np.eye(nb + q))
        S32 = S.astype(np.float32)
        beta = self.beta0.copy()
        # convergence is judged on the scale of the observed fit: a relative test
        # on the current iterate would accept a diverging iterate (|beta| -> inf)
        scale = 1.0 + float(np.max(np.abs(self.beta0)))
        for _ in range(maxiter):
            dense = (Xd @ beta[G:nb].astype(np.float32)).astype(np.float64)
            eta = beta[:G][c] + dense + S @ beta[nb:] + self.off
            mu, dmu = _inv_link(eta, self.link)
            r = self.w * (self.y - mu) * dmu / (mu * (1 - mu))
            r32 = r.astype(np.float32)
            g = np.concatenate([np.bincount(c, r, minlength=G),
                                (Xd.T @ r32).astype(np.float64), (S32.T @ r32).astype(np.float64)])
            step = Hinv @ g
            beta += step
            if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 100 * scale:
                return beta, False  # diverged: caller falls back to full IRLS
            if np.max(np.abs(step)) < tol * scale:
                return beta, True
        return beta, False  # caller falls back to full IRLS


# ---------------------------------------------------------------------------
# Mixed model (statsmodels BinomialBayesMixedGLM, Laplace)
# ---------------------------------------------------------------------------
def _neg_logpost_hessian(model: Any, vec: np.ndarray) -> np.ndarray:
    """Analytic Hessian of -log posterior of a logit BinomialBayesMixedGLM with
    one variance component whose exog_vc is a one-hot group matrix."""
    fep, vcp, vc = model._unpack(vec)
    X, Z = model.exog, model.exog_vc
    p, G = len(fep), len(vc)
    mu = expit(X @ fep + Z @ vc)
    W = mu * (1 - mu)
    Hbb, _ = _xtwx(X, W)
    Hbb += np.eye(p) / model.fe_p**2
    Hbu = np.asarray((Z.T @ (X * W[:, None])).T)  # (p, G)
    s2 = np.exp(-2 * vcp[0])
    Huu = np.diag(np.asarray(Z.T @ W).ravel() + s2)
    Hus = (-2 * vc * s2)[:, None]
    Hss = np.array([[2 * np.sum(vc**2) * s2 + 1 / model.vcp_p**2]])
    H = np.zeros((p + 1 + G, p + 1 + G))
    H[:p, :p] = Hbb
    H[:p, p + 1 :] = Hbu
    H[p + 1 :, :p] = Hbu.T
    H[p, p] = Hss[0, 0]
    H[p + 1 :, p] = Hus[:, 0]
    H[p, p + 1 :] = Hus[:, 0]
    H[p + 1 :, p + 1 :] = Huu
    return H


@dataclass
class MixedResult:
    fe_mean: np.ndarray
    fe_cov: np.ndarray
    vcp_mean: float
    vcp_sd: float
    vc_mean: np.ndarray
    converged: bool
    n_iter: int
    method: str
    model: Any = None
    mode: np.ndarray | None = None
    cov_full: np.ndarray | None = None
    # all starts: (log posterior, mode vector, converged), best first
    modes: list[tuple[float, np.ndarray, bool]] = field(default_factory=list)


def fit_mixed(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str], group_names: list[str],
    fe_p: float = 5.0, vcp_p: float = 2.0, vb: bool = False, seed: int = 42,
    sd_starts: Sequence[float] = (0.5, 3.0),
) -> MixedResult:
    """Laplace (or VB started at the Laplace mode) fit of a logit random-intercept model.

    The joint posterior of (fixed effects, dyad intercepts, log dyad SD) can be
    multimodal: dyad-level covariates (size difference, nearly constant dEODf)
    trade off against the random intercepts. The mode is searched from each
    initial dyad SD in ``sd_starts`` and the highest log posterior is kept;
    ``modes`` holds all of them for reporting.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    n = len(y)
    G = int(groups.max()) + 1
    Z = sparse.csr_matrix((np.ones(n), (np.arange(n), groups)), shape=(n, G))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM(
            np.asarray(y, float), X, Z, np.zeros(G, dtype=int), vcp_p=vcp_p, fe_p=fe_p,
            fep_names=list(names), vcp_names=["recording"], vc_names=list(group_names),
        )
    p = X.shape[1]
    modes: list[tuple[float, np.ndarray, bool, int]] = []
    for sd0 in sd_starts:
        x0 = np.zeros(p + 1 + G)
        x0[0] = float(np.log(y.mean() / (1 - y.mean())))
        x0[p] = np.log(float(sd0))
        r = optimize.minimize(
            lambda v: -model.logposterior(v), x0, jac=lambda v: -model.logposterior_grad(v),
            hess=lambda v: _neg_logpost_hessian(model, v), method="trust-exact",
            options=dict(maxiter=200, gtol=1e-6),
        )
        # trust-exact can stop on precision loss at an optimum on ~1e6 rows
        ok = bool(r.success) or float(np.max(np.abs(model.logposterior_grad(r.x)))) < 1e-3
        modes.append((float(-r.fun), r.x, ok, int(r.nit)))
    modes.sort(key=lambda m: (not m[2], -m[0]))  # converged first, then highest posterior
    lp, xbest, ok, nit = modes[0]
    H = _neg_logpost_hessian(model, xbest)
    cov = np.linalg.inv(H)
    cov = 0.5 * (cov + cov.T)
    res = MixedResult(
        fe_mean=xbest[:p], fe_cov=cov[:p, :p], vcp_mean=float(xbest[p]),
        vcp_sd=float(np.sqrt(cov[p, p])), vc_mean=xbest[p + 1 :], converged=ok,
        n_iter=nit, method="laplace", model=model, mode=xbest, cov_full=cov,
        modes=[(m[0], m[1], m[2]) for m in modes],
    )
    if vb:
        np.random.seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = model.fit_vb(mean=r.x.copy(), sd=np.sqrt(np.diag(cov)))
        res = MixedResult(
            fe_mean=v.fe_mean, fe_cov=np.diag(v.fe_sd**2), vcp_mean=float(v.vcp_mean[0]),
            vcp_sd=float(v.vcp_sd[0]), vc_mean=v.vc_mean,
            converged=bool(v.optim_retvals.success), n_iter=int(v.optim_retvals.nit),
            method="vb", model=model,
        )
    return res


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------
def _term_type(name: str) -> str:
    if name == "Intercept":
        return "intercept"
    if "_rise_h" in name:
        return "rise"
    for s in SENDERS:
        # _fut_ / _fgd_ / _fh: future-window chirp terms of the acausal control (hz_acausal)
        if name.startswith((f"{s}_h", f"{s}_any_", f"{s}_grd_", f"{s}_fut_", f"{s}_fgd_",
                            f"{s}_fh")):
            return f"chirp_{s}"
    if name.startswith(("el_s", "hr_s")):
        return "spline"
    return "control"


@dataclass
class _FitOut:
    """Natural-scale estimates of one fitted model (dummies excluded)."""

    model: str
    kind: str
    backend: str
    link: str
    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    df_t: int | None  # t/F reference df (cluster-robust), None = normal/chi2
    n_bins: int
    n_events: int
    n_rec: int
    fit_time: float
    converged: bool
    notes: list[str]
    extra: list[dict[str, Any]] = field(default_factory=list)
    n_rec_nonzero: dict[str, int] = field(default_factory=dict)  # per chirp column
    n_ev_nonzero: dict[str, int] = field(default_factory=dict)  # events with column != 0
    weak: dict[str, str] = field(default_factory=dict)  # term -> identifiability caveat
    late_from: float = 0.5  # s, first lag of the 'kernel_<s>_mean_late' summary


# GLMM terms that vary (almost) only between dyads: they trade off against the
# dyad random intercepts, and the joint posterior has several modes along that
# ridge (see fit_mixed). Their GLMM estimates are not interpretable.
WEAK_DYAD_TERMS = ("Intercept", "dsize", "deodf", "abs_deodf", "deodf_missing")
WEAK_NOTE = ("weakly identified: dyad-level covariate that trades off with the dyad random "
             "intercepts (GLMM posterior has several modes along this ridge); do not interpret")


def _natural(params: np.ndarray, cov: np.ndarray, M: _Matrix, shift: float = 0.0
             ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Back-transform standardized slopes (beta / sd); drop dummy columns.

    The intercept stays at the covariate means; ``shift`` is subtracted from it
    (case-control correction log(1/f)).
    """
    keep = [i for i, nm in enumerate(M.names) if not nm.startswith("dyad[")]
    D = np.diag(1.0 / M.sd[keep])
    b = params[keep] / M.sd[keep]
    names = [M.names[i] for i in keep]
    if "Intercept" in names:
        b[names.index("Intercept")] -= shift
    return names, b, D @ cov[np.ix_(keep, keep)] @ D


def _crit(df_t: int | None) -> float:
    return float(stats.t.ppf(0.975, df_t)) if df_t else float(stats.norm.ppf(0.975))


def _pval(z: np.ndarray, df_t: int | None) -> np.ndarray:
    return 2 * (stats.t.sf(np.abs(z), df_t) if df_t else stats.norm.sf(np.abs(z)))


def _effect_rows(f: _FitOut, lags: np.ndarray, B: np.ndarray) -> list[dict[str, Any]]:
    base = dict(model=f.model, kind=f.kind, backend=f.backend, link=f.link, n_bins=f.n_bins,
                n_events=f.n_events, n_recordings=f.n_rec, fit_time_s=round(f.fit_time, 2),
                converged=f.converged)
    note = "; ".join(f.notes)
    crit = _crit(f.df_t)
    se = np.sqrt(np.clip(np.diag(f.cov), 0, None))
    z = f.beta / np.where(se > 0, se, np.nan)
    pv = _pval(z, f.df_t)
    rows = []

    def support(names: list[str]) -> str:
        k = max((f.n_rec_nonzero.get(nm, 0) for nm in names), default=0)
        e = max((f.n_ev_nonzero.get(nm, 0) for nm in names), default=0)
        return f"term nonzero in {k}/{f.n_rec} recordings and at {e}/{f.n_events} events"

    for i, nm in enumerate(f.names):
        lo, hi = f.beta[i] - crit * se[i], f.beta[i] + crit * se[i]
        tt = _term_type(nm)
        nt = "intercept at covariate means" if tt == "intercept" else note
        if tt.startswith("chirp_"):
            nt = f"{support([nm])}; {nt}"
        if nm in f.weak:
            nt = f"{f.weak[nm]}; {nt}"
        rows.append(dict(base, term=nm, term_type=tt, lag=np.nan, estimate=f.beta[i], se=se[i],
                         ci_low=lo, ci_high=hi, p=pv[i], note=nt))
    idx = {nm: i for i, nm in enumerate(f.names)}
    for s in SENDERS:
        hn = [f"{s}_h{i + 1}" for i in range(B.shape[1])]
        cols = [idx.get(c) for c in hn]
        if any(c is None for c in cols):
            continue  # not a kernel model, or a basis column was dropped
        sup = support(hn)
        b, C = f.beta[cols], f.cov[np.ix_(cols, cols)]
        k = B @ b
        ks = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", B, C, B), 0, None))
        for lag, kv, sv in zip(lags, k, ks):
            rows.append(dict(base, term=f"kernel_{s}", term_type="kernel", lag=lag, estimate=kv,
                             se=sv, ci_low=kv - crit * sv, ci_high=kv + crit * sv,
                             p=_pval(np.array([kv / sv]), f.df_t)[0] if sv > 0 else np.nan,
                             note="log-odds/log-hazard change per chirp at lag (pointwise CI)"))
        c = B.mean(0)
        m, ms = float(c @ b), float(np.sqrt(c @ C @ c))
        rows.append(dict(base, term=f"kernel_{s}_mean", term_type="kernel_summary", lag=np.nan,
                         estimate=m, se=ms, ci_low=m - crit * ms, ci_high=m + crit * ms,
                         p=_pval(np.array([m / ms]), f.df_t)[0],
                         note=f"mean kernel over lags {lags[0]:g}-{lags[-1]:g} s "
                              f"(integrated effect / window); {sup}"))
        late = lags >= f.late_from - 1e-9
        if late.any() and not late.all():
            c = B[late].mean(0)
            m, ms = float(c @ b), float(np.sqrt(c @ C @ c))
            rows.append(dict(base, term=f"kernel_{s}_mean_late", term_type="kernel_summary",
                             lag=np.nan, estimate=m, se=ms, ci_low=m - crit * ms,
                             ci_high=m + crit * ms, p=_pval(np.array([m / ms]), f.df_t)[0],
                             note=f"mean kernel over lags {lags[late][0]:g}-{lags[-1]:g} s only "
                                  "(lags below the ~0.5 s chase-boundary precision left out); "
                                  f"{sup}"))
        try:
            wald = float(b @ np.linalg.solve(C, b))
        except np.linalg.LinAlgError:
            wald = np.nan
        q = len(b)
        if f.df_t:
            pj, stat, nt = float(stats.f.sf(wald / q, q, f.df_t)), wald / q, f"F({q},{f.df_t})"
        else:
            pj, stat, nt = float(stats.chi2.sf(wald, q)), wald, f"chi2({q})"
        rows.append(dict(base, term=f"kernel_{s}_joint", term_type="test", lag=np.nan,
                         estimate=stat, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=pj,
                         note=f"joint Wald test of all {s} basis terms, {nt}; {sup}"
                              + ("; cluster-robust joint tests are unreliable when few "
                                 "recordings carry the term" if f.df_t else "")))
    for r in f.extra:
        rows.append(dict(base, lag=np.nan, **r))
    for r in rows:
        exp_ok = r["term_type"] not in ("test", "re_sd")
        r["ratio"] = np.exp(r["estimate"]) if exp_ok else np.nan
        r["ratio_low"] = np.exp(r["ci_low"]) if exp_ok else np.nan
        r["ratio_high"] = np.exp(r["ci_high"]) if exp_ok else np.nan
    return rows


def _fit_one(
    design: pd.DataFrame, cfg: Config, model: str, backend: str, ctrl: _Block,
    cols: Sequence[str] | None = None,
) -> tuple[_FitOut, Any, _Matrix]:
    """One (model, backend) fit. ``cols`` replaces the model's chirp columns
    (``model`` is then only a label)."""
    kind = design.attrs["kind"]
    frac = float(design.attrs.get("control_fraction", 1.0))
    fe = backend == "glm_fe"
    M = _assemble(design, ctrl, _chirp_block(design, model, cols), fe)
    y = design["y"].to_numpy().astype(float)
    codes, recs = _groups(design)
    notes = list(M.notes)
    t0 = time.perf_counter()
    if fe:
        link = str(_opt(cfg, "glm_link"))
        w = design["w"].to_numpy() if frac < 1 and link == "cloglog" else None
        off = np.full(len(y), np.log(1 / frac)) if frac < 1 and link == "logit" else None
        res = fit_glm(M.X, y, link=link, offset=off, weights=w, groups=codes)
        cov = res.cov_cluster
        names, b, C = _natural(res.params, cov, M)
        conv, df_t = res.converged, max(res.n_groups - 1, 1)
        notes.append(f"cluster-robust SE by recording (G={res.n_groups}), t({df_t}) CIs")
        if frac < 1:
            notes.append(f"onset controls subsampled f={frac:g}: "
                         + ("IPW weights 1/f" if link == "cloglog" else "offset log(1/f)"))
        extra: list[dict[str, Any]] = []
        fit_obj: Any = res
    else:
        link = "logit"
        res = fit_mixed(M.X, y, codes, M.names, recs, fe_p=float(_opt(cfg, "fe_p")),
                        vcp_p=float(_opt(cfg, "vcp_p")), vb=backend == "mixed_vb", seed=cfg.seed,
                        sd_starts=tuple(_opt(cfg, "mixed_sd_starts")))
        shift = np.log(1 / frac) if frac < 1 else 0.0
        names, b, C = _natural(res.fe_mean, res.fe_cov, M, shift=shift)
        conv, df_t = res.converged, None
        notes.append("Laplace posterior mean +- 1.96 SD" if res.method == "laplace"
                     else "mean-field VB: marginal SDs only (tend to be too small)")
        if frac < 1:
            notes.append(f"onset controls subsampled f={frac:g}; intercept corrected by "
                         f"-log(1/f)={-shift:.3f}")
        if res.method == "laplace" and len(res.modes) > 1:
            notes.append(_mode_note(res, M, model, design.attrs["basis"].shape[1], cols))
        s, ss = res.vcp_mean, res.vcp_sd
        extra = [dict(term="sd_recording", term_type="re_sd", estimate=float(np.exp(s)),
                      se=np.nan, ci_low=float(np.exp(s - 1.96 * ss)),
                      ci_high=float(np.exp(s + 1.96 * ss)), p=np.nan,
                      note=f"SD of dyad random intercept (log-odds); {WEAK_NOTE}")]
        fit_obj = res
    dt = time.perf_counter() - t0
    ccols = list(cols) if cols else _chirp_columns(model, design.attrs["basis"].shape[1])
    nz = {c: int(design.loc[design[c] != 0, "recording"].nunique()) for c in ccols}
    nev = {c: int(((design[c] != 0) & (design["y"] == 1)).sum()) for c in ccols}
    weak = {} if fe else {t: WEAK_NOTE for t in WEAK_DYAD_TERMS}
    out = _FitOut(model=model, kind=kind, backend=backend, link=link, names=names, beta=b, cov=C,
                  df_t=df_t, n_bins=len(y), n_events=int(y.sum()), n_rec=len(recs), fit_time=dt,
                  converged=conv, notes=notes, extra=extra, n_rec_nonzero=nz,
                  n_ev_nonzero=nev, weak=weak, late_from=float(_opt(cfg, "guard_lag")))
    return out, fit_obj, M


def _mode_note(res: MixedResult, M: _Matrix, model: str, n_basis: int,
               cols: Sequence[str] | None = None) -> str:
    """How much the subordinate chirp terms move between the GLMM modes found."""
    p = len(res.fe_mean)
    sub = _sub_chirp_idx(M.names, model, n_basis, cols)
    best = res.modes[0]
    diffs = [float(np.max(np.abs((m[1][sub] - best[1][sub]) / M.sd[sub]))) if sub else 0.0
             for m in res.modes[1:]]
    alt = ", ".join(f"{m[0]:.2f} (dyad SD {np.exp(m[1][p]):.2f}"
                    + ("" if m[2] else ", not converged") + ")" for m in res.modes[1:])
    return (f"GLMM multi-start: kept log posterior {best[0]:.2f} (dyad SD "
            f"{np.exp(best[1][p]):.2f}); other starts {alt}; sub chirp coefficients differ "
            f"by <= {max(diffs, default=0.0):.3f} (log scale)")


def fit_models(
    design: pd.DataFrame, kind: str, cfg: Config, models: Sequence[str] | None = None,
    backends: Sequence[str] | None = None, verbose: bool = False,
) -> pd.DataFrame:
    """Fit all (model, backend) combinations for one kind; tidy effects table.

    models: subset of ``model_names(cfg)``; backends: subset of
    ('mixed_laplace', 'mixed_vb', 'glm_fe'). Kernel rows carry ``lag``.
    """
    if design.attrs.get("kind") != kind:
        raise ValueError("design was built for a different kind")
    models = list(models or model_names(cfg))
    backends = list(backends or _opt(cfg, "hazard_backends"))
    lags, B = design.attrs["lags"], design.attrs["basis"]
    ctrl: dict[bool, _Block] = {}
    rows: list[dict[str, Any]] = []
    for model in models:
        for backend in backends:
            fe = backend == "glm_fe"
            if fe not in ctrl:
                ctrl[fe] = _control_block(design, cfg, fe)
            f, _, _ = _fit_one(design, cfg, model, backend, ctrl[fe])
            if verbose:
                print(f"  {kind:6s} {model:10s} {backend:13s} {f.fit_time:6.1f} s "
                      f"converged={f.converged}", flush=True)
            rows.extend(_effect_rows(f, lags, B))
    eff = pd.DataFrame(rows)
    return eff.reindex(columns=EFFECT_COLUMNS)


# ---------------------------------------------------------------------------
# Jitter (permutation) test
# ---------------------------------------------------------------------------
def null_sub_trains(
    trials: Sequence[Trial], n: int, cfg: Config, null: str = "shift"
) -> dict[str, np.ndarray]:
    """recording -> (n + 1, m) subordinate chirp trains; row 0 is observed.

    Uses hz_desc.null_replicates, so the hazard jitter test and the
    descriptives share the same null trains (same seeds): ``shift`` circularly
    shifts in-chase and out-of-chase chirps on their state clocks, ``matched``
    moves each chirp to another eligible interval keeping its elapsed time
    (hz_desc.swap_matched). Chirps outside the valid
    (known-state) window stay where they are.
    """
    states = [trial_states(tr) for tr in trials]
    reps = null_replicates(list(trials), states, DescParams(n_jitter=n, seed=cfg.seed), null)
    out: dict[str, np.ndarray] = {}
    for tr, st in zip(trials, states):
        x = tr.chirps_sub
        fixed = x[~(_in_iv(x, st.iv["in"]) | _in_iv(x, st.iv["out"]))]
        blocks = [reps[(tr.recording, "sub", s)] for s in ("in", "out")]
        blocks.append(np.repeat(fixed[None, :], n + 1, axis=0))
        out[tr.recording] = np.sort(np.concatenate(blocks, axis=1), axis=1)
    return out


_JIT: dict[str, Any] = {}  # read-only state inherited by forked jitter workers


def _jitter_features(it: int) -> dict[str, np.ndarray]:
    J = _JIT
    feats: dict[str, np.ndarray] = {}
    for rec, (tr, sl) in J["slices"].items():
        f = _train_features(J["trains"][rec][it], "sub", float(tr.obs_start),
                            _n_bins(tr, J["cfg"].bin_width), J["k"][sl], J["cfg"], J["B"],
                            [J["tau"]])
        for c, v in f.items():
            feats.setdefault(c, np.empty(J["n"]))[sl] = v
    return feats


def _worker_init() -> None:
    """One BLAS thread per forked worker (the parent's OpenBLAS pool is 32 wide)."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:  # pragma: no cover
        pass


def _jitter_worker(its: Sequence[int]) -> list[dict[str, Any]]:
    """Refit both jitter models for iterations ``its`` (runs in a forked worker)."""
    J, rows = _JIT, []
    for it in its:
        feats = _jitter_features(it)
        for model, st in J["setups"].items():
            S = np.column_stack([(feats[c] - m) / sd for c, m, sd
                                 in zip(st["cols"], st["mean"], st["sd"])])
            ch = st["chord"]
            beta, ok = ch.refit(S)
            if not ok or not np.all(np.isfinite(beta)):
                # fall back to full IRLS on the assembled matrix, started at the
                # observed-data MLE (the chord iterate may have diverged)
                D = np.zeros((len(S), ch.G))
                D[np.arange(len(S)), ch.codes] = 1.0
                X = np.hstack([D, ch.Xd.astype(np.float64), S])
                res = fit_glm(X, J["y"], link=J["link"], offset=J["off"], weights=J["w"],
                              start=ch.beta0, cov=False)
                beta, ok = res.params, res.converged
            val = st["stat"](beta[len(beta) - S.shape[1]:] / st["sd"]) if ok else np.nan
            rows.append(dict(stat=st["name"], iter=it, value=val))
    return rows


def jitter_test(
    trials: Sequence[Trial], kind: str, cfg: Config, design: pd.DataFrame | None = None,
    n_jitter: int | None = None, null: str = "shift", n_workers: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Permutation null of the subordinate-chirp effect.

    Refits glm_fe (cfg glm_link, dyad intercepts, all controls) for the
    tau = jitter_tau indicator model and the kernel model under ``n_jitter``
    null subordinate chirp trains (``null`` = 'shift' or 'matched', see
    null_sub_trains), recomputing only the subordinate columns (chord-Newton
    refits, parallel over ``n_workers`` forked processes). Statistics: the
    tau-indicator coefficient and the kernel mean (log-hazard per chirp).
    Returns long rows [kind, null, stat, iter, value] (iter 0 = observed);
    ``attrs['summary']`` holds ``summarize_jitter`` of it (clear ``attrs``
    before pd.concat). cfg.quick caps n_jitter at 20.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    n = int(cfg.n_jitter if n_jitter is None else n_jitter)
    if n_jitter is None and cfg.quick:
        n = min(n, 20)
    n_workers = int(_opt(cfg, "n_workers") if n_workers is None else n_workers)
    design = build_design(trials, kind, cfg) if design is None else design
    B = design.attrs["basis"]
    tau = min(cfg.tau_grid, key=lambda t: abs(t - float(_opt(cfg, "jitter_tau"))))
    tau_model = f"tau_{_ms(tau)}"
    link = str(_opt(cfg, "glm_link"))
    frac = float(design.attrs.get("control_fraction", 1.0))
    y = design["y"].to_numpy().astype(float)
    w = design["w"].to_numpy() if frac < 1 and link == "cloglog" else None
    off = np.full(len(y), np.log(1 / frac)) if frac < 1 and link == "logit" else None
    ctrl = _control_block(design, cfg, fe=True)
    codes, _ = _groups(design)
    in_design = set(design["recording"].cat.categories)
    used = [tr for tr in trials if tr.recording in in_design]
    recs = design["recording"].to_numpy()
    slices = {}
    for tr in used:
        ii = np.flatnonzero(recs == tr.recording)
        slices[tr.recording] = (tr, slice(ii[0], ii[-1] + 1))

    kmean = B.mean(0)
    setups: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for model in (tau_model, "kernel"):
        M = _assemble(design, ctrl, _chirp_block(design, model), fe=True)
        sub = _sub_chirp_idx(M.names, model, B.shape[1])
        dense = [i for i in range(M.n_lead, len(M.names)) if i not in sub]
        order = list(range(M.n_lead)) + dense + sub
        res = fit_glm(M.X, y, link=link, offset=off, weights=w, cov=False)
        fn = (lambda b: float(b[0])) if model != "kernel" else (lambda b: float(kmean @ b))
        name = f"{tau_model}_sub" if model != "kernel" else "kernel_sub_mean"
        chord = _Chord.at(codes, M.X[:, dense], M.X[:, sub], y, link, res.params[order], off, w)
        setups[model] = dict(chord=chord, cols=[M.names[i] for i in sub], mean=M.mean[sub],
                             sd=M.sd[sub], stat=fn, name=name)
        rows.append(dict(stat=name, iter=0, value=fn(res.params[sub] / M.sd[sub])))
        del M
    _JIT.clear()
    _JIT.update(setups=setups, trains=null_sub_trains(used, n, cfg, null), slices=slices,
                k=design["k"].to_numpy(), cfg=cfg, B=B, tau=tau, n=len(y), y=y, link=link,
                off=off, w=w)
    t_start = time.perf_counter()
    its = list(range(1, n + 1))
    if n_workers > 1 and n > 1:
        chunks = [its[i::n_workers * 4] for i in range(min(n, n_workers * 4))]
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork"),
                                 initializer=_worker_init) as ex:
            for done, part in enumerate(ex.map(_jitter_worker, chunks), 1):
                rows.extend(part)
                if verbose and (done % max(1, len(chunks) // 4) == 0 or done == len(chunks)):
                    print(f"  jitter {kind}/{null}: {done}/{len(chunks)} chunks "
                          f"({time.perf_counter() - t_start:.0f} s)", flush=True)
    else:
        for it in its:
            rows.extend(_jitter_worker([it]))
            if verbose and (it % 25 == 0 or it == n):
                print(f"  jitter {kind}/{null}: {it}/{n} ({time.perf_counter() - t_start:.0f} s)",
                      flush=True)
    _JIT.clear()
    out = pd.DataFrame(rows).assign(kind=kind, null=null)
    out = out[["kind", "null", "stat", "iter", "value"]].sort_values(["stat", "iter"],
                                                                      ignore_index=True)
    out.attrs["summary"] = summarize_jitter(out)
    out.attrs["link"] = link
    out.attrs["elapsed_s"] = time.perf_counter() - t_start
    return out


def summarize_jitter(null: pd.DataFrame) -> pd.DataFrame:
    """Observed vs jitter null per (kind, stat), with permutation p-values.

    p_two_sided = min(1, 2 * min(p_less, p_greater)) with the +1 correction,
    i.e. relative to the null distribution, not to zero.
    """
    rows = []
    keys = [c for c in ("kind", "null", "stat") if c in null.columns]
    for key, g in null.groupby(keys, sort=False):
        lab = dict(zip(keys, key))
        obs = float(g.loc[g["iter"] == 0, "value"].iloc[0])
        v = g.loc[g["iter"] > 0, "value"].to_numpy()
        v = v[np.isfinite(v)]
        n = len(v)
        pg = (1 + np.sum(v >= obs)) / (n + 1)
        pl = (1 + np.sum(v <= obs)) / (n + 1)
        rows.append(dict(
            **lab, observed=obs, null_mean=float(v.mean()) if n else np.nan,
            null_sd=float(v.std(ddof=1)) if n > 1 else np.nan,
            null_q025=float(np.quantile(v, 0.025)) if n else np.nan,
            null_q975=float(np.quantile(v, 0.975)) if n else np.nan,
            obs_minus_null=obs - float(v.mean()) if n else np.nan,
            p_greater=pg, p_less=pl, p_two_sided=min(1.0, 2 * min(pg, pl)), n_jitter=n,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leave-one-dyad-out
# ---------------------------------------------------------------------------
def leave_one_dyad_out(design: pd.DataFrame, cfg: Config, model: str | None = None) -> pd.DataFrame:
    """glm_fe refits of one model with each recording left out, plus per-dyad support.

    ``model`` defaults to the jitter tau model. Rows: left_out, estimate (sub
    indicator coefficient, or kernel mean for 'kernel'), cluster-robust CI with
    t(G - 2), and for tau models the left-out dyad's own bins/events with the
    indicator on and its crude rate ratio (indicator on vs off). Shows whether
    a pooled effect rests on one or two dyads.
    """
    kind = design.attrs["kind"]
    if model is None:
        tau = min(cfg.tau_grid, key=lambda t: abs(t - float(_opt(cfg, "jitter_tau"))))
        model = f"tau_{_ms(tau)}"
    link = str(_opt(cfg, "glm_link"))
    frac = float(design.attrs.get("control_fraction", 1.0))
    M = _assemble(design, _control_block(design, cfg, fe=True), _chirp_block(design, model), fe=True)
    y = design["y"].to_numpy().astype(float)
    w_all = design["w"].to_numpy()
    codes, recs = _groups(design)
    sub = _sub_chirp_idx(M.names, model, design.attrs["basis"].shape[1])
    kmean = design.attrs["basis"].mean(0)
    ind = indicator_column(model, "sub")
    rows = []
    for g, rec in enumerate(recs):
        keep = codes != g
        cols = [c for c in range(M.X.shape[1]) if c != g]
        cg = codes[keep] - (codes[keep] > g)
        w = w_all[keep] if frac < 1 and link == "cloglog" else None
        off = np.full(int(keep.sum()), np.log(1 / frac)) if frac < 1 and link == "logit" else None
        res = fit_glm(M.X[keep][:, cols], y[keep], link=link, offset=off, weights=w, groups=cg)
        jj = [cols.index(i) for i in sub]
        b = res.params[jj] / M.sd[sub]
        C = res.cov_cluster[np.ix_(jj, jj)] / np.outer(M.sd[sub], M.sd[sub])
        c = np.array([1.0]) if ind else kmean
        est, se = float(c @ b), float(np.sqrt(c @ C @ c))
        crit = float(stats.t.ppf(0.975, max(res.n_groups - 1, 1)))
        r = dict(kind=kind, model=model, stat=ind or "kernel_sub_mean", left_out=rec,
                 estimate=est, se=se, ratio=np.exp(est), ratio_low=np.exp(est - crit * se),
                 ratio_high=np.exp(est + crit * se), converged=res.converged)
        if ind:
            m = codes == g
            on = design[ind].to_numpy()[m] != 0
            r.update(dyad_bins_on=int(on.sum()), dyad_events_on=int(y[m][on].sum()),
                     dyad_events_off=int(y[m][~on].sum()),
                     dyad_crude_rr=(y[m][on].mean() / y[m][~on].mean())
                     if on.any() and y[m][~on].any() else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Between-dyad heterogeneity: per-dyad observed / expected, random-effects meta
# ---------------------------------------------------------------------------
def dyad_oe(design: pd.DataFrame, cfg: Config, model: str = "tau_1000ms",
            term: str | None = None, cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-dyad observed vs expected events in bins with the sub indicator on.

    Fits glm_fe (cloglog, dyad intercepts, all controls, common sub slope),
    then sets the sub indicator to 0 in the linear predictor: E_g is the
    number of events dyad g would have in its indicator-on bins if a sub chirp
    had no effect, given its fitted baseline (IPW weights with subsampled
    controls). O_g / E_g is that dyad's own hazard ratio (to first order in
    the per-bin hazard). Exact Poisson (Garwood) 95% CIs; 'direction' is
    'higher' / 'lower' when the CI excludes 1. Dyads with E_g below
    meta_min_expected are marked uninformative.

    ``cols`` (explicit chirp columns of the fitted model) and ``term`` (the
    0/1 indicator whose effect is set to 0, default the model's sub
    indicator) let hz_acausal take O/E per term of a model with several
    indicators; the other terms stay at their fitted values.
    """
    ind = str(term or indicator_column(model, "sub"))
    link = str(_opt(cfg, "glm_link"))
    frac = float(design.attrs.get("control_fraction", 1.0))
    M = _assemble(design, _control_block(design, cfg, fe=True),
                  _chirp_block(design, model, cols), fe=True)
    y = design["y"].to_numpy().astype(float)
    w = design["w"].to_numpy() if frac < 1 and link == "cloglog" else None
    off = np.full(len(y), np.log(1 / frac)) if frac < 1 and link == "logit" else None
    codes, recs = _groups(design)
    res = fit_glm(M.X, y, link=link, offset=off, weights=w, groups=codes, cov=False)
    j = M.names.index(ind)
    x = design[ind].to_numpy().astype(float)
    b_nat = float(res.params[j] / M.sd[j])
    on = x != 0
    eta0 = M.X[on] @ res.params - b_nat * x[on] + (0.0 if off is None else off[on])
    mu0, _ = _inv_link(eta0, link)
    ww = np.ones(int(on.sum())) if w is None else w[on]
    G = len(recs)
    E = np.bincount(codes[on], ww * mu0, minlength=G)
    O = np.bincount(codes[on], y[on], minlength=G)
    n_on = np.bincount(codes[on], minlength=G)
    Es = np.where(E > 0, E, np.nan)
    lo = np.where(O > 0, stats.chi2.ppf(0.025, 2 * np.maximum(O, 1)) / 2, 0.0) / Es
    hi = stats.chi2.ppf(0.975, 2 * (O + 1)) / 2 / Es
    min_e = float(_opt(cfg, "meta_min_expected"))
    out = pd.DataFrame(dict(
        kind=design.attrs["kind"], model=model, term=ind, recording=recs, n_bins_on=n_on,
        observed=O.astype(int), expected=E, oe=O / Es, oe_low=lo, oe_high=hi,
        informative=E >= min_e,
        direction=np.where(lo > 1, "higher", np.where(hi < 1, "lower", "unclear")),
        common_hr=float(np.exp(b_nat)),
    ))
    out.loc[~(E > 0), "direction"] = "no bins"
    return out


def _pn_loglik(mu: float, tau: float, O: np.ndarray, logE: np.ndarray,
               nodes: np.ndarray, logw: np.ndarray) -> float:
    """Poisson-normal marginal log likelihood by Gauss-Hermite quadrature."""
    from scipy.special import gammaln, logsumexp

    lam_log = np.clip(logE[:, None] + mu + tau * nodes[None, :], -700.0, 700.0)
    lp = np.where(O[:, None] > 0, O[:, None] * lam_log, 0.0) - np.exp(lam_log) \
        - gammaln(O + 1)[:, None]
    return float(np.sum(logsumexp(lp + logw[None, :], axis=1)))


def re_meta(O: np.ndarray, E: np.ndarray, n_nodes: int = 40, tau_max: float = 5.0
            ) -> dict[str, float]:
    """Random-effects (Poisson-normal) meta-analysis of per-dyad O / E.

    O_g ~ Poisson(E_g exp(mu + b_g)), b_g ~ N(0, tau^2): mu is the mean log
    hazard ratio across dyads, tau the between-dyad SD. Maximum likelihood;
    the CI of mu is a profile-likelihood interval with a t(k-1)^2 cut-off
    (k = dyads with E_g > 0; a small-sample widening in the spirit of
    Hartung-Knapp), p from the same LR statistic against F(1, k-1).
    Heterogeneity: LR test of tau = 0 (boundary, 50:50 chi2 mixture) and the
    Poisson homogeneity G test (dyad-specific vs common ratio, k-1 df). Zero
    counts are handled exactly (no continuity correction).
    """
    m = np.asarray(E, float) > 0
    O, E = np.asarray(O, float)[m], np.asarray(E, float)[m]
    k = len(O)
    out = dict(mu=np.nan, mu_low=np.nan, mu_high=np.nan, p=np.nan, tau=np.nan, p_tau=np.nan,
               G=np.nan, G_df=float(k - 1), p_G=np.nan, k=float(k), O=float(O.sum()),
               E=float(E.sum()), mu_common=np.nan)
    if k < 2 or O.sum() == 0:
        return out
    nodes, wq = np.polynomial.hermite_e.hermegauss(n_nodes)
    logw = np.log(wq / wq.sum())
    logE = np.log(E)

    def ll(mu: float, tau: float) -> float:
        return _pn_loglik(mu, tau, O, logE, nodes, logw)

    mu0 = float(np.log(O.sum() / E.sum()))
    fits = [optimize.minimize(lambda v: -ll(float(v[0]), float(v[1])), [mu0, t0],
                              method="L-BFGS-B", bounds=[(mu0 - 10, mu0 + 10), (0.0, tau_max)])
            for t0 in (0.0, 0.5, 2.0)]
    best = min(fits, key=lambda r: r.fun)
    mu_hat, tau_hat, ll_max = float(best.x[0]), float(best.x[1]), -float(best.fun)

    grid = np.r_[0.0, np.geomspace(0.01, tau_max, 30)]

    def prof_t(mu: float) -> tuple[float, float]:
        """(max over tau of ll(mu, tau), argmax): grid search, then bounded
        refinement around the best grid point (the profile in tau can have two
        local maxima)."""
        v = np.array([ll(mu, t) for t in grid])
        i = int(np.nanargmax(v))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        best, t_best = float(v[i]), float(grid[i])
        if hi > lo:
            r = optimize.minimize_scalar(lambda t: -ll(mu, t), bounds=(lo, hi),
                                         method="bounded")
            if -float(r.fun) > best:
                best, t_best = -float(r.fun), float(r.x)
        return best, t_best

    def prof(mu: float) -> float:
        return prof_t(mu)[0]

    # the joint optimizer can stop at a local optimum: maximize the profile too
    r = optimize.minimize_scalar(lambda m: -prof(m), bounds=(mu_hat - 5, mu_hat + 5),
                                 method="bounded", options=dict(xatol=1e-4))
    if -float(r.fun) > ll_max:
        mu_hat = float(r.x)
        ll_max, tau_hat = prof_t(mu_hat)
    ll_max = max(ll_max, prof(mu_hat))
    q = float(stats.t.ppf(0.975, max(k - 1, 1)) ** 2)

    def root(sign: int) -> float:
        """Profile-likelihood CI limit; +-inf if the profile stays within the cut-off
        for 15 log units (an unbounded side, e.g. with very few events)."""
        def f(mu: float) -> float:
            return 2 * (ll_max - prof(mu)) - q
        step, a = 0.25, mu_hat
        fa = f(a)
        if not fa < 0:
            return float(a)
        while abs(a - mu_hat) < 15.0:
            b = a + sign * step
            fb = f(b)
            if not np.isfinite(fb):
                break
            if fb > 0:
                try:
                    return float(optimize.brentq(f, min(a, b), max(a, b), xtol=1e-4))
                except ValueError:  # pragma: no cover (non-monotone profile)
                    return float(b)
            a, fa, step = b, fb, step * 1.3
        return float(sign * np.inf)

    lr0 = max(0.0, 2 * (ll_max - prof(0.0)))
    lr_tau = max(0.0, 2 * (ll_max - ll(mu0, 0.0)))
    Eh = E * O.sum() / E.sum()
    G = float(2 * np.sum(np.where(O > 0, O * np.log(np.where(O > 0, O, 1) / Eh), 0.0) - (O - Eh)))
    lo_, hi_ = root(-1), root(+1)
    crit = float(np.sqrt(q))
    se_mu = (hi_ - lo_) / (2 * crit) if np.isfinite(hi_ - lo_) else np.nan
    half = crit * np.sqrt(se_mu**2 + tau_hat**2)
    out.update(mu=mu_hat, mu_low=lo_, mu_high=hi_, pred_low=mu_hat - half,
               pred_high=mu_hat + half,
               p=float(stats.f.sf(lr0, 1, max(k - 1, 1))), tau=tau_hat,
               p_tau=float(0.5 * stats.chi2.sf(lr_tau, 1)) if lr_tau > 1e-9 else 1.0,
               G=G, p_G=float(stats.chi2.sf(G, k - 1)), mu_common=mu0)
    return out


def heterogeneity(design: pd.DataFrame, cfg: Config, models: Sequence[str] | None = None,
                  term: str | None = None, cols: Sequence[str] | None = None,
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-dyad O/E tables and random-effects rows (effects-table format).

    Default models: the jitter tau model (1 s) and the lag-guarded model.
    ``term`` / ``cols`` are passed to ``dyad_oe`` (hz_acausal).
    """
    kind = design.attrs["kind"]
    if models is None:
        tau = min(cfg.tau_grid, key=lambda t: abs(t - float(_opt(cfg, "jitter_tau"))))
        models = [f"tau_{_ms(tau)}", guard_model(cfg)]
    min_e = float(_opt(cfg, "meta_min_expected"))
    tabs, rows = [], []
    for model in models:
        t0 = time.perf_counter()
        tab = dyad_oe(design, cfg, model, term=term, cols=cols)
        r = re_meta(tab["observed"].to_numpy(), tab["expected"].to_numpy())
        tabs.append(tab)
        inf = tab[tab["informative"]]
        n_hi = int((inf["direction"] == "higher").sum())
        n_lo = int((inf["direction"] == "lower").sum())
        kk = int(r["k"])
        base = dict(model=model, kind=kind, backend="re_meta", link="cloglog",
                    n_bins=int(tab["n_bins_on"].sum()), n_events=int(r["O"]), n_recordings=kk,
                    fit_time_s=round(time.perf_counter() - t0, 2),
                    converged=bool(np.isfinite(r["mu"])))
        crit = float(stats.t.ppf(0.975, max(kk - 1, 1)))
        width = r["mu_high"] - r["mu_low"]
        rows.append(dict(
            base, term=tab["term"].iloc[0], term_type="chirp_sub", estimate=r["mu"],
            se=width / (2 * crit) if np.isfinite(width) else np.nan,
            ci_low=r["mu_low"], ci_high=r["mu_high"], p=r["p"],
            note=(f"two-stage random-effects meta-analysis of per-dyad O/E (Poisson-normal ML, "
                  f"profile-likelihood CI with t({kk - 1})^2 cut-off): mean log HR across "
                  f"dyads, between-dyad SD tau = {r['tau']:.2f} (95% prediction range of a "
                  f"dyad's HR {np.exp(r.get('pred_low', np.nan)):.2f}-"
                  f"{np.exp(r.get('pred_high', np.nan)):.2f}); informative dyads "
                  f"(E >= {min_e:g}): {len(inf)}, O/E CI above 1 in {n_hi}, below 1 in {n_lo}; "
                  f"O = {r['O']:.0f}, E = {r['E']:.1f}")))
        rows.append(dict(base, term="dyad_sd_sub_effect", term_type="re_sd", estimate=r["tau"],
                         se=np.nan, ci_low=np.nan, ci_high=np.nan, p=r["p_tau"],
                         note="between-dyad SD of the sub-chirp log HR (Poisson-normal ML); p: LR "
                              "test of tau = 0 (50:50 chi2 mixture)"))
        rows.append(dict(base, term="heterogeneity_G", term_type="test", estimate=r["G"],
                         se=np.nan, ci_low=np.nan, ci_high=np.nan, p=r["p_G"],
                         note=f"Poisson homogeneity G test of per-dyad O/E, chi2({kk - 1}): "
                              "dyad-specific vs one common sub-chirp HR (baseline fixed)"))
    eff = pd.DataFrame(rows)
    rr = ~eff["term_type"].isin(["test", "re_sd"])
    for c, src in (("ratio", "estimate"), ("ratio_low", "ci_low"), ("ratio_high", "ci_high")):
        eff[c] = np.where(rr, np.exp(eff[src].astype(float)), np.nan)
    return pd.concat(tabs, ignore_index=True), eff.reindex(columns=EFFECT_COLUMNS)


# ---------------------------------------------------------------------------
# Figures (style, colours and saving shared with hz_desc)
# ---------------------------------------------------------------------------
BACKEND_LABEL = {
    "mixed_laplace": "logit GLMM, dyad random intercept (Laplace)",
    "mixed_vb": "logit GLMM (mean-field VB)",
    "glm_fe": "cloglog GLM, dyad fixed intercepts (cluster-robust)",
}
KIND_LABEL = {"offset": "Chase offset hazard (inside chases)",
              "onset": "Chase onset hazard (outside chases)"}
SENDER_LABEL = {"sub": "subordinate", "dom": "dominant"}


def _ratio_axis(ax: Any, lo: float, hi: float) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    from hz_desc import GRID, MUTED

    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
             if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _lims(est: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """Log-axis limits: all estimates visible, CIs shown up to 4x beyond them."""
    est = np.r_[np.asarray(est, float)[np.isfinite(est)], 1.0]
    lo = np.asarray(lo, float)[np.isfinite(lo)]
    hi = np.asarray(hi, float)[np.isfinite(hi)]
    lo_v = min(est.min(), max(lo.min() if len(lo) else 1.0, est.min() / 4)) / 1.15
    hi_v = max(est.max(), min(hi.max() if len(hi) else 1.0, est.max() * 4)) * 1.15
    return max(lo_v, 0.01), min(hi_v, 50.0)


def plot_kernels(effects: pd.DataFrame, out_dir: Path, name: str = "hazard_kernels",
                 primary: str = "mixed_laplace", check: str = "glm_fe") -> list[Path]:
    """Fitted chirp history kernels (ratio per chirp vs lag), senders x kinds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from hz_desc import COLORS, INK2, _save, _style

    _style()
    ker = effects[effects["term"].isin(["kernel_sub", "kernel_dom"])]
    kinds = [k for k in KINDS if k in set(ker["kind"])]
    if ker.empty or not kinds:
        return []
    fig, axes = plt.subplots(2, len(kinds), figsize=(3.6 * len(kinds) + 0.4, 5.4),
                             squeeze=False, constrained_layout=True)
    for j, kind in enumerate(kinds):
        for i, s in enumerate(SENDERS):
            ax = axes[i, j]
            k = ker[(ker["kind"] == kind) & (ker["term"] == f"kernel_{s}")]
            p = k[k["backend"] == primary].sort_values("lag")
            c = k[k["backend"] == check].sort_values("lag")
            if len(p):
                ax.fill_between(p["lag"], p["ratio_low"].clip(0.005, 100),
                                p["ratio_high"].clip(0.005, 100),
                                color=COLORS[s], alpha=0.18, lw=0, zorder=2)
                ax.plot(p["lag"], p["ratio"], color=COLORS[s], lw=1.8, zorder=4)
            if len(c):
                ax.plot(c["lag"], c["ratio"], color=COLORS[s], lw=1.2, ls=(0, (4, 2)), zorder=3)
            ref = p if len(p) else c
            _ratio_axis(ax, *_lims(k["ratio"].to_numpy(), ref["ratio_low"].to_numpy(),
                                   ref["ratio_high"].to_numpy()))
            summ = effects[(effects["kind"] == kind) & (effects["backend"] == primary)
                           & (effects["term"].isin([f"kernel_{s}_mean", f"kernel_{s}_joint"]))]
            txt = ""
            if len(summ) == 2:
                m = summ[summ["term"] == f"kernel_{s}_mean"].iloc[0]
                jt = summ[summ["term"] == f"kernel_{s}_joint"].iloc[0]
                txt = (f"mean over lags: {m['ratio']:.2f} [{m['ratio_low']:.2f}, "
                       f"{m['ratio_high']:.2f}]; joint p={jt['p']:.2g}")
            ax.set_title(f"{KIND_LABEL[kind].split(' (')[0]}: {SENDER_LABEL[s]} chirps\n{txt}",
                         loc="left", fontsize=8)
            ax.set_xlim(0, float(k["lag"].max()) if len(k) else 2.0)
            if i == 1:
                ax.set_xlabel("time since chirp (s)")
            if j == 0:
                ax.set_ylabel("odds ratio per chirp (log scale)")
            ax.tick_params(axis="y", colors=INK2)
    handles = [
        Line2D([], [], color=INK2, lw=1.8, label=f"{BACKEND_LABEL[primary]}, 95% band"),
        Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label=BACKEND_LABEL[check]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=7,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Chirp history kernels: change in chase-event hazard after one chirp",
                 fontsize=9, x=0.01, ha="left")
    return _save(fig, Path(out_dir), name)


def plot_tau_scan(effects: pd.DataFrame, out_dir: Path, name: str = "hazard_tau_scan",
                  backends: Sequence[str] = ("mixed_laplace", "glm_fe")) -> list[Path]:
    """Ratio for '>= 1 chirp in the preceding tau' vs tau, per kind, sender and backend."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from hz_desc import COLORS, INK2, _save, _style

    _style()
    tt = effects[effects["term"].str.contains("_any_", regex=False)
                 & effects["term_type"].str.startswith("chirp_")].copy()
    if tt.empty:
        return []
    tt["tau"] = tt["term"].str.extract(r"_any_(\d+)ms")[0].astype(float) / 1000
    tt["sender"] = tt["term"].str.split("_").str[0]
    taus = np.sort(tt["tau"].unique())
    kinds = [k for k in KINDS if k in set(tt["kind"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.7 * len(kinds) + 0.4, 3.3),
                             squeeze=False, constrained_layout=True)
    marks = {"mixed_laplace": ("o", True), "glm_fe": ("s", False), "mixed_vb": ("D", True)}
    combos = [(s, b) for s in SENDERS for b in backends]
    offs = np.linspace(-0.24, 0.24, len(combos))
    for j, kind in enumerate(kinds):
        ax = axes[0, j]
        d = tt[tt["kind"] == kind]
        for (s, b), dx in zip(combos, offs):
            r = d[(d["sender"] == s) & (d["backend"] == b)].sort_values("tau")
            if r.empty:
                continue
            x = np.searchsorted(taus, r["tau"].to_numpy()) + dx
            m, filled = marks.get(b, ("o", True))
            err = [r["ratio"] - r["ratio_low"], r["ratio_high"] - r["ratio"]]
            ax.errorbar(x, r["ratio"], yerr=err,
                        fmt=m, ms=4.5, color=COLORS[s], mfc=COLORS[s] if filled else "white",
                        mec=COLORS[s], ecolor=COLORS[s], elinewidth=1.0, capsize=0, lw=0,
                        alpha=1.0 if s == "sub" else 0.75, zorder=4 if s == "sub" else 3)
        sub = d[d["sender"] == "sub"]
        _ratio_axis(ax, *_lims(d["ratio"].to_numpy(), sub["ratio_low"].to_numpy(),
                               sub["ratio_high"].to_numpy()))
        ax.set_xticks(np.arange(len(taus)))
        ax.set_xticklabels([f"{t:g}" for t in taus])
        ax.set_xlabel("tau: chirp in the preceding tau seconds")
        if j == 0:
            ax.set_ylabel("odds ratio (log scale)")
        ax.set_title(KIND_LABEL[kind], loc="left", fontsize=8.3)
        ax.tick_params(axis="y", colors=INK2)
    handles = [Line2D([], [], color=COLORS[s], marker=marks[b][0], lw=0, ms=4.5,
                      mfc=COLORS[s] if marks[b][1] else "white",
                      label=f"{SENDER_LABEL[s]} chirp, {BACKEND_LABEL[b]}")
               for s, b in combos]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=7, bbox_to_anchor=(0.5, -0.3))
    fig.suptitle("Chase-event hazard after >= 1 chirp in the preceding tau (95% CI; one model "
                 "per tau, dominant chirp indicator as control)", fontsize=8.6, x=0.01, ha="left")
    return _save(fig, Path(out_dir), name)


def plot_dyad_oe(dyad_tab: pd.DataFrame, effects: pd.DataFrame, out_dir: Path,
                 name: str = "hazard_dyad_oe") -> list[Path]:
    """Forest plot of per-dyad observed / expected events (the verdicts' dyad-level basis).

    One panel per kind x model (1 s indicator, lag-guarded). Each row is a dyad:
    O/E with its exact Poisson 95% CI, marker area proportional to expected
    events; dyads below meta_min_expected are grey. Diamond: random-effects mean
    over dyads (re_meta) with 95% CI; dashed line: common-effect FE HR.
    """
    import matplotlib.pyplot as plt

    from hz_desc import COLORS, INK2, _save, _style

    if dyad_tab is None or dyad_tab.empty:
        return []
    _style()
    tab = dyad_tab[dyad_tab["expected"] > 0].copy()
    models = [m for m in ("tau_1000ms", "guard_500ms") if m in set(tab["model"])]
    kinds = [k for k in KINDS if k in set(tab["kind"])]
    n_rows = max(int((tab["model"] == m).groupby(tab["kind"]).sum().max()) for m in models)
    fig, axes = plt.subplots(len(models), len(kinds),
                             figsize=(3.8 * len(kinds) + 0.4, 0.22 * n_rows * len(models) + 1.6),
                             squeeze=False, constrained_layout=True)
    floor = 0.02  # O = 0 plotted at this value with a left-pointing marker
    for i, model in enumerate(models):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            d = tab[(tab["model"] == model) & (tab["kind"] == kind)].sort_values(
                "expected", ascending=True).reset_index(drop=True)
            if d.empty:
                ax.set_axis_off()
                continue
            ypos = np.arange(len(d)) + 1.5
            for y, r in zip(ypos, d.itertuples()):
                col = COLORS["sub"] if r.informative else "#b9b7b1"
                lo = max(r.oe_low, floor)
                hi = r.oe_high
                ax.plot([lo, hi], [y, y], color=col, lw=1.1, zorder=2)
                size = 12 + 6 * np.sqrt(r.expected)
                if r.observed == 0:
                    ax.scatter([floor], [y], marker="<", s=size, color=col, zorder=3)
                else:
                    ax.scatter([r.oe], [y], s=size, color=col, edgecolor="white", lw=0.5,
                               zorder=3)
            ax.set_yticks(ypos)
            ax.set_yticklabels([f"{rec[5:10]} ({o}/{e:.1f})" for rec, o, e in
                                zip(d["recording"], d["observed"], d["expected"])], fontsize=6.5)
            term = str(d["term"].iloc[0])
            re = effects[(effects["model"] == model) & (effects["kind"] == kind)
                         & (effects["backend"] == "re_meta") & (effects["term"] == term)]
            if len(re):
                r0 = re.iloc[0]
                ax.fill([r0["ratio_low"], r0["ratio"], r0["ratio_high"], r0["ratio"]],
                        [0.5, 0.75, 0.5, 0.25], color=INK2, zorder=3)
                ax.text(r0["ratio_high"] * 1.1, 0.5,
                        f"RE mean {r0['ratio']:.2f} [{r0['ratio_low']:.2f}, "
                        f"{r0['ratio_high']:.2f}]", va="center", fontsize=6.3, color=INK2)
            ax.axvline(1.0, color="#8a8883", lw=0.8, zorder=1)
            ax.axvline(float(d["common_hr"].iloc[0]), color=COLORS["sub"], lw=0.8, ls="--",
                       zorder=1)
            ax.set_xscale("log")
            xmax = max(12.0, float(np.nanmax(d["oe_high"].to_numpy())) * 1.2)
            ax.set_xlim(floor / 1.5, xmax)
            ax.set_ylim(0, len(d) + 1)
            ax.set_xlabel("observed / expected events (log scale)")
            win = "0.5-1.5 s" if model.startswith("guard") else "0-1 s"
            ax.set_title(f"{KIND_LABEL[kind]}\nsub chirp {win} before the bin", loc="left",
                         fontsize=7.8)
    fig.suptitle("Per-dyad O/E in bins after a subordinate chirp (exact 95% CI; size ~ expected "
                 "events; grey = < min expected; dashed = common FE HR; diamond = RE mean)",
                 fontsize=8.0, x=0.01, ha="left")
    return _save(fig, Path(out_dir), name)


def plot_jitter(null: pd.DataFrame, out_dir: Path, name: str = "hazard_jitter_null") -> list[Path]:
    """Null distributions of the subordinate-chirp statistics vs the observed value."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from hz_desc import COLORS, NULL_FILL, NULL_LINE, _fmt_p, _save, _style

    _style()
    if null.empty:
        return []
    summ = summarize_jitter(null)
    kinds = [k for k in KINDS if k in set(null["kind"])]
    stats_ = list(dict.fromkeys(null["stat"]))
    nulls = list(dict.fromkeys(null["null"]))
    fig, axes = plt.subplots(len(kinds), len(stats_), figsize=(3.5 * len(stats_) + 0.4,
                             2.4 * len(kinds) + 0.5), squeeze=False, constrained_layout=True)
    styles = {"shift": dict(histtype="stepfilled", color=NULL_FILL, ec=NULL_FILL),
              "matched": dict(histtype="step", color=NULL_LINE, lw=1.2)}
    for i, kind in enumerate(kinds):
        for j, st in enumerate(stats_):
            ax = axes[i, j]
            g = null[(null["kind"] == kind) & (null["stat"] == st)]
            g0 = g.loc[g["iter"] == 0, "value"]
            obs = float(g0.iloc[0]) if len(g0) else np.nan
            vals = g.loc[g["iter"] > 0, "value"].dropna()
            lo, hi = min(vals.min(), obs), max(vals.max(), obs)
            bins = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 40)
            lines = []
            for nl in nulls:
                v = g.loc[(g["iter"] > 0) & (g["null"] == nl), "value"].dropna()
                ax.hist(v, bins=bins, **styles.get(nl, styles["matched"]))
                s = summ[(summ["kind"] == kind) & (summ["stat"] == st) & (summ["null"] == nl)]
                if len(s):
                    lines.append(f"{nl}: p={_fmt_p(float(s['p_two_sided'].iloc[0]))}")
            ax.axvline(obs, color=COLORS["sub"], lw=1.8)
            ax.axvline(0, color=NULL_LINE, lw=0.8, ls=(0, (3, 2)))
            lab = ("tau = 1 s indicator: log hazard ratio" if st.startswith("tau_")
                   else "kernel mean: log hazard ratio per chirp")
            ax.set_xlabel(lab)
            ax.set_title(f"{KIND_LABEL[kind].split(' (')[0]}\nobserved {obs:+.2f}; "
                         + ", ".join(lines), loc="left", fontsize=7.8)
            ax.set_yticks([])
    handles = [Patch(color=NULL_FILL, label="within-state circular shift null"),
               Patch(fill=False, ec=NULL_LINE, label="elapsed-time-matched swap null"),
               Line2D([], [], color=COLORS["sub"], lw=1.8, label="observed")]
    fig.legend(handles=[h for h, n in zip(handles, ["shift", "matched", "obs"])
                        if n == "obs" or n in nulls],
               loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Subordinate-chirp effect vs permutation nulls (dyad-FE cloglog GLM refits)",
                 fontsize=8.6, x=0.01, ha="left")
    return _save(fig, Path(out_dir), name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def headline(effects: pd.DataFrame, jitter_summary: pd.DataFrame | None = None,
             lodo: pd.DataFrame | None = None, primary: str = "mixed_laplace",
             check: str = "glm_fe") -> str:
    """Concise text answer to (a) offset and (b) onset."""
    def pick(kind: str, backend: str, term: str) -> pd.Series | None:
        r = effects[(effects["kind"] == kind) & (effects["backend"] == backend)
                    & (effects["term"] == term)]
        return r.iloc[0] if len(r) else None

    def ci(r: pd.Series | None) -> str:
        if r is None:
            return "n/a"
        return f"{r['ratio']:.2f} [{r['ratio_low']:.2f}, {r['ratio_high']:.2f}], p={r['p']:.2g}"

    out = []
    q = {"offset": "(a) Does a subordinate chirp raise the hazard that an ongoing chase ends?",
         "onset": "(b) Does a subordinate chirp lower the hazard that a new chase starts?"}
    for kind in KINDS:
        tau = effects[(effects["kind"] == kind) & effects["term"].str.startswith("sub_any_")]
        if tau.empty and effects[effects["kind"] == kind].empty:
            continue
        r1 = pick(kind, primary, "sub_any_1000ms")
        n_ev = int(r1["n_events"]) if r1 is not None else 0
        n_bins = int(r1["n_bins"]) if r1 is not None else 0
        out.append(q[kind])
        bw = effects.attrs.get("bin_width", 0.05)
        out.append(f"  {n_ev} events in {n_bins:,} bins of {bw:g} s")
        out.append(f"  OR, >=1 sub chirp in preceding 1 s: GLMM {ci(r1)}; "
                   f"FE-cloglog {ci(pick(kind, check, 'sub_any_1000ms'))}")
        out.append(f"  OR per sub chirp, kernel mean over 0-2 s: GLMM "
                   f"{ci(pick(kind, primary, 'kernel_sub_mean'))}")
        j = pick(kind, primary, "kernel_sub_joint")
        if j is not None:
            out.append(f"  joint test of sub kernel: p={j['p']:.2g}")
        pm = tau[tau["backend"] == primary].copy()
        if len(pm):
            pm["tau"] = pm["term"].str.extract(r"(\d+)ms")[0].astype(float) / 1000
            out.append("  tau scan (GLMM OR): " + ", ".join(
                f"{t:g}s {o:.2f}" for t, o in zip(pm.sort_values('tau')['tau'],
                                                  pm.sort_values('tau')['ratio'])))
        re_ = pick(kind, "re_meta", "sub_any_1000ms")
        if re_ is not None:
            out.append(f"  random-effects mean over dyads (per-dyad O/E): {ci(re_)}")
        g_ = pick(kind, primary, "sub_grd_500ms")
        if g_ is not None:
            out.append(f"  lag-guarded (sub chirp 0.5-1.5 s before): GLMM {ci(g_)}; FE-cloglog "
                       f"{ci(pick(kind, check, 'sub_grd_500ms'))}")
        dm = pick(kind, primary, "dom_any_1000ms")
        out.append(f"  comparison, dominant chirp in preceding 1 s: GLMM {ci(dm)}")
        if lodo is not None and len(lodo) and (lodo["kind"] == kind).any():
            lk = lodo[lodo["kind"] == kind]
            i_lo, i_hi = lk["ratio"].idxmin(), lk["ratio"].idxmax()
            out.append(f"  leave-one-dyad-out ({lk['stat'].iloc[0]}, FE-cloglog OR): "
                       f"{lk.loc[i_lo, 'ratio']:.2f} (without {lk.loc[i_lo, 'left_out'][:10]}) to "
                       f"{lk.loc[i_hi, 'ratio']:.2f} (without {lk.loc[i_hi, 'left_out'][:10]})")
            if "dyad_events_on" in lk:
                n_on = int(lk["dyad_events_on"].sum())
                n_d = int((lk["dyad_events_on"] > 0).sum())
                out.append(f"  events preceded by a sub chirp (indicator on): {n_on}, in "
                           f"{n_d}/{len(lk)} dyads")
        if jitter_summary is not None and len(jitter_summary):
            js = jitter_summary[jitter_summary["kind"] == kind]
            for _, r in js.iterrows():
                out.append(f"  jitter[{r['null']}] {r['stat']}: obs {r['observed']:+.3f}, null "
                           f"{r['null_mean']:+.3f} +- {r['null_sd']:.3f}, p={r['p_two_sided']:.3g} "
                           f"(n={int(r['n_jitter'])})")
    return "\n".join(out)


def run_hazard(
    trials: Sequence[Trial], cfg: Config, kinds: Sequence[str] = KINDS,
    jitter_nulls: Sequence[str] = ("shift", "matched"), vb_check: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Designs, all fits, jitter tests, tables and figures for both kinds.

    Writes to cfg.out_dir/'hazard' (tables) and cfg.out_dir/'figures'.
    Returns dict(effects, jitter, jitter_summary, design_summary,
    leave_one_dyad_out, timings, text, figures).
    """
    out = Path(cfg.out_dir) / "hazard"
    fig_dir = Path(cfg.out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    effs, nulls, books, timings, lodos, oes = [], [], [], [], [], []
    for kind in kinds:
        t0 = time.perf_counter()
        d = build_design(trials, kind, cfg)
        timings.append(dict(kind=kind, step="build_design", seconds=time.perf_counter() - t0))
        bk = d.attrs["bookkeeping"].assign(kind=kind, min_elapsed=d.attrs["min_elapsed"],
                                           control_fraction=d.attrs["control_fraction"])
        books.append(bk)
        if verbose:
            print(f"[{kind}] {len(d):,} bins, {int(d['y'].sum())} events, "
                  f"min_elapsed {d.attrs['min_elapsed']:.3f} s", flush=True)
        t0 = time.perf_counter()
        e = fit_models(d, kind, cfg, verbose=verbose)
        timings.append(dict(kind=kind, step="fit_models", seconds=time.perf_counter() - t0))
        if vb_check and kind == "offset":
            t0 = time.perf_counter()
            e = pd.concat([e, fit_models(d, kind, cfg, models=["tau_1000ms"],
                                         backends=["mixed_vb"], verbose=verbose)])
            timings.append(dict(kind=kind, step="mixed_vb", seconds=time.perf_counter() - t0))
        effs.append(e)
        t0 = time.perf_counter()
        oe, het = heterogeneity(d, cfg)
        oes.append(oe)
        effs.append(het)
        timings.append(dict(kind=kind, step="heterogeneity", seconds=time.perf_counter() - t0))
        t0 = time.perf_counter()
        lodos.append(leave_one_dyad_out(d, cfg))
        timings.append(dict(kind=kind, step="leave_one_dyad_out", seconds=time.perf_counter() - t0))
        for nl in jitter_nulls:
            t0 = time.perf_counter()
            nulls.append(jitter_test(trials, kind, cfg, design=d, null=nl, verbose=verbose))
            timings.append(dict(kind=kind, step=f"jitter_{nl}", seconds=time.perf_counter() - t0))
        del d
    effects = pd.concat(effs, ignore_index=True)
    effects.attrs["bin_width"] = cfg.bin_width
    for x in nulls:  # attrs hold DataFrames, which pd.concat cannot compare
        x.attrs = {}
    null = pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame(
        columns=["kind", "null", "stat", "iter", "value"])
    jsum = summarize_jitter(null) if len(null) else pd.DataFrame()
    books_df = pd.concat(books, ignore_index=True)
    tim = pd.DataFrame(timings)
    effects.to_csv(out / "hazard_effects.csv", index=False)
    null.to_csv(out / "hazard_jitter_null.csv", index=False)
    jsum.to_csv(out / "hazard_jitter_summary.csv", index=False)
    books_df.to_csv(out / "hazard_design_summary.csv", index=False)
    tim.to_csv(out / "hazard_timings.csv", index=False)
    lodo = pd.concat(lodos, ignore_index=True)
    lodo.to_csv(out / "hazard_leave_one_dyad_out.csv", index=False)
    dyad_tab = pd.concat(oes, ignore_index=True)
    dyad_tab.to_csv(out / "hazard_dyad_oe.csv", index=False)
    figs = plot_kernels(effects, fig_dir) + plot_tau_scan(effects, fig_dir)
    figs += plot_dyad_oe(dyad_tab, effects, fig_dir)
    if len(null):
        figs += plot_jitter(null, fig_dir)
    text = headline(effects, jsum, lodo)
    (out / "hazard_summary.txt").write_text(text + "\n")
    return dict(effects=effects, jitter=null, jitter_summary=jsum, design_summary=books_df,
                leave_one_dyad_out=lodo, dyad_oe=dyad_tab, timings=tim, text=text,
                figures=figs)


def main() -> None:
    from hz_data import load_trials

    ap = argparse.ArgumentParser(description="Discrete-time chirp -> chase hazard models.")
    ap.add_argument("data_root", type=Path, nargs="?", default=Config().data_root)
    ap.add_argument("--out", type=Path, default=Config().out_dir)
    ap.add_argument("--bin-width", type=float, default=None)
    ap.add_argument("--hist-max", type=float, default=None)
    ap.add_argument("--n-jitter", type=int, default=None)
    ap.add_argument("--max-time", type=float, default=None, help="s; <=0 disables the cap")
    ap.add_argument("--use-rises", action="store_true")
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--include-draws", action="store_true")
    ap.add_argument("--onset-control-fraction", type=float, default=None)
    ap.add_argument("--nulls", default="shift,matched", help="comma list of jitter nulls or ''")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--vb", action="store_true", help="add a mean-field VB fit (offset, tau 1 s)")
    ap.add_argument("--quick", action="store_true", help="20 jitters per null")
    a = ap.parse_args()
    cfg = Config(data_root=a.data_root, out_dir=a.out).updated(
        bin_width=a.bin_width, hist_max=a.hist_max, n_jitter=a.n_jitter, max_time=a.max_time,
        use_rises=a.use_rises or None, onset_control_fraction=a.onset_control_fraction,
        exclude_bad_recordings=False if a.include_bad else None,
        exclude_draws=False if a.include_draws else None, quick=a.quick or None,
    )
    if a.n_workers is not None:
        cfg.extra["n_workers"] = a.n_workers
    t0 = time.perf_counter()
    trials, flags, _ = load_trials(cfg)
    print(f"{len(trials)} trials loaded ({time.perf_counter() - t0:.1f} s)")
    res = run_hazard(trials, cfg, jitter_nulls=[x for x in a.nulls.split(",") if x],
                     vb_check=a.vb)
    print("\n" + res["text"])
    print("\ntimings (s):\n" + res["timings"].to_string(index=False))
    print(f"\ntotal {time.perf_counter() - t0:.0f} s; tables in {Path(cfg.out_dir) / 'hazard'}")


if __name__ == "__main__":
    main()
