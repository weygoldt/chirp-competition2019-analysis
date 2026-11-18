"""Simulation checks for the discrete-time chirp -> chase hazard models.

Units: seconds and Hz. Synthetic trials are :class:`hz_data.Trial` objects, so
``hz_hazard.build_design`` and the glm_fe / mixed fits run unchanged on real
and simulated data (``fit_fe`` is the glm_fe path of ``hz_hazard.fit_models``
plus the model-based covariance; a test checks they agree).

Generative model (one synthetic trial per real trial, 0.05 s analysis bins)
---------------------------------------------------------------------------
Chase state  Alternating out (inter-chase gap) and in (chase) episodes on the
             analysis bin grid. An episode that began at t_s ends in bin k with
             P = 1 - exp(-dL0(e_k) * m_k * exp(beta * x_k)), e_k = t_k - t_s,
             dL0 the baseline cumulative hazard over the bin (grouped-time
             proportional hazards: hz_hazard's cloglog GLM is correctly
             specified except that it fits one elapsed-time spline shape for
             all dyads). Baseline per real trial: Gaussian KDE (half the
             Silverman bandwidth) of log chase durations / log gaps, truncated
             at the shortest duration / gap of all trials (BORIS floor ~1 s /
             ~2 s). Pooled chase counts, medians and means match the data to
             ~2-4%; single trials deviate more (chase counts up to ~15%, see
             sim_match.csv and sim_summary.txt section 6).
Sub chirps   Poisson cluster (burst) process. Parent rate = trial rate in the
             state x pooled relative shape of elapsed state time (in chase:
             rises ~3.5x over the chase; after an offset: ~5x elevated for
             ~1 s, decaying over ~20 s). Parents start geometric bursts
             (continuation probability q per trial, fitted to the share of
             inter-chirp intervals < 1.5 s above chance) with intervals
             resampled from the pooled observed within-burst intervals.
             Per-state expected counts are matched to each real trial.
Dom chirps   Same burst process with constant per-state rates. No effect.
Effect       x_k = 1 if >= 1 sub chirp in bins k - J .. k - 1 (J = window / bw;
             'box', default 1 s: hz_hazard's tau = 1 s indicator is then
             correctly specified, the kernel model is not), or x_k = sum_i
             exp(-lag_i / window) ('exp', additive per chirp). beta_off acts
             inside chases, beta_on outside. A chirp in the outcome bin never
             acts on it (as in the design). With beta_off_sd / beta_on_sd > 0
             each synthetic trial draws its own beta ~ N(beta, sd^2)
             (between-dyad heterogeneity, 'het' scenarios).
Calibration  estimate_params matches every synthetic trial to its real trial
             under no effect. Effect scenarios ('calib') use per-trial
             multipliers of the baseline hazards and chirp rates, iterated
             until the effect scenario reproduces each trial's chase count,
             time in chase and in/out-of-chase sub chirp counts
             (calibrate_params).
Contacts     Poisson in out-of-chase time at the trial's rate; sizes, sexes and
             baseline EODf tracks (dEODf covariates) are copied from the trial.

Options (SimParams fields; each exercised in at least one scenario)
------------------------------------------------------------------
annotation   boundary_sd: Gaussian error on every chase onset / offset
             (truncated at 0.4 x the adjacent chase or gap); boundary_step:
             rounding to a lattice (the data suggest 0.5 s annotation steps).
detection    p_detect_in / p_detect_out: each chirp is detected independently
             (by its true chase state); generation rates are divided by p so
             detected rates stay matched. Effects act through all true chirps.
trigger      Hidden common trigger (offset null): with probability pi per
             chase, a latent event at T = offset - U(trigger_offset_delay)
             ends the chase and elicits one sub chirp at
             T + U(trigger_chirp_delay) (default 0-0.5 s and 0-0.3 s). Chase
             durations are unchanged, trigger chirps have no causal effect, and
             the baseline in-chase chirp rate is lowered so counts stay matched.
             Strength trigger_share = expected share of in-chase sub chirps
             that are trigger chirps landing before the offset (pi = share x
             n_in / (n_chases x p_before), capped at 1, p_before = P(chirp
             delay < offset delay) = 0.7 by default; the ~30% of trigger
             chirps that land after the offset are extra out-of-chase chirps).
             Reports use the realized share from the simulated data.
lag          offset_lag: every annotated offset = true offset + lag (capped at
             0.4 x the following gap); no causal effect. The post-offset chirp
             burst then falls before the annotated offset.
hidden state Onset analogue (an addition to the brief): a latent two-state
             process (fraction hidden_frac of time, mean episode
             hidden_mean_dur s) that multiplies the out-of-chase sub chirp rate
             by hidden_chirp_mult and the onset hazard by hidden_onset_mult
             (both renormalised), with no causal chirp effect (e.g. the
             subordinate retreats and chirps more while it is away).
offset-locked chirps (used by hz_acausal_sim; off by default)
             At every TRUE chase offset: a response burst of Poisson(resp_n)
             sub chirps (TrialParams.resp_n) at off + L, L drawn from the
             piecewise-uniform density (resp_edges, resp_density); and/or a
             latent symmetric cluster, present with probability
             sym_n / sym_m, that puts Poisson(sym_m) sub chirps at off - L and
             Poisson(sym_m) at off + L (L from sym_edges / sym_density), i.e.
             sym_n chirps per side per offset on average. With
             resp_bout_mult != 1 the response mean is multiplied by it after
             offsets with a sub chirp in the resp_bout_window s before the true
             offset (a chirp bout that continues past the chase end). These chirps act
             on later chases through the effect covariate like any other
             chirp; the pre-offset side of the cluster is added after its chase
             was simulated, so it cannot act on that chase (use it with
             beta_off = 0). Own random stream (8th SeedSequence child), so the
             other streams and all scenarios without these chirps are unchanged.
drift        (hz_acausal_sim; off by default) sub_mod_frac / sub_mod_dur /
             sub_mod_mult: a slow latent two-state process (own 9th stream)
             multiplies the sub chirp parent rate in BOTH chase states and
             the offset-locked chirps (response burst mean, cluster
             probability) at each offset (renormalised; no effect on the
             hazards), i.e. minute-scale co-variation of chirping within a
             dyad. burst_cross: bursts
             begun before a chase boundary keep their offspring after it
             (default: every burst ends at the state change). Same random
             draws as without it; only the bookkeeping changes.

Checks (run_simulation_checks; sizes scale with cfg.n_sims, see
default_scenarios)
------------------
recovery     beta_off = log 2, beta_on = log 0.5 (1 s box): bias, SD, SE
             calibration and 95% CI coverage of the 1 s indicator and the
             kernel mean (glm_fe: cloglog, dyad fixed intercepts, model-based
             and cluster-robust t(G-1) intervals). mixed_laplace (the primary
             backend of the real analysis) and the random-effects meta-analysis
             (re_meta) on every no_effect / recovery / het replicate.
             The same effects with annotation error and with missed chirps;
             an exp-shaped kernel (offset).
no_effect    beta = 0: rejection rates at 0.05 of model-based Wald,
             cluster-robust t / F and the jitter test (hz_hazard.jitter_test,
             both nulls, fewer replicates; rank p with 39 draws is exact at
             p <= 0.05). The jitter null SD is also compared with the sampling
             SD of the statistic across replicates.
trigger      beta = 0 plus the hidden trigger at several strengths (offset):
             spurious tau scan and kernel; a slow-lag variant (offset 0.5-1.5 s
             after the latent event) and one with annotation error.
het          dyad-specific effects with SD fitted to the data (main script passes
             the random-effects tau): het_no_effect (mean 0) and het_recovery
             (mean HR 2 / 0.5); GLMM, FE and random-effects meta calibration.
lag          systematic late offset annotation by 0.25 / 0.5 / 0.75 s.
hidden_state onset analogue with a slow (30 s) and a short (3 s) latent state.

Onset designs of simulated data keep all event bins and 10% of non-event bins
(hz_hazard's case-control option; IPW-weighted cloglog). Full onset designs
(2.9e6 bins) make each fit memory-bandwidth bound and throughput stops scaling
at ~2 concurrent fits. The mixed-check replicates are also fitted on all bins
(backend 'glm_fe_full'): the two agree to ~0.01 SE. The 'observed' reference
(real data) always uses all bins, as in the real analysis.

Replicates use SeedSequence([seed, crc32(key), rep]) with separate streams per
trial and component (chase clock, chirps, trigger, observation, latent state,
dom chirps), so scenarios with the same key and rep share randomness (common
random numbers): with no causal effect the chase sequence is identical across
trigger strengths, and the jitter replicates re-use the no_effect datasets.
Work runs in forked processes with one BLAS thread each.

Outputs: cfg.out_dir/'simulation'/sim_*.csv, sim_summary.txt; figures
sim_recovery, sim_fpr, sim_confounds in cfg.out_dir/'figures'. The no-effect scenario
is called 'no_effect', not 'null', which pandas reads as NaN.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parent))
import hz_hazard as hz  # noqa: E402
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402
from hz_desc import _complement, _in_iv, trial_states  # noqa: E402

KINDS = ("offset", "onset")
LOG2 = float(np.log(2.0))
KDE_BW_FACTOR = 0.5  # x Silverman bandwidth for the duration / gap KDEs
EXTRA_S = 600.0  # s simulated past obs_end so the last chase gets an offset
CHUNK = {"in": 400, "out": 2000}  # bins processed per step inside an episode
IN_EDGES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, np.inf])
OUT_EDGES = np.array([0, 0.5, 1, 2, 3, 5, 10, 20, 40, 80, 160, 320, np.inf])

# Simulation options, read from cfg.extra (key 'sim_<name>') or these defaults.
SIM_DEFAULTS: dict[str, Any] = dict(
    # Forked worker processes. The fits stream large matrices and are memory-
    # bandwidth bound: on a 16-core Ryzen 5950X throughput stops improving at
    # ~6-8 concurrent fits (each slows ~3x), so more workers only cost memory.
    workers=8,
    jitter_reps_offset=12,  # no_effect replicates that also get the jitter test
    jitter_draws_offset=39,  # draws per null (39: min rank p = 0.05, exact level)
    jitter_reps_onset=8,
    jitter_draws_onset=39,
    mixed_reps=4,  # recovery replicates also fitted on all onset bins (glm_fe_full, GLMM)
    # Onset designs of simulated data keep all event bins and this fraction of
    # non-event bins (hz_hazard case-control option, IPW-weighted cloglog).
    # Full onset designs (2.9e6 bins) make every fit memory-bandwidth bound:
    # throughput stops scaling at ~2 concurrent fits on this machine.
    onset_fraction=0.1,
    hidden=True,  # run the hidden-state onset scenario
    burst_ici_max=1.5,  # s, inter-chirp interval that continues a burst
)


def _sopt(cfg: Config, name: str) -> Any:
    return cfg.extra.get(f"sim_{name}", SIM_DEFAULTS[name])


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class SurvTab:
    """Log-survival of a baseline duration distribution, tabulated in log e."""

    floor: float
    x: np.ndarray  # log elapsed grid (s)
    logS: np.ndarray  # 0 at floor, decreasing
    e_max: float
    h_end: float  # hazard used to extrapolate beyond e_max (1/s)
    bandwidth: float  # KDE bandwidth (log s)

    def logsurv(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, float)
        out = np.interp(np.log(np.maximum(e, self.floor)), self.x, self.logS)
        over = e > self.e_max
        if np.any(over):
            out = np.where(over, self.logS[-1] - self.h_end * (e - self.e_max), out)
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Inverse-transform draws (for tests)."""
        u = -np.log(rng.uniform(size=n))  # -log S at the draw
        return np.exp(np.interp(-u, self.logS[::-1], self.x[::-1]))


def _kde_logsurv(ld: np.ndarray, h: float, e: np.ndarray) -> np.ndarray:
    z = (np.log(e)[:, None] - ld[None, :]) / h
    ls = logsumexp(stats.norm.logsf(z), axis=1) - np.log(len(ld))
    return ls - ls[0]


def surv_table(d: np.ndarray, floor: float, n_grid: int = 2000) -> SurvTab:
    """Gaussian KDE of log durations, truncated at floor.

    Bandwidth = KDE_BW_FACTOR x Silverman's rule: log-scale smoothing inflates
    the mean (by ~exp(h^2 / 2)); at half the Silverman bandwidth simulated
    chase counts, medians and means stay within ~3% of the data (checked).
    """
    d = np.asarray(d, float)
    ld = np.log(d)
    n = len(ld)
    if n < 2:
        raise ValueError("need >= 2 durations")
    sd = float(ld.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(ld, [75, 25])))
    s = min(sd, iqr / 1.34) if iqr > 0 else sd
    h = max(KDE_BW_FACTOR * 0.9 * s * n**-0.2, 0.02)
    e = np.geomspace(floor, 2.0 * float(np.exp(ld.max() + 8 * h)), n_grid)
    z = (np.log(e)[:, None] - ld[None, :]) / h
    ls = logsumexp(stats.norm.logsf(z), axis=1) - np.log(n)
    ls = ls - ls[0]
    h_end = max(float((ls[-2] - ls[-1]) / (e[-1] - e[-2])), 1e-3)
    return SurvTab(
        floor=float(floor), x=np.log(e), logS=ls, e_max=float(e[-1]), h_end=h_end, bandwidth=h
    )


@dataclass
class TrialParams:
    """Per-trial generative parameters matched to one real trial."""

    recording: str
    template: Trial  # sizes, sexes, ids, EODf tracks, window
    dur: SurvTab
    gap: SurvTab
    sub_base_in: float  # 1/s, multiplies the pooled in-chase shape
    sub_base_out: float  # 1/s, multiplies the pooled post-offset shape
    dom_rate_in: float
    dom_rate_out: float
    sub_q: float  # burst continuation probability
    dom_q: float
    contact_rate: float  # 1/s of out-of-chase time
    n_chases: int
    n_sub_in: int
    # calibration multipliers (1 = matched under no effect; see calibrate_params)
    h_mult_in: float = 1.0  # baseline chase-offset hazard
    h_mult_out: float = 1.0  # baseline chase-onset hazard
    sub_mult_in: float = 1.0  # sub chirp rate in chases
    sub_mult_out: float = 1.0  # sub chirp rate outside chases
    # offset-locked chirps (hz_acausal_sim; 0 = off): expected response-burst
    # chirps per offset, and expected symmetric-cluster chirps per side per offset
    resp_n: float = 0.0
    sym_n: float = 0.0


@dataclass
class SimParams:
    """Generative model: per-trial parameters, pooled shapes, effects, options."""

    trials: list[TrialParams]
    shape_in: np.ndarray  # relative sub chirp rate vs time since onset (IN_EDGES)
    shape_out: np.ndarray  # relative sub chirp rate vs time since offset (OUT_EDGES)
    burst_icis: np.ndarray  # pooled within-burst inter-chirp intervals (s)
    dur_floor: float
    gap_floor: float
    # causal effects (log hazard ratios)
    beta_off: float = 0.0
    beta_on: float = 0.0
    # between-dyad SD of the effects: each synthetic trial draws its own
    # beta ~ N(beta_off/on, sd^2) (dyad-specific effects, 'het' scenarios)
    beta_off_sd: float = 0.0
    beta_on_sd: float = 0.0
    effect_kernel: str = "box"  # 'box' | 'exp'
    effect_window: float = 1.0  # s: box width or exp time constant
    # hidden common trigger (offset)
    trigger_share: float = 0.0
    trigger_chirp_delay: tuple[float, float] = (0.0, 0.3)
    trigger_offset_delay: tuple[float, float] = (0.0, 0.5)
    # hidden state (onset)
    hidden_frac: float = 0.0
    hidden_mean_dur: float = 30.0
    hidden_chirp_mult: float = 1.0
    hidden_onset_mult: float = 1.0
    # observation model
    boundary_sd: float = 0.0
    boundary_step: float = 0.0
    offset_lag: float = 0.0  # s, systematic late annotation of every chase offset
    p_detect_in: float = 1.0
    p_detect_out: float = 1.0
    bursts: bool = True
    # offset-locked chirps (latency densities; per-trial sizes in TrialParams)
    resp_edges: np.ndarray | None = None  # s, latency bins of the response burst
    resp_density: np.ndarray | None = None  # probability per bin (sums to 1)
    sym_edges: np.ndarray | None = None  # s, |latency| bins of the symmetric cluster
    sym_density: np.ndarray | None = None
    sym_m: float = 1.0  # mean chirps per side of one cluster
    # 'bout': the response burst mean is x resp_bout_mult after offsets with a sub
    # chirp in the resp_bout_window s before the true offset (1 = no dependence)
    resp_bout_mult: float = 1.0
    resp_bout_window: float = 1.0
    # slow latent modulation of the sub chirp rate in BOTH chase states (hz_acausal_sim
    # 'drift'; 0 = off): a two-state process, sub_mod_frac of the time in a high state with
    # mean episode sub_mod_dur s, multiplies the parent rate by sub_mod_mult there
    # (renormalised, so the mean rate is unchanged); no effect on the chase hazards
    sub_mod_frac: float = 0.0
    sub_mod_dur: float = 300.0
    sub_mod_mult: float = 1.0
    # True: chirp bursts continue across chase boundaries (offspring of a parent drawn before
    # the state change are kept after it); False: every burst ends at the state change
    burst_cross: bool = False

    def effect_lags(self, bw: float) -> np.ndarray:
        """Effect kernel g at lags j * bw, j = 1..J (box: ones)."""
        w = float(self.effect_window)
        if self.effect_kernel == "box":
            return np.ones(max(1, int(round(w / bw))))
        if self.effect_kernel == "exp":
            j = np.arange(1, int(np.ceil(6 * w / bw)) + 1)
            return np.exp(-j * bw / w)
        raise ValueError(f"unknown effect_kernel {self.effect_kernel}")

    def table(self) -> pd.DataFrame:
        rows = []
        for tp in self.trials:
            rows.append(
                dict(
                    recording=tp.recording,
                    n_chases=tp.n_chases,
                    n_sub_in=tp.n_sub_in,
                    dur_kde_bw=tp.dur.bandwidth,
                    gap_kde_bw=tp.gap.bandwidth,
                    sub_base_in=tp.sub_base_in,
                    sub_base_out=tp.sub_base_out,
                    dom_rate_in=tp.dom_rate_in,
                    dom_rate_out=tp.dom_rate_out,
                    sub_q=tp.sub_q,
                    dom_q=tp.dom_q,
                    contact_rate=tp.contact_rate,
                    h_mult_in=tp.h_mult_in,
                    h_mult_out=tp.h_mult_out,
                    sub_mult_in=tp.sub_mult_in,
                    sub_mult_out=tp.sub_mult_out,
                    dur_floor=self.dur_floor,
                    gap_floor=self.gap_floor,
                )
            )
        return pd.DataFrame(rows)


def _shape_at(shape: np.ndarray, edges: np.ndarray, e: np.ndarray) -> np.ndarray:
    return shape[np.clip(np.searchsorted(edges, e, side="right") - 1, 0, len(shape) - 1)]


def _shape_integral(shape: np.ndarray, edges: np.ndarray, L: np.ndarray) -> np.ndarray:
    """int_0^L shape(e) de for each L."""
    lo, hi = edges[:-1], edges[1:]
    ov = np.clip(np.minimum(L[:, None], hi[None, :]) - lo[None, :], 0, None)
    return ov @ shape


def _elapsed_counts(
    t: np.ndarray, iv: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Chirp counts and exposure (s) per elapsed-time bin over intervals iv."""
    cnt = np.zeros(len(edges) - 1)
    if len(iv):
        i = np.searchsorted(iv[:, 0], t, side="right") - 1
        ok = (i >= 0) & (t < iv[np.maximum(i, 0), 1])
        cnt = np.histogram(t[ok] - iv[i[ok], 0], edges)[0].astype(float)
    L = iv[:, 1] - iv[:, 0] if len(iv) else np.zeros(0)
    expo = np.clip(np.minimum(L[:, None], edges[None, 1:]) - edges[None, :-1], 0, None).sum(0)
    return cnt, expo


def _burst_q(t: np.ndarray, local_rate: np.ndarray, ici_max: float) -> float:
    """Burst continuation probability: share of intervals < ici_max above chance."""
    if len(t) < 5:
        return 0.0
    q_obs = float(np.mean(np.diff(t) < ici_max))
    p0 = float(np.mean(1 - np.exp(-ici_max * local_rate[:-1])))
    return float(np.clip((q_obs - p0) / max(1 - p0, 1e-6), 0.0, 0.8))


def estimate_params(trials: Sequence[Trial], cfg: Config | None = None) -> SimParams:
    """Per-trial parameters matched to the real trials (effects set to zero)."""
    cfg = cfg or Config()
    ici_max = float(_sopt(cfg, "burst_ici_max"))
    states = [trial_states(tr) for tr in trials]
    durs, gaps = [], []
    for st in states:
        f_in = st.starts_event["in"] & st.ends_event["in"]
        f_out = st.starts_event["out"] & st.ends_event["out"]
        durs.append(np.diff(st.iv["in"][f_in], axis=1).ravel())
        gaps.append(np.diff(st.iv["out"][f_out], axis=1).ravel())
    dur_floor = float(np.concatenate(durs).min())
    gap_floor = float(np.concatenate(gaps).min())

    # pooled relative sub chirp rate vs elapsed state time (relative to each
    # trial's own state rate, so dyads with many chirps do not set the shape)
    num = {"in": np.zeros(len(IN_EDGES) - 1), "out": np.zeros(len(OUT_EDGES) - 1)}
    den = {"in": np.zeros(len(IN_EDGES) - 1), "out": np.zeros(len(OUT_EDGES) - 1)}
    icis = []
    for tr, st in zip(trials, states):
        for s, edges in (("in", IN_EDGES), ("out", OUT_EDGES)):
            iv = st.iv[s]
            T = float(np.sum(iv[:, 1] - iv[:, 0]))
            x = tr.chirps_sub[_in_iv(tr.chirps_sub, iv)]
            rate = len(x) / T if T > 0 else 0.0
            ivs = iv[st.starts_event[s]]
            c, e = _elapsed_counts(x, ivs, edges)
            num[s] += c
            den[s] += rate * e
        v = tr.chirps_sub[(tr.chirps_sub >= tr.obs_start) & (tr.chirps_sub < tr.obs_end)]
        dv = np.diff(np.sort(v))
        icis.append(dv[(dv > 0.05) & (dv < ici_max)])
    shape = {s: np.clip((num[s] + 2.0) / (den[s] + 2.0), 0.05, 20.0) for s in num}

    tps = []
    for tr, st, d, g in zip(trials, states, durs, gaps):
        base, local = {}, []
        for s, edges in (("in", IN_EDGES), ("out", OUT_EDGES)):
            iv = st.iv[s]
            x = tr.chirps_sub[_in_iv(tr.chirps_sub, iv)]
            L = iv[:, 1] - iv[:, 0]
            integ = np.where(
                st.starts_event[s], _shape_integral(shape[s], edges, L), shape[s][-1] * L
            )
            base[s] = len(x) / integ.sum() if integ.sum() > 0 else 0.0
            if len(x):
                i = np.searchsorted(iv[:, 0], x, side="right") - 1
                sh = np.where(
                    st.starts_event[s][i], _shape_at(shape[s], edges, x - iv[i, 0]), shape[s][-1]
                )
                local.append(np.c_[x, base[s] * sh])
        loc = np.concatenate(local) if local else np.zeros((0, 2))
        loc = loc[np.argsort(loc[:, 0])]
        T_in = float(np.sum(np.diff(st.iv["in"], axis=1)))
        T_out = float(np.sum(np.diff(st.iv["out"], axis=1)))
        dom_in = int(_in_iv(tr.chirps_dom, st.iv["in"]).sum())
        dom_out = int(_in_iv(tr.chirps_dom, st.iv["out"]).sum())
        dv = tr.chirps_dom[_in_iv(tr.chirps_dom, st.valid)]
        dom_local = np.where(
            _in_iv(dv, st.iv["in"]), dom_in / max(T_in, 1e-9), dom_out / max(T_out, 1e-9)
        )
        n_contacts = int(_in_iv(tr.contacts, st.valid).sum())
        tps.append(
            TrialParams(
                recording=tr.recording,
                template=tr,
                dur=surv_table(d, dur_floor),
                gap=surv_table(g, gap_floor),
                sub_base_in=base["in"],
                sub_base_out=base["out"],
                dom_rate_in=dom_in / max(T_in, 1e-9),
                dom_rate_out=dom_out / max(T_out, 1e-9),
                sub_q=_burst_q(loc[:, 0], loc[:, 1], ici_max),
                dom_q=_burst_q(dv, dom_local, ici_max),
                contact_rate=n_contacts / max(T_out, 1e-9),
                n_chases=len(st.events["offset"]),
                n_sub_in=int(_in_iv(tr.chirps_sub, st.iv["in"]).sum()),
            )
        )
    return SimParams(
        trials=tps,
        shape_in=shape["in"],
        shape_out=shape["out"],
        burst_icis=np.concatenate(icis),
        dur_floor=dur_floor,
        gap_floor=gap_floor,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _offspring(
    parents: np.ndarray, q: float, icis: np.ndarray, rng: np.random.Generator,
    with_parent: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Parents plus geometric bursts (continuation probability q); with ``with_parent``
    also the parent time of every returned chirp (same random draws either way)."""
    if q <= 0 or len(parents) == 0:
        return (parents, parents.copy()) if with_parent else parents
    m = rng.geometric(1.0 - q, size=len(parents)) - 1
    tot = int(m.sum())
    if tot == 0:
        return (parents, parents.copy()) if with_parent else parents
    cs = np.cumsum(rng.choice(icis, size=tot))
    first = np.cumsum(m) - m  # index of each group's first interval
    base = np.zeros(len(parents))
    nz = m > 0
    base[nz] = np.where(first[nz] > 0, cs[np.maximum(first[nz] - 1, 0)], 0.0)
    kids = np.repeat(parents, m) + cs - np.repeat(base, m)
    if with_parent:
        return np.concatenate([parents, kids]), np.concatenate([parents, np.repeat(parents, m)])
    return np.concatenate([parents, kids])


def _state_clock_points(iv: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniform points on the union of intervals iv."""
    if n == 0 or len(iv) == 0:
        return np.zeros(0)
    L = iv[:, 1] - iv[:, 0]
    c = np.concatenate([[0.0], np.cumsum(L)])
    u = rng.uniform(0, c[-1], n)
    j = np.clip(np.searchsorted(c, u, side="right") - 1, 0, len(iv) - 1)
    return iv[j, 0] + (u - c[j])


def _flat_train(
    iv_in: np.ndarray,
    iv_out: np.ndarray,
    r_in: float,
    r_out: float,
    q: float,
    icis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    parts = []
    for iv, r in ((iv_in, r_in), (iv_out, r_out)):
        T = float(np.sum(iv[:, 1] - iv[:, 0])) if len(iv) else 0.0
        n = int(rng.poisson(r * (1 - q) * T)) if r > 0 else 0
        parts.append(_offspring(_state_clock_points(iv, n, rng), q, icis, rng))
    return np.sort(np.concatenate(parts))


def _annotate(
    ch: np.ndarray, sd: float, step: float, t0: float, rng: np.random.Generator,
    lag: float = 0.0,
) -> np.ndarray:
    """Chase boundaries as annotated: systematic offset delay ``lag`` (capped at 0.4 x
    the following gap), Gaussian error (truncated) and lattice rounding."""
    n = len(ch)
    if n == 0 or (sd <= 0 and step <= 0 and lag == 0):
        return ch
    on, off = ch[:, 0].copy(), ch[:, 1].copy()
    if lag != 0:
        g_next = np.r_[on[1:] - off[:-1], np.inf]
        off = off + np.clip(lag, -0.4 * (off - on), 0.4 * g_next)
        ch = np.c_[on, off]
    dur = off - on
    g_before = np.r_[np.inf, on[1:] - off[:-1]]
    g_after = np.r_[on[1:] - off[:-1], np.inf]
    if sd > 0:
        lim_on, lim_off = 0.4 * np.minimum(dur, g_before), 0.4 * np.minimum(dur, g_after)
        on = on + np.clip(rng.normal(0, sd, n), -lim_on, lim_on)
        off = off + np.clip(rng.normal(0, sd, n), -lim_off, lim_off)
    if step > 0:
        on = t0 + np.round((on - t0) / step) * step
        off = t0 + np.round((off - t0) / step) * step
    out = np.c_[on, off]
    for _ in range(3):  # revert chases whose boundaries collide
        seq = out.ravel()
        bad = np.flatnonzero(np.diff(seq) <= 1e-6)
        if len(bad) == 0:
            return out
        idx = np.unique(np.r_[bad // 2, (bad + 1) // 2])
        out[idx] = ch[idx]
    return ch


def draw_latencies(edges: np.ndarray, density: np.ndarray, n: int, rng: np.random.Generator
                   ) -> np.ndarray:
    """n latencies (s) from a piecewise-uniform density over the bins ``edges``."""
    if n <= 0:
        return np.zeros(0)
    e = np.asarray(edges, float)
    p = np.asarray(density, float)
    j = rng.choice(len(p), size=n, p=p / p.sum())
    return e[j] + rng.uniform(size=n) * (e[j + 1] - e[j])


def trigger_p_before(P: SimParams) -> float:
    """P(trigger chirp lands before the chase offset): chirp delay after the latent
    event, U(trigger_chirp_delay), shorter than the offset delay, U(trigger_offset_delay)."""
    (a, b), (c, d) = P.trigger_offset_delay, P.trigger_chirp_delay
    u = (np.arange(400) + 0.5) / 400
    off_d = a + (b - a) * u[:, None]
    ch_d = c + (d - c) * u[None, :]
    return float(np.mean(ch_d < off_d))


class _TrialSim:
    """Sequential episode simulation of one synthetic trial."""

    def __init__(
        self, tp: TrialParams, P: SimParams, bw: float, ss: np.random.SeedSequence, tag: str
    ) -> None:
        self.tp, self.P, self.bw, self.tag = tp, P, float(bw), tag
        tr = tp.template
        self.t0 = float(tr.obs_start)
        self.obs_end = float(tr.obs_end)
        self.K = int(np.floor((self.obs_end - self.t0) / bw + 1e-9))
        self.Kx = self.K + int(round(EXTRA_S / bw))
        self.g = P.effect_lags(bw)
        self.J = len(self.g)
        self.box = P.effect_kernel == "box"
        self.pad = self.J + 2
        self.cnt = np.zeros(self.pad + self.Kx + 1)
        # SeedSequence children are indexed, so the 7th stream (dyad effects)
        # leaves the first six (and all earlier scenarios) unchanged
        (self.r_chase, self.r_chirp, self.r_trig, self.r_obs, self.r_hid, self.r_dom,
         self.r_eff) = [np.random.default_rng(s) for s in ss.spawn(7)]
        # offset-locked chirps: an 8th child, drawn from only when they are on
        self.r_off = np.random.default_rng(ss.spawn(1)[0])
        z = self.r_eff.normal(size=2)  # always drawn: common random numbers
        self.b_off = float(P.beta_off) + float(P.beta_off_sd) * float(z[0])
        self.b_on = float(P.beta_on) + float(P.beta_on_sd) * float(z[1])
        self.sub: list[np.ndarray] = []
        self.n_offset_locked = 0
        self.trig: list[float] = []
        self.trig_in: list[bool] = []
        # hidden common trigger: pi = expected trigger chirps per chase, set so
        # that trigger chirps landing BEFORE the offset make up trigger_share of
        # the in-chase sub chirps (p_before = P(chirp delay < offset delay))
        share = float(P.trigger_share)
        pb = trigger_p_before(P) if share > 0 else 1.0
        self.pi_trig = (min(1.0, share * tp.n_sub_in / max(tp.n_chases * pb, 1e-9))
                        if share > 0 else 0.0)
        realized = self.pi_trig * pb * tp.n_chases / max(tp.n_sub_in, 1) if share > 0 else 0.0
        self.in_scale = tp.sub_mult_in * max(0.0, 1.0 - realized) / P.p_detect_in
        self.out_scale = tp.sub_mult_out / P.p_detect_out
        self.hidden = self._hidden_track() if P.hidden_frac > 0 else None
        # slow sub chirp rate modulation: a 9th stream, drawn from only when it is on
        self.mod = None
        if P.sub_mod_frac > 0 and P.sub_mod_mult != 1.0:
            phi, m = float(P.sub_mod_frac), float(P.sub_mod_mult)
            hi = self._two_state(np.random.default_rng(ss.spawn(1)[0]), phi, float(P.sub_mod_dur))
            self.mod = np.where(hi, m, 1.0) / (1 - phi + phi * m)

    # -- helpers -------------------------------------------------------------
    def _two_state(self, rng: np.random.Generator, phi: float, d1: float) -> np.ndarray:
        """Per-bin indicator of a two-state process: fraction phi of the time in the 'on'
        state, exponential episodes of mean d1 s (on) and d1 (1 - phi) / phi s (off)."""
        d0 = d1 * (1 - phi) / phi
        out = np.zeros(self.Kx + 1, bool)
        t, state = 0.0, bool(rng.uniform() < phi)
        T = (self.Kx + 1) * self.bw
        while t < T:
            dur = rng.exponential(d1 if state else d0)
            if state:
                a, b = int(t / self.bw), int(min(t + dur, T) / self.bw) + 1
                out[a:b] = True
            t += dur
            state = not state
        return out

    def _hidden_track(self) -> np.ndarray:
        P = self.P
        return self._two_state(self.r_hid, float(P.hidden_frac), float(P.hidden_mean_dur))

    def _add(self, t: np.ndarray, sign: float) -> None:
        if len(t) == 0:
            return
        idx = np.floor((np.asarray(t) - self.t0) / self.bw).astype(np.int64) + self.pad
        m = (idx >= 0) & (idx < len(self.cnt))
        np.add.at(self.cnt, idx[m], sign)

    def _x(self, ka: int, kb: int) -> np.ndarray:
        """Effect covariate for bins ka..kb-1 from chirps in bins k - J .. k - 1."""
        seg = self.cnt[self.pad + ka - self.J : self.pad + kb - 1]
        s = np.convolve(seg, self.g)[self.J - 1 : self.J - 1 + kb - ka]
        return (s > 0.5).astype(float) if self.box else s

    def _gen_sub(self, state: str, t_s: float, first: bool, ga: float, gb: float
                 ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sub chirps in [ga, gb) for an episode of ``state`` that began at t_s (with
        burst_cross: sorted chirps and the parent time of each)."""
        P, tp, bw, t0 = self.P, self.tp, self.bw, self.t0
        ka = int(np.floor((ga - t0) / bw))
        kb = int(np.ceil((gb - t0) / bw - 1e-9))
        k = np.arange(ka, kb)
        lo = np.maximum(t0 + k * bw, ga)
        hi = np.minimum(t0 + (k + 1) * bw, gb)
        expo = np.clip(hi - lo, 0, None)
        e = 0.5 * (lo + hi) - t_s
        if state == "in":
            rate = tp.sub_base_in * self.in_scale * _shape_at(P.shape_in, IN_EDGES, e)
        else:
            sh = np.full(len(e), P.shape_out[-1]) if first else _shape_at(P.shape_out, OUT_EDGES, e)
            rate = tp.sub_base_out * self.out_scale * sh
            if self.hidden is not None:
                m = float(P.hidden_chirp_mult)
                phi = float(P.hidden_frac)
                rate = (
                    rate
                    * np.where(self.hidden[np.clip(k, 0, self.Kx)], m, 1.0)
                    / (1 - phi + phi * m)
                )
        if self.mod is not None:
            rate = rate * self.mod[np.clip(k, 0, self.Kx)]
        q = tp.sub_q if P.bursts else 0.0
        n = self.r_chirp.poisson(rate * expo * (1 - q))
        par = np.repeat(lo, n) + self.r_chirp.uniform(size=int(n.sum())) * np.repeat(expo, n)
        if P.burst_cross:
            x, pa = _offspring(par, q, P.burst_icis, self.r_chirp, with_parent=True)
            o = np.argsort(x, kind="stable")
            return x[o], pa[o]
        return np.sort(_offspring(par, q, P.burst_icis, self.r_chirp))

    # -- episodes ------------------------------------------------------------
    def _episode(self, state: str, t_s: float, first: bool) -> float | None:
        """Simulate one episode from t_s; return its end time (None: censored)."""
        P, tp, bw, t0 = self.P, self.tp, self.bw, self.t0
        tab = tp.dur if state == "in" else tp.gap
        beta = self.b_off if state == "in" else self.b_on
        h_mult = float(tp.h_mult_in if state == "in" else tp.h_mult_out)
        kmax = self.Kx if state == "in" else self.K
        E = float(self.r_chase.exponential())
        u_ev = float(self.r_chase.uniform())
        ka = int(np.ceil((t_s - t0) / bw - 1e-9))  # first bin with start >= t_s
        ga, cum = t_s, 0.0
        chirps: list[np.ndarray] = []
        parents: list[np.ndarray] = []
        while ka < kmax:
            kb = min(ka + CHUNK[state], kmax)
            gb = t0 + kb * bw
            new = self._gen_sub(state, t_s, first, ga, gb)
            if P.burst_cross:
                new, pa = new
                parents.append(pa)
            self._add(new, +1.0)
            chirps.append(new)
            e = t0 + np.arange(ka, kb) * bw - t_s
            dL = np.maximum(tab.logsurv(e) - tab.logsurv(e + bw), 0.0) * h_mult
            if state == "out" and self.hidden is not None:
                m, phi = float(P.hidden_onset_mult), float(P.hidden_frac)
                dL = dL * np.where(self.hidden[ka:kb], m, 1.0) / (1 - phi + phi * m)
            if beta != 0.0:
                dL = dL * np.exp(beta * self._x(ka, kb))
            c = cum + np.cumsum(dL)
            i = int(np.searchsorted(c, E, side="left"))
            if i < len(c):
                t_ev = t0 + (ka + i) * bw + u_ev * bw
                allc = np.concatenate(chirps)
                late = allc >= t_ev
                if P.burst_cross:  # bursts begun before the state change continue after it
                    late &= np.concatenate(parents) >= t_ev
                self._add(allc[late], -1.0)
                self.sub.append(allc[~late])
                return t_ev
            cum, ka, ga = float(c[-1]), kb, gb
        allc = np.concatenate(chirps) if chirps else np.zeros(0)
        self.sub.append(allc[allc < t0 + kmax * bw])
        return None

    def _trigger(self, t_on: float, t_off: float) -> None:
        P = self.P
        u = self.r_trig.uniform(size=3)  # always drawn: common random numbers
        if self.pi_trig <= 0 or u[0] >= self.pi_trig:
            return
        (a, b), (c, d) = P.trigger_offset_delay, P.trigger_chirp_delay
        T = max(t_off - (a + (b - a) * u[1]), t_on)
        tc = T + c + (d - c) * u[2]
        self._add(np.array([tc]), +1.0)
        self.sub.append(np.array([tc]))
        self.trig.append(tc)
        self.trig_in.append(bool(tc < t_off))

    def _offset_chirps(self, t_off: float) -> None:
        """Response burst and symmetric cluster at the true offset t_off (module doc)."""
        P, tp, rng = self.P, self.tp, self.r_off
        new = []
        # with the slow modulation (drift) on, offset-locked chirping follows the same state
        mf = 1.0
        if self.mod is not None:
            mf = float(self.mod[int(np.clip(np.floor((t_off - self.t0) / self.bw), 0, self.Kx))])
        if tp.resp_n > 0 and P.resp_density is not None:
            mu = float(tp.resp_n) * mf
            if P.resp_bout_mult != 1.0:
                w = float(P.resp_bout_window)
                recent = any(bool(np.any((x >= t_off - w) & (x < t_off))) for x in self.sub[-6:])
                mu *= float(P.resp_bout_mult) if recent else 1.0
            new.append(t_off + draw_latencies(P.resp_edges, P.resp_density,
                                              int(rng.poisson(mu)), rng))
        if tp.sym_n > 0 and P.sym_density is not None:
            m = float(P.sym_m)
            if rng.uniform() < min(1.0, tp.sym_n * mf / m):
                a, b = rng.poisson(m, size=2)
                new.append(t_off - draw_latencies(P.sym_edges, P.sym_density, int(a), rng))
                new.append(t_off + draw_latencies(P.sym_edges, P.sym_density, int(b), rng))
        if new:
            x = np.concatenate(new)
            self._add(x, +1.0)
            self.sub.append(x)
            self.n_offset_locked += len(x)

    def run(self) -> Trial:
        P, tp, t0 = self.P, self.tp, self.t0
        tr = tp.template
        chases: list[tuple[float, float]] = []
        t, first = t0, True
        while True:
            t_on = self._episode("out", t, first)
            first = False
            if t_on is None or t_on >= self.obs_end:
                break
            t_off = self._episode("in", t_on, False)
            censored = t_off is None
            if t_off is None:
                t_off = t0 + self.Kx * self.bw
            chases.append((t_on, t_off))
            self._trigger(t_on, t_off)
            if not censored:
                self._offset_chirps(t_off)
            if t_off >= self.obs_end:
                break
            t = t_off
        true_ch = np.array(chases, float).reshape(-1, 2)
        t_hi = self.obs_end
        sub = np.sort(np.concatenate(self.sub)) if self.sub else np.zeros(0)
        sub = sub[(sub >= t0) & (sub < t_hi)]
        in_iv = np.c_[np.maximum(true_ch[:, 0], t0), np.minimum(true_ch[:, 1], t_hi)]
        in_iv = in_iv[in_iv[:, 1] > in_iv[:, 0]]
        out_iv = _complement(in_iv, t0, t_hi)
        dom = _flat_train(
            in_iv,
            out_iv,
            tp.dom_rate_in / P.p_detect_in,
            tp.dom_rate_out / P.p_detect_out,
            tp.dom_q if P.bursts else 0.0,
            P.burst_icis,
            self.r_dom,
        )
        dom = dom[(dom >= t0) & (dom < t_hi)]
        T_out = float(np.sum(out_iv[:, 1] - out_iv[:, 0]))
        contacts = np.sort(
            _state_clock_points(
                out_iv, int(self.r_obs.poisson(tp.contact_rate * T_out)), self.r_obs
            )
        )
        trig = np.array(self.trig)
        # observation model: detection (by true state), then annotation error
        n_true_sub = len(sub)
        if P.p_detect_in < 1 or P.p_detect_out < 1:
            p_sub = np.where(_in_iv(sub, in_iv), P.p_detect_in, P.p_detect_out)
            sub = sub[self.r_obs.uniform(size=len(sub)) < p_sub]
            p_dom = np.where(_in_iv(dom, in_iv), P.p_detect_in, P.p_detect_out)
            dom = dom[self.r_obs.uniform(size=len(dom)) < p_dom]
        ann = _annotate(true_ch, float(P.boundary_sd), float(P.boundary_step), t0, self.r_obs,
                        lag=float(P.offset_lag))
        t_end = max(self.obs_end, float(ann[:, 1].max()) if len(ann) else self.obs_end)
        off_true = true_ch[true_ch[:, 1] < t_hi, 1]
        pre = np.zeros(0, bool)
        if len(trig) and len(off_true):
            j = np.searchsorted(off_true, trig, side="left")
            pre = (j < len(off_true)) & (off_true[np.minimum(j, len(off_true) - 1)] - trig <= 1.0)
        meta = dict(
            synthetic=True,
            template=tr.recording,
            tag=self.tag,
            n_true_sub=n_true_sub,
            n_trigger_chirps=int(len(trig)),
            n_trigger_in_chase=int(np.sum(self.trig_in)),
            n_trigger_before_offset=int(pre.sum()),
            n_true_offsets=int(len(off_true)),
            pi_trigger=self.pi_trig,
            beta_off_trial=self.b_off,
            beta_on_trial=self.b_on,
            n_offset_locked=self.n_offset_locked,
        )
        return Trial(
            recording=f"{tr.recording}#{self.tag}",
            group=tr.group,
            dom_id=tr.dom_id,
            sub_id=tr.sub_id,
            dom_fish=tr.dom_fish,
            sub_fish=tr.sub_fish,
            t_start=tr.t_start,
            t_end=t_end,
            chases=ann,
            contacts=contacts,
            chirps_dom=dom,
            chirps_sub=sub,
            rises_dom=None,
            rises_sub=None,
            eodf_t=tr.eodf_t,
            eodf_dom=tr.eodf_dom,
            eodf_sub=tr.eodf_sub,
            size_dom=tr.size_dom,
            size_sub=tr.size_sub,
            sex_dom=tr.sex_dom,
            sex_sub=tr.sex_sub,
            meta=meta,
            obs_start=t0,
            obs_end=self.obs_end,
        )


def _simulate(params: SimParams, bw: float, ss: np.random.SeedSequence, tag: str) -> list[Trial]:
    kids = ss.spawn(len(params.trials))
    return [_TrialSim(tp, params, bw, k, tag).run() for tp, k in zip(params.trials, kids)]


def simulate_trials(
    params: SimParams,
    cfg: Config,
    rng: np.random.Generator | np.random.SeedSequence,
    tag: str = "sim",
) -> list[Trial]:
    """One synthetic dataset (one Trial per real trial) on cfg.bin_width bins."""
    ss = (
        rng
        if isinstance(rng, np.random.SeedSequence)
        else np.random.SeedSequence(int(rng.integers(2**63)))
    )
    return _simulate(params, cfg.bin_width, ss, tag)


# ---------------------------------------------------------------------------
# Summary statistics of a trial (real or synthetic)
# ---------------------------------------------------------------------------
def trial_stats(tr: Trial) -> dict[str, Any]:
    """Counts and timing statistics used to check the simulation against data."""
    st = trial_states(tr)
    f_in = st.starts_event["in"] & st.ends_event["in"]
    f_out = st.starts_event["out"] & st.ends_event["out"]
    d = np.diff(st.iv["in"][f_in], axis=1).ravel()
    g = np.diff(st.iv["out"][f_out], axis=1).ravel()
    sub = np.sort(tr.chirps_sub[_in_iv(tr.chirps_sub, st.valid)])
    dom = tr.chirps_dom[_in_iv(tr.chirps_dom, st.valid)]
    T_in, T_out = st.state_time("in"), st.state_time("out")

    def pre(ev: np.ndarray, w: float = 1.0) -> int:
        if len(ev) == 0:
            return 0
        n = np.searchsorted(sub, ev, "left") - np.searchsorted(sub, ev - w, "left")
        return int(np.sum(n > 0))

    ici = np.diff(sub)
    n_sub_in = int(_in_iv(sub, st.iv["in"]).sum())
    n_off, n_on = len(st.events["offset"]), len(st.events["onset"])
    k_off, k_on = pre(st.events["offset"]), pre(st.events["onset"])
    return dict(
        n_chases=n_off,
        n_onsets=n_on,
        median_dur=float(np.median(d)) if len(d) else np.nan,
        mean_dur=float(np.mean(d)) if len(d) else np.nan,
        median_gap=float(np.median(g)) if len(g) else np.nan,
        mean_gap=float(np.mean(g)) if len(g) else np.nan,
        time_in=T_in,
        time_out=T_out,
        n_sub=len(sub),
        n_sub_in=n_sub_in,
        n_sub_out=len(sub) - n_sub_in,
        rate_sub_in=n_sub_in / T_in if T_in else np.nan,
        rate_sub_out=(len(sub) - n_sub_in) / T_out if T_out else np.nan,
        n_dom=len(dom),
        n_dom_in=int(_in_iv(dom, st.iv["in"]).sum()),
        n_contacts=int(_in_iv(tr.contacts, st.valid).sum()),
        frac_ici_lt1=float(np.mean(ici < 1.0)) if len(ici) else np.nan,
        n_off_sub1s=k_off,
        n_on_sub1s=k_on,
        frac_off_sub1s=k_off / n_off if n_off else np.nan,
        frac_on_sub1s=k_on / n_on if n_on else np.nan,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    check: str  # recovery | no_effect | observation | trigger | hidden
    overrides: dict[str, Any]
    n_reps: int
    models: dict[str, tuple[str, ...]]  # kind -> models
    label: str
    x: float = np.nan  # sweep value (trigger share, lag)
    seed_key: str = "base"
    scan_reps: int = -1  # >= 0: reps from this index on fit only the 1 s and kernel models
    calib: str = ""  # per-trial rates calibrated under this effect (calibrate_params)

    def models_for(self, rep: int) -> dict[str, tuple[str, ...]]:
        if 0 <= self.scan_reps <= rep:
            return {
                k: tuple(m for m in v if m in ("tau_1000ms", "kernel"))
                for k, v in self.models.items()
            }
        return self.models


def het_sd(cfg: Config, kind: str) -> float:
    """Between-dyad SD of the chirp effect in the 'het' scenarios (cfg.extra
    'sim_het_sd_<kind>', set by the main script to the random-effects estimate
    of the real data; else a default near it)."""
    return float(cfg.extra.get(f"sim_het_sd_{kind}", {"offset": 0.4, "onset": 1.0}[kind]))


MIXED_SCENARIOS = ("no_effect", "recovery", "het_no_effect", "het_recovery")


def default_scenarios(cfg: Config) -> list[Scenario]:
    """The scenario set of run_simulation_checks (sizes scale with cfg.n_sims).

    n = n_sims replicates for no_effect, recovery and the two heterogeneous
    scenarios; 2n/5 for the observation-error scenarios; n/4 for the
    offset-only trigger / annotation-lag sweeps and the exp kernel; n/5 for
    the hidden-state scenarios. The tau scan (all models) runs on all trigger,
    lag and hidden-state replicates and on the first n/4 recovery and
    annotation replicates. Effect scenarios use per-trial rates calibrated
    under the effect ('calib'), so chirp and chase counts match the data.
    """
    n = int(cfg.n_sims)
    if cfg.quick:
        n = min(n, 8)
    obs_n, quarter, fifth = max(2, 2 * n // 5), max(2, n // 4), max(2, n // 5)
    main = ("tau_1000ms", "kernel")
    allm = tuple(hz.model_names(cfg))
    both = {"offset": main, "onset": main}
    eff = dict(beta_off=LOG2, beta_on=-LOG2)
    het = dict(beta_off_sd=het_sd(cfg, "offset"), beta_on_sd=het_sd(cfg, "onset"))
    ann = dict(boundary_sd=0.3, boundary_step=0.5)
    det = dict(p_detect_in=0.6, p_detect_out=0.85)
    sc = [
        Scenario("no_effect", "no_effect", {}, n, both, "no chirp effect"),
        Scenario("recovery", "recovery", eff, n, {"offset": allm, "onset": allm},
                 "HR 2 (offset), 0.5 (onset), 1 s box", scan_reps=quarter, calib="box"),
        Scenario("annotation", "observation", {**eff, **ann}, obs_n,
                 {"offset": allm, "onset": allm},
                 "as recovery + boundary error SD 0.3 s, 0.5 s lattice", scan_reps=quarter,
                 calib="box"),
        Scenario("detection", "observation", {**eff, **det}, obs_n, both,
                 "as recovery + chirp detection 0.6 in / 0.85 out of chases", calib="box"),
        Scenario("recovery_exp", "recovery",
                 dict(beta_off=LOG2, beta_on=-LOG2, effect_kernel="exp", effect_window=0.5),
                 quarter, {"offset": allm}, "HR 2 per chirp, exp kernel (0.5 s), offset only",
                 calib="exp"),
        Scenario("het_no_effect", "no_effect", het, n, both,
                 f"no mean effect; dyad effects SD {het['beta_off_sd']:.2f} (offset), "
                 f"{het['beta_on_sd']:.2f} (onset)"),
        Scenario("het_recovery", "recovery", {**eff, **het}, n, both,
                 f"mean HR 2 / 0.5, dyad effects SD {het['beta_off_sd']:.2f} / "
                 f"{het['beta_on_sd']:.2f}", calib="box"),
    ]
    for phi in (0.1, 0.25, 0.5, 1.0):
        sc.append(Scenario(f"trigger_{phi:g}", "trigger", dict(trigger_share=phi), quarter,
                           {"offset": allm}, f"hidden trigger, share {phi:g}", x=phi))
    sc.append(Scenario("trigger_0.5_slow", "trigger",
                       dict(trigger_share=0.5, trigger_offset_delay=(0.5, 1.5)), quarter,
                       {"offset": allm}, "hidden trigger, share 0.5, offset 0.5-1.5 s after",
                       x=0.5))
    sc.append(Scenario("trigger_0.25_annotation", "trigger", dict(trigger_share=0.25, **ann),
                       quarter, {"offset": allm}, "hidden trigger, share 0.25, + boundary error",
                       x=0.25))
    for lag in (0.25, 0.5, 0.75):
        sc.append(Scenario(f"lag_{lag:g}", "lag", dict(offset_lag=lag), quarter,
                           {"offset": allm},
                           f"no chirp effect; every offset annotated {lag:g} s late", x=lag))
    if _sopt(cfg, "hidden"):
        sc.append(Scenario(
            "hidden_state", "hidden",
            dict(hidden_frac=0.2, hidden_mean_dur=30.0, hidden_chirp_mult=4.0,
                 hidden_onset_mult=0.25),
            fifth, {"onset": allm},
            "latent state (20% of time, 30 s): sub chirps x4, onset hazard x0.25"))
        sc.append(Scenario(
            "hidden_state_short", "hidden",
            dict(hidden_frac=0.1, hidden_mean_dur=3.0, hidden_chirp_mult=6.0,
                 hidden_onset_mult=0.1),
            fifth, {"onset": allm},
            "latent state (10% of time, 3 s): sub chirps x6, onset hazard x0.1"))
    return sc


def _truths(P: SimParams, kind: str, bw: float, lags: np.ndarray) -> dict[str, Any]:
    beta = float(P.beta_off if kind == "offset" else P.beta_on)
    g = P.effect_lags(bw)
    gl = np.zeros(len(lags))
    j = np.rint(lags / bw).astype(int)
    ok = j <= len(g)
    gl[ok] = g[j[ok] - 1]
    out: dict[str, Any] = {
        "kernel_sub_mean": beta * float(gl.mean()),
        "kernel_curve": beta * gl,
        "kernel_dom_mean": 0.0,
    }
    for ms in ("250", "500", "1000", "2000", "3000", "5000"):
        tau = int(ms) / 1000
        exact = P.effect_kernel == "box" and abs(tau - P.effect_window) < 1e-9
        out[f"sub_any_{ms}ms"] = beta if (beta == 0 or exact) else np.nan  # no exact truth
        out[f"dom_any_{ms}ms"] = 0.0
    g_ind = hz.indicator_column(hz.guard_model(Config()), "sub")
    out[str(g_ind)] = 0.0 if beta == 0 else np.nan
    out["dyad_sd_sub_effect"] = float(P.beta_off_sd if kind == "offset" else P.beta_on_sd)
    return out


# ---------------------------------------------------------------------------
# Fitting (same code path as hz_hazard.fit_models' glm_fe backend)
# ---------------------------------------------------------------------------
def fit_fe(
    design: pd.DataFrame, cfg: Config, model: str, ctrl: Any = None
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """glm_fe fit of one model: sub/dom chirp terms with model-based and
    cluster-robust inference, plus the sub kernel curve (kernel model).

    Mirrors hz_hazard._fit_one(backend='glm_fe') (same matrices, solver and
    back-transformation) but also returns the model-based covariance.
    """
    ctrl = hz._control_block(design, cfg, fe=True) if ctrl is None else ctrl
    M = hz._assemble(design, ctrl, hz._chirp_block(design, model), fe=True)
    y = design["y"].to_numpy().astype(float)
    codes, _ = hz._groups(design)
    link = str(hz._opt(cfg, "glm_link"))
    frac = float(design.attrs.get("control_fraction", 1.0))
    w = design["w"].to_numpy() if frac < 1 and link == "cloglog" else None
    off = np.full(len(y), np.log(1 / frac)) if frac < 1 and link == "logit" else None
    res = hz.fit_glm(M.X, y, link=link, offset=off, weights=w, groups=codes)
    names, b, C_cl = hz._natural(res.params, res.cov_cluster, M)
    _, _, C_mb = hz._natural(res.params, res.cov, M)
    df_t = max(res.n_groups - 1, 1)
    t_crit, z_crit = float(stats.t.ppf(0.975, df_t)), float(stats.norm.ppf(0.975))
    idx = {nm: i for i, nm in enumerate(names)}
    B = design.attrs["basis"]
    rows: list[dict[str, Any]] = []

    def add(term: str, c: np.ndarray, cols: list[int]) -> None:
        est = float(c @ b[cols])
        s_mb = float(np.sqrt(max(c @ C_mb[np.ix_(cols, cols)] @ c, 0)))
        s_cl = float(np.sqrt(max(c @ C_cl[np.ix_(cols, cols)] @ c, 0)))
        rows.append(
            dict(
                term=term,
                estimate=est,
                se_naive=s_mb,
                se_cluster=s_cl,
                p_naive=float(2 * stats.norm.sf(abs(est) / s_mb)) if s_mb > 0 else np.nan,
                p_cluster=float(2 * stats.t.sf(abs(est) / s_cl, df_t)) if s_cl > 0 else np.nan,
                lo_naive=est - z_crit * s_mb,
                hi_naive=est + z_crit * s_mb,
                lo_cluster=est - t_crit * s_cl,
                hi_cluster=est + t_crit * s_cl,
            )
        )

    curve = None
    for s in hz.SENDERS:
        if model == "kernel":
            hn = [f"{s}_h{i + 1}" for i in range(B.shape[1])]
            if not all(h in idx for h in hn):
                continue
            cols = [idx[h] for h in hn]
            add(f"kernel_{s}_mean", B.mean(0), cols)
            bb = b[cols]
            q = len(cols)
            try:
                w_mb = float(bb @ np.linalg.solve(C_mb[np.ix_(cols, cols)], bb))
                w_cl = float(bb @ np.linalg.solve(C_cl[np.ix_(cols, cols)], bb))
            except np.linalg.LinAlgError:
                w_mb = w_cl = np.nan
            rows.append(
                dict(
                    term=f"kernel_{s}_joint",
                    estimate=np.nan,
                    p_naive=float(stats.chi2.sf(w_mb, q)),
                    p_cluster=float(stats.f.sf(w_cl / q, q, df_t)),
                )
            )
            if s == "sub":
                curve = B @ bb
        else:
            nm = str(hz.indicator_column(model, s))
            if nm in idx:
                add(nm, np.array([1.0]), [idx[nm]])
    ind = hz.indicator_column(model, "sub")
    n_on = int(((design[ind] != 0) & (design["y"] == 1)).sum()) if ind in design else np.nan
    for r in rows:
        r.update(
            model=model,
            n_bins=len(y),
            n_events=int(y.sum()),
            n_events_on=n_on,
            n_recordings=res.n_groups,
            converged=bool(res.converged),
        )
    return rows, curve


def re_meta_rows(design: pd.DataFrame, cfg: Config, model: str = "tau_1000ms"
                 ) -> list[dict[str, Any]]:
    """Per-dyad O/E random-effects meta-analysis (hz_hazard.dyad_oe + re_meta) as
    estimate rows: the mean log HR (CI and p in the *_naive slots), the
    between-dyad SD and the homogeneity G test."""
    tab = hz.dyad_oe(design, cfg, model)
    r = hz.re_meta(tab["observed"].to_numpy(), tab["expected"].to_numpy())
    k = int(r["k"])
    crit = float(stats.t.ppf(0.975, max(k - 1, 1)))
    w = r["mu_high"] - r["mu_low"]
    base = dict(model=model, backend="re_meta", n_events_on=int(r["O"]), n_recordings=k,
                converged=bool(np.isfinite(r["mu"])))
    return [
        dict(base, term=str(hz.indicator_column(model, "sub")), estimate=r["mu"],
             se_naive=w / (2 * crit) if np.isfinite(w) else np.nan, lo_naive=r["mu_low"],
             hi_naive=r["mu_high"], p_naive=r["p"]),
        dict(base, term="dyad_sd_sub_effect", estimate=r["tau"], p_naive=r["p_tau"]),
        dict(base, term="heterogeneity_G", estimate=r["G"], p_naive=r["p_G"]),
    ]


# ---------------------------------------------------------------------------
# Worker tasks
# ---------------------------------------------------------------------------
_SIM: dict[str, Any] = {}  # read-only state inherited by forked workers


def _worker_init() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:  # pragma: no cover
        pass


def _task(spec: dict[str, Any]) -> dict[str, Any]:
    """Simulate (or take the real trials), build designs, fit, optional jitter/mixed."""
    G = _SIM
    cfg: Config = G["cfg"]
    t_task = time.perf_counter()
    name, rep = spec["scenario"], spec["rep"]
    out: dict[str, Any] = dict(est=[], kern=[], jit=[], stats=[], meta=[], timing=[])
    if name == "observed":
        trials = G["trials"]
        P = None
        models = {"offset": tuple(hz.model_names(cfg)), "onset": tuple(hz.model_names(cfg))}
    else:
        scen: Scenario = G["scenarios"][name]
        P = replace(G["params"], **scen.overrides)
        if scen.calib:
            P = replace(P, trials=G["calib"][scen.calib])
        ss = np.random.SeedSequence([int(cfg.seed), zlib.crc32(scen.seed_key.encode()), int(rep)])
        trials = _simulate(P, cfg.bin_width, ss, tag=f"{name}.{rep}")
        models = scen.models_for(int(rep))
    base = dict(scenario=name, rep=rep)
    t_sim = time.perf_counter() - t_task
    if spec.get("stats", True):
        tst = [trial_stats(tr) for tr in trials]
        for tr, x in zip(trials, tst):
            out["stats"].append(dict(base, recording=tr.meta.get("template", tr.recording), **x))
        if P is not None:
            m = [tr.meta for tr in trials]
            n_in = sum(x["n_sub_in"] for x in tst)
            out["meta"].append(
                dict(
                    base,
                    n_trigger_chirps=sum(x["n_trigger_chirps"] for x in m),
                    n_trigger_in_chase=sum(x["n_trigger_in_chase"] for x in m),
                    n_trigger_before_offset=sum(x["n_trigger_before_offset"] for x in m),
                    n_true_offsets=sum(x["n_true_offsets"] for x in m),
                    n_sub_in=n_in,
                    n_true_sub=sum(x["n_true_sub"] for x in m),
                )
            )
    for kind in spec.get("kinds", list(models)):
        mods = models.get(kind, ())
        if not mods and not spec.get("jitter"):
            continue
        taus = tuple(sorted({hz._tau_of(m) for m in mods if m.startswith("tau_")} | {1.0}))
        cfg_full = cfg.updated(tau_grid=taus)
        cfg_k = cfg_full
        if kind == "onset" and P is not None:  # case-control subsampling (see module doc)
            cfg_k = cfg_full.updated(onset_control_fraction=float(_sopt(cfg, "onset_fraction")))
        kb = dict(base, kind=kind)
        t0 = time.perf_counter()
        d = hz.build_design(trials, kind, cfg_k)
        lags = d.attrs["lags"]
        tru = _truths(P, kind, cfg.bin_width, lags) if P is not None else {}
        if spec.get("jitter"):
            n_draws, null = spec["jitter"]
            jt = hz.jitter_test(
                trials, kind, cfg_k, design=d, n_jitter=n_draws, null=null, n_workers=1
            )
            for _, r in hz.summarize_jitter(jt).iterrows():
                z = (r["observed"] - r["null_mean"]) / r["null_sd"] if r["null_sd"] > 0 else np.nan
                out["jit"].append(
                    dict(
                        kb,
                        null=null,
                        stat=r["stat"],
                        observed=r["observed"],
                        null_mean=r["null_mean"],
                        null_sd=r["null_sd"],
                        p_rank=r["p_two_sided"],
                        z=z,
                        p_z=float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                        n_draws=int(r["n_jitter"]),
                        control_fraction=d.attrs["control_fraction"],
                        seconds=time.perf_counter() - t0,
                    )
                )
            out["timing"].append(dict(kb, step=f"jitter_{null}", seconds=time.perf_counter() - t0))
            continue
        designs = [("glm_fe", d, cfg_k, mods)]
        if spec.get("full_check") and d.attrs["control_fraction"] < 1:
            # the same replicate on all bins: subsampled vs full-data estimates
            designs.append(
                (
                    "glm_fe_full",
                    hz.build_design(trials, kind, cfg_full),
                    cfg_full,
                    ("tau_1000ms", "kernel"),
                )
            )
        for backend, dd, cc, mm in designs:
            ctrl = hz._control_block(dd, cc, fe=True)
            for model in mm:
                t1 = time.perf_counter()
                rows, curve = fit_fe(dd, cc, model, ctrl)
                for r in rows:
                    r.update(
                        kb,
                        backend=backend,
                        truth=tru.get(r["term"], np.nan),
                        control_fraction=dd.attrs["control_fraction"],
                        fit_s=time.perf_counter() - t1,
                    )
                out["est"].extend(rows)
                if curve is not None and backend == "glm_fe":
                    tc = tru.get("kernel_curve", np.full(len(lags), np.nan))
                    out["kern"].extend(
                        dict(kb, lag=float(l), estimate=float(v), truth=float(t))
                        for l, v, t in zip(lags, curve, tc)
                    )
            del ctrl
        if "tau_1000ms" in mods:  # two-stage random-effects meta-analysis (hz_hazard)
            t1 = time.perf_counter()
            out["est"].extend(dict(r, **kb, truth=tru.get(r["term"], np.nan),
                                   control_fraction=d.attrs["control_fraction"],
                                   fit_s=time.perf_counter() - t1)
                              for r in re_meta_rows(d, cfg_k))
        if spec.get("mixed") or spec.get("full_check"):
            # primary backend of the real analysis. Logit case-control sampling
            # leaves slopes unchanged, so the subsampled onset design is valid;
            # the full-check replicates also fit it on all bins.
            dd, cc = (designs[-1][1], designs[-1][2]) if spec.get("full_check") else (d, cfg_k)
            t1 = time.perf_counter()
            e = hz.fit_models(dd, kind, cc, models=["tau_1000ms"], backends=["mixed_laplace"])
            r = e[e["term"] == "sub_any_1000ms"].iloc[0]
            out["est"].append(
                dict(
                    kb,
                    model="tau_1000ms",
                    backend="mixed_laplace",
                    term="sub_any_1000ms",
                    estimate=r["estimate"],
                    se_naive=r["se"],
                    lo_naive=r["ci_low"],
                    hi_naive=r["ci_high"],
                    p_naive=r["p"],
                    truth=tru.get("sub_any_1000ms", np.nan),
                    n_bins=r["n_bins"],
                    n_events=r["n_events"],
                    converged=bool(r["converged"]),
                    control_fraction=dd.attrs["control_fraction"],
                    fit_s=time.perf_counter() - t1,
                )
            )
        del d, designs
        out["timing"].append(dict(kb, step="design+fits", seconds=time.perf_counter() - t0))
    out["timing"].append(dict(base, kind="", step="simulate", seconds=t_sim))
    out["seconds"] = time.perf_counter() - t_task
    return out


def _task_list(cfg: Config, scenarios: list[Scenario]) -> list[dict[str, Any]]:
    """All work items, most expensive first (costs: single-thread seconds, rough)."""
    q = cfg.quick
    tasks: list[tuple[float, dict[str, Any]]] = [(60.0, dict(scenario="observed", rep=0))]
    n_full = min(int(_sopt(cfg, "mixed_reps")), 1 if q else 99)
    for sc in scenarios:
        for rep in range(sc.n_reps):
            m = sc.models_for(rep)
            full = sc.name == "recovery" and rep < n_full
            mixed = sc.name in MIXED_SCENARIOS
            cost = 2.5 + 0.7 * len(m.get("offset", ())) + 1.0 * len(m.get("onset", ()))
            cost += 30.0 if full else (6.0 if mixed else 0.0)
            tasks.append((cost, dict(scenario=sc.name, rep=rep, mixed=mixed, full_check=full)))
    ref = next((s for s in scenarios if s.name == "no_effect"), None)
    if ref is not None:
        for kind in KINDS:
            n_rep = int(_sopt(cfg, f"jitter_reps_{kind}"))
            n_dr = int(_sopt(cfg, f"jitter_draws_{kind}"))
            if q:
                n_rep, n_dr = min(n_rep, 2), min(n_dr, 19)
            for rep in range(min(n_rep, ref.n_reps)):
                for nl in ("shift", "matched"):
                    tasks.append(
                        (
                            2.0 + 0.6 * n_dr,
                            dict(
                                scenario="no_effect",
                                rep=rep,
                                kinds=[kind],
                                jitter=(n_dr, nl),
                                stats=False,
                            ),
                        )
                    )
    tasks.sort(key=lambda x: -x[0])
    return [t for _, t in tasks]


# ---------------------------------------------------------------------------
# Calibration of per-trial rates under a causal effect
# ---------------------------------------------------------------------------
CALIB_EFFECTS: dict[str, dict[str, Any]] = {
    "box": dict(beta_off=LOG2, beta_on=-LOG2),
    "exp": dict(beta_off=LOG2, beta_on=-LOG2, effect_kernel="exp", effect_window=0.5),
}


def _calib_worker(rep: int) -> list[dict[str, Any]]:
    G = _SIM
    # same seeds in every iteration (common random numbers): changes between
    # iterations reflect the new multipliers, not simulation noise
    ss = np.random.SeedSequence([int(G["cfg"].seed), zlib.crc32(G["calib_key"].encode()),
                                 int(rep)])
    trials = _simulate(G["calib_P"], G["cfg"].bin_width, ss, tag=f"calib.{rep}")
    extra = G.get("calib_extra")
    return [dict(recording=tr.meta["template"], **trial_stats(tr), **(extra(tr) if extra else {}))
            for tr in trials]


def calibrate_params(
    params: SimParams, overrides: dict[str, Any], trials: Sequence[Trial], cfg: Config,
    key: str, n_iter: int = 3, n_reps: int = 8, workers: int = 8,
    extra_stats: Any = None, extra_update: Any = None,
) -> tuple[list[TrialParams], pd.DataFrame]:
    """Per-trial rate multipliers that make an effect scenario match the data.

    estimate_params matches each synthetic trial to its real trial under no
    effect. A causal effect changes the data it generates (a chirp that ends
    chases shortens the chases in which the subordinate chirps, which removes
    in-chase chirps). Each iteration simulates ``n_reps`` datasets with the
    effect and rescales, per trial, the baseline offset / onset hazards by
    the ratio of simulated to real mean chase duration / gap, and the in- /
    out-of-chase sub chirp rates by real / simulated counts. Per-trial log
    ratios are shrunk toward the pooled one with weight n / (n + 100) (n =
    the trial's real count of chases or chirps), so trials with few events
    follow the pooled correction; steps are clipped to [0.75, 1.33]. The same
    seeds are used in every iteration. Returns the calibrated TrialParams and
    a log of the pooled ratios per iteration.

    Optional hooks (hz_acausal_sim): ``extra_stats(trial) -> dict`` adds
    per-trial statistics (real and simulated; their pooled ratios are logged)
    and ``extra_update(tps, sim, real) -> tps`` adjusts further per-trial
    parameters after each update, from the same simulations (sim / real:
    per-recording means of all statistics).
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    real = pd.DataFrame([dict(recording=tr.recording, **trial_stats(tr),
                              **(extra_stats(tr) if extra_stats else {})) for tr in trials]
                        ).set_index("recording")
    extra_cols = [c for c in (extra_stats(trials[0]) if extra_stats and len(trials) else {})]
    tps = list(params.trials)
    log = []
    for it in range(n_iter + 1):
        P = replace(params, trials=tps, **overrides)
        _SIM.update(calib_P=P, calib_key=key, calib_iter=it, cfg=cfg, calib_extra=extra_stats)
        if workers > 1:
            with ProcessPoolExecutor(min(workers, n_reps), mp_context=mp.get_context("fork"),
                                     initializer=_worker_init) as ex:
                res = list(ex.map(_calib_worker, range(n_reps)))
        else:
            res = [_calib_worker(r) for r in range(n_reps)]
        sim = pd.DataFrame([x for r in res for x in r]).groupby("recording").mean(
            numeric_only=True)
        tot = sim.reindex(real.index)
        log.append(dict(calib=key, iteration=it, **{
            f"ratio_{c}": float(tot[c].sum() / real[c].sum())
            for c in ("n_chases", "time_in", "n_sub_in", "n_sub_out", "n_off_sub1s",
                      "n_on_sub1s")},
            max_abs_dev_n_chases=float(np.nanmax(np.abs(tot["n_chases"] / real["n_chases"] - 1))),
            max_abs_dev_n_sub_in=float(np.nanmax(np.abs(
                (tot["n_sub_in"] + 1) / (real["n_sub_in"] + 1) - 1)))))
        log[-1].update({f"ratio_{c}": float(tot[c].sum() / max(real[c].sum(), 1e-9))
                        for c in extra_cols})
        if it == n_iter:
            break
        # log correction per trial and quantity, shrunk toward the pooled one
        sm = sim.reindex(real.index)
        md_s = np.log(sm["time_in"] / sm["n_chases"].clip(lower=1e-9))
        md_r = np.log(real["time_in"] / real["n_chases"].clip(lower=1))
        mg_s = np.log(sm["time_out"] / sm["n_chases"].clip(lower=1e-9))
        mg_r = np.log(real["time_out"] / real["n_chases"].clip(lower=1))
        corr = {
            "h_mult_in": (md_s - md_r, real["n_chases"],
                          np.log(sm["time_in"].sum() / sm["n_chases"].sum())
                          - np.log(real["time_in"].sum() / real["n_chases"].sum())),
            "h_mult_out": (mg_s - mg_r, real["n_chases"],
                           np.log(sm["time_out"].sum() / sm["n_chases"].sum())
                           - np.log(real["time_out"].sum() / real["n_chases"].sum())),
            "sub_mult_in": (np.log((real["n_sub_in"] + 1) / (sm["n_sub_in"] + 1)),
                            real["n_sub_in"], np.log(real["n_sub_in"].sum()
                                                     / sm["n_sub_in"].sum())),
            "sub_mult_out": (np.log((real["n_sub_out"] + 1) / (sm["n_sub_out"] + 1)),
                             real["n_sub_out"], np.log(real["n_sub_out"].sum()
                                                       / sm["n_sub_out"].sum())),
        }
        new = []
        for tp in tps:
            upd = {}
            for name, (lr, n, pooled) in corr.items():
                x = float(lr.get(tp.recording, np.nan))
                w = float(n.get(tp.recording, 0)) / (float(n.get(tp.recording, 0)) + 100.0)
                x = w * x + (1 - w) * pooled if np.isfinite(x) else pooled
                upd[name] = getattr(tp, name) * float(np.exp(np.clip(x, np.log(0.75),
                                                                     np.log(1.33))))
            new.append(replace(tp, **upd))
        tps = new
        if extra_update is not None:
            tps = list(extra_update(tps, sm, real))
    for k in ("calib_P", "calib_key", "calib_iter", "calib_extra"):
        _SIM.pop(k, None)
    return tps, pd.DataFrame(log)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _wilson(k: float, n: float) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = 1.959964
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def summarize_estimates(
    est: pd.DataFrame, scenarios: Sequence[Scenario] | pd.DataFrame
) -> pd.DataFrame:
    """Bias, spread, SE calibration, coverage and rejection rates per estimand."""
    if isinstance(scenarios, pd.DataFrame):  # sim_scenarios.csv
        scenarios = [
            Scenario(
                r["scenario"], r["check"], {}, int(r["n_reps"]), {}, r["label"], x=float(r["x"])
            )
            for _, r in scenarios.iterrows()
        ]
    meta = {s.name: s for s in scenarios}
    rows = []
    keys = ["scenario", "kind", "model", "backend", "term"]
    for key, g in est[est["scenario"] != "observed"].groupby(keys, sort=False):
        lab = dict(zip(keys, key))
        sc = meta.get(lab["scenario"])
        truth = float(g["truth"].iloc[0]) if g["truth"].notna().any() else np.nan
        v = g["estimate"].to_numpy(float)
        n = int(np.isfinite(v).sum()) if not lab["term"].endswith("_joint") else len(g)
        r = dict(
            lab, check=sc.check if sc else "", x=sc.x if sc else np.nan, n_reps=len(g), truth=truth
        )
        if not lab["term"].endswith("_joint"):
            sd = float(np.nanstd(v, ddof=1)) if n > 1 else np.nan
            r.update(
                mean=float(np.nanmean(v)),
                median=float(np.nanmedian(v)),
                sd=sd,
                q025=float(np.nanquantile(v, 0.025)),
                q975=float(np.nanquantile(v, 0.975)),
                ratio_mean=float(np.exp(np.nanmean(v))),
                ratio_median=float(np.exp(np.nanmedian(v))),
                ratio_q025=float(np.exp(np.nanquantile(v, 0.025))),
                ratio_q975=float(np.exp(np.nanquantile(v, 0.975))),
                bias=float(np.nanmean(v) - truth) if np.isfinite(truth) else np.nan,
                rmse=float(np.sqrt(np.nanmean((v - truth) ** 2))) if np.isfinite(truth) else np.nan,
                mean_se_naive=float(g["se_naive"].mean()),
                mean_se_cluster=float(g["se_cluster"].mean()) if "se_cluster" in g else np.nan,
            )
            r["sd_over_se_naive"] = sd / r["mean_se_naive"] if r["mean_se_naive"] else np.nan
            r["sd_over_se_cluster"] = (
                sd / r["mean_se_cluster"]
                if r["mean_se_cluster"] and np.isfinite(r["mean_se_cluster"])
                else np.nan
            )
            for s in ("naive", "cluster"):
                lo, hi = g.get(f"lo_{s}"), g.get(f"hi_{s}")
                if lo is None or not np.isfinite(truth) or lo.isna().all():
                    r[f"coverage_{s}"] = np.nan
                    continue
                cov = ((lo <= truth) & (truth <= hi)).to_numpy()
                r[f"coverage_{s}"] = float(cov.mean())
                r[f"coverage_{s}_lo"], r[f"coverage_{s}_hi"] = _wilson(cov.sum(), len(cov))
        for s in ("naive", "cluster"):
            p = g.get(f"p_{s}")
            if p is None or p.isna().all():
                continue
            k = float((p < 0.05).sum())
            r[f"reject_{s}"] = k / p.notna().sum()
            r[f"reject_{s}_lo"], r[f"reject_{s}_hi"] = _wilson(k, p.notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)


def summarize_jitter_fpr(jit: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    """Jitter-test rejection rates under the null, and null-SD calibration.

    ``sampling_sd`` is the SD of the observed statistic across all null
    replicates (the true sampling SD); ``null_sd_mean / sampling_sd`` < 1
    means the permutation null is too narrow (anti-conservative).
    """
    if jit.empty:
        return pd.DataFrame()
    stat_term = {"tau_1000ms_sub": "sub_any_1000ms", "kernel_sub_mean": "kernel_sub_mean"}
    rows = []
    for (kind, null, stat), g in jit.groupby(["kind", "null", "stat"], sort=False):
        e = est[
            (est["scenario"] == "no_effect")
            & (est["kind"] == kind)
            & (est["backend"] == "glm_fe")
            & (est["term"] == stat_term.get(stat, stat))
        ]
        samp = float(e["estimate"].std(ddof=1)) if len(e) > 1 else np.nan
        n = len(g)
        kr = float((g["p_rank"] <= 0.05 + 1e-12).sum())  # exact level: reject at p <= alpha
        kz = float((g["p_z"] < 0.05).sum())
        rows.append(
            dict(
                kind=kind,
                null=null,
                stat=stat,
                n_reps=n,
                n_draws=int(g["n_draws"].iloc[0]),
                reject_rank=kr / n,
                reject_rank_lo=_wilson(kr, n)[0],
                reject_rank_hi=_wilson(kr, n)[1],
                reject_z=kz / n,
                reject_z_lo=_wilson(kz, n)[0],
                reject_z_hi=_wilson(kz, n)[1],
                mean_observed=float(g["observed"].mean()),
                mean_null_mean=float(g["null_mean"].mean()),
                null_sd_mean=float(g["null_sd"].mean()),
                sampling_sd=samp,
                null_sd_over_sampling_sd=float(g["null_sd"].mean()) / samp if samp else np.nan,
                min_p_rank=2.0 / (int(g["n_draws"].iloc[0]) + 1),
                mean_seconds=float(g["seconds"].mean()),
            )
        )
    return pd.DataFrame(rows)


def summarize_kernels(kern: pd.DataFrame) -> pd.DataFrame:
    if kern.empty:
        return kern
    g = kern.groupby(["scenario", "kind", "lag"], sort=False)
    out = (
        g["estimate"]
        .agg(
            mean="mean",
            median="median",
            q025=lambda v: np.quantile(v, 0.025),
            q975=lambda v: np.quantile(v, 0.975),
            n="size",
        )
        .reset_index()
    )
    out["truth"] = g["truth"].first().to_numpy()
    return out


# pooled ('ALL') statistics: fractions are event-weighted (sum of counts / sum of
# events), counts and times are summed, other statistics are means over trials
POOLED_FRAC = {"frac_off_sub1s": ("n_off_sub1s", "n_chases"),
               "frac_on_sub1s": ("n_on_sub1s", "n_onsets"),
               "rate_sub_in": ("n_sub_in", "time_in"), "rate_sub_out": ("n_sub_out", "time_out")}


def _pool(df: pd.DataFrame, c: str) -> tuple[float, str]:
    if c in POOLED_FRAC:
        a, b = POOLED_FRAC[c]
        return float(df[a].sum() / df[b].sum()) if df[b].sum() else np.nan, "event-weighted"
    if c.startswith("n_") or c.startswith("time_"):
        return float(df[c].sum()), "sum"
    return float(df[c].mean()), "mean over trials"


def summarize_match(st: pd.DataFrame) -> pd.DataFrame:
    """Real vs simulated per-trial statistics, and pooled ('ALL') rows.

    Scenarios: no_effect (rates matched under no effect) and the effect
    scenarios recovery / het_recovery (rates calibrated under the effect).
    Per-trial rows carry the relative deviation sim_mean / real - 1.
    """
    if st.empty:
        return st
    cols = [c for c in st.columns if c not in ("scenario", "rep", "recording")]
    real = st[st["scenario"] == "observed"].set_index("recording")[cols]
    rows = []
    for sc in ("no_effect", "recovery", "het_recovery"):
        s = st[st["scenario"] == sc]
        if s.empty:
            continue
        m = s.groupby("recording")[cols].mean()
        for rec in real.index:
            if rec not in m.index:
                continue
            for c in cols:
                rv, sv = real.loc[rec, c], m.loc[rec, c]
                rows.append(dict(scenario=sc, recording=rec, stat=c, real=rv, sim_mean=sv,
                                 rel_dev=sv / rv - 1 if rv else np.nan))
        for c in cols:
            pooled_real, how = _pool(real, c)
            per_rep = pd.Series({r: _pool(g, c)[0] for r, g in s.groupby("rep")})
            rows.append(dict(scenario=sc, recording="ALL", stat=c, pooling=how, real=pooled_real,
                             sim_mean=float(per_rep.mean()),
                             sim_q025=float(per_rep.quantile(0.025)),
                             sim_q975=float(per_rep.quantile(0.975))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
SLOT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}  # dataviz categorical order


def _log_ratio_axis(ax: Any, vals: np.ndarray) -> None:
    v = np.exp(np.asarray(vals, float)[np.isfinite(vals)])
    lo, hi = (min(v.min(), 1.0) / 1.2, max(v.max(), 1.0) * 1.2) if len(v) else (0.5, 2)
    hz._ratio_axis(ax, max(lo, 0.02), min(hi, 50))


def plot_recovery(summ: pd.DataFrame, est: pd.DataFrame, out: Path) -> list[Path]:
    """Estimated hazard ratios per replicate vs the generative value."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from hz_desc import INK, INK2, MUTED, _save, _style

    _style()
    order = ["no_effect", "recovery", "annotation", "detection", "recovery_exp"]
    labels = {
        "no_effect": "no\neffect",
        "recovery": "effect",
        "annotation": "effect +\nboundary\nerror",
        "detection": "effect +\nmissed\nchirps",
        "recovery_exp": "effect,\nexp\nkernel",
    }
    terms = [
        ("sub_any_1000ms", ">= 1 sub chirp in preceding 1 s"),
        ("kernel_sub_mean", "kernel mean per sub chirp, 0.05-2 s"),
    ]
    e = est[(est["backend"] == "glm_fe") & est["scenario"].isin(order)]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)
    for j, kind in enumerate(KINDS):
        for i, (term, tlab) in enumerate(terms):
            ax = axes[i, j]
            g = e[(e["kind"] == kind) & (e["term"] == term)]
            scs = [s for s in order if s in set(g["scenario"])]
            for x, sc in enumerate(scs):
                v = g.loc[g["scenario"] == sc, "estimate"].to_numpy(float)
                ax.scatter(
                    x + rng.uniform(-0.18, 0.18, len(v)),
                    np.exp(v),
                    s=9,
                    color=SLOT[1],
                    alpha=0.45,
                    lw=0,
                    zorder=3,
                )
                q = np.quantile(v, [0.025, 0.5, 0.975])
                ax.plot([x + 0.27] * 2, np.exp(q[[0, 2]]), color=INK, lw=1.2, zorder=4)
                ax.plot(x + 0.27, np.exp(q[1]), "o", ms=4.5, color=INK, zorder=5)
                tr = g.loc[g["scenario"] == sc, "truth"].iloc[0]
                # kernel mean under a box (indicator) effect: nominal, not exact
                approx = term == "kernel_sub_mean" and sc != "recovery_exp" and tr != 0
                if np.isfinite(tr):
                    ax.plot(
                        [x - 0.32, x + 0.36],
                        [np.exp(tr)] * 2,
                        color=INK2,
                        lw=1.0,
                        ls=(0, (1, 1.5)) if approx else (0, (3, 1.5)),
                        zorder=2,
                    )
                s = summ[
                    (summ["scenario"] == sc)
                    & (summ["kind"] == kind)
                    & (summ["term"] == term)
                    & (summ["backend"] == "glm_fe")
                ]
                if len(s) and np.isfinite(s["coverage_cluster"].iloc[0]) and not approx:
                    ax.text(
                        x,
                        1.0,
                        f"cov {100 * s['coverage_cluster'].iloc[0]:.0f}%",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=6.3,
                        color=MUTED,
                    )
            _log_ratio_axis(ax, g["estimate"].to_numpy(float))
            ax.set_xticks(range(len(scs)))
            ax.set_xticklabels([labels[s] for s in scs], fontsize=6.5)
            ax.set_xlim(-0.6, len(scs) - 0.4)
            ax.set_title(f"{hz.KIND_LABEL[kind].split(' (')[0]}: {tlab}", loc="left", fontsize=7.8)
            if j == 0:
                ax.set_ylabel("estimated hazard ratio (log scale)")
    handles = [
        Line2D(
            [], [], color=SLOT[1], marker="o", lw=0, ms=4, alpha=0.6, label="one simulated dataset"
        ),
        Line2D([], [], color=INK, marker="o", lw=1.2, ms=4, label="median and 95% range"),
        Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 1.5)), label="generative value"),
        Line2D(
            [],
            [],
            color=INK2,
            lw=1.0,
            ls=(0, (1, 1.5)),
            label="nominal value (kernel model misspecified for a 1 s indicator effect)",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, fontsize=6.8, bbox_to_anchor=(0.5, -0.07)
    )
    fig.suptitle(
        "Recovery of known subordinate-chirp effects (glm_fe; cov = coverage of the "
        "cluster-robust 95% CI)",
        fontsize=8.2,
        x=0.01,
        ha="left",
    )
    return _save(fig, out, "sim_recovery")


def plot_fpr(summ: pd.DataFrame, jfpr: pd.DataFrame, out: Path) -> list[Path]:
    """Rejection rates at alpha 0.05 under no effect, per test."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from hz_desc import GRID, INK2, MUTED, _save, _style

    _style()
    terms = [
        ("sub_any_1000ms", "tau_1000ms_sub", "sub\n1 s indicator"),
        ("kernel_sub_mean", "kernel_sub_mean", "sub\nkernel mean"),
        ("kernel_sub_joint", None, "sub\nkernel joint"),
        ("dom_any_1000ms", None, "dom\n1 s indicator"),
        ("kernel_dom_joint", None, "dom\nkernel joint"),
    ]
    tests = [
        ("naive", "FE GLM Wald (model-based)"),
        ("cluster", "FE GLM cluster-robust t(G-1) / F(q, G-1)"),
        ("shift", "jitter test, circular-shift null"),
        ("matched", "jitter test, matched null"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), constrained_layout=True, squeeze=False)
    width = 0.19
    top = 0.3
    for j, kind in enumerate(KINDS):
        ax = axes[0, j]
        for i, (term, jstat, _) in enumerate(terms):
            s = summ[
                (summ["scenario"] == "no_effect")
                & (summ["kind"] == kind)
                & (summ["term"] == term)
                & (summ["backend"] == "glm_fe")
            ]
            for k, (t, _) in enumerate(tests):
                x = i + (k - 1.5) * width
                if t in ("naive", "cluster"):
                    if s.empty or f"reject_{t}" not in s:
                        continue
                    r, lo, hi = (float(s[f"reject_{t}{suf}"].iloc[0]) for suf in ("", "_lo", "_hi"))
                    lab = ""
                else:
                    js = (
                        jfpr[(jfpr["kind"] == kind) & (jfpr["null"] == t) & (jfpr["stat"] == jstat)]
                        if jstat and len(jfpr)
                        else pd.DataFrame()
                    )
                    if js.empty:
                        continue
                    use = "rank" if js["min_p_rank"].iloc[0] <= 0.05 + 1e-12 else "z"
                    r, lo, hi = (
                        float(js[f"reject_{use}{suf}"].iloc[0]) for suf in ("", "_lo", "_hi")
                    )
                    lab = f"n={int(js['n_reps'].iloc[0])}"
                top = max(top, hi)
                ax.bar(x, max(r, 0.004), width * 0.9, color=SLOT[k + 1], zorder=3)
                ax.errorbar(
                    x,
                    r,
                    yerr=[[max(r - lo, 0.0)], [max(hi - r, 0.0)]],
                    color=INK2,
                    lw=0.9,
                    capsize=0,
                    zorder=4,
                )
                if lab:
                    ax.text(
                        x,
                        hi + 0.01,
                        lab,
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color=MUTED,
                        rotation=90,
                    )
        ax.axhline(0.05, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_xticks(range(len(terms)))
        ax.set_xticklabels([t[2] for t in terms], fontsize=6.8)
        ax.set_title(
            f"{hz.KIND_LABEL[kind].split(' (')[0]}: no chirp effect", loc="left", fontsize=8
        )
        if j == 0:
            ax.set_ylabel("rejection rate at p < 0.05")
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[0]:
        ax.set_ylim(0, min(1.05, top + 0.12))
    axes[0, 0].legend(
        handles=[Patch(color=SLOT[k + 1], label=lab) for k, (_, lab) in enumerate(tests)],
        loc="upper right", fontsize=5.8, frameon=False,
    )
    # bottom row: sub 1 s indicator, backends under homogeneous vs heterogeneous null
    backs = [("glm_fe", "naive", "FE GLM Wald"), ("glm_fe", "cluster", "FE cluster-robust"),
             ("mixed_laplace", "naive", "GLMM (primary)"),
             ("re_meta", "naive", "random-effects meta")]
    scs = [("no_effect", "one common effect\n(zero)"),
           ("het_no_effect", "dyad effects vary,\nmean zero")]
    top2 = 0.3
    for j, kind in enumerate(KINDS):
        ax = axes[1, j]
        for i, (sc, _) in enumerate(scs):
            for k, (b, t, _) in enumerate(backs):
                s = summ[(summ["scenario"] == sc) & (summ["kind"] == kind)
                         & (summ["term"] == "sub_any_1000ms") & (summ["backend"] == b)]
                if s.empty or f"reject_{t}" not in s or not np.isfinite(s[f"reject_{t}"].iloc[0]):
                    continue
                r, lo, hi = (float(s[f"reject_{t}{suf}"].iloc[0]) for suf in ("", "_lo", "_hi"))
                x = i + (k - 1.5) * width
                top2 = max(top2, hi)
                ax.bar(x, max(r, 0.004), width * 0.9, color=SLOT[k + 1], zorder=3)
                ax.errorbar(x, r, yerr=[[max(r - lo, 0.0)], [max(hi - r, 0.0)]], color=INK2,
                            lw=0.9, capsize=0, zorder=4)
                ax.text(x, hi + 0.01, f"n={int(s['n_reps'].iloc[0])}", ha="center",
                        va="bottom", fontsize=5.5, color=MUTED, rotation=90)
        ax.axhline(0.05, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_xticks(range(len(scs)))
        ax.set_xticklabels([x[1] for x in scs], fontsize=6.8)
        ax.set_ylim(0, min(1.05, top2 + 0.12))
        ax.set_title(f"{hz.KIND_LABEL[kind].split(' (')[0]}: sub 1 s indicator, by backend",
                     loc="left", fontsize=8)
        if j == 0:
            ax.set_ylabel("rejection rate at p < 0.05")
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[1, 0].legend(
        handles=[Patch(color=SLOT[k + 1], label=lab) for k, (_, _, lab) in enumerate(backs)],
        loc="upper left", fontsize=5.8, frameon=False,
    )
    fig.suptitle(
        "False-positive rates under no mean chirp effect (Wilson 95% CI; dashed: nominal 5%; "
        "jitter: rank p, n = replicates)",
        fontsize=8,
        x=0.01,
        ha="left",
    )
    return _save(fig, out, "sim_fpr")


def _tau_scan(est: pd.DataFrame, scenario: str, kind: str) -> pd.DataFrame:
    e = est[
        (est["scenario"] == scenario)
        & (est["kind"] == kind)
        & (est["backend"] == "glm_fe")
        & est["term"].str.match(r"sub_any_\d+ms")
    ].copy()
    if e.empty:
        return e
    e["tau"] = e["term"].str.extract(r"(\d+)ms")[0].astype(float) / 1000
    return (
        e.groupby("tau")["estimate"]
        .agg(
            mean="mean", q025=lambda v: np.quantile(v, 0.025), q975=lambda v: np.quantile(v, 0.975)
        )
        .reset_index()
    )


def realized_trigger_share(meta: pd.DataFrame) -> dict[str, float]:
    """Mean realized share of in-chase sub chirps that are trigger chirps, per scenario."""
    if meta is None or meta.empty or "n_trigger_in_chase" not in meta:
        return {}
    m = meta[meta["n_sub_in"] > 0]
    return (m["n_trigger_in_chase"] / m["n_sub_in"]).groupby(m["scenario"]).mean().to_dict()


def plot_confounds(
    summ: pd.DataFrame,
    est: pd.DataFrame,
    kern: pd.DataFrame,
    out: Path,
    meta: pd.DataFrame | None = None,
) -> list[Path]:
    """Hidden trigger (offset) and hidden state (onset) vs causal effect vs data."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from hz_desc import INK, INK2, _save, _style

    _style()
    has_hidden = (est["scenario"] == "hidden_state").any()
    ncol = 3
    nrow = 2 if has_hidden else 1
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(10.2, 3.2 * nrow), constrained_layout=True, squeeze=False
    )
    obs = est[(est["scenario"] == "observed") & (est["backend"] == "glm_fe")]
    # A: spurious OR vs trigger share
    ax = axes[0, 0]
    tr_s = summ[
        (summ["check"].isin(["trigger", "no_effect"]))
        & (summ["kind"] == "offset")
        & (summ["term"] == "sub_any_1000ms")
        & (summ["backend"] == "glm_fe")
        & (~summ["scenario"].str.contains("slow|annotation"))
    ].copy()
    tr_s.loc[tr_s["scenario"] == "no_effect", "x"] = 0.0
    real_share = realized_trigger_share(meta) if meta is not None else {}
    if real_share:
        tr_s["x"] = [0.0 if sc == "no_effect" else real_share.get(sc, np.nan)
                     for sc in tr_s["scenario"]]
    tr_s = tr_s.sort_values("x")
    ax.errorbar(
        tr_s["x"],
        tr_s["ratio_median"],
        yerr=[tr_s["ratio_median"] - tr_s["ratio_q025"], tr_s["ratio_q975"] - tr_s["ratio_median"]],
        fmt="o-",
        color=SLOT[3],
        ms=4.5,
        lw=1.2,
        elinewidth=1.0,
        capsize=0,
        zorder=4,
    )
    o1 = obs[(obs["kind"] == "offset") & (obs["term"] == "sub_any_1000ms")]
    vals = np.log(np.r_[tr_s["ratio_q025"], tr_s["ratio_q975"]])
    if len(o1):
        ax.axhline(np.exp(o1["estimate"].iloc[0]), color=INK, lw=1.2, ls=(0, (4, 2)))
        ax.text(
            0.0, np.exp(o1["estimate"].iloc[0]), " observed", va="bottom", fontsize=6.8, color=INK
        )
        vals = np.r_[vals, o1["estimate"].iloc[0]]
    _log_ratio_axis(ax, vals)
    ax.set_xlabel("realized trigger share of in-chase sub chirps"
                  if real_share else "nominal trigger share of in-chase sub chirps")
    ax.set_ylabel("hazard ratio, sub chirp in preceding 1 s")
    ax.set_title(
        "A  Offset: spurious effect of a hidden\n    common trigger (no causal effect)",
        loc="left",
        fontsize=8,
    )

    def curves(ax_tau: Any, ax_k: Any, kind: str, series: list[tuple[str, str, Any, str]]) -> None:
        allv = []
        for sc, lab, col, ls in series:
            if sc == "observed":
                e = obs[(obs["kind"] == kind) & obs["term"].str.match(r"sub_any_\d+ms")].copy()
                if e.empty:
                    continue
                e["tau"] = e["term"].str.extract(r"(\d+)ms")[0].astype(float) / 1000
                e = e.sort_values("tau")
                ax_tau.plot(
                    e["tau"],
                    np.exp(e["estimate"]),
                    marker="o",
                    color=col,
                    ls=ls,
                    lw=1.4,
                    ms=3.5,
                    zorder=5,
                    label=lab,
                )
                allv.append(e["estimate"].to_numpy())
                k = kern[(kern["scenario"] == "observed") & (kern["kind"] == kind)]
                if len(k):
                    ax_k.plot(k["lag"], np.exp(k["mean"]), color=col, ls=ls, lw=1.4, zorder=5)
                continue
            t = _tau_scan(est, sc, kind)
            if len(t):
                ax_tau.plot(
                    t["tau"],
                    np.exp(t["mean"]),
                    marker="o",
                    color=col,
                    ls=ls,
                    lw=1.2,
                    ms=3.5,
                    label=lab,
                    zorder=4,
                )
                ax_tau.fill_between(
                    t["tau"], np.exp(t["q025"]), np.exp(t["q975"]), color=col, alpha=0.12, lw=0
                )
                allv.append(t["mean"].to_numpy())
            k = kern[(kern["scenario"] == sc) & (kern["kind"] == kind)]
            if len(k):
                ax_k.plot(k["lag"], np.exp(k["mean"]), color=col, ls=ls, lw=1.2, zorder=4)
                ax_k.fill_between(
                    k["lag"], np.exp(k["q025"]), np.exp(k["q975"]), color=col, alpha=0.10, lw=0
                )
        v = np.concatenate(allv) if allv else np.zeros(1)
        _log_ratio_axis(ax_tau, v)
        kv = kern[kern["kind"] == kind]["mean"].to_numpy()
        _log_ratio_axis(ax_k, kv if len(kv) else v)
        ax_tau.set_xscale("log")
        ax_tau.set_xticks([0.25, 0.5, 1, 2, 5])
        ax_tau.set_xticklabels(["0.25", "0.5", "1", "2", "5"])
        ax_tau.set_xlabel("tau (s): sub chirp in preceding tau")
        ax_k.set_xlabel("time since sub chirp (s)")
        ax_k.set_ylabel("hazard ratio per chirp")
        ax_tau.set_ylabel("hazard ratio")

    ser_off = [
        ("observed", "observed data", INK, "-"),
        ("recovery", "causal effect (HR 2, 0-1 s)", SLOT[1], "-"),
        ("trigger_0.5", "hidden trigger, offset 0-0.5 s", SLOT[3], "-"),
        ("trigger_0.25_annotation", "trigger (share 0.25) + boundary error", SLOT[3],
         (0, (3, 1.5))),
        ("trigger_0.5_slow", "hidden trigger, offset 0.5-1.5 s", SLOT[2], "-"),
        ("lag_0.5", "no effect, offsets annotated 0.5 s late", SLOT[4], "-"),
    ]
    curves(axes[0, 1], axes[0, 2], "offset", ser_off)
    axes[0, 1].set_title("B  Offset: tau scan (mean, 95% range)", loc="left", fontsize=8)
    axes[0, 2].set_title("C  Offset: fitted sub-chirp kernel", loc="left", fontsize=8)
    h = [Line2D([], [], color=c, ls=ls, lw=1.3, label=lab) for _, lab, c, ls in ser_off]
    axes[0, 0].legend(handles=h, fontsize=6.0, loc="upper left", frameon=False)
    if has_hidden:
        ser_on = [
            ("observed", "observed data", INK, "-"),
            ("recovery", "causal effect (HR 0.5, 0-1 s)", SLOT[1], "-"),
            ("hidden_state", "slow hidden state (30 s), no causal effect", SLOT[3], "-"),
            ("hidden_state_short", "short hidden state (3 s), no causal effect", SLOT[4], "-"),
        ]
        curves(axes[1, 1], axes[1, 2], "onset", ser_on)
        ax = axes[1, 0]
        hs = summ[
            (summ["scenario"].isin(["hidden_state", "hidden_state_short", "no_effect",
                                    "recovery"]))
            & (summ["kind"] == "onset")
            & (summ["term"] == "sub_any_1000ms")
            & (summ["backend"] == "glm_fe")
        ]
        for x, (sc, col) in enumerate(
            [("no_effect", INK2), ("recovery", SLOT[1]), ("hidden_state", SLOT[3]),
             ("hidden_state_short", SLOT[4])]
        ):
            r = hs[hs["scenario"] == sc]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(
                x,
                r["ratio_median"],
                yerr=[[r["ratio_median"] - r["ratio_q025"]], [r["ratio_q975"] - r["ratio_median"]]],
                fmt="o",
                color=col,
                ms=5,
                capsize=0,
            )
        o2 = obs[(obs["kind"] == "onset") & (obs["term"] == "sub_any_1000ms")]
        vv = np.log(np.r_[hs["ratio_q025"], hs["ratio_q975"]])
        if len(o2):
            ax.axhline(np.exp(o2["estimate"].iloc[0]), color=INK, lw=1.2, ls=(0, (4, 2)))
            vv = np.r_[vv, o2["estimate"].iloc[0]]
        _log_ratio_axis(ax, vv)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["no effect", "causal\neffect", "slow\nhidden\nstate",
                            "short\nhidden\nstate"], fontsize=7)
        ax.set_ylabel("hazard ratio, sub chirp in preceding 1 s")
        ax.set_title(
            "D  Onset: hidden states vs causal effect\n    (dashed: observed)",
            loc="left",
            fontsize=8,
        )
        axes[1, 1].set_title("E  Onset: tau scan", loc="left", fontsize=8)
        axes[1, 2].set_title("F  Onset: fitted sub-chirp kernel", loc="left", fontsize=8)
        h = [Line2D([], [], color=c, ls=ls, lw=1.3, label=lab) for _, lab, c, ls in ser_on]
        axes[1, 0].legend(handles=h, fontsize=6.0, loc="lower right", frameon=False)
    fig.suptitle(
        "Confounders that mimic a chirp effect (glm_fe fits of simulated datasets)",
        fontsize=8.6,
        x=0.01,
        ha="left",
    )
    return _save(fig, out, "sim_confounds")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def lag_profile(est: pd.DataFrame, scenario: str, kind: str) -> dict[str, float]:
    """Tau-scan shape: short = log HR(0.25 s) / log HR(1 s), long = log HR(5 s) /
    log HR(1 s) of the glm_fe tau models, per replicate; median and 95% range."""
    e = est[(est["scenario"] == scenario) & (est["kind"] == kind) & (est["backend"] == "glm_fe")
            & est["term"].isin(["sub_any_250ms", "sub_any_1000ms", "sub_any_5000ms"])]
    nan = dict(short=np.nan, short_lo=np.nan, short_hi=np.nan, long=np.nan, long_lo=np.nan,
               long_hi=np.nan, n=0)
    if e.empty:
        return nan
    pv = e.pivot_table(index="rep", columns="term", values="estimate").dropna()
    if pv.empty or not {"sub_any_250ms", "sub_any_1000ms", "sub_any_5000ms"} <= set(pv):
        return nan
    sh = (pv["sub_any_250ms"] / pv["sub_any_1000ms"]).to_numpy()
    lo = (pv["sub_any_5000ms"] / pv["sub_any_1000ms"]).to_numpy()
    q = lambda v, a: float(np.quantile(v, a)) if len(v) else np.nan  # noqa: E731
    return dict(short=q(sh, 0.5), short_lo=q(sh, 0.025), short_hi=q(sh, 0.975),
                long=q(lo, 0.5), long_lo=q(lo, 0.025), long_hi=q(lo, 0.975), n=len(pv))


def kernel_peak(kern: pd.DataFrame, scenario: str, kind: str, sign: int = 1) -> float:
    """Lag (s) at which the mean fitted sub kernel of a scenario peaks (sign * log HR)."""
    k = kern[(kern["scenario"] == scenario) & (kern["kind"] == kind)] if len(kern) else kern
    if k is None or k.empty:
        return np.nan
    col = "mean" if "mean" in k else "estimate"
    return float(k.loc[(sign * k[col]).idxmax(), "lag"])


def summary_text(
    summ: pd.DataFrame,
    jfpr: pd.DataFrame,
    est: pd.DataFrame,
    meta: pd.DataFrame,
    match: pd.DataFrame,
    elapsed: float,
    cfg: Config,
    calib_log: pd.DataFrame | None = None,
    kern: pd.DataFrame | None = None,
) -> str:
    def row(sc: str, kind: str, term: str, backend: str = "glm_fe") -> pd.Series | None:
        r = summ[
            (summ["scenario"] == sc)
            & (summ["kind"] == kind)
            & (summ["term"] == term)
            & (summ["backend"] == backend)
        ]
        return r.iloc[0] if len(r) else None

    def pct(v: float) -> str:
        return f"{100 * v:.0f}%" if np.isfinite(v) else "n/a"

    def rej(r: pd.Series | None, t: str) -> str:
        if r is None or not np.isfinite(r.get(f"reject_{t}", np.nan)):
            return "n/a"
        return (f"{pct(r[f'reject_{t}'])} [{pct(r[f'reject_{t}_lo'])}, "
                f"{pct(r[f'reject_{t}_hi'])}]")

    kern = kern if kern is not None else pd.DataFrame()
    g_term = str(hz.indicator_column(hz.guard_model(cfg), "sub"))
    L = [
        f"Simulation checks (bin {cfg.bin_width:g} s, {len(summ['scenario'].unique())} scenarios; "
        f"runtime {elapsed / 60:.1f} min)",
        "Estimates are log hazard ratios; HR = exp. Backends: glm_fe = cloglog GLM with dyad "
        "fixed intercepts ('mb' = its model-based Wald test, 'cl' = cluster-robust t(G-1)); "
        "mixed_laplace = the primary logit GLMM (dyad random intercept); re_meta = two-stage "
        "random-effects meta-analysis of per-dyad O/E. Rates are given with Wilson 95% CIs.",
        f"Simulated onset designs keep all events and {_sopt(cfg, 'onset_fraction'):g} of "
        "non-event bins (IPW for glm_fe, case-control offset for the GLMM); the observed "
        "reference uses all bins.",
        "",
        "1. Recovery (generative HR 2 for offsets, 0.5 for onsets; >=1 sub chirp in the preceding "
        "1 s; per-trial rates calibrated under the effect). The kernel-mean 'truth' is nominal: "
        "the kernel model is misspecified for a 1 s box effect.",
    ]
    for sc in ("recovery", "annotation", "detection", "recovery_exp", "het_recovery"):
        for kind in KINDS:
            for term in ("sub_any_1000ms", "kernel_sub_mean", g_term):
                r = row(sc, kind, term)
                if r is None:
                    continue
                tr = f"{np.exp(r['truth']):.2f}" if np.isfinite(r["truth"]) else "n/a"
                L.append(
                    f"  {sc:12s} {kind:6s} {term:15s} truth HR {tr}; median est HR "
                    f"{r['ratio_median']:.2f} [{r['ratio_q025']:.2f}, {r['ratio_q975']:.2f}], "
                    f"bias {r['bias']:+.3f}, SD {r['sd']:.3f}, "
                    f"SD/SE cl {r['sd_over_se_cluster']:.2f} mb {r['sd_over_se_naive']:.2f}, "
                    f"coverage cl {pct(r['coverage_cluster'])} mb {pct(r['coverage_naive'])}, "
                    f"power cl {pct(r.get('reject_cluster', np.nan))} (n={int(r['n_reps'])})"
                )
    for sc in ("recovery", "het_recovery"):
        for kind in KINDS:
            for b, lab in (("mixed_laplace", "GLMM"), ("re_meta", "RE meta")):
                r = row(sc, kind, "sub_any_1000ms", b)
                if r is None:
                    continue
                L.append(f"  {sc:12s} {kind:6s} {lab:7s} median est HR {r['ratio_median']:.2f}, "
                         f"bias {r['bias']:+.3f}, coverage {pct(r['coverage_naive'])}, power "
                         f"{pct(r.get('reject_naive', np.nan))} (n={int(r['n_reps'])})")
    rec = est[(est["scenario"] == "recovery") & (est["term"] == "sub_any_1000ms")]
    for kind in KINDS:
        f = rec[rec["kind"] == kind]
        pv = f.pivot_table(index="rep", columns="backend", values="estimate")
        se = f[f["backend"] == "glm_fe"].set_index("rep")["se_naive"]
        for other, what in (("mixed_laplace", "GLMM vs glm_fe"),
                            ("glm_fe_full", "glm_fe on all onset bins vs 10% controls")):
            if other not in pv or "glm_fe" not in pv:
                continue
            dd = (pv[other] - pv["glm_fe"]).dropna()
            if len(dd):
                L.append(f"  {what} ({kind}, 1 s indicator, {len(dd)} reps): max |diff| "
                         f"{float(np.max(np.abs(dd))):.3f} log HR = "
                         f"{float(np.max(np.abs(dd) / se.reindex(dd.index))):.2f} SE")
    L.append("")
    L.append("2. False-positive rates at p < 0.05 when the mean chirp effect is zero")
    L.append("  (a) one common (zero) effect, 'no_effect'; FE GLM model-based Wald (mb) and "
             "cluster-robust (cl):")
    for kind in KINDS:
        for term in ("sub_any_1000ms", "kernel_sub_mean", "kernel_sub_joint", "dom_any_1000ms",
                     "kernel_dom_mean", "kernel_dom_joint"):
            r = row("no_effect", kind, term)
            if r is None:
                continue
            L.append(f"    {kind:6s} {term:16s} mb {rej(r, 'naive')}, cl {rej(r, 'cluster')} "
                     f"(n={int(r['n_reps'])}); median est {r.get('median', np.nan):+.3f}")
    L.append("  (b) sub 1 s indicator by backend, common zero effect (no_effect) vs dyad effects "
             "with mean zero (het_no_effect, SD as fitted to the data):")
    for kind in KINDS:
        for sc in ("no_effect", "het_no_effect"):
            parts = []
            for b, t, lab in (("glm_fe", "naive", "FE mb"), ("glm_fe", "cluster", "FE cl"),
                              ("mixed_laplace", "naive", "GLMM"), ("re_meta", "naive", "RE meta")):
                r = row(sc, kind, "sub_any_1000ms", b)
                if r is not None:
                    parts.append(f"{lab} {rej(r, t)}")
            g = row(sc, kind, "heterogeneity_G", "re_meta")
            if g is not None:
                parts.append(f"heterogeneity G test rejects {rej(g, 'naive')}")
            if parts:
                n = row(sc, kind, "sub_any_1000ms")
                L.append(f"    {kind:6s} {sc:13s} " + "; ".join(parts)
                         + (f" (n={int(n['n_reps'])})" if n is not None else ""))
    obs = est[(est["scenario"] == "observed") & (est["backend"] == "glm_fe")
              & (est["term"] == "sub_any_1000ms")]
    for kind in KINDS:
        o = obs[obs["kind"] == kind]
        o_ratio = float(o["se_cluster"].iloc[0] / o["se_naive"].iloc[0]) if len(o) else np.nan
        parts = []
        for sc in ("no_effect", "het_no_effect", "recovery", "het_recovery"):
            e = est[(est["scenario"] == sc) & (est["kind"] == kind) & (est["backend"] == "glm_fe")
                    & (est["term"] == "sub_any_1000ms")]
            if len(e):
                v = (e["se_cluster"] / e["se_naive"]).to_numpy()
                parts.append(f"{sc} {np.median(v):.2f} [{np.quantile(v, 0.025):.2f}, "
                             f"{np.quantile(v, 0.975):.2f}]")
        if parts:
            L.append(f"  SE ratio cluster / model-based ({kind}, 1 s): observed {o_ratio:.2f}; "
                     + "; ".join(parts))
    for _, r in jfpr.iterrows():
        use = "rank" if r["min_p_rank"] <= 0.05 + 1e-12 else "z"
        k = int(round(r[f"reject_{use}"] * r["n_reps"]))
        L.append(
            f"  jitter {r['kind']:6s} {r['null']:7s} {r['stat']:15s} null SD / sampling SD "
            f"{r['null_sd_over_sampling_sd']:.2f} (calibration measure: ~1 = calibrated); "
            f"{k} of {int(r['n_reps'])} datasets rejected ({use} p, {int(r['n_draws'])} draws; "
            f"Wilson CI {pct(r[f'reject_{use}_lo'])}-{pct(r[f'reject_{use}_hi'])})"
        )
    L.append("")
    L.append("3. Hidden common trigger (offset; no causal chirp effect). Realized share = "
             "trigger chirps inside the chase / all in-chase sub chirps.")
    o = obs[obs["kind"] == "offset"]
    o_hr = float(np.exp(o["estimate"].iloc[0])) if len(o) else np.nan
    share = realized_trigger_share(meta)
    trig = summ[(summ["check"] == "trigger") & (summ["kind"] == "offset")
                & (summ["term"] == "sub_any_1000ms") & (summ["backend"] == "glm_fe")]
    for _, r in trig.iterrows():
        mt = meta[meta["scenario"] == r["scenario"]]
        pre = (mt["n_trigger_before_offset"] / mt["n_true_offsets"]).mean() if len(mt) else np.nan
        L.append(
            f"  {r['scenario']:23s} realized share {pct(share.get(r['scenario'], np.nan))}; "
            f"spurious HR {r['ratio_median']:.2f} [{r['ratio_q025']:.2f}, {r['ratio_q975']:.2f}]; "
            f"offsets with a trigger chirp in the preceding 1 s {pct(pre)}; cl rejects "
            f"{pct(r.get('reject_cluster', np.nan))}"
        )
    phi = trigger_match(summ, meta, np.log(o_hr) if np.isfinite(o_hr) else np.nan)
    if np.isfinite(phi[0]):
        L.append(f"  observed HR {o_hr:.2f} is matched by a realized trigger share of ~{phi[0]:.2f} "
                 f"of in-chase sub chirps (~{pct(phi[1])} of offsets; linear interpolation of the "
                 "median)")
    L.append("")
    L.append("4. Late annotation of chase offsets (no causal effect; every offset marked lag s "
             "after the true offset)")
    for lag in (0.25, 0.5, 0.75):
        sc = f"lag_{lag:g}"
        r = row(sc, "offset", "sub_any_1000ms")
        if r is None:
            continue
        rg = row(sc, "offset", g_term)
        pk = kernel_peak(kern, sc, "offset")
        L.append(f"  {sc:9s} spurious HR (1 s) {r['ratio_median']:.2f} [{r['ratio_q025']:.2f}, "
                 f"{r['ratio_q975']:.2f}]"
                 + (f"; lag-guarded HR {rg['ratio_median']:.2f}" if rg is not None else "")
                 + (f"; fitted kernel peaks at {pk:.2f} s" if np.isfinite(pk) else "")
                 + f"; cl rejects {pct(r.get('reject_cluster', np.nan))} (n={int(r['n_reps'])})")
    L.append("")
    L.append("5. Tau-scan shape per replicate: short = log HR(0.25 s) / log HR(1 s), long = "
             "log HR(5 s) / log HR(1 s); median [95% range over replicates]. It does not "
             "separate a causal effect from a trigger or annotation lag with 0.5-1 s delays.")
    for kind, scs in (("offset", ("observed", "recovery", "annotation", "recovery_exp",
                                  "trigger_0.1", "trigger_0.25", "trigger_0.5",
                                  "trigger_0.25_annotation", "trigger_0.5_slow", "lag_0.25",
                                  "lag_0.5", "lag_0.75")),
                      ("onset", ("observed", "recovery", "annotation", "hidden_state",
                                 "hidden_state_short"))):
        for sc in scs:
            lp = lag_profile(est, sc, kind)
            if not lp["n"]:
                continue
            rng = (lambda a, b: "" if sc == "observed" else f" [{a:.2f}, {b:.2f}]")  # noqa: E731
            L.append(f"  {kind:6s} {sc:23s} short {lp['short']:.2f}"
                     f"{rng(lp['short_lo'], lp['short_hi'])}, long {lp['long']:.2f}"
                     f"{rng(lp['long_lo'], lp['long_hi'])} (n={lp['n']})")
    for sc, lab in (("hidden_state", "slow (30 s episodes)"),
                    ("hidden_state_short", "short (3 s episodes)")):
        hs = row(sc, "onset", "sub_any_1000ms")
        if hs is None:
            continue
        t = _tau_scan(est, sc, "onset")
        L.append(f"  hidden state, {lab}: spurious onset HR {hs['ratio_median']:.2f} "
                 f"[{hs['ratio_q025']:.2f}, {hs['ratio_q975']:.2f}]; tau scan (mean HR) "
                 + ", ".join(f"{a:g}s {np.exp(b):.2f}" for a, b in zip(t["tau"], t["mean"])))
    L.append("  A slow latent state depresses the onset hazard at long tau as well; a short one "
             "gives a local association like a causal effect.")
    if len(match):
        L.append("")
        L.append("6. Match to the data (ALL = pooled over trials: counts summed, fractions "
                 "event-weighted, other statistics averaged over trials; sim mean [95% range])")
        keep = ("n_chases", "mean_dur", "median_gap", "n_sub_in", "n_sub_out", "n_dom",
                "frac_ici_lt1", "frac_off_sub1s", "frac_on_sub1s")
        for sc in ("no_effect", "recovery", "het_recovery"):
            mm = match[(match["scenario"] == sc) & (match["recording"] == "ALL")
                       & match["stat"].isin(keep)]
            if mm.empty:
                continue
            L.append(f"  {sc}:")
            for _, r in mm.iterrows():
                L.append(f"    {r['stat']:15s} real {r['real']:.4g}  sim {r['sim_mean']:.4g} "
                         f"[{r['sim_q025']:.4g}, {r['sim_q975']:.4g}] ({r.get('pooling', '')})")
            per = match[(match["scenario"] == sc) & (match["recording"] != "ALL")]
            for stat in ("n_chases", "n_sub_in", "mean_dur", "mean_gap"):
                v = per[per["stat"] == stat]
                v = v[np.isfinite(v["rel_dev"].astype(float))]
                if len(v):
                    i = v["rel_dev"].abs().idxmax()
                    L.append(f"    per trial {stat:9s}: median |dev| "
                             f"{100 * v['rel_dev'].abs().median():.0f}%, max "
                             f"{100 * v.loc[i, 'rel_dev']:+.0f}% ({v.loc[i, 'recording'][:10]})")
    if calib_log is not None and len(calib_log):
        L.append("  calibration under the effect (pooled sim / real per iteration): " + "; ".join(
            f"{r['calib']} it{int(r['iteration'])}: chases {r['ratio_n_chases']:.3f}, sub in "
            f"{r['ratio_n_sub_in']:.3f}, sub out {r['ratio_n_sub_out']:.3f}"
            for _, r in calib_log.iterrows()))
    return "\n".join(L)


def trigger_match(summ: pd.DataFrame, meta: pd.DataFrame, obs_log_hr: float
                  ) -> tuple[float, float]:
    """Realized trigger share (and share of offsets preceded by a trigger chirp)
    at which the fast-trigger scenarios reproduce ``obs_log_hr``; linear
    interpolation of the median spurious log HR (glm_fe, 1 s indicator)."""
    s = summ[(summ["check"] == "trigger") & (summ["kind"] == "offset")
             & (summ["term"] == "sub_any_1000ms") & (summ["backend"] == "glm_fe")
             & ~summ["scenario"].str.contains("slow|annotation")]
    null = summ[(summ["scenario"] == "no_effect") & (summ["kind"] == "offset")
                & (summ["term"] == "sub_any_1000ms") & (summ["backend"] == "glm_fe")]
    share = realized_trigger_share(meta)
    if s.empty or null.empty or not np.isfinite(obs_log_hr) or not share:
        return np.nan, np.nan
    s = s.assign(xr=[share.get(sc, np.nan) for sc in s["scenario"]]).dropna(subset=["xr"])
    s = s.sort_values("xr")
    xs = np.r_[0.0, s["xr"].to_numpy(float)]
    ys = np.r_[float(null["median"].iloc[0]), np.log(s["ratio_median"].to_numpy(float))]
    if not (np.all(np.diff(ys) > 0) and ys[0] <= obs_log_hr <= ys[-1]):
        return np.nan, np.nan
    phi = float(np.interp(obs_log_hr, ys, xs))
    fr = [0.0]
    for sc in s["scenario"]:
        mt = meta[meta["scenario"] == sc]
        fr.append(float((mt["n_trigger_before_offset"] / mt["n_true_offsets"]).mean())
                  if len(mt) else np.nan)
    return phi, float(np.interp(phi, xs, fr))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_simulation_checks(
    trials: Sequence[Trial], cfg: Config, verbose: bool = True
) -> dict[str, Any]:
    """Estimate parameters, run all scenarios in parallel, write tables and figures.

    Tables go to cfg.out_dir/'simulation', figures to cfg.out_dir/'figures'.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t_all = time.perf_counter()
    out = Path(cfg.out_dir) / "simulation"
    fig_dir = Path(cfg.out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    params = estimate_params(trials, cfg)
    params.table().to_csv(out / "sim_params.csv", index=False)
    pd.DataFrame(
        dict(edge_lo=IN_EDGES[:-1], edge_hi=IN_EDGES[1:], shape_in=params.shape_in)
    ).to_csv(out / "sim_shape_in.csv", index=False)
    pd.DataFrame(
        dict(edge_lo=OUT_EDGES[:-1], edge_hi=OUT_EDGES[1:], shape_out=params.shape_out)
    ).to_csv(out / "sim_shape_out.csv", index=False)
    scenarios = default_scenarios(cfg)
    tasks = _task_list(cfg, scenarios)
    n_workers = max(1, min(int(_sopt(cfg, "workers")), os.cpu_count() or 1, len(tasks)))
    _SIM.clear()
    _SIM.update(cfg=cfg)
    calib, calib_logs = {}, []
    t_cal = time.perf_counter()
    for key in sorted({s.calib for s in scenarios if s.calib}):
        calib[key], lg = calibrate_params(params, CALIB_EFFECTS[key], trials, cfg, key,
                                          n_iter=2 if cfg.quick else 3,
                                          n_reps=4 if cfg.quick else 8, workers=n_workers)
        calib_logs.append(lg)
        pd.DataFrame([dict(recording=tp.recording, h_mult_in=tp.h_mult_in,
                           h_mult_out=tp.h_mult_out, sub_mult_in=tp.sub_mult_in,
                           sub_mult_out=tp.sub_mult_out) for tp in calib[key]]
                     ).to_csv(out / f"sim_calib_params_{key}.csv", index=False)
    calib_log = pd.concat(calib_logs, ignore_index=True) if calib_logs else pd.DataFrame()
    calib_log.to_csv(out / "sim_calibration.csv", index=False)
    if verbose and len(calib_log):
        print(f"[sim] calibration ({time.perf_counter() - t_cal:.0f} s):\n"
              + calib_log.round(3).to_string(index=False), flush=True)
    _SIM.clear()
    _SIM.update(
        cfg=cfg, params=params, trials=list(trials), scenarios={s.name: s for s in scenarios},
        calib=calib,
    )
    if verbose:
        print(
            f"[sim] {len(tasks)} tasks, {len(scenarios)} scenarios, {n_workers} workers", flush=True
        )
    results: list[dict[str, Any]] = []
    if n_workers > 1:
        with ProcessPoolExecutor(
            n_workers, mp_context=mp.get_context("fork"), initializer=_worker_init
        ) as ex:
            futs = [ex.submit(_task, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                if verbose and (i % max(1, len(tasks) // 10) == 0 or i == len(tasks)):
                    print(
                        f"[sim] {i}/{len(tasks)} tasks ({time.perf_counter() - t_all:.0f} s)",
                        flush=True,
                    )
    else:
        for t in tasks:
            results.append(_task(t))
    _SIM.clear()

    def cat(key: str) -> pd.DataFrame:
        rows = [r for res in results for r in res.get(key, [])]
        return pd.DataFrame(rows)

    est, kern_raw, jit, st, meta, tim = (
        cat(k) for k in ("est", "kern", "jit", "stats", "meta", "timing")
    )
    summ = summarize_estimates(est, scenarios)
    jfpr = summarize_jitter_fpr(jit, est)
    kern = summarize_kernels(kern_raw)
    match = summarize_match(st)
    elapsed = time.perf_counter() - t_all
    task_s = pd.DataFrame(
        [
            dict(
                scenario=r["timing"][0]["scenario"],
                rep=r["timing"][0]["rep"],
                steps=";".join(x["step"] for x in r["timing"]),
                seconds=r["seconds"],
            )
            for r in results
        ]
    )
    scen_tab = pd.DataFrame(
        [
            dict(
                scenario=s.name,
                check=s.check,
                label=s.label,
                n_reps=s.n_reps,
                kinds=",".join(s.models),
                x=s.x,
                overrides=repr(s.overrides),
            )
            for s in scenarios
        ]
    )
    est.to_csv(out / "sim_estimates.csv", index=False)
    summ.to_csv(out / "sim_summary.csv", index=False)
    jit.to_csv(out / "sim_jitter.csv", index=False)
    jfpr.to_csv(out / "sim_jitter_fpr.csv", index=False)
    kern.to_csv(out / "sim_kernels.csv", index=False)
    match.to_csv(out / "sim_match.csv", index=False)
    meta.to_csv(out / "sim_trigger_meta.csv", index=False)
    scen_tab.to_csv(out / "sim_scenarios.csv", index=False)
    tim.to_csv(out / "sim_step_timings.csv", index=False)
    task_s.to_csv(out / "sim_task_timings.csv", index=False)
    elapsed = time.perf_counter() - t_all
    pd.DataFrame(
        [
            dict(
                step="total",
                seconds=elapsed,
                workers=n_workers,
                n_tasks=len(tasks),
                cpu_seconds=float(task_s["seconds"].sum()),
            )
        ]
    ).to_csv(out / "sim_timings.csv", index=False)
    figs = (
        plot_recovery(summ, est, fig_dir)
        + plot_fpr(summ, jfpr, fig_dir)
        + plot_confounds(summ, est, kern, fig_dir, meta)
    )
    text = summary_text(summ, jfpr, est, meta, match, elapsed, cfg, calib_log, kern)
    (out / "sim_summary.txt").write_text(text + "\n")
    return dict(
        params=params,
        calibration=calib_log,
        scenarios=scen_tab,
        estimates=est,
        summary=summ,
        jitter=jit,
        jitter_fpr=jfpr,
        kernels=kern,
        match=match,
        trigger_meta=meta,
        timings=task_s,
        text=text,
        figures=figs,
        elapsed=elapsed,
    )


def report_from_tables(cfg: Config) -> dict[str, Any]:
    """Rebuild summaries, figures and sim_summary.txt from saved sim_*.csv tables."""
    out = Path(cfg.out_dir) / "simulation"

    def rd(name: str) -> pd.DataFrame:
        f = out / name
        return (
            pd.read_csv(f, keep_default_na=False, na_values=[""]) if f.exists() else pd.DataFrame()
        )

    # sim_kernels.csv and sim_match.csv already hold the per-scenario summaries
    est, jit, kern, match, meta = (
        rd(f"sim_{k}.csv") for k in ("estimates", "jitter", "kernels", "match", "trigger_meta")
    )
    scen = rd("sim_scenarios.csv")
    summ = summarize_estimates(est, scen)
    jfpr = summarize_jitter_fpr(jit, est)
    tim = rd("sim_timings.csv")
    elapsed = float(tim["seconds"].iloc[0]) if len(tim) else np.nan
    summ.to_csv(out / "sim_summary.csv", index=False)
    jfpr.to_csv(out / "sim_jitter_fpr.csv", index=False)
    fig_dir = Path(cfg.out_dir) / "figures"
    figs = (
        plot_recovery(summ, est, fig_dir)
        + plot_fpr(summ, jfpr, fig_dir)
        + plot_confounds(summ, est, kern, fig_dir, meta)
    )
    calib_log = rd("sim_calibration.csv")
    text = summary_text(summ, jfpr, est, meta, match, elapsed, cfg, calib_log, kern)
    (out / "sim_summary.txt").write_text(text + "\n")
    return dict(summary=summ, jitter_fpr=jfpr, text=text, figures=figs)


def main() -> None:
    from hz_data import load_trials

    ap = argparse.ArgumentParser(
        description="Simulation checks of the chirp -> chase hazard models."
    )
    ap.add_argument("data_root", type=Path, nargs="?", default=Config().data_root)
    ap.add_argument("--out", type=Path, default=Config().out_dir)
    ap.add_argument("--bin-width", type=float, default=None)
    ap.add_argument("--n-sims", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-time", type=float, default=None, help="s; <=0 disables the cap")
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--no-hidden", action="store_true", help="skip the hidden-state onset scenario")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild figures and sim_summary.txt from existing sim_*.csv tables",
    )
    a = ap.parse_args()
    cfg = Config(data_root=a.data_root, out_dir=a.out).updated(
        bin_width=a.bin_width,
        n_sims=a.n_sims,
        max_time=a.max_time,
        quick=a.quick or None,
        exclude_bad_recordings=False if a.include_bad else None,
    )
    if a.workers is not None:
        cfg.extra["sim_workers"] = a.workers
    if a.no_hidden:
        cfg.extra["sim_hidden"] = False
    if a.report_only:
        print(report_from_tables(cfg)["text"])
        return
    t0 = time.perf_counter()
    trials, _, _ = load_trials(cfg)
    print(f"{len(trials)} trials loaded ({time.perf_counter() - t0:.1f} s)", flush=True)
    res = run_simulation_checks(trials, cfg)
    print("\n" + res["text"])
    print(f"\ntotal {time.perf_counter() - t0:.0f} s; tables in {Path(cfg.out_dir) / 'simulation'}")


if __name__ == "__main__":
    main()
