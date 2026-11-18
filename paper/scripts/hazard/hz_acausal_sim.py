"""Simulation calibration of the acausal control (hz_acausal).

The acausal control compares sub chirps before and after the outcome bin (A:
past vs future windows, two-sided kernel) and real in-chase chirps with
post-offset chirps reflected into the chase (B: restricted real vs mirror).
None of these comparisons has a null of 1: the offset risk set holds in-chase
bins only, sub chirps are ~3.3x more frequent inside chases, the post-offset
chirp burst reaches into the future windows of pre-offset bins, and chase
boundaries were annotated on a ~0.5 s lattice. This module simulates what each
account of the association produces for exactly these statistics, with chase
and chirp rates matched to the real dyads, and puts the observed values next
to them. Offsets (a) get the full set of accounts; onsets (b) get a small set
(below), because the after-window of an onset bin lies inside the chase the
outcome starts.

Accounts for chase offsets (no onset effect, beta_on = 0)
---------------------------------------------------------
All start from hz_sim's generator matched to the real trials
(hz_sim.estimate_params), with the post-offset part of the elapsed-time chirp
profile removed: the out-of-chase rate in the first 5 s after an offset is set
to its 5-10 s level ('flat_early'), so that the chirps locked to the offset
come only from the explicit mechanisms below. Unless named '_exact', chase
boundaries are annotated with hz_sim's boundary error (Gaussian SD 0.3 s,
rounded to a 0.5 s lattice), the observation model of the real annotations.

  causal_only            HR 2 for 0-1 s after each sub chirp (hz_sim 'box'), no
                         post-offset response burst.
  burst_only(_exact)     no causal effect; a response burst after every true
                         offset: Poisson(resp_n) sub chirps at off + L, L from the
                         EMPIRICAL post-offset excess profile (below), resp_n per
                         dyad matched to its post-offset counts.
  causal_plus_burst(_exact)  both.
  causal_lag_plus_burst  as causal_plus_burst with the effect lasting 0-1.5 s
                         after a chirp (the data's association outlasts 1 s:
                         guarded HR ~2.0, kernel peak ~0.65 s).
  causal_het_plus_burst  as causal_plus_burst with dyad-specific effects (log HR
                         ~ N(log 2, sd^2), sd = the data's random-effects SD).
  causal_plus_burst_drift  as causal_plus_burst on a more realistic background:
                         a slow latent two-state modulation of the sub chirp rate
                         (both chase states; strength matched to the data's
                         co-occurrence of pre- and post-offset chirps at
                         NEIGHBOURING offsets, see calibrate_drift) and chirp
                         bursts that continue across chase boundaries.
  causal_plus_burst_matched  causal effect raised so that the standard 1 s HR
                         reproduces the observed one.
  causal_plus_bout       as matched, with a larger response after offsets
                         preceded by a sub chirp (1 s): the data's same-offset
                         rate ratio DIVIDED by its neighbouring-offset analogue,
                         i.e. the offset-specific part of the dependence.
  common_trigger(_exact) hz_sim's hidden trigger (a latent event ends the chase
                         0-0.5 s later and elicits a sub chirp 0-0.3 s after it; no
                         causal effect) at the nominal share that reproduced the
                         observed FE HR in hz_sim (~12% realized), plus the response
                         burst.
  symmetric_cluster      no causal effect; a latent cluster centred on the TRUE
                         offset: with probability sym_n / sym_m per offset,
                         Poisson(sym_m) sub chirps at off - L and Poisson(sym_m) at
                         off + L, L from the empirical PRE-offset excess profile;
                         sym_n per dyad matched to its pre-offset counts (sym_m = 1 is
                         not calibrated). The post-offset excess the cluster does not
                         cover is a residual response burst.

Accounts for chase onsets (no offset effect; hz_sim generator without
flat_early; lattice boundary error): no effect; no mean effect with
dyad-specific effects (sd = the data's onset RE SD); a short latent
'subordinate away' state (hz_sim's hidden_state_short); a causal HR 0.5 for
0-1 s after a sub chirp (rates calibrated under the effect). The default
generator produces fewer sub chirps in the first 1-1.5 s of a chase than the
data (every chirp burst starts afresh at the onset: 0.5-1 s after annotated
onsets ~23 vs 36 real), so the after-window of an onset bin is less raised
than in the data. Two further accounts ('chase start matched': no effect, and
causal HR 0.5) scale the in-chase chirp-rate profile in the first 2 s of a
chase (four 0.5 s bins, pooled over dyads; calibrate_start) until the pooled
counts after annotated onsets match the real ones. Onset designs keep all
event bins and 10% of the other bins (hz_hazard case-control option, IPW
cloglog); the 'observed' onset values use the same subsampling.

Empirical latency profiles
--------------------------
Real sub chirps per latency bin (0.25 s to 2 s, 0.5 s to 5 s) before offsets
(in-chase, own chase) and after them (out of chase, before the next onset) minus
the same counts in 'flat_early' simulations without offset-locked chirps
and without effects (exact boundaries): the excess before and after the
annotated offset. resp = excess after; sym = excess before; residual = excess
after minus excess before (each clipped at 0 and normalised; the small negative
pre-offset excess at 2-4 s is clipped). The background of the default
generator has fewer chirps just after an offset than later (every chirp burst
ends at the chase end), so part of the 'response' replaces burst continuations;
the drift account, whose bursts cross the boundary, gets its own profiles
estimated against its own background. The real profiles are relative to
annotated offsets and are applied at true offsets, so the annotation error of
the lattice scenarios is applied on top.

Calibration
-----------
hz_sim.calibrate_params per scenario (baseline hazards and in/out chirp rates
matched per trial, shrunk toward the pooled correction), extended by an
additive per-dyad update of resp_n (post-offset chirps within 2 s) and sym_n
(pre-offset chirps within 2 s). Counts that no mechanism targets (pre-offset
chirps in the causal, burst and trigger scenarios) are free: whether a scenario
reproduces them, pooled and per dyad, is part of the result.

Statistics per replicate (glm_fe = cloglog, dyad FE; the hz_acausal functions)
----------------------------------------------------------------------------
Offsets: standard 1 s model HR; joint 1 s model HR_past, HR_future and their
ratio; the same for the guarded windows; two-sided kernel mean before / after
and their ratio; restricted real and mirror arms (W = 2 s) with the 1 s and
guarded indicators and the paired ratios; chirp counts in 0-1 and 0-2 s before
/ after offsets; co-occurrence of a sub chirp in the last 1 s of a chase and the
first 1 s after its offset relative to (i) within-dyad independence, (ii) the
same pairing across NEIGHBOURING offsets (pre of chase i x post of chase i +- 1:
the slow co-variation baseline) and their ratio (iii) 'local' = the
offset-specific excess; and the pre 0-1 s x post 2-3 s pairing on the same
offset. Onsets: standard 1 s and guarded HR, joint HR_past / HR_future and
their ratios, joint / standalone HR_past, and sub chirp counts in the first
1 s and 0.5-1.5 s of chases and in the 1 s before onsets.

Criteria. Each observed statistic is compared with every account either by
its EXCEEDANCE fraction, the share of simulated datasets at or beyond the
observed value on the observed value's side (min(share >= observed, share <=
observed)), or, where the observed data show more between-dyad spread than the
simulated fits (observed cluster-robust SE above the simulated fits' own SEs),
by z = (log observed - log median) / observed SE and its two-sided p (see
``criterion``). Status per account: 'inconsistent' if the exceedance fraction
is < 2.5% (z: p < 0.05), 'borderline' if 2.5-10% (z: 0.05 <= p < 0.2),
'consistent' otherwise (``status_of``). Verdicts count borderline accounts as
compatible and say when a verdict would change if they were counted as
inconsistent.

Outputs: cfg.out_dir/'simulation'/acausal_sim_*.csv, acausal_sim_summary.txt;
figure cfg.out_dir/'figures'/acausal_sim_calibration.(png|pdf). The run ends by
rebuilding acausal/acausal_summary.txt (hz_acausal.rewrite_summary).

Run: <venv>/bin/python src_hazard/hz_acausal_sim.py DATA --out DIR [--quick] [--n-reps N]
     [--workers N]   (defaults: 100 datasets x 13 offset + 6 onset accounts, ~80 min with 4
     workers; --quick 4 datasets)
     <venv>/bin/python src_hazard/hz_acausal_sim.py --out DIR --report-only
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hz_acausal as ha  # noqa: E402
import hz_sim as S  # noqa: E402
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402

LATTICE = dict(boundary_sd=0.3, boundary_step=0.5)  # hz_sim's boundary-error scenario
PROFILE_EDGES = np.r_[np.arange(0.0, 2.0, 0.25), np.arange(2.0, 5.01, 0.5)]  # s
MATCH_W = 2.0  # s, window of the per-dyad counts matched by the calibration
FAR = (2.0, 3.0)  # s, non-adjacent post-offset window of the co-occurrence check

START_EDGES = S.IN_EDGES[S.IN_EDGES <= 2.0 + 1e-9]  # s after an onset: chase-start profile bins
# status of an account given the observed value: exceedance fraction (share of simulated
# datasets at or beyond the observed value) or, under the z criterion, the two-sided p
EXC_INCONSISTENT, EXC_BORDERLINE = 0.025, 0.10
P_INCONSISTENT, P_BORDERLINE = 0.05, 0.20

ACSIM_DEFAULTS: dict[str, Any] = dict(
    acsim_n_reps=100,  # replicates per offset account (13 accounts)
    acsim_quick_reps=4,
    acsim_trigger_share=0.12,  # nominal hz_sim trigger share (~12% realized)
    acsim_sym_m=1.0,  # mean chirps per side of one symmetric cluster (not calibrated)
    acsim_calib_iter=3,
    acsim_calib_reps=8,
    acsim_profile_reps=8,
    acsim_lag_window=1.5,  # s, effect window of causal_lag_plus_burst
    acsim_drift_frac=0.25,  # share of time in the high-chirping state (drift account)
    acsim_drift_dur=300.0,  # s, mean high-state episode (drift account)
    acsim_drift_grid=(1.0, 2.0, 3.0, 4.0, 5.5, 7.5),  # high / low rate ratios tried (drift)
    acsim_drift_reps=10,  # datasets per grid value
    acsim_onset_reps=100,  # replicates per onset account (6 accounts)
    acsim_onset_fraction=0.1,  # share of onset non-event bins kept (case-control)
    acsim_start_iter=3,  # updates of the chase-start profile ('chase start matched' accounts)
    acsim_start_reps=32,  # datasets per round of that calibration
)


def _opt(cfg: Config, name: str) -> Any:
    return cfg.extra.get(name, ACSIM_DEFAULTS[name])


def data_het_sd(cfg: Config, kind: str) -> float:
    """Between-dyad SD of the chirp effect: cfg.extra 'sim_het_sd_<kind>' (set by the main
    script), else the random-effects estimate in out_dir/hazard/hazard_effects.csv, else
    hz_sim's default."""
    key = f"sim_het_sd_{kind}"
    if key in cfg.extra:
        return float(cfg.extra[key])
    f = Path(cfg.out_dir) / "hazard" / "hazard_effects.csv"
    if f.exists():
        e = pd.read_csv(f, usecols=["kind", "model", "backend", "term", "estimate"])
        r = e[(e["kind"] == kind) & (e["model"] == "tau_1000ms") & (e["backend"] == "re_meta")
              & (e["term"] == "dyad_sd_sub_effect")]
        if len(r) and np.isfinite(r["estimate"].iloc[0]):
            return float(r["estimate"].iloc[0])
    return S.het_sd(cfg, kind)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcScenario:
    name: str
    label: str
    overrides: dict[str, Any]  # SimParams fields
    locked: str  # offset-locked chirps: '' | 'resp' | 'sym'
    short: str  # label in figures
    matched: bool = False  # causal effect set so the standard 1 s HR matches the observed one
    bout: bool = False  # response x bout_mult after offsets preceded by a sub chirp (1 s)
    drift: bool = False  # slow chirp-rate modulation + bursts across chase boundaries

    @property
    def causal(self) -> bool:
        return float(self.overrides.get("beta_off", 0.0)) != 0.0


def scenarios(cfg: Config, het_off: float | None = None) -> list[AcScenario]:
    share = float(_opt(cfg, "acsim_trigger_share"))
    lagw = float(_opt(cfg, "acsim_lag_window"))
    het = float(data_het_sd(cfg, "offset") if het_off is None else het_off)
    L2 = S.LOG2
    return [
        AcScenario("causal_only", "causal HR 2 (0-1 s after a sub chirp), no post-offset response "
                   "burst; lattice boundary error", dict(beta_off=L2, **LATTICE), "",
                   "causal only"),
        AcScenario("burst_only", "no causal effect; empirical post-offset response burst; lattice "
                   "boundary error", dict(LATTICE), "resp", "burst only"),
        AcScenario("burst_only_exact", "as burst_only, exact boundaries", {}, "resp",
                   "burst only, exact"),
        AcScenario("causal_plus_burst", "causal HR 2 + post-offset response burst; lattice "
                   "boundary error", dict(beta_off=L2, **LATTICE), "resp", "causal + burst"),
        AcScenario("causal_plus_burst_exact", "as causal_plus_burst, exact boundaries",
                   dict(beta_off=L2), "resp", "causal + burst, exact"),
        AcScenario("causal_lag_plus_burst", f"causal HR 2 for 0-{lagw:g} s after a sub chirp "
                   "(effect outlasts the 1 s window) + post-offset response burst; lattice "
                   "boundary error", dict(beta_off=L2, effect_window=lagw, **LATTICE), "resp",
                   f"causal 0-{lagw:g} s + burst"),
        AcScenario("causal_het_plus_burst", "causal + burst with dyad-specific effects (mean log "
                   f"HR log 2, SD {het:.2f} = the data's random-effects SD); lattice boundary "
                   "error", dict(beta_off=L2, beta_off_sd=het, **LATTICE), "resp",
                   "causal, dyads differ, + burst"),
        AcScenario("causal_plus_burst_drift", "causal HR 2 + burst on a background with "
                   "minute-scale co-variation of sub chirping (slow latent rate modulation matched "
                   "to the neighbouring-offset co-occurrence of the data) and chirp bursts that "
                   "continue across chase boundaries; lattice boundary error",
                   dict(beta_off=L2, burst_cross=True, **LATTICE), "resp",
                   "causal + burst, drift", drift=True),
        AcScenario("causal_plus_burst_matched", "as causal_plus_burst, causal effect set so that "
                   "the standard 1 s HR reproduces the observed one", dict(beta_off=L2, **LATTICE),
                   "resp", "causal + burst, HR matched", matched=True),
        AcScenario("causal_plus_bout", "as causal_plus_burst_matched, response burst larger after "
                   "offsets preceded by a sub chirp within 1 s (the data's same-offset rate ratio "
                   "divided by its neighbouring-offset analogue)", dict(beta_off=L2, **LATTICE),
                   "resp", "causal + bout", matched=True, bout=True),
        AcScenario("common_trigger", "no causal effect; hidden trigger (nominal share "
                   f"{share:g}) + post-offset response burst; lattice boundary error",
                   dict(trigger_share=share, **LATTICE), "resp", "common trigger"),
        AcScenario("common_trigger_exact", "as common_trigger, exact boundaries (hz_sim's trigger "
                   "scenario)", dict(trigger_share=share), "resp", "common trigger, exact"),
        AcScenario("symmetric_cluster", "no causal effect; latent chirp cluster centred on the "
                   "true offset (pre-offset excess mirrored after it; 1 chirp per side per "
                   "cluster, not calibrated) + residual response burst; lattice boundary error",
                   dict(LATTICE), "sym", "symmetric cluster"),
    ]


@dataclass(frozen=True)
class OnScenario:
    name: str
    label: str
    overrides: dict[str, Any]
    short: str
    calib: bool = False  # rates calibrated under the effect (hz_sim.calibrate_params)
    start: bool = False  # chase-start chirp profile matched to the data (calibrate_start)

    @property
    def causal(self) -> bool:
        return float(self.overrides.get("beta_on", 0.0)) != 0.0


def onset_scenarios(cfg: Config, het_on: float | None = None) -> list[OnScenario]:
    het = float(data_het_sd(cfg, "onset") if het_on is None else het_on)
    start = ("in-chase sub chirp rate in the first 2 s of a chase (0.5 s bins, pooled over "
             "dyads) scaled so that the counts after annotated onsets match the data")
    return [
        OnScenario("onset_no_effect", "no chirp effect; lattice boundary error", dict(LATTICE),
                   "no effect"),
        OnScenario("onset_start_matched", f"no chirp effect; {start}; lattice boundary error",
                   dict(LATTICE), "no effect + chase-start match", start=True),
        OnScenario("onset_het_no_effect", f"no mean effect, dyad-specific effects (SD {het:.2f} = "
                   "the data's random-effects SD); lattice boundary error",
                   dict(beta_on_sd=het, **LATTICE), "no mean effect, dyads differ"),
        OnScenario("onset_hidden_short", "no chirp effect; short latent 'subordinate away' state "
                   "(10% of time, 3 s episodes: sub chirps x6, onset hazard x0.1; hz_sim's "
                   "hidden_state_short); lattice boundary error",
                   dict(hidden_frac=0.1, hidden_mean_dur=3.0, hidden_chirp_mult=6.0,
                        hidden_onset_mult=0.1, **LATTICE), "short hidden state"),
        OnScenario("onset_causal", "causal HR 0.5 for 0-1 s after a sub chirp (onset), rates "
                   "calibrated under the effect; lattice boundary error",
                   dict(beta_on=-S.LOG2, **LATTICE), "causal HR 0.5", calib=True),
        OnScenario("onset_causal_start", f"causal HR 0.5 as above; {start}; rates calibrated "
                   "under the effect; lattice boundary error", dict(beta_on=-S.LOG2, **LATTICE),
                   "causal HR 0.5 + chase-start match", calib=True, start=True),
    ]


def flat_early(shape_out: np.ndarray) -> np.ndarray:
    """hz_sim's post-offset chirp shape with its first 5 s set to the 5-10 s level."""
    s = np.asarray(shape_out, float).copy()
    j = int(np.flatnonzero(np.isclose(S.OUT_EDGES[:-1], 5.0))[0])
    s[:j] = s[j]
    return s


# ---------------------------------------------------------------------------
# Counts around chase offsets
# ---------------------------------------------------------------------------
def _offset_window(tr: Trial) -> np.ndarray:
    ch = tr.chases
    return (ch[:, 1] >= tr.obs_start) & (ch[:, 1] < tr.obs_end) if len(ch) else np.zeros(0, bool)


def _pair_oe(a: np.ndarray, b: np.ndarray, win: np.ndarray, shifts: Sequence[int]
             ) -> tuple[float, float]:
    """Observed and expected (independence within the pair set) number of pairs (i, i + s)
    with a[i] and b[i + s], both offsets inside the window, pooled over ``shifts``."""
    n = len(a)
    O = E = 0.0
    idx = np.arange(n)
    for s in shifts:
        j = idx + s
        ok = (j >= 0) & (j < n)
        i_, j_ = idx[ok], j[ok]
        ok = win[i_] & win[j_]
        A, B = a[i_[ok]], b[j_[ok]]
        if len(A):
            O += float((A & B).sum())
            E += float(A.sum() * B.sum() / len(A))
    return O, E


def window_stats(tr: Trial) -> dict[str, float]:
    """Sub chirps within 1 / 2 s before (in-chase, own chase) and after (out of chase, before the
    next onset) the offsets inside the analysis window, as hz_acausal counts them, and the
    co-occurrence counts: offsets with a chirp on both sides within 1 s (both_1s) and the number
    expected if the sides were independent within the dyad (both_1s_indep); the same pairing of
    the pre side of offset i with the post side of offsets i +- 1 (nb_*: slow co-variation,
    no shared bout possible); and pre 0-1 s x post 2-3 s of the same offset, on offsets whose
    next onset / unknown state is >= 3 s later (far_*)."""
    win = _offset_window(tr)
    n = len(win)
    _, ip, lp = ha.pre_offset_chirps(tr, MATCH_W)
    _, ia, la = ha.post_offset_chirps(tr, FAR[1])
    out: dict[str, float] = dict(n_off_win=float(win.sum()))
    for w, lab in ((1.0, "1s"), (2.0, "2s")):
        cp = np.bincount(ip[lp < w], minlength=n)[win]
        ca = np.bincount(ia[la < w], minlength=n)[win]
        out[f"pre_{lab}"] = float(cp.sum())
        out[f"post_{lab}"] = float(ca.sum())
    pre1 = np.bincount(ip[lp < 1.0], minlength=n) > 0
    post1 = np.bincount(ia[la < 1.0], minlength=n) > 0
    a, b = pre1[win], post1[win]
    out["off_pre_1s"] = float(a.sum())
    out["both_1s"] = float((a & b).sum())
    out["both_1s_indep"] = float(a.sum() * b.sum() / max(win.sum(), 1))
    out["nb_both_1s"], out["nb_both_1s_indep"] = _pair_oe(pre1, post1, win, (1, -1))
    out["far_both"] = out["far_indep"] = 0.0
    if n:
        ch = tr.chases
        nxt_on = np.r_[ch[1:, 0], np.inf]
        ex = np.sort(tr.exclude_intervals[:, 0]) if len(tr.exclude_intervals) else np.zeros(0)
        nxt_ex = np.r_[ex, np.inf][np.searchsorted(ex, ch[:, 1], side="right")]
        avail = win & (nxt_on >= ch[:, 1] + FAR[1]) & (nxt_ex >= ch[:, 1] + FAR[1])
        far = np.bincount(ia[(la >= FAR[0]) & (la < FAR[1])], minlength=n) > 0
        A, B = pre1[avail], far[avail]
        out["far_both"] = float((A & B).sum())
        out["far_indep"] = float(A.sum() * B.sum() / max(avail.sum(), 1))
    return out


def latency_counts(tr: Trial, edges: np.ndarray = PROFILE_EDGES) -> tuple[np.ndarray, np.ndarray]:
    """Pre- and post-offset sub chirp counts per latency bin (offsets inside the window)."""
    win = _offset_window(tr)
    W = float(edges[-1])
    _, ip, lp = ha.pre_offset_chirps(tr, W)
    _, ia, la = ha.post_offset_chirps(tr, W)
    return (np.histogram(lp[win[ip]], edges)[0].astype(float),
            np.histogram(la[win[ia]], edges)[0].astype(float))


# ---------------------------------------------------------------------------
# Parallel helper (forked workers read the module state _AC)
# ---------------------------------------------------------------------------
_AC: dict[str, Any] = {}


def _pool_map(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(min(workers, len(items)), mp_context=mp.get_context("fork"),
                             initializer=S._worker_init) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------------------
# Empirical latency profiles
# ---------------------------------------------------------------------------
def _profile_worker(rep: int) -> list[dict[str, Any]]:
    G = _AC
    ss = np.random.SeedSequence([int(G["cfg"].seed), zlib.crc32(b"acsim_profile"), int(rep)])
    sims = S._simulate(G["P_bg"], G["cfg"].bin_width, ss, tag=f"acsim_profile.{rep}")
    rows = []
    for tr in sims:
        pre, post = latency_counts(tr)
        rows.append(dict(recording=tr.meta["template"], pre=pre, post=post, **window_stats(tr)))
    return rows


def _mass_below(density: np.ndarray, w: float, edges: np.ndarray = PROFILE_EDGES) -> float:
    hi = edges[1:]
    return float(np.sum(density[hi <= w + 1e-9]))


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 0, None)
    if x.sum() <= 0:
        x = np.ones_like(x)
    return x / x.sum()


def estimate_profiles(trials: Sequence[Trial], P_bg: S.SimParams, cfg: Config, n_reps: int,
                      workers: int) -> dict[str, Any]:
    """Empirical pre / post-offset excess profiles and initial per-dyad sizes (module doc)."""
    _AC.update(cfg=cfg, P_bg=P_bg)
    res = _pool_map(_profile_worker, list(range(n_reps)), workers)
    _AC.pop("P_bg", None)
    recs = [tr.recording for tr in trials]
    nb = len(PROFILE_EDGES) - 1
    bg_pre, bg_post = np.zeros(nb), np.zeros(nb)
    bgw: dict[str, dict[str, float]] = {r: {} for r in recs}
    for rows in res:
        for r in rows:
            bg_pre += r["pre"] / n_reps
            bg_post += r["post"] / n_reps
            for k in ("n_off_win", "pre_2s", "post_2s"):
                bgw[r["recording"]][k] = bgw[r["recording"]].get(k, 0.0) + r[k] / n_reps
    real_pre, real_post = np.zeros(nb), np.zeros(nb)
    rw = {}
    bout_raw = bout_rate_ratio(trials)
    bout_nb = bout_rate_ratio(trials, shifts=(1, -1))
    for tr in trials:
        a, b = latency_counts(tr)
        real_pre += a
        real_post += b
        rw[tr.recording] = window_stats(tr)
    ex_pre, ex_post = real_pre - bg_pre, real_post - bg_post
    dens = dict(resp=_norm(ex_post), sym=_norm(ex_pre),
                resid=_norm(ex_post - np.clip(ex_pre, 0, None)))
    m2 = {k: _mass_below(v, MATCH_W) for k, v in dens.items()}
    init: dict[str, dict[str, float]] = {"resp": {}, "sym": {}, "resid": {}}
    for rec in recs:
        n_off = max(rw[rec]["n_off_win"], 1.0)
        d_post = max(0.0, rw[rec]["post_2s"] - bgw[rec].get("post_2s", 0.0))
        d_pre = max(0.0, rw[rec]["pre_2s"] - bgw[rec].get("pre_2s", 0.0))
        init["resp"][rec] = d_post / (n_off * m2["resp"])
        init["sym"][rec] = d_pre / (n_off * m2["sym"])
        # the cluster's post side covers d_pre x (sym mass < 2 s after) of the post excess
        init["resid"][rec] = max(0.0, d_post - d_pre) / (n_off * m2["resid"])
    table = pd.DataFrame(dict(
        edge_lo=PROFILE_EDGES[:-1], edge_hi=PROFILE_EDGES[1:], real_pre=real_pre,
        real_post=real_post, background_pre=bg_pre, background_post=bg_post,
        excess_pre=ex_pre, excess_post=ex_post, density_resp=dens["resp"],
        density_sym=dens["sym"], density_resid=dens["resid"]))
    return dict(density=dens, mass2=m2, init=init, table=table,
                bout_mult=max(1.0, bout_raw / bout_nb) if bout_nb > 0 else bout_raw,
                bout_raw=bout_raw, bout_nb=bout_nb, real=rw)


def bout_rate_ratio(trials: Sequence[Trial], w_pre: float = 1.0, w_post: float = MATCH_W,
                    shifts: Sequence[int] = (0,)) -> float:
    """Within-dyad (Mantel-Haenszel) rate ratio of post-offset sub chirps (0-w_post s) after
    offset i + s with vs without a sub chirp in the w_pre s before offset i, pooled over
    ``shifts`` (0: the same offset; +-1: the neighbouring offsets, where no bout can be
    shared, i.e. the slow co-variation part)."""
    num = den = 0.0
    for tr in trials:
        win = _offset_window(tr)
        n = len(win)
        if not win.any():
            continue
        _, ip, lp = ha.pre_offset_chirps(tr, w_pre)
        _, ia, la = ha.post_offset_chirps(tr, w_post)
        pre = np.bincount(ip, minlength=n) > 0
        post = np.bincount(ia, minlength=n).astype(float)
        idx = np.arange(n)
        for s in shifts:
            j = idx + s
            ok = (j >= 0) & (j < n)
            i_, j_ = idx[ok], j[ok]
            ok = win[i_] & win[j_]
            A, Bv = pre[i_[ok]], post[j_[ok]]
            N = float(len(A))
            if N == 0:
                continue
            num += Bv[A].sum() * (~A).sum() / N
            den += Bv[~A].sum() * A.sum() / N
    return float(num / den) if den > 0 else 1.0


# ---------------------------------------------------------------------------
# Slow modulation (drift account): strength matched to the neighbouring-offset co-occurrence
# ---------------------------------------------------------------------------
def drift_params(P_bg: S.SimParams, cfg: Config, mult: float) -> S.SimParams:
    return replace(P_bg, burst_cross=True, sub_mod_frac=float(_opt(cfg, "acsim_drift_frac")),
                   sub_mod_dur=float(_opt(cfg, "acsim_drift_dur")), sub_mod_mult=float(mult))


def _drift_worker(spec: tuple[float, int]) -> dict[str, Any]:
    m, rep = spec
    G = _AC
    ss = np.random.SeedSequence([int(G["cfg"].seed), zlib.crc32(b"acsim_drift"), int(rep)])
    sims = S._simulate(G["drift_P"][m], G["cfg"].bin_width, ss, tag=f"acsim_drift.{rep}")
    tot: dict[str, float] = {}
    for tr in sims:
        for k, v in window_stats(tr).items():
            tot[k] = tot.get(k, 0.0) + v
    return dict(mult=m, rep=rep, **tot)


def calibrate_drift(trials: Sequence[Trial], make: Callable[[float], S.SimParams], cfg: Config,
                    n_reps: int, workers: int) -> tuple[float, pd.DataFrame]:
    """High / low sub chirp rate ratio of the slow modulation whose pooled neighbouring-offset
    co-occurrence ratio matches the data's. ``make(mult)`` builds the generator (drift
    background with bursts crossing chase boundaries plus the uncalibrated response burst, no
    causal effect). The ratio is measured on n_reps datasets per grid value and interpolated
    linearly on its running maximum (monotone in the multiplier up to noise)."""
    real: dict[str, float] = {}
    for tr in trials:
        for k, v in window_stats(tr).items():
            real[k] = real.get(k, 0.0) + v
    target = real["nb_both_1s"] / max(real["nb_both_1s_indep"], 1e-9)
    grid = [float(m) for m in _opt(cfg, "acsim_drift_grid")]
    _AC.update(cfg=cfg, drift_P={m: make(m) for m in grid})
    res = _pool_map(_drift_worker, [(m, r) for m in grid for r in range(n_reps)], workers)
    _AC.pop("drift_P", None)
    g = pd.DataFrame(res).groupby("mult").sum(numeric_only=True)
    tab = pd.DataFrame(dict(
        mult=g.index.to_numpy(float),
        nb_ratio_1s=(g["nb_both_1s"] / g["nb_both_1s_indep"]).to_numpy(),
        both_ratio_1s=(g["both_1s"] / g["both_1s_indep"]).to_numpy(),
        far_ratio=(g["far_both"] / g["far_indep"]).to_numpy(), n_reps=n_reps))
    tab["local_ratio_1s"] = tab["both_ratio_1s"] / tab["nb_ratio_1s"]
    x = tab["mult"].to_numpy()
    y = np.maximum.accumulate(tab["nb_ratio_1s"].to_numpy())
    if target <= y[0]:
        m_star = float(x[0])
    elif target >= y[-1]:
        m_star = float(x[-1])
    else:
        j = int(np.flatnonzero(y >= target)[0]) - 1
        m_star = float(x[j] + (target - y[j]) * (x[j + 1] - x[j]) / max(y[j + 1] - y[j], 1e-9))
    tab["target_nb_ratio_1s"] = target
    tab["chosen_mult"] = m_star
    return m_star, tab


# ---------------------------------------------------------------------------
# Per-scenario parameters and calibration
# ---------------------------------------------------------------------------
def scenario_params(sc: AcScenario, P_bg: S.SimParams, prof: dict[str, Any], cfg: Config
                    ) -> S.SimParams:
    """Uncalibrated generative parameters of one scenario."""
    P = replace(P_bg, **sc.overrides)
    d, init = prof["density"], prof["init"]
    if sc.locked == "resp":
        bm = float(prof["bout_mult"]) if sc.bout else 1.0

        def mean_mult(rec: str) -> float:  # expected bout multiplier per offset (real data)
            r = prof["real"][rec]
            return 1.0 + (bm - 1.0) * r["off_pre_1s"] / max(r["n_off_win"], 1.0)

        P = replace(P, resp_edges=PROFILE_EDGES, resp_density=d["resp"], resp_bout_mult=bm,
                    trials=[replace(tp, resp_n=init["resp"][tp.recording] / mean_mult(tp.recording))
                            for tp in P.trials])
    elif sc.locked == "sym":
        if P.beta_off != 0:
            raise ValueError("the symmetric cluster is only defined without a causal offset effect")
        P = replace(P, resp_edges=PROFILE_EDGES, resp_density=d["resid"],
                    sym_edges=PROFILE_EDGES, sym_density=d["sym"],
                    sym_m=float(_opt(cfg, "acsim_sym_m")),
                    trials=[replace(tp, resp_n=init["resid"][tp.recording],
                                    sym_n=init["sym"][tp.recording]) for tp in P.trials])
    return P


def _locked_update(sc: AcScenario, prof: dict[str, Any]
                   ) -> Callable[..., list[S.TrialParams]] | None:
    """Additive per-dyad update of resp_n / sym_n toward the real 0-2 s counts."""
    if not sc.locked:
        return None
    m2 = prof["mass2"]
    m_resp = m2["resp"] if sc.locked == "resp" else m2["resid"]
    bm = float(prof["bout_mult"]) if sc.bout else 1.0

    def upd(tps: list[S.TrialParams], sim: pd.DataFrame, real: pd.DataFrame
            ) -> list[S.TrialParams]:
        # per-offset rates, so that noise in the simulated number of chases does not feed back
        out = []
        for tp in tps:
            r, s = real.loc[tp.recording], sim.loc[tp.recording]
            n_r, n_s = max(float(r["n_off_win"]), 1.0), float(s["n_off_win"])
            if not (np.isfinite(n_s) and n_s > 0):
                out.append(tp)
                continue
            d_post = r["post_2s"] / n_r - s["post_2s"] / n_s
            u = {}
            if sc.locked == "sym":
                sym = max(0.0, tp.sym_n + (r["pre_2s"] / n_r - s["pre_2s"] / n_s) / m2["sym"])
                d_post -= (sym - tp.sym_n) * m2["sym"]  # the cluster's post side moves too
                u["sym_n"] = sym
            mult = 1.0 + (bm - 1.0) * float(s["off_pre_1s"]) / n_s
            u["resp_n"] = max(0.0, tp.resp_n + d_post / (m_resp * mult))
            out.append(replace(tp, **{k: float(v) for k, v in u.items()}))
        return out

    return upd


def calibrate_scenario(sc: AcScenario, P_bg: S.SimParams, prof: dict[str, Any],
                       trials: Sequence[Trial], cfg: Config, n_iter: int, n_reps: int,
                       workers: int) -> tuple[S.SimParams, pd.DataFrame]:
    P = scenario_params(sc, P_bg, prof, cfg)
    tps, log = S.calibrate_params(P, {}, trials, cfg, "acsim", n_iter=n_iter, n_reps=n_reps,
                                  workers=workers, extra_stats=window_stats,
                                  extra_update=_locked_update(sc, prof))
    return replace(P, trials=tps), log.assign(calib=sc.name)


# ---------------------------------------------------------------------------
# Chase-start chirping (onset accounts)
# ---------------------------------------------------------------------------
def onset_start_stats(tr: Trial) -> dict[str, float]:
    """Sub chirps per latency bin after the (annotated) onsets inside the analysis window,
    inside their chase (START_EDGES; a bin stops at the chase's offset), and in the 1 s before
    those onsets; the onsets as in hz_acausal.state_rates."""
    nb = len(START_EDGES) - 1
    out = {f"on_start_{i}": 0.0 for i in range(nb)}
    out.update(on_pre_1s=0.0, n_on_win=0.0)
    ch = tr.chases
    if not len(ch):
        return out
    ons = ch[(ch[:, 0] >= tr.obs_start) & (ch[:, 0] < tr.obs_end)]
    x = np.sort(np.asarray(tr.chirps_sub, float))
    h = np.zeros(nb)
    for on, off in ons:
        a, b = np.searchsorted(x, [on, min(off, on + float(START_EDGES[-1]))])
        h += np.histogram(x[a:b] - on, START_EDGES)[0]
    pre = np.searchsorted(x, ons[:, 0]) - np.searchsorted(x, ons[:, 0] - 1.0)
    out.update({f"on_start_{i}": float(h[i]) for i in range(nb)})
    out.update(on_pre_1s=float(pre.sum()), n_on_win=float(len(ons)))
    return out


def _start_shape(shape_in: np.ndarray, boost: np.ndarray) -> np.ndarray:
    s = np.asarray(shape_in, float).copy()
    s[:len(boost)] *= boost
    return s


def calibrate_start(P: S.SimParams, trials: Sequence[Trial], cfg: Config, key: str,
                    n_iter: int, n_reps: int, workers: int
                    ) -> tuple[S.SimParams, pd.DataFrame, np.ndarray]:
    """Chase-start profile: multipliers on the first len(START_EDGES) - 1 bins of shape_in
    (0-2 s after an onset) that make the pooled simulated sub chirp counts per bin after
    annotated onsets match the real ones. Each round runs hz_sim.calibrate_params with one
    rate update (chase hazards and in / out-of-chase chirp rates per dyad stay matched, under
    the scenario's effect), then rescales the multipliers by real / simulated (steps clipped
    to [0.5, 2]); the last round only evaluates. Every round uses the same seeds (common random
    numbers), so the changes between rounds reflect the multipliers, not simulation noise.
    Pooled over dyads: ~60 real chirps fall in the first 1 s of all chases together."""
    nb = len(START_EDGES) - 1
    cols = [f"on_start_{i}" for i in range(nb)]
    real = np.zeros(nb)
    for tr in trials:
        s = onset_start_stats(tr)
        real += np.array([s[c] for c in cols])
    shape0 = np.asarray(P.shape_in, float).copy()
    boost = np.ones(nb)
    logs = []
    for it in range(n_iter + 1):
        Pb = replace(P, shape_in=_start_shape(shape0, boost))
        tps, lg = S.calibrate_params(Pb, {}, trials, cfg, f"{key}.start", n_iter=1,
                                     n_reps=n_reps, workers=workers,
                                     extra_stats=onset_start_stats)
        P = replace(P, trials=tps)
        ratio = np.array([float(lg[f"ratio_{c}"].iloc[-1]) for c in cols])
        logs.append(lg.assign(start_round=it, **{f"start_mult_{i}": float(boost[i])
                                                 for i in range(nb)},
                              **{f"real_{c}": float(real[i]) for i, c in enumerate(cols)}))
        if it < n_iter:  # the last round's multipliers are the ones evaluated and returned
            step = np.where(np.isfinite(ratio) & (ratio > 0), 1.0 / np.maximum(ratio, 1e-9), 1.0)
            boost = boost * np.clip(step, 0.5, 2.0)
    return replace(P, shape_in=_start_shape(shape0, boost)), pd.concat(logs, ignore_index=True), \
        boost


# ---------------------------------------------------------------------------
# Statistics of one dataset
# ---------------------------------------------------------------------------
def _v(E: pd.DataFrame, model: str, term: str, col: str = "ratio") -> float:
    x = ha._pick(E, model=model, term=term)
    return float(x[col]) if x is not None else np.nan


def acausal_sim_stats(trials: Sequence[Trial], cfg: Config, W: float = MATCH_W
                      ) -> dict[str, float]:
    """The offset acausal statistics (glm_fe; same functions as run_acausal)."""
    c1 = cfg.updated(tau_grid=(1.0,))
    g = ha._gname(c1)
    spec_all = ha.two_sided_specs(c1, (1.0,))
    specs = {m: spec_all[m] for m in ("past_1000ms", "pf_1000ms", f"pf_guard_{g}",
                                      "kernel_two_sided")}
    with np.errstate(over="ignore"):  # CI bounds of unused dominant-chirp terms may overflow
        r = ha._two_sided_block(trials, "offset", c1, specs, ("glm_fe",), (), (), False)
    A = r["effects"]
    out = dict(
        hr_primary=_v(A, "past_1000ms", "sub_any_1000ms"),
        hr_past=_v(A, "pf_1000ms", "sub_any_1000ms"),
        hr_future=_v(A, "pf_1000ms", "sub_fut_1000ms"),
        ratio_pf=_v(A, "pf_1000ms", "sub_any_1000ms-sub_fut_1000ms"),
        p_pf=_v(A, "pf_1000ms", "sub_any_1000ms-sub_fut_1000ms", "p"),
        hr_past_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}"),
        hr_future_grd=_v(A, f"pf_guard_{g}", f"sub_fgd_{g}"),
        ratio_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}-sub_fgd_{g}"),
        p_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}-sub_fgd_{g}", "p"),
        k_before=_v(A, "kernel_two_sided", "kernel_sub_mean"),
        k_after=_v(A, "kernel_two_sided", "kernel_sub_fut_mean"),
        ratio_k=_v(A, "kernel_two_sided", "kernel_sub_mean-kernel_sub_fut_mean"),
        p_k=_v(A, "kernel_two_sided", "kernel_sub_mean-kernel_sub_fut_mean", "p"),
        p_kshape=_v(A, "kernel_two_sided", "kernel_sub_shape_diff_joint", "p"),
        se_primary=_v(A, "past_1000ms", "sub_any_1000ms", "se"),
        se_pf=_v(A, "pf_1000ms", "sub_any_1000ms-sub_fut_1000ms", "se"),
        se_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}-sub_fgd_{g}", "se"),
        se_k=_v(A, "kernel_two_sided", "kernel_sub_mean-kernel_sub_fut_mean", "se"),
        n_dropped_bins=float(r["bookkeeping"]["n_dropped_future_past_end"].sum()),
        converged=float(A["converged"].astype(bool).all()),
    )
    arms, _, _ = ha.build_arms(trials, c1, (W,), W, ())
    for (lab, _W), ov in arms.items():
        with np.errstate(over="ignore"):
            e, _, _ = ha.fit_arm(trials, c1, ov, backends=("glm_fe",),
                                 models=["tau_1000ms", f"guard_{g}"])
        key = "restricted" if lab == "restricted_real" else "mirror"
        out[f"hr_{key}"] = _v(e, "tau_1000ms", "sub_any_1000ms")
        out[f"hr_{key}_grd"] = _v(e, f"guard_{g}", f"sub_grd_{g}")
        out["converged"] = float(out["converged"] and bool(e["converged"].astype(bool).all()))
    out["ratio_rm"] = out["hr_restricted"] / out["hr_mirror"]
    out["ratio_rm_grd"] = out["hr_restricted_grd"] / out["hr_mirror_grd"]
    return out


def onset_sim_stats(trials: Sequence[Trial], cfg: Config) -> dict[str, float]:
    """Onset acausal statistics (glm_fe) on a case-control onset design (all event bins, a
    fraction acsim_onset_fraction of the others; IPW cloglog)."""
    c1 = cfg.updated(tau_grid=(1.0,),
                     onset_control_fraction=float(_opt(cfg, "acsim_onset_fraction")))
    g = ha._gname(c1)
    sa = ha.two_sided_specs(c1, (1.0,))
    specs = {m: sa[m] for m in ("past_1000ms", "pf_1000ms", f"past_guard_{g}", f"pf_guard_{g}",
                                "fut_1000ms", f"fut_guard_{g}")}
    with np.errstate(over="ignore"):
        r = ha._two_sided_block(trials, "onset", c1, specs, ("glm_fe",), (), (), False)
    A = r["effects"]
    out = dict(
        on_hr_primary=_v(A, "past_1000ms", "sub_any_1000ms"),
        on_hr_past=_v(A, "pf_1000ms", "sub_any_1000ms"),
        on_hr_future=_v(A, "pf_1000ms", "sub_fut_1000ms"),
        on_ratio_pf=_v(A, "pf_1000ms", "sub_any_1000ms-sub_fut_1000ms"),
        on_p_pf=_v(A, "pf_1000ms", "sub_any_1000ms-sub_fut_1000ms", "p"),
        on_se_pf=_v(A, "pf_1000ms", "sub_any_1000ms-sub_fut_1000ms", "se"),
        on_hr_primary_grd=_v(A, f"past_guard_{g}", f"sub_grd_{g}"),
        on_hr_past_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}"),
        on_hr_future_grd=_v(A, f"pf_guard_{g}", f"sub_fgd_{g}"),
        on_ratio_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}-sub_fgd_{g}"),
        on_p_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}-sub_fgd_{g}", "p"),
        on_se_grd=_v(A, f"pf_guard_{g}", f"sub_grd_{g}-sub_fgd_{g}", "se"),
        on_se_primary=_v(A, "past_1000ms", "sub_any_1000ms", "se"),
        # the future window alone (not adjusted for the past window): chase-start chirping
        on_hr_future_alone=_v(A, "fut_1000ms", "sub_fut_1000ms"),
        on_hr_future_alone_grd=_v(A, f"fut_guard_{g}", f"sub_fgd_{g}"),
        on_n_events=float(ha._pick(A, model="past_1000ms", term="sub_any_1000ms")["n_events"])
        if ha._pick(A, model="past_1000ms", term="sub_any_1000ms") is not None else np.nan,
        on_converged=float(A["converged"].astype(bool).all()),
    )
    out["on_joint_shift"] = out["on_hr_past"] / out["on_hr_primary"]
    out["on_joint_shift_grd"] = out["on_hr_past_grd"] / out["on_hr_primary_grd"]
    # chase-start chirping (what raises the after-window of an onset bin) and pre-onset chirps
    tot: dict[str, float] = {}
    for tr in trials:
        for k, v in onset_start_stats(tr).items():
            tot[k] = tot.get(k, 0.0) + v
    out["on_n_start_1s"] = tot["on_start_0"] + tot["on_start_1"]
    out["on_n_start_grd"] = tot["on_start_1"] + tot["on_start_2"]
    out["on_n_pre_1s"] = tot["on_pre_1s"]
    out["on_n_onsets"] = tot["n_on_win"]
    return out


def dataset_counts(trials: Sequence[Trial]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Pooled and per-dyad chirp counts around offsets; in / out-of-chase totals."""
    per = []
    tot: dict[str, float] = {}
    n_in = n_out = n_trig = 0.0
    for tr in trials:
        w = window_stats(tr)
        per.append(dict(recording=tr.meta.get("template", tr.recording), **w))
        for k, x in w.items():
            tot[k] = tot.get(k, 0.0) + x
        v = tr.valid(tr.chirps_sub)
        ic = tr.in_chase(tr.chirps_sub)
        n_in += float((v & ic).sum())
        n_out += float((v & ~ic).sum())
        n_trig += float(tr.meta.get("n_trigger_in_chase", 0))
    both = tot["both_1s"] / max(tot["both_1s_indep"], 1e-9)
    nb = tot["nb_both_1s"] / max(tot["nb_both_1s_indep"], 1e-9)
    out = dict(n_offsets=tot["n_off_win"], n_pre_1s=tot["pre_1s"], n_post_1s=tot["post_1s"],
               n_pre_2s=tot["pre_2s"], n_post_2s=tot["post_2s"], n_both_1s=tot["both_1s"],
               n_both_1s_indep=tot["both_1s_indep"], both_ratio_1s=both,
               n_nb_both_1s=tot["nb_both_1s"], n_nb_both_1s_indep=tot["nb_both_1s_indep"],
               nb_ratio_1s=nb, local_ratio_1s=both / nb if nb > 0 else np.nan,
               far_ratio=tot["far_both"] / max(tot["far_indep"], 1e-9),
               n_sub_in=n_in, n_sub_out=n_out, trigger_share=n_trig / max(n_in, 1.0))
    return out, per


def _rep_task(spec: tuple[str, int]) -> dict[str, Any]:
    G = _AC
    name, rep = spec
    t0 = time.perf_counter()
    if name == "observed":
        trials = G["trials"]
    else:
        ss = np.random.SeedSequence([int(G["cfg"].seed), zlib.crc32(b"acausal_sim"), int(rep)])
        trials = S._simulate(G["params"][name], G["cfg"].bin_width, ss, tag=f"{name}.{rep}")
    t_sim = time.perf_counter() - t0
    st = acausal_sim_stats(trials, G["cfg"])
    cnt, per = dataset_counts(trials)
    rows = [dict(scenario=name, rep=rep, stat=k, value=float(x)) for k, x in {**st, **cnt}.items()]
    return dict(rows=rows, per=[dict(scenario=name, rep=rep, **p) for p in per],
                timing=dict(scenario=name, rep=rep, simulate_s=t_sim,
                            total_s=time.perf_counter() - t0))


def _onset_task(spec: tuple[str, int]) -> dict[str, Any]:
    G = _AC
    name, rep = spec
    t0 = time.perf_counter()
    if name == "observed":
        trials = G["trials"]
    else:
        ss = np.random.SeedSequence([int(G["cfg"].seed), zlib.crc32(b"acausal_sim_onset"),
                                     int(rep)])
        trials = S._simulate(G["on_params"][name], G["cfg"].bin_width, ss, tag=f"{name}.{rep}")
    st = onset_sim_stats(trials, G["cfg"])
    return dict(rows=[dict(scenario=name, rep=rep, stat=k, value=float(x)) for k, x in st.items()],
                timing=dict(scenario=name, rep=rep, simulate_s=np.nan,
                            total_s=time.perf_counter() - t0))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
STATS: dict[str, str] = {  # stat -> label (ratios are hazard ratios or ratios of them)
    "hr_primary": "HR, standard 1 s model",
    "hr_past": "HR_past, joint 1 s model",
    "hr_future": "HR_future, joint 1 s model",
    "ratio_pf": "HR_past / HR_future, 1 s",
    "hr_past_grd": "HR_past, guarded 0.5-1.5 s",
    "hr_future_grd": "HR_future, guarded 0.5-1.5 s",
    "ratio_grd": "HR_past / HR_future, guarded",
    "k_before": "kernel mean, before the bin",
    "k_after": "kernel mean, after the bin",
    "ratio_k": "kernel mean before / after",
    "hr_restricted": "restricted real arm, 1 s (W = 2 s)",
    "hr_mirror": "mirror arm, 1 s (W = 2 s)",
    "ratio_rm": "restricted real / mirror, 1 s",
    "hr_restricted_grd": "restricted real arm, guarded",
    "hr_mirror_grd": "mirror arm, guarded",
    "ratio_rm_grd": "restricted real / mirror, guarded",
    "n_pre_1s": "sub chirps 0-1 s before offsets",
    "n_post_1s": "sub chirps 0-1 s after offsets",
    "n_pre_2s": "sub chirps 0-2 s before offsets",
    "n_post_2s": "sub chirps 0-2 s after offsets",
    "n_both_1s": "offsets with a sub chirp on both sides (1 s)",
    "both_ratio_1s": "both sides (1 s), same offset / expected if independent within dyads",
    "nb_ratio_1s": "pre side of offset i x post side of offsets i +- 1 / expected (slow "
                   "co-variation baseline)",
    "local_ratio_1s": "both sides, same offset / neighbouring offsets (offset-specific excess)",
    "far_ratio": "pre 0-1 s x post 2-3 s, same offset / expected",
    "n_offsets": "chase offsets",
    "n_sub_in": "in-chase sub chirps",
    "n_sub_out": "out-of-chase sub chirps",
    "trigger_share": "trigger chirps / in-chase sub chirps",
}
ON_STATS: dict[str, str] = {
    "on_hr_primary": "HR, standard 1 s model (onset)",
    "on_hr_primary_grd": "HR, standard guarded model (onset)",
    "on_hr_past": "HR_past, joint 1 s model",
    "on_hr_future": "HR_future, joint 1 s model",
    "on_ratio_pf": "HR_past / HR_future, 1 s",
    "on_hr_past_grd": "HR_past, joint guarded model",
    "on_hr_future_grd": "HR_future, joint guarded model",
    "on_hr_future_alone": "HR_future, future window alone, 1 s",
    "on_hr_future_alone_grd": "HR_future, future window alone, guarded",
    "on_ratio_grd": "HR_past / HR_future, guarded",
    "on_joint_shift": "joint / standalone HR_past, 1 s",
    "on_joint_shift_grd": "joint / standalone HR_past, guarded",
    "on_n_start_1s": "sub chirps in the first 1 s of chases",
    "on_n_start_grd": "sub chirps 0.5-1.5 s after chase onsets",
    "on_n_pre_1s": "sub chirps in the 1 s before chase onsets",
    "on_n_onsets": "chase onsets",
}
CONTRASTS = {"ratio_pf": "p_pf", "ratio_grd": "p_grd", "ratio_k": "p_k",
             "on_ratio_pf": "on_p_pf", "on_ratio_grd": "on_p_grd"}
KEY_STATS = ("ratio_pf", "ratio_grd", "ratio_k", "ratio_rm", "ratio_rm_grd")
COOC_STATS = ("both_ratio_1s", "nb_ratio_1s", "local_ratio_1s", "far_ratio")
COOC_SUMS = {"both_ratio_1s": ("both_1s", "both_1s_indep"),
             "nb_ratio_1s": ("nb_both_1s", "nb_both_1s_indep"),
             "far_ratio": ("far_both", "far_indep")}
SE_OF = {"hr_primary": "se_primary", "ratio_pf": "se_pf", "ratio_grd": "se_grd", "ratio_k": "se_k",
         "on_hr_primary": "on_se_primary", "on_ratio_pf": "on_se_pf", "on_ratio_grd": "on_se_grd"}
# real-data bootstrap rows of hz_acausal (acausal/acausal_effects.csv) for the mirror contrasts
BOOT_TERMS = {"ratio_rm": "sub_any_1000ms", "ratio_rm_grd": "sub_grd_500ms"}
T14 = 2.1447866879169273  # t(14) 97.5% quantile (15 dyads)


def observed_uncertainty(est: pd.DataFrame, per: pd.DataFrame, out_dir: Path | None,
                         n_boot: int = 2000, seed: int = 42,
                         est_on: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uncertainty of the observed key statistics (log scale): the cluster-robust SE of the
    real fit (standard HR, past/future and kernel contrasts, onset ones), hz_acausal's dyad
    bootstrap of the restricted real - mirror contrast (W = 2 s), and a dyad bootstrap of the
    co-occurrence ratios."""
    o = est[est["scenario"] == "observed"].set_index("stat")["value"]
    if est_on is not None and len(est_on):
        o = pd.concat([o, est_on[est_on["scenario"] == "observed"].set_index("stat")["value"]])
    rows = []
    for stat, se in SE_OF.items():
        if stat in o and se in o:
            rows.append(dict(stat=stat, observed=float(o[stat]), se_log=float(o[se]),
                             method="cluster-robust SE of the real fit (t(14) CI)"))
    ae = Path(out_dir) / "acausal" / "acausal_effects.csv" if out_dir is not None else None
    if ae is not None and ae.exists():
        a = pd.read_csv(ae, usecols=["variant", "subset", "term", "estimate", "se", "ci_low",
                                     "ci_high"], float_precision="round_trip")
        a = a[(a["variant"] == "restricted_real-mirror")
              & (a["subset"] == f"w_{int(MATCH_W * 1000)}ms")]
        for stat, term in BOOT_TERMS.items():
            r = a[a["term"] == term]
            if len(r) and stat in o:
                rows.append(dict(stat=stat, observed=float(o[stat]),
                                 se_log=float(r["se"].iloc[0]),
                                 ci_low=float(np.exp(r["ci_low"].iloc[0])),
                                 ci_high=float(np.exp(r["ci_high"].iloc[0])),
                                 method="hz_acausal dyad bootstrap (SD, percentile CI; "
                                        "acausal_effects.csv)"))
    ob = per[per["scenario"] == "observed"] if per is not None and len(per) else pd.DataFrame()
    if len(ob) and all(c in ob for c in ("nb_both_1s", "far_both")):
        rng = np.random.default_rng([seed, 1957])
        idx = rng.integers(0, len(ob), size=(n_boot, len(ob)))
        logs = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for stat, (num, den) in COOC_SUMS.items():
                b, e = ob[num].to_numpy(float), ob[den].to_numpy(float)
                logs[stat] = np.log(b[idx].sum(1) / e[idx].sum(1))
            logs["local_ratio_1s"] = logs["both_ratio_1s"] - logs["nb_ratio_1s"]
        for stat in COOC_STATS:
            v = logs[stat]
            v = v[np.isfinite(v)]
            if stat in o and len(v) > 1:
                rows.append(dict(stat=stat, observed=float(o[stat]),
                                 se_log=float(np.std(v, ddof=1)),
                                 ci_low=float(np.exp(np.quantile(v, 0.025))),
                                 ci_high=float(np.exp(np.quantile(v, 0.975))),
                                 method=f"dyad bootstrap ({len(v)} resamples, percentile CI)"))
    out = pd.DataFrame(rows).reindex(columns=["stat", "observed", "se_log", "ci_low", "ci_high",
                                              "method"])
    if len(out):
        m = out["ci_low"].isna()
        e = np.exp(T14 * out.loc[m, "se_log"])
        out.loc[m, "ci_low"] = out.loc[m, "observed"] / e
        out.loc[m, "ci_high"] = out.loc[m, "observed"] * e
    return out


def exceedance(x: np.ndarray, o: float) -> dict[str, Any]:
    """Position of the observed value o among simulated values x (finite): datasets strictly
    below / above it, the side on which it lies ('above' when at most as many datasets are at
    or above it as at or below it), the number of datasets it lies beyond on that side, and
    the exceedance fraction min(share >= o, share <= o)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0 or not np.isfinite(o):
        return dict(n_below_obs=np.nan, n_above_obs=np.nan, exceed_side="", exceed_n=np.nan,
                    exceed_frac=np.nan)
    lt, gt = int(np.sum(x < o)), int(np.sum(x > o))
    f_ge, f_le = (n - lt) / n, (n - gt) / n
    side = "above" if f_ge <= f_le else "below"
    return dict(n_below_obs=lt, n_above_obs=gt, exceed_side=side,
                exceed_n=lt if side == "above" else gt, exceed_frac=float(min(f_ge, f_le)))


def status_of(frac: float = np.nan, p: float = np.nan) -> str:
    """'inconsistent' / 'borderline' / 'consistent' from an exceedance fraction (< 2.5%,
    2.5-10%, else) or, if p is given, from a two-sided p (< 0.05, 0.05-0.2, else); '' if
    neither is finite."""
    if np.isfinite(p):
        return ("inconsistent" if p < P_INCONSISTENT else "borderline" if p < P_BORDERLINE
                else "consistent")
    if np.isfinite(frac):
        return ("inconsistent" if frac < EXC_INCONSISTENT else "borderline"
                if frac < EXC_BORDERLINE else "consistent")
    return ""


def add_status(summ: pd.DataFrame) -> pd.DataFrame:
    """Per row: status by the exceedance fraction (status_range), by z (status_z, where an
    observed SE exists), the statistic's criterion (``criterion``) and the status by it."""
    s = summ.copy()
    if s.empty:
        return s
    s["status_range"] = [status_of(f) for f in s["exceed_frac"].to_numpy(float)]
    pz = s["p_obs"].to_numpy(float) if "p_obs" in s else np.full(len(s), np.nan)
    s["status_z"] = [status_of(p=p) if np.isfinite(p) else "" for p in pz]
    crit = {st: criterion(s, st)[0] for st in pd.unique(s["stat"])}
    s["criterion"] = s["stat"].map(crit)
    s["status"] = np.where((s["criterion"] == "z") & (s["status_z"] != ""), s["status_z"],
                           s["status_range"])
    return s


def summarize(est: pd.DataFrame, obs_unc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Median, 95% range and the observed value's position per scenario and statistic
    (percentile, exceedance: ``exceedance``); with obs_unc also z = (log observed - log
    scenario median) / SE of the observed statistic and its two-sided p; for statistics with a
    simulated cluster-robust SE, its 97.5% quantile and the share of replicates whose t(14)
    interval covers the replicate mean (calibration of the SE); last, the status of each
    account by exceedance, by z and by the statistic's criterion (``add_status``)."""
    from scipy import stats as st

    obs = est[est["scenario"] == "observed"].set_index("stat")["value"]
    unc = obs_unc.set_index("stat") if obs_unc is not None and len(obs_unc) else pd.DataFrame()
    sim = est[est["scenario"] != "observed"]
    wide = sim.pivot_table(index=["scenario", "rep"], columns="stat", values="value")
    skip = ("p_", "se_", "on_p_", "on_se_")
    rows = []
    for sc in pd.unique(sim["scenario"]):
        w = wide.loc[sc]
        for stat in [s for s in wide.columns if not s.startswith(skip)]:
            x = w[stat].to_numpy(float)
            x = x[np.isfinite(x)]
            o = float(obs.get(stat, np.nan))
            r = dict(scenario=sc, stat=stat, label={**STATS, **ON_STATS}.get(stat, stat),
                     n=len(x), median=float(np.median(x)) if len(x) else np.nan,
                     q025=float(np.quantile(x, 0.025)) if len(x) else np.nan,
                     q975=float(np.quantile(x, 0.975)) if len(x) else np.nan,
                     mean=float(np.mean(x)) if len(x) else np.nan, observed=o,
                     obs_pct=(100 * (np.mean(x < o) + 0.5 * np.mean(x == o))
                              if len(x) and np.isfinite(o) else np.nan))
            r["obs_inside"] = bool(np.isfinite(o) and r["q025"] <= o <= r["q975"])
            r.update(exceedance(x, o))
            if stat in CONTRASTS and CONTRASTS[stat] in w:
                p = w[CONTRASTS[stat]].to_numpy(float)
                xs = w[stat].to_numpy(float)
                ok = np.isfinite(p) & np.isfinite(xs)
                r["reject_pos"] = float(np.mean((p[ok] < 0.05) & (xs[ok] > 1)))
                r["reject_neg"] = float(np.mean((p[ok] < 0.05) & (xs[ok] < 1)))
                r["observed_p"] = float(obs.get(CONTRASTS[stat], np.nan))
            lx = np.log(x[x > 0]) if len(x) else np.zeros(0)
            if len(lx) > 1:
                r["sim_sd_log"] = float(np.std(lx, ddof=1))
            if stat in SE_OF and SE_OF[stat] in w:
                xs = w[stat].to_numpy(float)
                se = w[SE_OF[stat]].to_numpy(float)
                ok = np.isfinite(xs) & (xs > 0) & np.isfinite(se)
                if ok.sum() > 1:
                    lxx = np.log(xs[ok])
                    r["sim_mean_se"] = float(np.mean(se[ok]))
                    r["sim_se_q975"] = float(np.quantile(se[ok], 0.975))
                    r["sim_cr_cover"] = float(np.mean(np.abs(lxx - lxx.mean()) <= T14 * se[ok]))
            if stat in unc.index and np.isfinite(o) and o > 0 and r["median"] > 0:
                u = unc.loc[stat]
                r.update(obs_se_log=float(u["se_log"]), obs_ci_low=float(u["ci_low"]),
                         obs_ci_high=float(u["ci_high"]))
                z = (np.log(o) - np.log(r["median"])) / float(u["se_log"])
                r["z_obs"] = float(z)
                r["p_obs"] = float(2 * st.norm.sf(abs(z)))
            rows.append(r)
    return add_status(pd.DataFrame(rows))


def dyad_counts(per: pd.DataFrame) -> pd.DataFrame:
    """Per-dyad mean simulated counts around offsets, with the real counts."""
    cols = [c for c in ("n_off_win", "pre_1s", "pre_2s", "post_1s", "post_2s", "both_1s",
                        "both_1s_indep", "nb_both_1s", "nb_both_1s_indep") if c in per]
    obs = per[per["scenario"] == "observed"].set_index("recording")[cols]
    sim = per[per["scenario"] != "observed"].groupby(["scenario", "recording"])[cols].mean()
    out = sim.reset_index()
    for c in cols:
        out[f"real_{c}"] = out["recording"].map(obs[c])
    return out


# ---------------------------------------------------------------------------
# Criteria and reading
# ---------------------------------------------------------------------------
def _fmt(x: float, count: bool = False) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.0f}" if count else f"{x:.2f}"


def _is_count(stat: str) -> bool:
    return stat.startswith(("n_", "on_n_"))


def _row(summ: pd.DataFrame, sc: str, stat: str) -> pd.Series | None:
    r = summ[(summ["scenario"] == sc) & (summ["stat"] == stat)]
    return r.iloc[0] if len(r) else None


def _cell(summ: pd.DataFrame, sc: str, stat: str) -> str:
    r = _row(summ, sc, stat)
    if r is None:
        return "n/a"
    c = _is_count(stat)
    return f"{_fmt(r['median'], c)} [{_fmt(r['q025'], c)}, {_fmt(r['q975'], c)}]"


def _obs(summ: pd.DataFrame, stat: str) -> float:
    r = summ[summ["stat"] == stat]
    return float(r["observed"].iloc[0]) if len(r) else np.nan


def _exc(r: pd.Series | None) -> str:
    """'above 29/30': the observed value lies above 29 of the 30 simulated datasets (its side
    is the one with fewer datasets at or beyond it); 'below ...' likewise."""
    if r is None or not np.isfinite(float(r.get("exceed_n", np.nan))):
        return "n/a"
    return f"{r['exceed_side']} {int(r['exceed_n'])}/{int(r['n'])}"


def _crit_of(r: pd.Series | None, crit: str | None) -> str:
    if crit:
        return crit
    c = r.get("criterion", "range") if r is not None else "range"
    return c if c in ("range", "z") else "range"


def _pos(r: pd.Series | None, crit: str | None = None) -> str:
    """Observed position for one account: exceedance, plus z's p under the z criterion."""
    s = _exc(r)
    if _crit_of(r, crit) == "z" and r is not None and np.isfinite(float(r.get("p_obs", np.nan))):
        s += f"; z p = {float(r['p_obs']):.2f}"
    return s


def _status(summ: pd.DataFrame, sc: str, stat: str, crit: str) -> str | None:
    """Status of account ``sc`` for ``stat`` by criterion ``crit`` ('range': exceedance
    fraction; 'z': two-sided p of z, falling back to the exceedance without an observed SE);
    None without an observed value."""
    r = _row(summ, sc, stat)
    if r is None or not np.isfinite(float(r.get("observed", np.nan))):
        return None
    p = float(r.get("p_obs", np.nan))
    if crit == "z" and np.isfinite(p):
        return status_of(p=p)
    return status_of(float(r.get("exceed_frac", np.nan))) or None


def _cellx(summ: pd.DataFrame, sc: str, stat: str, crit: str | None = None) -> str:
    """Median [95% range] (observed position[, status if not consistent])."""
    r = _row(summ, sc, stat)
    if r is None:
        return "n/a"
    c = _crit_of(r, crit)
    st = _status(summ, sc, stat, c)
    flag = f", {st}" if st in ("borderline", "inconsistent") else ""
    return f"{_cell(summ, sc, stat)} ({_pos(r, c)}{flag})"


SHORT = {"hr_primary": "standard HR", "ratio_pf": "past/future 1 s",
         "ratio_grd": "past/future guarded", "ratio_k": "kernel ratio",
         "ratio_rm": "restricted/mirror 1 s", "ratio_rm_grd": "restricted/mirror guarded",
         "n_pre_1s": "chirps 0-1 s before", "n_post_1s": "chirps 0-1 s after",
         "n_pre_2s": "chirps 0-2 s before", "n_post_2s": "chirps 0-2 s after",
         "both_ratio_1s": "both sides vs independence", "nb_ratio_1s": "neighbour baseline",
         "local_ratio_1s": "both sides, local", "far_ratio": "pre x post 2-3 s",
         "on_hr_primary": "standard HR", "on_ratio_pf": "past/future 1 s",
         "on_ratio_grd": "past/future guarded", "on_hr_future": "HR_future 1 s",
         "on_hr_future_grd": "HR_future guarded", "on_hr_past": "joint HR_past 1 s",
         "on_hr_past_grd": "joint HR_past guarded", "on_n_start_1s": "chirps in the first 1 s",
         "on_n_start_grd": "chirps 0.5-1.5 s after onset", "on_n_pre_1s": "chirps 1 s before",
         "on_hr_future_alone": "future window alone 1 s",
         "on_hr_future_alone_grd": "future window alone guarded"}
FIT_STATS = ("hr_primary", "n_pre_1s", "n_post_1s")  # what an offset account must reproduce
COUNT_STATS = ("n_pre_1s", "n_post_1s", "n_pre_2s", "n_post_2s")
VERDICT_STATS = (("ratio_pf", "HR_past / HR_future, 1 s"),
                 ("ratio_rm", "restricted real / mirror, 1 s"),
                 ("ratio_grd", "HR_past / HR_future, guarded"),
                 ("ratio_k", "kernel before / after"),
                 ("ratio_rm_grd", "restricted real / mirror, guarded"),
                 ("local_ratio_1s", "co-occurrence on both sides, same / neighbouring offsets"))
ON_VERDICT_STATS = (("on_ratio_pf", "HR_past / HR_future, 1 s"),
                    ("on_ratio_grd", "HR_past / HR_future, guarded"))
# what tells whether an onset account reproduces chase-start chirping
START_CHECK = ("on_n_start_1s", "on_n_start_grd", "on_hr_future_alone", "on_hr_future_alone_grd")
COMPAT = ("consistent", "borderline")
# statistics whose causal / non-causal accounts differ in more than causality
STAT_CAVEAT = {
    "local_ratio_1s": ("caveat: only a causal account has the slow co-variation of chirping "
                       "(drift), the trigger has no offset-specific co-occurrence and the "
                       "symmetric cluster's size is not calibrated, so this verdict does not "
                       "separate causal from non-causal timing"),
}
STAT_CAVEAT_SHORT = {"local_ratio_1s": "not a causal / non-causal contrast, see caveat"}
CRIT_NAME = {"range": "exceedance", "z": "z"}
CAT_TEXT = {
    "none": "compatible with causal and non-causal accounts alike: does not discriminate",
    "causal": "compatible only with causal accounts, inconsistent with every non-causal one",
    "noncausal": "compatible only with non-causal accounts, inconsistent with every causal one",
    "above": "inconsistent with every account (above all of them)",
    "below": "inconsistent with every account (below all of them)",
    "between": "inconsistent with every account",
}
CAT_SHORT = {"none": "does not discriminate", "causal": "causal accounts only",
             "noncausal": "non-causal accounts only", "above": "above every account",
             "below": "below every account", "between": "inconsistent with every account"}
STATUS_RULE = ("status per account: inconsistent if fewer than 2.5% of its datasets lie at or "
               "beyond the observed value (z criterion: p < 0.05), borderline if 2.5-10% (z: "
               "p 0.05-0.2), consistent otherwise; 'above k/n' = the observed value lies above k "
               "of the n datasets")


def _se_ratio(summ: pd.DataFrame) -> pd.Series:
    """Median over scenarios of (mean cluster-robust SE of the simulated fits / SD of their
    estimates), per statistic."""
    if "sim_mean_se" not in summ:
        return pd.Series(dtype=float)
    d = summ.dropna(subset=["sim_mean_se", "sim_sd_log"])
    return (d["sim_mean_se"] / d["sim_sd_log"]).groupby(d["stat"]).median()


def criterion(summ: pd.DataFrame, stat: str) -> tuple[str, str, bool]:
    """Which comparison is calibrated for ``stat``: 'range' (the exceedance fraction among the
    simulated datasets) when the observed data show no more between-dyad spread than the
    simulated fits, else 'z' (observed SE); plus the evidence and whether the choice is firm.
    Evidence: the observed SE against the simulated fits' own cluster-robust SEs (97.5%
    quantile, median over accounts; exact-boundary variants excluded; firm), or, without a
    simulated SE, the observed SE / simulated SD ratio (range up to 1.5, z above; firm only
    outside 1.25-2). _verdicts reports a statistic with a non-firm criterion whose verdict
    changes with it."""
    s = summ[(summ["stat"] == stat) & ~summ["scenario"].astype(str).str.endswith("_exact")]
    if s.empty or "obs_se_log" not in s or not s["obs_se_log"].notna().any():
        return "range", "no observed SE", True
    ose = float(s["obs_se_log"].dropna().iloc[0])
    if "sim_se_q975" in s and s["sim_se_q975"].notna().any():
        q = float(s["sim_se_q975"].median())
        cov = float(s["sim_cr_cover"].median()) if "sim_cr_cover" in s else np.nan
        cv = f"; their t(14) intervals cover the replicate mean in {100 * cov:.0f}%" if np.isfinite(
            cov) else ""
        if ose <= q:
            return "range", (f"observed SE {ose:.3f} within the simulated fits' cluster-robust SEs "
                             f"(97.5% quantile {q:.3f}{cv}): no between-dyad spread beyond the "
                             "simulated one"), True
        return "z", (f"observed SE {ose:.3f} above the simulated fits' cluster-robust SEs (97.5% "
                     f"quantile {q:.3f}{cv}): the dyads differ more than in the simulation"), True
    sd = float(s["sim_sd_log"].median()) if "sim_sd_log" in s else np.nan
    if not np.isfinite(sd) or sd <= 0:
        return "range", "no simulated spread", True
    r = ose / sd
    firm = r <= 1.25 or r >= 2.0
    return (("range" if r <= 1.5 else "z"),
            f"observed SE / simulated SD {r:.2f}" + ("" if firm else ", borderline"), firm)


def _order(summ: pd.DataFrame, scen: pd.DataFrame | None) -> list[str]:
    if scen is not None and len(scen):
        return [sc for sc in scen["scenario"] if (summ["scenario"] == sc).any()]
    return list(pd.unique(summ["scenario"]))


def _short_names(scen: pd.DataFrame | None) -> dict[str, str]:
    if scen is None or not len(scen) or "short" not in scen:
        return {}
    return dict(zip(scen["scenario"], scen["short"]))


def _causal_map(scen: pd.DataFrame | None) -> dict[str, bool]:
    if scen is None or not len(scen):
        return {}
    if "causal" in scen:
        return dict(zip(scen["scenario"], scen["causal"].astype(bool)))
    col = "beta_off" if "beta_off" in scen else None
    return dict(zip(scen["scenario"], scen[col].astype(float) != 0)) if col else {}


def consistency(summ: pd.DataFrame, scen: pd.DataFrame | None = None,
                stats: Sequence[str] = KEY_STATS, mode: str = "range"
                ) -> dict[str, dict[str, list[str]]]:
    """Scenario -> {'inconsistent': [...], 'borderline': [...]}: the statistics with that
    status by the exceedance fraction (mode 'range'), by z's p given the observed SE (mode
    'se'; statistics without an observed SE are skipped), or by each statistic's criterion
    (mode 'criterion', see ``criterion``)."""
    crit = {st: criterion(summ, st)[0] for st in stats} if mode == "criterion" else {}
    out: dict[str, dict[str, list[str]]] = {}
    for sc in _order(summ, scen):
        d: dict[str, list[str]] = {"inconsistent": [], "borderline": []}
        for st in stats:
            if mode == "se":
                r = _row(summ, sc, st)
                p = float(r.get("p_obs", np.nan)) if r is not None else np.nan
                s = status_of(p=p) if np.isfinite(p) else None
            else:
                s = _status(summ, sc, st, "range" if mode == "range" else crit[st])
            if s in d:
                d[s].append(st)
        out[sc] = d
    return out


def _lab(b: list[str]) -> str:
    return ", ".join(SHORT.get(x, x) for x in b) if b else "none"


def _lab2(d: dict[str, list[str]]) -> str:
    s = _lab(d["inconsistent"])
    return s + (f" (borderline: {_lab(d['borderline'])})" if d["borderline"] else "")


def _span(summ: pd.DataFrame, scs: Sequence[str], stat: str) -> str:
    s = summ[(summ["stat"] == stat) & summ["scenario"].isin(list(scs))]
    if s.empty:
        return "n/a"
    return f"{s['median'].min():.2f}-{s['median'].max():.2f}"


def _med_span(summ: pd.DataFrame, scs: Sequence[str], stat: str) -> str:
    s = summ[(summ["stat"] == stat) & summ["scenario"].isin(list(scs))]["median"]
    if s.empty:
        return "n/a"
    c = _is_count(stat)
    lo, hi = _fmt(float(s.min()), c), _fmt(float(s.max()), c)
    return lo if lo == hi else f"{lo}-{hi}"


def _exc_span(summ: pd.DataFrame, scs: Sequence[str], stat: str) -> str:
    """Observed position over several accounts, e.g. 'above 97-100/100'."""
    rows = [r for r in (_row(summ, sc, stat) for sc in scs)
            if r is not None and np.isfinite(float(r.get("exceed_n", np.nan)))]
    if not rows:
        return "n/a"
    sides = {str(r["exceed_side"]) for r in rows}
    if len(sides) > 1:
        return ", ".join(_exc(r) for r in rows)
    k = [int(r["exceed_n"]) for r in rows]
    n = sorted({int(r["n"]) for r in rows})
    kk = f"{min(k)}" if min(k) == max(k) else f"{min(k)}-{max(k)}"
    return f"{sides.pop()} {kk}/{'/'.join(str(x) for x in n)}"


def _category(sts: dict[str, str | None], causal: dict[str, bool],
              compat: Sequence[str]) -> str:
    cons = [sc for sc, s in sts.items() if s in compat]
    cc = any(causal.get(sc, False) for sc in cons)
    nc = any(not causal.get(sc, False) for sc in cons)
    return "none" if cc and nc else "causal" if cc else "noncausal" if nc else ""


def _verdicts(summ: pd.DataFrame, scen: pd.DataFrame | None, fit_stats: Sequence[str],
              vstats: Sequence[tuple[str, str]], exclude_exact: bool = True
              ) -> tuple[list[str], list[str], list[str], dict[str, dict[str, Any]]]:
    """Accounts that fit (no fit statistic inconsistent, each by its criterion; borderline is
    allowed and returned separately) and per-statistic verdicts among them. Category ('cat'):
    'none' (compatible, i.e. consistent or borderline, with causal and non-causal accounts),
    'causal' / 'noncausal' (only with those), 'above' / 'below' / 'between' (inconsistent with
    every fitting account); 'strict' = the category when borderline accounts count as
    inconsistent; 'alt' = the category by the other criterion when the criterion is not firm
    and the category changes with it (else None)."""
    names = _short_names(scen)
    causal = _causal_map(scen)
    order = [sc for sc in _order(summ, scen) if not (exclude_exact and sc.endswith("_exact"))]
    crit = {st: criterion(summ, st)[0] for st in list(fit_stats) + [s for s, _ in vstats]}
    fit = [sc for sc in order if all(_status(summ, sc, st, crit[st]) != "inconsistent"
                                     for st in fit_stats)]
    fit_bl = [sc for sc in fit if any(_status(summ, sc, st, crit[st]) == "borderline"
                                      for st in fit_stats)]
    L: list[str] = []
    verdict: dict[str, dict[str, Any]] = {}
    base = fit if fit else order

    def cat_of(stat: str, c: str, compat: Sequence[str] = COMPAT
               ) -> tuple[str, dict[str, str | None]]:
        sts = {sc: _status(summ, sc, stat, c) for sc in base}
        k = _category(sts, causal, compat)
        if not k:
            sides = {str(r["exceed_side"]) for r in (_row(summ, sc, stat) for sc in base)
                     if r is not None}
            k = "above" if sides == {"above"} else "below" if sides == {"below"} else "between"
        return k, sts

    for stat, lab in vstats:
        if not (summ["stat"] == stat).any():
            continue
        o = _obs(summ, stat)
        c = crit[stat]
        cat, sts = cat_of(stat, c)
        strict, _ = cat_of(stat, c, ("consistent",))
        alt = "z" if c == "range" else "range"
        cat2, _ = cat_of(stat, alt)
        dep = cat2 != cat and not criterion(summ, stat)[2]
        verdict[stat] = dict(cat=cat, strict=strict, alt=cat2 if dep else None, crit=c,
                             status=sts)
        txt = CAT_TEXT[cat]
        if strict != cat:
            txt += f" (counting borderline accounts as inconsistent: {CAT_SHORT[strict]})"
        if dep:
            txt += f"; by the other criterion ({CRIT_NAME[alt]}): {CAT_SHORT[cat2]}"
        if stat in STAT_CAVEAT:
            txt += f" ({STAT_CAVEAT[stat]})"
        grp = {k: [sc for sc in base if sts[sc] == k]
               for k in ("consistent", "borderline", "inconsistent")}

        def lst(scs: list[str]) -> str:
            return ", ".join(f"{names.get(sc, sc)} ({_pos(_row(summ, sc, stat), c)})"
                             for sc in scs) if scs else "none"

        bc = [sc for sc in base if causal.get(sc, False)]
        bn = [sc for sc in base if not causal.get(sc, False)]
        rej = ""
        r_ = summ[(summ["stat"] == stat) & summ["scenario"].isin(bc)]
        if "reject_pos" in r_ and r_["reject_pos"].notna().any():
            col = "reject_pos" if not stat.startswith("on_") else "reject_neg"
            rej = (f"; causal accounts: datasets with p < 0.05 in the direction of the primary "
                   f"association {100 * r_[col].min():.0f}-{100 * r_[col].max():.0f}%")
        other = [sc for sc in order if sc not in base
                 and _status(summ, sc, stat, c) in COMPAT]
        oth = (f"; also compatible, among accounts that do not fit: "
               f"{', '.join(names.get(sc, sc) for sc in other)}" if other else "")
        L.append(f"{lab} (observed {_fmt(o)}; {CRIT_NAME[c]}): {txt}; medians causal "
                 f"{_span(summ, bc, stat)}, non-causal {_span(summ, bn, stat)}; consistent: "
                 f"{lst(grp['consistent'])}; borderline: {lst(grp['borderline'])}; "
                 f"inconsistent: {lst(grp['inconsistent'])}{rej}{oth}")
    return fit, fit_bl, L, verdict


def reading(summ: pd.DataFrame, scen: pd.DataFrame | None) -> list[str]:
    """Data-driven reading of the offset control: which accounts reproduce the standard HR and
    counts, and, per acausal statistic, whether the observed value separates causal from
    non-causal accounts (statuses by exceedance or z, see ``status_of``)."""
    names = _short_names(scen)
    fit, fit_bl, L, verdict = _verdicts(summ, scen, FIT_STATS, VERDICT_STATS)
    head = ("accounts that reproduce the standard 1 s HR and the pooled sub chirp counts within "
            "1 s before and after offsets (none of the three inconsistent, each by its "
            "criterion): " + (", ".join(names.get(sc, sc) for sc in fit) if fit else
                             "none (verdicts below use all accounts)")
            + (f" (borderline on at least one: {', '.join(names.get(sc, sc) for sc in fit_bl)})"
               if fit_bl else ""))
    labs = dict(VERDICT_STATS)
    intro = {"none": "does not discriminate",
             "causal": "compatible only with the causal accounts",
             "noncausal": "compatible only with non-causal accounts",
             "above": "inconsistent with every account that fits, above all of them (more "
                      "before-specific)",
             "below": "inconsistent with every account that fits, below all of them",
             "between": "inconsistent with every account that fits"}
    short = []
    for k in intro:
        items = []
        for s, v in verdict.items():
            if v["cat"] != k:
                continue
            q = []
            if v["strict"] != k:
                q.append(f"hinges on borderline accounts; counting them as inconsistent: "
                         f"{CAT_SHORT[v['strict']]}")
            if v["alt"]:
                q.append(f"by the other criterion: {CAT_SHORT[v['alt']]}")
            if s in STAT_CAVEAT_SHORT:
                q.append(STAT_CAVEAT_SHORT[s])
            items.append(labs[s] + (f" ({'; '.join(q)})" if q else ""))
        if items:
            short.append(f"{intro[k]}: " + "; ".join(items))
    crit = {st: v["crit"] for st, v in verdict.items()}
    allok = [sc for sc in fit if all(_status(summ, sc, st, crit[st]) != "inconsistent"
                                     for st in verdict)]
    clean = [sc for sc in allok if all(_status(summ, sc, st, crit[st]) == "consistent"
                                       for st in verdict)]
    short.append("accounts compatible with every statistic above (none inconsistent): "
                 + (", ".join(names.get(sc, sc) for sc in allok) if allok else "none")
                 + (f"; of these, consistent with all (none borderline): "
                    f"{', '.join(names.get(sc, sc) for sc in clean) if clean else 'none'}"
                    if allok else ""))
    return [head] + L + ["in short: " + " | ".join(short)]


def onset_reading(summ: pd.DataFrame, scen: pd.DataFrame | None) -> list[str]:
    """Reading of the onset control: which accounts reproduce the standard onset HR; each
    contrast against every account; how much each account chirps at chase start (HR_future
    and counts, by exceedance), because the contrasts lie beyond an account for that reason
    alone if the account under-produces chase-start chirping."""
    names = _short_names(scen)
    fit, fit_bl, L, verdict = _verdicts(summ, scen, ("on_hr_primary",), ON_VERDICT_STATS, False)
    head = ("accounts that reproduce the standard onset HR (by its criterion): "
            + (", ".join(names.get(sc, sc) for sc in fit) if fit else "none")
            + (f" (borderline: {', '.join(names.get(sc, sc) for sc in fit_bl)})"
               if fit_bl else ""))
    causal = _causal_map(scen)
    order = _order(summ, scen)
    nonc = [sc for sc in order if not causal.get(sc, False)]

    def nm(scs: Sequence[str]) -> str:
        x = [names.get(sc, sc) for sc in scs]
        return x[0] if len(x) == 1 else ", ".join(x[:-1]) + " and " + x[-1] if x else "none"

    # chase-start chirping: pooled counts after onsets and the future window alone (the joint
    # HR_future is adjusted for the past window and so also reflects the pre-onset chirps)
    chk = [st for st in START_CHECK if (summ["stat"] == st).any()]
    joint = [st for st in ("on_hr_future", "on_hr_future_grd") if (summ["stat"] == st).any()]
    if not chk:  # tables of an older version: the joint HR_future is all there is
        chk = joint

    def under(sc: str) -> bool:  # less chase-start chirping than observed
        for st in chk:
            r = _row(summ, sc, st)
            if (r is not None and r["exceed_side"] == "above"
                    and _status(summ, sc, st, "range") in ("inconsistent", "borderline")):
                return True
        return False

    def over(sc: str) -> bool:  # more chase-start chirping than observed
        for st in chk:
            r = _row(summ, sc, st)
            if (r is not None and r["exceed_side"] == "below"
                    and _status(summ, sc, st, "range") == "inconsistent"):
                return True
        return False

    def lp(scs: list[str], stats: Sequence[str]) -> str:
        return ", ".join(
            f"{names.get(sc, sc)} ({' / '.join(_exc(_row(summ, sc, st)) for st in stats)})"
            for sc in scs) if scs else "none"

    if chk:
        und = [sc for sc in order if under(sc)]
        ovr = [sc for sc in order if over(sc) and sc not in und]
        mat = [sc for sc in order if sc not in und and sc not in ovr]
        L.append("chase-start chirping, by " + " / ".join(SHORT.get(st, st) for st in chk)
                 + " (observed " + " / ".join(_fmt(_obs(summ, st), _is_count(st)) for st in chk)
                 + "; exceedance; ranges in the lines above): less than observed (inconsistent "
                 f"or borderline): {lp(und, chk)}; reproduced: {lp(mat, chk)}"
                 + (f"; more than observed: {lp(ovr, chk)}" if ovr else ""))
    if joint and chk != joint:
        oj = [_obs(summ, st) for st in joint]
        oa = [_obs(summ, st) for st in ("on_hr_future_alone", "on_hr_future_alone_grd")
              if (summ["stat"] == st).any()]
        L.append("joint HR_future 1 s / guarded (observed " + " / ".join(_fmt(x) for x in oj)
                 + (", future window alone " + " / ".join(_fmt(x) for x in oa) if oa else "")
                 + "; exceedance): " + lp(order, joint)
                 + ("; the joint HR_future is adjusted for the past window, which in the data "
                    "holds few chirps before onsets, and is larger than the future window alone, "
                    "so it overstates chase-start chirping"
                    if len(oa) == len(oj) and all(j > 1.05 * a for j, a in zip(oj, oa)) else ""))
    # the accounts that could explain the contrasts by chase-start chirping alone: no-effect
    # accounts that reproduce the standard onset HR
    nc_fit = [sc for sc in nonc if sc in fit]
    nc_under = [sc for sc in nc_fit if under(sc)]
    nc_match = [sc for sc in nc_fit if not under(sc) and not over(sc)]
    def under_on(sc: str, st: str) -> bool:
        r = _row(summ, sc, st)
        return bool(r is not None and r["exceed_side"] == "above"
                    and _status(summ, sc, st, "range") in ("inconsistent", "borderline"))

    parts_u = []
    for st in chk:
        accs = [sc for sc in nc_under if under_on(sc, st)]
        if accs:
            whose = "their" if accs == nc_under else f"{nm(accs)}'s"
            bl = all(_status(summ, sc, st, "range") == "borderline" for sc in accs)
            parts_u.append(f"{SHORT.get(st, st)} {_fmt(_obs(summ, st), _is_count(st))} lies "
                           f"{_exc_span(summ, accs, st)} of {whose} datasets"
                           + (", borderline" if bl else ""))
    fut_txt = "; ".join(parts_u)
    comp = {st: [sc for sc in nonc if _status(summ, sc, st, v["crit"]) in COMPAT]
            for st, v in verdict.items()}
    comp_strict = {st: [sc for sc in nonc if _status(summ, sc, st, v["crit"]) == "consistent"]
                   for st, v in verdict.items()}
    vl = dict(ON_VERDICT_STATS)
    tail = ""
    if not verdict:
        tail = "in short: no onset contrast available"
    elif not any(comp.values()):
        if not nc_match:
            why = (f"those that reproduce the standard onset HR ({nm(nc_under)}) produce less "
                   f"chirping at chase start than observed ({fut_txt})" if nc_under else
                   "none of those that reproduce the standard onset HR also reproduces the "
                   "observed chase-start chirping")
            tail = ("in short: every onset contrast lies beyond the simulated no-effect "
                    f"accounts ({nm(nonc)}), but {why}, so the contrasts lying beyond them do "
                    "not show an asymmetry beyond chase-start chirping; they are not evidence "
                    "for (b), whose verdict rests on the standard past-only HR and on the dyads' "
                    "disagreement")
        else:
            m = nc_match
            hr_rows = [(sc, _row(summ, sc, "on_hr_primary")) for sc in m]
            hr_txt = ", ".join(
                f"z p = {float(r['p_obs']):.2f} ({_status(summ, sc, 'on_hr_primary', 'z')})"
                for sc, r in hr_rows
                if r is not None and np.isfinite(float(r.get("p_obs", np.nan))))
            past = "; ".join(
                f"{SHORT[st]} observed {_fmt(_obs(summ, st), _is_count(st))} vs median "
                f"{_med_span(summ, m, st)} ({_exc_span(summ, m, st)})"
                for st in ("on_hr_past", "on_hr_past_grd", "on_n_pre_1s")
                if (summ["stat"] == st).any())
            tail = ("in short: every onset contrast lies beyond every simulated no-effect "
                    f"account, including {nm(m)}, whose chase-start chirping matches the data "
                    "(chirp counts after onsets and future window alone, above)"
                    + (f"; the other no-effect accounts that reproduce the standard onset HR "
                       f"({nm(nc_under)}) produce less chirping at chase start than observed "
                       f"({fut_txt})" if nc_under else "")
                    + f". Against the matched account{'s' if len(m) > 1 else ''}, the asymmetry "
                    f"is on the pre-onset side ({past}): fewer sub chirps before onsets than "
                    f"{'they produce' if len(m) > 1 else 'it produces'}, the deficit "
                    "that the standard past-only HR measures, here compared within dyads with "
                    "the chase-start rate"
                    + (f" (the standard past-only HR itself, judged by z because the dyads "
                       f"disagree, gives {hr_txt} against {nm(m)})" if hr_txt else "")
                    + "; this points the same way as (b) but does not change its verdict, which "
                    "rests on the standard past-only HR and on the dyads' disagreement")
        tail += ("; accounts that were not simulated (other latent processes, other chirp "
                 "patterns around onsets) remain possible")
    else:
        parts = []
        for st, scs in comp.items():
            if scs:
                bl = [sc for sc in scs if sc not in comp_strict[st]]
                parts.append(f"{vl[st]} compatible with {nm(scs)}"
                             + (f" ({nm(bl)} borderline)" if bl else "")
                             + ("" if all(sc in nc_match for sc in scs) else
                                " (an account that under-produces chase-start chirping)"
                                if all(sc in nc_under for sc in scs) else ""))
            else:
                parts.append(f"{vl[st]} beyond every no-effect account")
        tail = ("in short: " + "; ".join(parts) + "; at least one onset contrast is compatible "
                "with a no-effect account, so the onset control gives no evidence for (b)"
                + (f" (no-effect accounts that under-produce chase-start chirping: "
                   f"{nm(nc_under)}; {fut_txt})" if nc_under else ""))
    return [head] + L + [tail]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def _obs_ci(summ: pd.DataFrame, stat: str) -> str:
    r = summ[(summ["stat"] == stat)]
    if r.empty or "obs_ci_low" not in r or not np.isfinite(r["obs_ci_low"].iloc[0]):
        return _fmt(_obs(summ, stat), _is_count(stat))
    r = r.iloc[0]
    return f"{r['observed']:.2f} [{r['obs_ci_low']:.2f}, {r['obs_ci_high']:.2f}]"


def _per_dyad_dev(cnt: pd.DataFrame | None, sc: str, col: str) -> float:
    if cnt is None or not len(cnt) or col not in cnt:
        return np.nan
    x = cnt[cnt["scenario"] == sc]
    return float(np.abs(x[col] - x[f"real_{col}"]).sum()) if len(x) else np.nan


def _criteria_line(summ: pd.DataFrame, stats: Sequence[str]) -> str:
    parts = []
    for st in stats:
        if not (summ["stat"] == st).any():
            continue
        c, why, _ = criterion(summ, st)
        parts.append(f"{SHORT.get(st, st)}: {CRIT_NAME[c]} ({why})")
    return "; ".join(parts)


def summary_lines(t: dict[str, pd.DataFrame] | None, kind: str = "offset") -> list[str]:
    """Simulation block for summary.txt and acausal_summary.txt, per kind."""
    if t is None:
        return []
    if kind == "onset":
        return _onset_lines(t)
    if not len(t.get("summary", [])):
        return []
    summ, scen = t["summary"], t.get("scenarios")
    names = _short_names(scen)
    order = [sc for sc in _order(summ, scen) if not sc.endswith("_exact")]
    summ = summ[summ["scenario"].isin(order)]
    scen = scen[scen["scenario"].isin(order)] if scen is not None and len(scen) else scen
    n = int(summ["n"].max())
    acc = ("; ".join(f"{r['short']} = {r['label']}" for _, r in scen.iterrows())
           if scen is not None and len(scen) and "label" in scen else "")
    L = [f"    Simulation of this control (hz_acausal_sim; {n} datasets per account; chase and "
         "chirp rates matched to these dyads; FE; median [95% range] over datasets, then the "
         f"observed value's position among them; {STATUS_RULE}; exact-boundary variants and "
         "all statistics in simulation/acausal_sim_summary.txt). Accounts: " + acc + "."]
    for stat, lab in (("hr_primary", "standard 1 s HR"), ("ratio_pf", "HR_past / HR_future 1 s"),
                      ("ratio_grd", "HR_past / HR_future guarded"),
                      ("ratio_k", "kernel before / after"),
                      ("ratio_rm", "restricted real / mirror 1 s, W = 2 s"),
                      ("ratio_rm_grd", "restricted real / mirror guarded, W = 2 s")):
        if (summ["stat"] == stat).any():
            c = criterion(summ, stat)[0]
            L.append(f"      {lab} (observed {_obs_ci(summ, stat)}; {CRIT_NAME[c]}): "
                     + "; ".join(f"{names.get(sc, sc)} {_cellx(summ, sc, stat, c)}"
                                 for sc in order) + ".")

    def med(sc: str, stat: str) -> str:
        r = _row(summ, sc, stat)
        return _fmt(float(r["median"]), True) if r is not None else "n/a"

    L.append(f"      sub chirps within 1 s before / after offsets, pooled over dyads (observed "
             f"{_fmt(_obs(summ, 'n_pre_1s'), True)} / {_fmt(_obs(summ, 'n_post_1s'), True)}), "
             "median (observed position): " + "; ".join(
                 f"{names.get(sc, sc)} {med(sc, 'n_pre_1s')} / {med(sc, 'n_post_1s')} "
                 f"({_exc(_row(summ, sc, 'n_pre_1s'))}; {_exc(_row(summ, sc, 'n_post_1s'))})"
                 for sc in order) + ".")
    cnt = t.get("counts")
    if cnt is not None and len(cnt) and "real_pre_2s" in cnt:
        rp = float(cnt.groupby("recording")["real_pre_2s"].first().sum())
        L.append(f"      Per dyad, the pre-offset counts (0-2 s; sum over dyads of |simulated mean "
                 f"- real|, real total {rp:.0f}): " + "; ".join(
                     f"{names.get(sc, sc)} {_per_dyad_dev(cnt, sc, 'pre_2s'):.0f}" for sc in order)
                 + ". Only the symmetric cluster is calibrated per dyad on these counts; the "
                 "causal and trigger accounts carry one common (or randomly varying) effect, so "
                 "their poorer per-dyad match is not by itself evidence against them, and a "
                 "per-dyad calibrated causal effect was not simulated. The post-offset counts are "
                 "calibrated per dyad in every account with a response burst.")
    if (summ["stat"] == "local_ratio_1s").any():
        drift = t.get("drift")
        dtxt = ""
        if drift is not None and len(drift) and "chosen_mult" in drift:
            dtxt = (f" (the drift account's slow modulation, high / low chirp rate "
                    f"{float(drift['chosen_mult'].iloc[0]):.1f}, is matched to it)")
        cl = criterion(summ, "local_ratio_1s")[0]
        L.append(
            "      Co-occurrence of sub chirps on both sides of an offset (last 1 s of the chase "
            f"and first 1 s after it): same offset {_obs_ci(summ, 'both_ratio_1s')} x the number "
            "expected if the two sides were independent within each dyad; but the pre side of "
            "offset i pairs with the post side of the NEIGHBOURING offsets (i +- 1, tens of "
            f"seconds apart) at {_obs_ci(summ, 'nb_ratio_1s')} x, and with chirps 2-3 s after "
            f"the same offset at {_fmt(_obs(summ, 'far_ratio'))} x: sub chirping co-varies over "
            "seconds to minutes within dyads, so within-dyad independence is not the null. "
            "Offset-specific excess (same offset / neighbouring offsets) "
            f"{_obs_ci(summ, 'local_ratio_1s')}. Simulated, local excess (neighbour baseline): "
            + "; ".join(f"{names.get(sc, sc)} {_cellx(summ, sc, 'local_ratio_1s', cl)} "
                        f"({_fmt(float(_row(summ, sc, 'nb_ratio_1s')['median']))})"
                        if _row(summ, sc, 'nb_ratio_1s') is not None else names.get(sc, sc)
                        for sc in order)
            + f". Accounts without drift have no slow co-variation (neighbour baseline ~1){dtxt}. "
            "The symmetric cluster's value depends on its cluster size (1 chirp per side), which "
            "was not calibrated, so matching it is not evidence for that account.")
    crit_stats = ("hr_primary",) + KEY_STATS + ("local_ratio_1s",)
    L.append("      Criteria per statistic: " + _criteria_line(summ, crit_stats) + ".")
    bad = consistency(summ, scen, crit_stats + COUNT_STATS[:2], mode="criterion")
    L.append("      Inconsistent with the observed value (each statistic by its criterion; "
             "borderline in parentheses): "
             + "; ".join(f"{names.get(sc, sc)}: {_lab2(b)}" for sc, b in bad.items()) + ".")
    L += ["      Reading: " + "; ".join(reading(summ, scen)) + "."]
    return L


def _onset_lines(t: dict[str, pd.DataFrame]) -> list[str]:
    summ, scen = t.get("onset_summary"), t.get("onset_scenarios")
    if summ is None or not len(summ):
        return ["    Simulation of the onset control: not run (hz_acausal_sim onset accounts)."]
    names = _short_names(scen)
    order = _order(summ, scen)
    n = int(summ["n"].max())
    acc = ("; ".join(f"{r['short']} = {r['label']}" for _, r in scen.iterrows())
           if scen is not None and len(scen) and "label" in scen else "")
    L = [f"    Simulation of the onset control (hz_acausal_sim; {n} datasets per account; hz_sim "
         "generator matched to these dyads; FE on onset designs with all event bins and 10% of "
         "the others, IPW cloglog; the observed values here use the same subsampling; median "
         f"[95% range], then the observed value's position; {STATUS_RULE}). Accounts: "
         + acc + "."]
    for stat, lab in (("on_hr_primary", "standard 1 s HR"),
                      ("on_hr_past", "joint HR_past, 1 s"),
                      ("on_hr_future", "joint HR_future, 1 s"),
                      ("on_hr_future_grd", "joint HR_future, guarded"),
                      ("on_hr_future_alone", "HR_future, future window alone, 1 s"),
                      ("on_hr_future_alone_grd", "HR_future, future window alone, guarded"),
                      ("on_ratio_pf", "HR_past / HR_future, 1 s"),
                      ("on_ratio_grd", "HR_past / HR_future, guarded"),
                      ("on_joint_shift", "joint / standalone HR_past, 1 s"),
                      ("on_joint_shift_grd", "joint / standalone HR_past, guarded"),
                      ("on_n_start_1s", "sub chirps in the first 1 s of chases"),
                      ("on_n_start_grd", "sub chirps 0.5-1.5 s after onsets"),
                      ("on_n_pre_1s", "sub chirps in the 1 s before onsets")):
        if (summ["stat"] == stat).any():
            c = criterion(summ, stat)[0]
            L.append(f"      {lab} (observed {_obs_ci(summ, stat)}; {CRIT_NAME[c]}): " + "; ".join(
                f"{names.get(sc, sc)} {_cellx(summ, sc, stat, c)}" for sc in order) + ".")
    est = t.get("onset_estimates")
    if est is not None and len(est):
        w = est[est["scenario"] != "observed"].pivot_table(index=["scenario", "rep"],
                                                           columns="stat", values="value")
        nc = [sc for sc in order if not _causal_map(scen).get(sc, False)]
        parts = []
        for st in ("on_joint_shift", "on_joint_shift_grd"):
            if st in w:
                x = w.loc[w.index.get_level_values(0).isin(nc), st].to_numpy(float)
                x = x[np.isfinite(x)]
                if len(x):
                    parts.append(f"{'1 s' if st == 'on_joint_shift' else 'guarded'} "
                                 f"{100 * np.mean(x < 1):.0f}%")
        if parts:
            L.append("      Without a chirp effect, the joint HR_past falls below the standalone "
                     "HR_past in " + ", ".join(parts) + " of the no-effect datasets (adjusting "
                     "for the after-window biases it toward < 1).")
    L.append("      Criteria per statistic: " + _criteria_line(
        summ, ("on_hr_primary", "on_ratio_pf", "on_ratio_grd", "on_hr_future",
               "on_hr_future_grd")) + ".")
    L.append("      Reading: " + "; ".join(onset_reading(summ, scen)) + ".")
    return L


def _detail_line(summ: pd.DataFrame, sc: str, stat: str, name: str, onset: bool = False) -> str:
    """One account in the detailed listing of acausal_sim_summary.txt."""
    r = _row(summ, sc, stat)
    zv = float(r.get("z_obs", np.nan))
    zt = f"  z {zv:+5.2f}" if np.isfinite(zv) else ""
    fr = float(r.get("exceed_frac", np.nan))
    pos = f"{_exc(r)} ({100 * fr:.1f}% at or beyond)" if np.isfinite(fr) else "n/a"
    c = _crit_of(r, None)
    st = _status(summ, sc, stat, c)
    flag = f"  {st.upper()}" if st in ("borderline", "inconsistent") else ""
    extra = ""
    a, b = ("reject_neg", "reject_pos") if onset else ("reject_pos", "reject_neg")
    if stat in CONTRASTS and np.isfinite(float(r.get(a, np.nan))):
        extra = (f"; p < 0.05 with ratio {'<' if onset else '>'} 1 in {100 * r[a]:.0f}%, "
                 f"{'>' if onset else '<'} 1 in {100 * r[b]:.0f}%")
    return (f"    {name:30s} {_cell(summ, sc, stat):26s} pct {r['obs_pct']:5.1f}{zt}  {pos}"
            f"{flag}{extra}")


def summary_text(t: dict[str, pd.DataFrame]) -> str:
    summ, scen, prof = t["summary"], t.get("scenarios"), t.get("profiles")
    cal, tim, cnt = t.get("calibration"), t.get("timings"), t.get("counts")
    names = _short_names(scen)
    order = _order(summ, scen)
    L = ["Simulation calibration of the acausal control",
         "=" * 46,
         "Each account simulated with chase and sub chirp rates matched to the 15 real dyads "
         "(hz_sim generator; per-dyad calibration of baseline hazards, chirp rates and the "
         "offset-locked chirps); statistics as in hz_acausal (glm_fe = cloglog, dyad FE, "
         "cluster-robust). Values: median [2.5%, 97.5%] over replicates; 'obs' = the real data "
         "(same functions, with its own 95% CI where available); 'pct' = percent of simulated "
         "values below the observed one; 'above k/n' = the observed value lies above k of the n "
         "datasets (its side is the one with fewer datasets at or beyond it), with the "
         "exceedance fraction = share of datasets at or beyond it; 'z' = (log observed - log "
         "median) / SE of the observed statistic (cluster-robust or dyad bootstrap). Which of the "
         "two comparisons is calibrated is decided per statistic (Criteria): the exceedance when "
         "the observed SE lies within the simulated fits' own cluster-robust SEs (no between-dyad "
         "spread beyond the simulated one), z when it exceeds them (the dyads differ more than "
         f"simulated). Flags by that criterion: {STATUS_RULE}.", ""]
    if scen is not None and len(scen):
        L.append("Offset accounts")
        for _, r in scen.iterrows():
            L.append(f"  {r['scenario']:26s} {r['label']} ({int(r['n_reps'])} replicates)")
        L.append("")
    L.append("Statistics (offset)")
    for stat, lab in STATS.items():
        s = summ[summ["stat"] == stat]
        if s.empty:
            continue
        c = _is_count(stat)
        ob = _fmt(float(s["observed"].iloc[0]), c) if c else _obs_ci(s, stat)
        L.append(f"  {lab} (obs {ob}; criterion {CRIT_NAME[criterion(summ, stat)[0]]}):")
        for sc in order:
            if _row(s, sc, stat) is not None:
                L.append(_detail_line(s, sc, stat, names.get(sc, sc)))
    L.append("")
    crit_stats = ("hr_primary",) + KEY_STATS + COOC_STATS
    L.append("Criteria per statistic: " + _criteria_line(summ, crit_stats) + ".")
    if "sim_sd_log" in summ and "obs_se_log" in summ:
        k = summ[summ["stat"].isin(KEY_STATS + ("hr_primary",) + COOC_STATS)].dropna(
            subset=["sim_sd_log", "obs_se_log"])
        g = k.groupby("stat")[["sim_sd_log", "obs_se_log"]].median()
        sr = _se_ratio(summ)
        L.append("Spread (log scale): observed SE / median simulated SD per statistic: "
                 + ", ".join(f"{st_} {r['obs_se_log']:.3f} / {r['sim_sd_log']:.3f}"
                             for st_, r in g.iterrows())
                 + (". Mean cluster-robust SE of the simulated fits / their SD: " + ", ".join(
                     f"{st_} {v:.2f}" for st_, v in sr.items()) if len(sr) else "") + ".")

    def fl(d: dict[str, dict[str, list[str]]]) -> str:
        return "; ".join(f"{sc}: {_lab2(b)}" for sc, b in d.items())

    L.append("Key contrasts (" + ", ".join(KEY_STATS) + ") by the exceedance fraction "
             "(inconsistent: < 2.5% of datasets at or beyond the observed value; borderline "
             "2.5-10% in parentheses): " + fl(consistency(summ, scen, KEY_STATS)) + ".")
    L.append("Primary HR, chirp counts and co-occurrence by the exceedance fraction: "
             + fl(consistency(summ, scen, ("hr_primary",) + COUNT_STATS + COOC_STATS)) + ".")
    L.append("By z given the observed SE (inconsistent p < 0.05, borderline 0.05-0.2): "
             + fl(consistency(summ, scen, ("hr_primary",) + KEY_STATS + COOC_STATS, mode="se"))
             + ".")
    L.append("By each statistic's criterion: " + fl(consistency(
        summ, scen, ("hr_primary",) + KEY_STATS + ("local_ratio_1s",) + COUNT_STATS[:2],
        mode="criterion")) + ".")
    L.append("Reading: " + "; ".join(reading(summ, scen)) + ".")
    L.append("")
    drift = t.get("drift")
    if drift is not None and len(drift):
        L.append("Drift calibration (no causal effect, uncalibrated response burst, bursts "
                 "crossing chase boundaries, lattice error; pooled over "
                 f"{int(drift['n_reps'].iloc[0])} datasets per value; "
                 f"target neighbour ratio {float(drift['target_nb_ratio_1s'].iloc[0]):.2f}): "
                 + "; ".join(f"high / low {r['mult']:g}: neighbour {r['nb_ratio_1s']:.2f}, same "
                             f"offset {r['both_ratio_1s']:.2f}, local {r['local_ratio_1s']:.2f}, "
                             f"post 2-3 s {r['far_ratio']:.2f}" for _, r in drift.iterrows())
                 + f"; chosen {float(drift['chosen_mult'].iloc[0]):.2f}.")
        L.append("")
    if prof is not None and len(prof):
        L.append("Empirical latency profiles (all offsets, 0-3 h; real minus simulated background "
                 "without offset-locked chirps; negative excess is clipped to 0 in the densities):")
        L.append("  bin (s)    real pre  real post  bg pre  bg post  excess pre  excess post")
        for _, r in prof.iterrows():
            L.append(f"  {r['edge_lo']:4.2f}-{r['edge_hi']:4.2f}  {r['real_pre']:8.0f}  "
                     f"{r['real_post']:9.0f}  {r['background_pre']:6.1f}  "
                     f"{r['background_post']:7.1f}  {r['excess_pre']:10.1f}  "
                     f"{r['excess_post']:11.1f}")
        L.append("  The default background has fewer chirps just after an offset than later "
                 "(every burst ends at the chase end), so part of the post-offset excess at short "
                 "latencies replaces burst continuations; the drift account uses profiles "
                 "estimated against its own background (acausal_sim_profiles_drift.csv).")
        L.append("")
    if cnt is not None and len(cnt):
        g = cnt.groupby("scenario")[["real_pre_2s", "real_post_2s"]].sum()
        rp, ro = g["real_pre_2s"].iloc[0], g["real_post_2s"].iloc[0]
        L.append("Per-dyad match of the 0-2 s counts (sum over dyads of |sim mean - real|; real "
                 f"totals pre {rp:.0f}, post {ro:.0f}): "
                 + "; ".join(f"{names.get(sc, sc)} pre {_per_dyad_dev(cnt, sc, 'pre_2s'):.0f} / "
                             f"post {_per_dyad_dev(cnt, sc, 'post_2s'):.0f}"
                             for sc in order if sc in g.index)
                 + "; per dyad in acausal_sim_counts.csv. Only the symmetric cluster is calibrated "
                   "per dyad on the pre-offset counts.")
        L.append("")
    so, sco = t.get("onset_summary"), t.get("onset_scenarios")
    if so is not None and len(so):
        on_names = _short_names(sco)
        L.append("Onset accounts")
        if sco is not None and len(sco):
            for _, r in sco.iterrows():
                L.append(f"  {r['scenario']:26s} {r['label']} ({int(r['n_reps'])} replicates)")
        L.append("Statistics (onset; case-control designs, observed with the same subsampling)")
        for stat, lab in ON_STATS.items():
            s = so[so["stat"] == stat]
            if s.empty:
                continue
            L.append(f"  {lab} (obs {_obs_ci(s, stat)}; criterion "
                     f"{CRIT_NAME[criterion(so, stat)[0]]}):")
            for sc in _order(so, sco):
                if _row(s, sc, stat) is not None:
                    L.append(_detail_line(s, sc, stat, on_names.get(sc, sc), onset=True))
        L.append("Criteria (onset): " + _criteria_line(
            so, ("on_hr_primary", "on_ratio_pf", "on_ratio_grd", "on_hr_future",
                 "on_hr_future_grd")) + ".")
        L.append("Reading (onset): " + "; ".join(onset_reading(so, sco)) + ".")
        L.append("")
    if cal is not None and len(cal):
        c2 = cal.copy()
        c2["_round"] = c2["start_round"].fillna(0) if "start_round" in c2 else 0
        last = c2.sort_values(["_round", "iteration"], kind="mergesort").groupby("calib").tail(1)
        L.append("Calibration, last iteration (simulated / real pooled): " + "; ".join(
            f"{names.get(r['calib'], r['calib'])}: chases {r['ratio_n_chases']:.3f}, sub in "
            f"{r['ratio_n_sub_in']:.3f}, out {r['ratio_n_sub_out']:.3f}"
            + (f", pre 2 s {r['ratio_pre_2s']:.3f}, post 2 s {r['ratio_post_2s']:.3f}"
               if np.isfinite(float(r.get("ratio_pre_2s", np.nan))) else "")
            + (", chase start (0-0.5 / 0.5-1 / 1-1.5 / 1.5-2 s after onsets) "
               + " / ".join(f"{float(r[f'ratio_on_start_{i}']):.2f}"
                            for i in range(len(START_EDGES) - 1))
               + " with multipliers " + " / ".join(f"{float(r[f'start_mult_{i}']):.2f}"
                                                    for i in range(len(START_EDGES) - 1))
               if np.isfinite(float(r.get("start_mult_0", np.nan))) else "")
            for _, r in last.iterrows()) + ".")
    if tim is not None and len(tim):
        L.append("Runtime: " + ", ".join(f"{r['step']} {r['seconds']:.0f} s"
                                         for _, r in tim.iterrows()) + ".")
    L += ["", "Reading rule: an acausal statistic separates accounts only where their simulated "
          "values differ; an observed value compatible with causal and non-causal accounts alike "
          "does not choose between them. Verdicts count borderline accounts as compatible and "
          "say when they would change if borderline accounts were counted as inconsistent. "
          "Accounts that were not simulated (mixtures, other latent processes, a per-dyad "
          "calibrated causal effect) remain possible."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
PANELS = (("hr_primary", "standard 1 s model: HR", True),
          ("hr_past", "joint 1 s: HR_past", True),
          ("hr_future", "joint 1 s: HR_future", True),
          ("ratio_pf", "HR_past / HR_future, 1 s", True),
          ("ratio_grd", "HR_past / HR_future, guarded", True),
          ("ratio_k", "kernel mean before / after", True),
          ("ratio_rm", "restricted real / mirror, 1 s", True),
          ("ratio_rm_grd", "restricted real / mirror, guarded", True),
          ("n_pre_1s", "sub chirps 0-1 s before offsets", False),
          ("n_post_1s", "sub chirps 0-1 s after offsets", False),
          ("nb_ratio_1s", "co-occurrence, neighbouring offsets\n(slow co-variation baseline)",
           True),
          ("local_ratio_1s", "co-occurrence, same / neighbouring\noffsets (offset-specific)",
           True))
C_OBS = "#c2452d"


def plot_calibration(summ: pd.DataFrame, scen: pd.DataFrame | None, out_dir: Path,
                     name: str = "acausal_sim_calibration") -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, NullLocator

    from hz_desc import GRID, INK, INK2, MUTED, _save, _style

    _style()
    names = _short_names(scen)
    order = _order(summ, scen)
    ny = len(order)
    fig, axes = plt.subplots(3, 4, figsize=(11.6, 9.4), constrained_layout=True, sharey=True)
    for ax, (stat, title, logx) in zip(axes.ravel(), PANELS):
        s = summ[summ["stat"] == stat].set_index("scenario")
        vals = []
        for i, sc in enumerate(order):
            if sc not in s.index:
                continue
            r = s.loc[sc]
            y = ny - 1 - i
            exact = sc.endswith("_exact")
            ax.plot([r["q025"], r["q975"]], [y, y], color=INK2, lw=1.6, solid_capstyle="round",
                    zorder=3)
            ax.plot(r["median"], y, "o", ms=6, mfc="white" if exact else INK2, mec=INK2, mew=1.4,
                    zorder=4)
            vals += [r["q025"], r["q975"], r["median"]]
        o = float(s["observed"].iloc[0]) if len(s) else np.nan
        if np.isfinite(o):
            ax.axvline(o, color=C_OBS, lw=1.6, zorder=2)
            vals.append(o)
            if "obs_ci_low" in s and np.isfinite(float(s["obs_ci_low"].iloc[0])):
                a_, b_ = float(s["obs_ci_low"].iloc[0]), float(s["obs_ci_high"].iloc[0])
                ax.axvspan(a_, b_, color=C_OBS, alpha=0.10, lw=0, zorder=0)
                vals += [a_, b_]
        v = np.array([x for x in vals if np.isfinite(x) and (x > 0 or not logx)])
        if logx and len(v):
            lo, hi = float(v.min()) / 1.25, float(v.max()) * 1.25
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 20, 30)
                     if lo <= t <= hi]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            if lo <= 1 <= hi:
                ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        elif len(v):
            pad = 0.08 * (v.max() - v.min() + 1)
            ax.set_xlim(max(0, v.min() - pad), v.max() + pad)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        crit = criterion(summ, stat)[0] if stat in ("hr_primary",) + KEY_STATS + COOC_STATS \
            else "range"
        ct = " (criterion: z)" if crit == "z" else ""
        ax.set_title(f"{title}\nobserved {o:.2f}{ct}" if logx and np.isfinite(o) else
                     f"{title}\nobserved {o:.0f}" if np.isfinite(o) else title,
                     loc="left", fontsize=7.8, color=INK)
    for ax in axes[:, 0]:
        ax.set_yticks(range(ny))
        ax.set_yticklabels([names.get(sc, sc) for sc in order[::-1]], fontsize=7.2)
        ax.set_ylim(-0.6, ny - 0.4)
    handles = [Line2D([], [], color=INK2, marker="o", mfc=INK2, mec=INK2, lw=1.6, ms=6,
                      label="simulated median and 95% range, lattice boundary error"),
               Line2D([], [], color=INK2, marker="o", mfc="white", mec=INK2, lw=1.6, ms=6,
                      label="same, exact boundaries"),
               Line2D([], [], color=C_OBS, lw=1.6, label="observed (real data, same statistic)"),
               Patch(color=C_OBS, alpha=0.18, lw=0,
                     label="its 95% CI (cluster-robust or dyad bootstrap)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.4,
               bbox_to_anchor=(0.5, -0.03))
    n = int(summ["n"].max()) if len(summ) else 0
    fig.suptitle("Acausal control under simulated accounts of the offset association (chase "
                 f"offsets; {n} replicates per account; glm_fe, dyad FE). Rows 1-2: hazard ratios "
                 "and acausal contrasts; row 3: counts and co-occurrence around offsets.",
                 fontsize=8.6, x=0.01, ha="left")
    return _save(fig, Path(out_dir), name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_acausal_sim(trials: Sequence[Trial], cfg: Config, verbose: bool = True,
                    n_reps: int | None = None, workers: int | None = None) -> dict[str, Any]:
    """Estimate profiles, calibrate every scenario, simulate, summarise; write tables and figure;
    rebuild acausal/acausal_summary.txt."""
    t_all = time.perf_counter()
    out = Path(cfg.out_dir) / "simulation"
    out.mkdir(parents=True, exist_ok=True)
    quick = bool(cfg.quick)
    n_reps = int(_opt(cfg, "acsim_n_reps") if n_reps is None else n_reps)
    if quick:
        n_reps = min(n_reps, int(_opt(cfg, "acsim_quick_reps")))
    on_reps = min(n_reps, int(_opt(cfg, "acsim_onset_reps")))
    workers = int(S._sopt(cfg, "workers") if workers is None else workers)
    n_iter = 2 if quick else int(_opt(cfg, "acsim_calib_iter"))
    c_reps = 4 if quick else int(_opt(cfg, "acsim_calib_reps"))
    p_reps = 4 if quick else int(_opt(cfg, "acsim_profile_reps"))
    d_reps = 2 if quick else int(_opt(cfg, "acsim_drift_reps"))
    tim = []

    def step(name: str, t0: float) -> None:
        tim.append(dict(step=name, seconds=time.perf_counter() - t0))
        if verbose:
            print(f"[acausal sim] {name} ({tim[-1]['seconds']:.0f} s)", flush=True)

    t0 = time.perf_counter()
    P0 = S.estimate_params(trials, cfg)
    P_bg = replace(P0, shape_out=flat_early(P0.shape_out))
    prof = estimate_profiles(trials, P_bg, cfg, p_reps, workers)
    step("profiles", t0)
    t0 = time.perf_counter()
    burst = next(sc for sc in scenarios(cfg) if sc.name == "burst_only")
    m_star, drift_tab = calibrate_drift(
        trials, lambda m: scenario_params(burst, drift_params(P_bg, cfg, m), prof, cfg), cfg,
        d_reps, workers)
    P_bg_d = drift_params(P_bg, cfg, m_star)
    prof_d = estimate_profiles(trials, P_bg_d, cfg, p_reps, workers)
    step(f"drift calibration (high / low {m_star:.2f})", t0)

    het_off = data_het_sd(cfg, "offset")
    dov = dict(sub_mod_frac=P_bg_d.sub_mod_frac, sub_mod_dur=P_bg_d.sub_mod_dur,
               sub_mod_mult=P_bg_d.sub_mod_mult)
    scen = [replace(sc, overrides={**sc.overrides, **dov}) if sc.drift else sc
            for sc in scenarios(cfg, het_off)]
    params: dict[str, S.SimParams] = {}
    logs, res = [], []
    _AC.clear()
    _AC.update(cfg=cfg, params=params, trials=list(trials))

    def stage(scs: list[AcScenario], with_obs: bool, tag: str) -> None:
        t0 = time.perf_counter()
        for sc in scs:
            pb, pr = (P_bg_d, prof_d) if sc.drift else (P_bg, prof)
            params[sc.name], lg = calibrate_scenario(sc, pb, pr, trials, cfg, n_iter, c_reps,
                                                     workers)
            logs.append(lg)
        step(f"calibration {tag}", t0)
        if verbose:
            last = pd.concat(logs[-len(scs):]).groupby("calib").tail(1)
            print(last[["calib", "ratio_n_chases", "ratio_n_sub_in", "ratio_n_sub_out",
                        "ratio_pre_2s", "ratio_post_2s"]].round(3).to_string(index=False),
                  flush=True)
        t0 = time.perf_counter()
        _AC.update(cfg=cfg, params=params, trials=list(trials))
        tasks = ([("observed", 0)] if with_obs else []) + [
            (sc.name, r) for r in range(n_reps) for sc in scs]
        res.extend(_pool_map(_rep_task, tasks, workers))
        step(f"replicates {tag} ({len(tasks)} datasets)", t0)

    stage([sc for sc in scen if not sc.matched], True, "stage 1")
    # causal effect that reproduces the observed standard 1 s HR, from the attenuation of
    # HR 2 in causal_plus_burst (log HR estimated / log HR true, median over replicates)
    e1 = pd.DataFrame([r for x in res for r in x["rows"]])
    hp = e1[e1["stat"] == "hr_primary"]
    h_obs = float(hp.loc[hp["scenario"] == "observed", "value"].iloc[0])
    h2 = float(hp.loc[hp["scenario"] == "causal_plus_burst", "value"].median())
    atten = np.log(h2) / S.LOG2 if h2 > 1 else 1.0
    beta_m = float(np.clip(np.log(h_obs) / max(atten, 0.2), np.log(1.2), np.log(8.0)))
    scen = [replace(sc, overrides={**sc.overrides, "beta_off": beta_m},
                    label=sc.label + f" (true HR {np.exp(beta_m):.2f}; HR 2 gave {h2:.2f})")
            if sc.matched else sc for sc in scen]
    stage([sc for sc in scen if sc.matched], False, "stage 2 (HR matched)")

    # onsets: no offset effect, hz_sim's generator (with its post-offset chirp profile)
    t0 = time.perf_counter()
    on_scen = onset_scenarios(cfg)
    on_params: dict[str, S.SimParams] = {}
    start_mult: dict[str, np.ndarray] = {}
    s_iter = 2 if quick else int(_opt(cfg, "acsim_start_iter"))
    s_reps = 4 if quick else int(_opt(cfg, "acsim_start_reps"))
    for i, osc in enumerate(on_scen):
        P = replace(P0, **osc.overrides)
        if osc.start:  # chase-start profile (and rates, under the effect) calibrated
            P, lg, mult = calibrate_start(P, trials, cfg, f"acsim_{osc.name}", s_iter, s_reps,
                                          workers)
            logs.append(lg.assign(calib=osc.name))
            start_mult[osc.name] = mult
            on_scen[i] = replace(osc, label=osc.label + " (multipliers 0-0.5 / 0.5-1 / 1-1.5 / "
                                 "1.5-2 s: " + " / ".join(f"{m:.2f}" for m in mult) + ")")
        elif osc.calib:
            tps, lg = S.calibrate_params(P, {}, trials, cfg, f"acsim_{osc.name}", n_iter=n_iter,
                                         n_reps=c_reps, workers=workers)
            P = replace(P, trials=tps)
            logs.append(lg.assign(calib=osc.name))
        on_params[osc.name] = P
    step("onset calibration", t0)
    t0 = time.perf_counter()
    _AC.clear()
    _AC.update(cfg=cfg, on_params=on_params, trials=list(trials))
    on_tasks = [("observed", 0)] + [(osc.name, r) for r in range(on_reps) for osc in on_scen]
    res_on = _pool_map(_onset_task, on_tasks, workers)
    step(f"onset accounts ({len(on_tasks)} datasets)", t0)
    _AC.clear()

    calib = pd.concat(logs, ignore_index=True)
    est = pd.DataFrame([r for x in res for r in x["rows"]])
    per = pd.DataFrame([r for x in res for r in x["per"]])
    est_on = pd.DataFrame([r for x in res_on for r in x["rows"]])
    task_t = pd.DataFrame([x["timing"] for x in res] + [x["timing"] for x in res_on])
    obs_unc = observed_uncertainty(est, per, Path(cfg.out_dir), seed=int(cfg.seed),
                                   est_on=est_on)
    summ = summarize(est, obs_unc)
    summ_on = summarize(est_on, obs_unc)
    cnt = dyad_counts(per)
    scen_tab = pd.DataFrame([dict(scenario=s.name, label=s.label, short=s.short,
                                  locked=s.locked, matched=s.matched, bout=s.bout, drift=s.drift,
                                  causal=s.causal,
                                  bout_mult=float(prof["bout_mult"]) if s.bout else 1.0,
                                  bout_rate_ratio_same=float(prof["bout_raw"]),
                                  bout_rate_ratio_neighbour=float(prof["bout_nb"]),
                                  beta_off=float(s.overrides.get("beta_off", 0.0)),
                                  overrides=repr(s.overrides), n_reps=n_reps)
                             for s in scen])
    on_tab = pd.DataFrame([dict(scenario=s.name, label=s.label, short=s.short, causal=s.causal,
                                calib=s.calib, start=s.start, overrides=repr(s.overrides),
                                n_reps=on_reps,
                                **{f"start_mult_{i}": (float(start_mult[s.name][i])
                                                       if s.name in start_mult else 1.0)
                                   for i in range(len(START_EDGES) - 1)})
                           for s in on_scen])
    par = pd.DataFrame([dict(scenario=nm, recording=tp.recording, h_mult_in=tp.h_mult_in,
                             h_mult_out=tp.h_mult_out, sub_mult_in=tp.sub_mult_in,
                             sub_mult_out=tp.sub_mult_out, resp_n=tp.resp_n, sym_n=tp.sym_n)
                        for nm, P in {**params, **on_params}.items() for tp in P.trials])
    tim.append(dict(step="total", seconds=time.perf_counter() - t_all))
    timings = pd.DataFrame(tim).assign(workers=workers, n_reps=n_reps,
                                       cpu_seconds_replicates=float(task_t["total_s"].sum()))
    est.to_csv(out / "acausal_sim_estimates.csv", index=False)
    est_on.to_csv(out / "acausal_sim_onset_estimates.csv", index=False)
    obs_unc.to_csv(out / "acausal_sim_observed.csv", index=False)
    summ.to_csv(out / "acausal_sim_summary.csv", index=False)
    summ_on.to_csv(out / "acausal_sim_onset_summary.csv", index=False)
    cnt.to_csv(out / "acausal_sim_counts.csv", index=False)
    prof["table"].to_csv(out / "acausal_sim_profiles.csv", index=False)
    prof_d["table"].to_csv(out / "acausal_sim_profiles_drift.csv", index=False)
    drift_tab.to_csv(out / "acausal_sim_drift.csv", index=False)
    calib.to_csv(out / "acausal_sim_calibration.csv", index=False)
    scen_tab.to_csv(out / "acausal_sim_scenarios.csv", index=False)
    on_tab.to_csv(out / "acausal_sim_onset_scenarios.csv", index=False)
    par.to_csv(out / "acausal_sim_params.csv", index=False)
    timings.to_csv(out / "acausal_sim_timings.csv", index=False)
    task_t.to_csv(out / "acausal_sim_task_timings.csv", index=False)
    t = dict(summary=summ, scenarios=scen_tab, profiles=prof["table"], calibration=calib,
             timings=timings, counts=cnt, estimates=est, onset_summary=summ_on,
             onset_scenarios=on_tab, onset_estimates=est_on, drift=drift_tab, observed=obs_unc)
    text = summary_text(t)
    (out / "acausal_sim_summary.txt").write_text(text + "\n")
    figs = plot_calibration(summ, scen_tab, Path(cfg.out_dir) / "figures")
    ha.rewrite_summary(Path(cfg.out_dir))
    return dict(t, text=text, figures=figs, params=params)


TABLES = ("estimates", "summary", "scenarios", "profiles", "calibration", "timings", "counts",
          "observed", "onset_estimates", "onset_summary", "onset_scenarios", "drift")


def load_tables(out_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Saved tables of a run (for --report-only and the summaries), or None."""
    d = Path(out_dir) / "simulation"
    if not (d / "acausal_sim_estimates.csv").exists():
        return None

    def rd(nm: str) -> pd.DataFrame:
        f = d / f"acausal_sim_{nm}.csv"
        return (pd.read_csv(f, keep_default_na=False, na_values=[""],
                            float_precision="round_trip") if f.exists() else pd.DataFrame())

    t = {k: rd(k) for k in TABLES}
    # summaries written by an older version (without exceedance / status): rebuild them from
    # the estimates (not written back; hz_acausal_sim --report-only rewrites the csv files)
    for k, e in (("summary", "estimates"), ("onset_summary", "onset_estimates")):
        if len(t[e]) and (not len(t[k]) or "exceed_frac" not in t[k]):
            t[k] = summarize(t[e], t["observed"])
    return t


def report_from_tables(out_dir: Path) -> str:
    """Rebuild the summaries (csv / txt), the figure and acausal_summary.txt from the saved
    estimates."""
    t = load_tables(out_dir)
    if t is None:
        raise SystemExit(f"no {Path(out_dir) / 'simulation' / 'acausal_sim_estimates.csv'}")
    d = Path(out_dir) / "simulation"
    t["summary"] = summarize(t["estimates"], t["observed"])
    t["summary"].to_csv(d / "acausal_sim_summary.csv", index=False)
    if len(t["onset_estimates"]):
        t["onset_summary"] = summarize(t["onset_estimates"], t["observed"])
        t["onset_summary"].to_csv(d / "acausal_sim_onset_summary.csv", index=False)
    text = summary_text(t)
    (d / "acausal_sim_summary.txt").write_text(text + "\n")
    plot_calibration(t["summary"], t["scenarios"], Path(out_dir) / "figures")
    ha.rewrite_summary(Path(out_dir))
    return text


def main(argv: list[str] | None = None) -> None:
    from hz_data import load_trials

    ap = argparse.ArgumentParser(description="Simulation calibration of the acausal control.")
    ap.add_argument("data_root", type=Path, nargs="?", default=Config().data_root)
    ap.add_argument("--out", type=Path, required=True, help="output directory of the pipeline")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the summary and figure from simulation/acausal_sim_*.csv")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 4 replicates per scenario, lighter calibration")
    ap.add_argument("--n-reps", type=int, default=None, help="replicates per scenario (100)")
    ap.add_argument("--workers", type=int, default=None, help="forked workers (8)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if a.report_only:
        print(report_from_tables(a.out))
        return
    cfg = Config(data_root=a.data_root, out_dir=a.out).updated(quick=a.quick or None, seed=a.seed)
    t0 = time.perf_counter()
    trials, _, _ = load_trials(cfg)
    print(f"{len(trials)} trials loaded ({time.perf_counter() - t0:.1f} s)", flush=True)
    res = run_acausal_sim(trials, cfg, n_reps=a.n_reps, workers=a.workers)
    print("\n" + res["text"])
    print(f"\ntotal {time.perf_counter() - t0:.0f} s; tables in "
          f"{Path(cfg.out_dir) / 'simulation'}")


if __name__ == "__main__":
    main()
