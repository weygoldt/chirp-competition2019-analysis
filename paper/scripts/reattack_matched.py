"""Chirp-triggered re-attack rate against matched moments: the hazard model's covariate switch as an event.

The re-attack hazard (``reattack_hazard.py``, ``stats_reattack_hazard.py``) asks whether the winner's next chase
onset comes sooner or later once the loser has chirped (``chirped``: 1 from the loser's first chirp at or after
max(offset - 5 s, onset) on). This module makes the same comparison as an event-triggered average, the way
Fig. 6 shows chirp rates around chase offsets (``stats_reattack_matched.py`` runs it; Fig. 8 B and the supplement
figure ``plot_receiver_matched_checks.py`` draw it).

Triggers
    ``first_chirp_triggers``: in every interval (chase offset -> next onset) whose loser first chirps after the
    offset, that chirp, at s seconds after the offset: exactly the moments at which the model's ``chirped``
    switches from 0 to 1 after the offset (an interval whose first counted chirp falls in the chase's last 5 s
    is chirped from the offset on and gives no trigger). At most one per interval.
    ``all_chirp_triggers`` (check): every loser chirp between the offset and the next onset.
Matched controls
    For a trigger at s in competition c: the other intervals of c that at s are still at risk (no re-attack yet,
    length > s) and not yet chirped (their ``chirped`` still 0 at s). Controls are NOT required to stay
    silent afterwards (that would keep the intervals that ended quickly: conditioning on the outcome); a
    control leaves the risk set (is censored) when its own loser chirps, as in the hazard model. Every
    eligible interval is used with weight 1 / (number eligible), so each trigger's controls weigh as much as
    the trigger (``n_controls`` samples a fixed number instead, with a seed).
Follow-up
    From the trigger (u = 0) to the interval's end: an event if the interval ends in a chase onset (the first
    re-attack after the trigger), censored at the end of the analysis window; controls also censored at
    their own first chirp.

Estimators (``MatchedSummary``, per competition so that competitions can be resampled)
    kernel-smoothed hazard: the re-attacks after the trigger convolved with Fig. 6's Gaussian kernel
    (sigma 0.3 s; reflected at u = 0, where no re-attack can precede the trigger) divided by the
    (weighted) number of intervals still at risk, per minute; Kaplan-Meier "not yet re-attacked";
    events / time at risk in a window [0, T] and their ratio (triggers / controls).

Surrogates (``shuffle_first_chirps``, ``simulate_reattacks``) check that the matched design returns 1 without
an effect and recovers a planted hazard ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import reattack_hazard as rh
from paper_config import SEED

COMP = rh.COMP
TRIGGER, CONTROL = "trigger", "control"
GROUPS = (TRIGGER, CONTROL)
KERNEL_SIGMA_S = 0.3  # Fig. 6 (analyze_event_time_rates_trial.gaussian_kde_params)
GRID_DT_S = 0.1  # Fig. 6's grid step
HORIZON_S = 30.0  # follow-up shown and summarised (Fig. 6's half-width)
CONTROL_RULES = ("unchirped_at_s", "never_chirped")  # the second is the selection trap (surrogate check only)


# ---------------------------------------------------------------------------
# triggers and matched controls
# ---------------------------------------------------------------------------
def switch_times(intervals: pd.DataFrame) -> np.ndarray:
    """Per interval, the time since the offset at which ``chirped`` switches on: max(first chirp - offset, 0);
    inf if the loser does not chirp before the interval ends."""
    rel = (intervals["first_chirp"] - intervals["offset"]).to_numpy(float)
    return np.where(np.isfinite(rel), np.maximum(rel, 0.0), np.inf)


def _check_intervals(intervals: pd.DataFrame) -> None:
    need = [COMP, "offset", "length", "event", "first_chirp"]
    missing = [c for c in need if c not in intervals.columns]
    if missing:
        raise KeyError(f"interval columns missing: {missing}")
    if not intervals.index.is_unique:
        raise ValueError("interval index must be unique (it identifies the intervals)")
    if (intervals["length"] <= 0).any():
        raise ValueError("an interval of zero or negative length")
    rel = (intervals["first_chirp"] - intervals["offset"]).to_numpy(float)
    late = np.isfinite(rel) & (rel >= intervals["length"].to_numpy(float))
    if late.any():
        raise ValueError(f"{int(late.sum())} first chirps at or after the interval's end")


def first_chirp_triggers(intervals: pd.DataFrame) -> pd.DataFrame:
    """The loser's first chirp after the offset where ``chirped`` was still 0: columns recording, interval
    (index of ``intervals``), s (time since the offset)."""
    _check_intervals(intervals)
    rel = (intervals["first_chirp"] - intervals["offset"]).to_numpy(float)
    keep = np.isfinite(rel) & (rel > 0)
    return pd.DataFrame({COMP: intervals[COMP].to_numpy()[keep], "interval": intervals.index.to_numpy()[keep],
                         "s": rel[keep]})


def all_chirp_triggers(intervals: pd.DataFrame, chirps: dict[str, np.ndarray]) -> pd.DataFrame:
    """Every loser chirp strictly between the offset and the interval's end (check: several per interval)."""
    _check_intervals(intervals)
    rows = []
    for idx, row in intervals.iterrows():
        c = np.asarray(chirps[row[COMP]], float) - float(row["offset"])
        s = c[(c > 0) & (c < float(row["length"]))]
        rows.append(pd.DataFrame({COMP: row[COMP], "interval": idx, "s": s}))
    if not rows:
        return pd.DataFrame({COMP: [], "interval": [], "s": []})
    return pd.concat(rows, ignore_index=True)


def matched_followups(intervals: pd.DataFrame, triggers: pd.DataFrame, n_controls: int | None = None,
                      seed: int = SEED, rule: str = "unchirped_at_s") -> pd.DataFrame:
    """Follow-up rows of every trigger and its matched controls.

    Columns: set (trigger row of ``triggers``), group ('trigger' / 'control'), recording, interval, s,
    weight, exit (time after the trigger at which the row leaves the risk set), event (1 if it leaves by a
    re-attack) and by_chirp (1 if a control leaves because its own loser chirps). A trigger without an
    eligible control is dropped (``attrs['n_without_controls']``).

    ``rule`` 'unchirped_at_s' (the design): controls at risk and not yet chirped at s, censored at their
    own first chirp. 'never_chirped' (the selection trap, for the surrogate check only): controls at risk
    at s that never chirp before their re-attack; this keeps short intervals and biases the comparison.
    ``n_controls``: None uses every eligible control with weight 1 / n_eligible; an integer samples that many
    (fewer if fewer are eligible) without replacement, weight 1 / n_sampled.
    """
    _check_intervals(intervals)
    if rule not in CONTROL_RULES:
        raise ValueError(f"unknown control rule {rule!r} ({CONTROL_RULES})")
    if n_controls is not None and n_controls < 1:
        raise ValueError("n_controls must be >= 1 or None")
    unknown = set(triggers["interval"]) - set(intervals.index)
    if unknown:
        raise ValueError(f"{len(unknown)} trigger intervals are not in the interval table")
    rng = np.random.default_rng(seed)
    sw_all = pd.Series(switch_times(intervals), index=intervals.index)
    out, n_without = [], 0
    for rec, trg in triggers.groupby(COMP, sort=True):
        ivc = intervals[intervals[COMP] == rec]
        idx = ivc.index.to_numpy()
        L = ivc["length"].to_numpy(float)
        ev = ivc["event"].to_numpy(int)
        sw = sw_all.loc[idx].to_numpy(float)
        s = trg["s"].to_numpy(float)
        ti = trg["interval"].to_numpy()
        pos = pd.Index(idx).get_indexer(ti)
        if (s >= L[pos]).any() or (s < 0).any():
            raise ValueError(f"{rec}: a trigger outside its interval")
        elig = (L[None, :] > s[:, None]) & (idx[None, :] != ti[:, None])
        if rule == "unchirped_at_s":
            elig &= sw[None, :] > s[:, None]
            c_exit_abs, c_event = np.minimum(L, sw), np.where(np.isfinite(sw), 0, ev)
            c_by_chirp = np.isfinite(sw).astype(int)  # a first chirp always lies before the interval's end
        else:
            elig &= ~np.isfinite(sw)
            c_exit_abs, c_event, c_by_chirp = L, ev, np.zeros(len(L), int)
        kk, jj = np.nonzero(elig)  # (trigger, control) pairs, sorted by trigger
        if n_controls is not None and len(kk):
            # a random rank within each trigger's eligible controls; keep the first n_controls
            order = np.lexsort((rng.random(len(kk)), kk))
            kk, jj = kk[order], jj[order]
            start = np.searchsorted(kk, kk, side="left")
            keep = np.arange(len(kk)) - start < n_controls
            kk, jj = kk[keep], jj[keep]
            o = np.lexsort((jj, kk))
            kk, jj = kk[o], jj[o]
        n = np.bincount(kk, minlength=len(s))
        has = n > 0
        n_without += int((~has).sum())
        set_ids = trg.index.to_numpy()
        t_rows = pd.DataFrame({"set": set_ids[has], "group": TRIGGER, COMP: rec, "interval": ti[has],
                               "s": s[has], "weight": 1.0, "exit": L[pos[has]] - s[has],
                               "event": ev[pos[has]].astype(int), "by_chirp": 0})
        c_rows = pd.DataFrame({"set": set_ids[kk], "group": CONTROL, COMP: rec, "interval": idx[jj],
                               "s": s[kk], "weight": 1.0 / n[kk], "exit": c_exit_abs[jj] - s[kk],
                               "event": c_event[jj].astype(int), "by_chirp": c_by_chirp[jj]})
        out.extend([t_rows, c_rows])
    cols = ["set", "group", COMP, "interval", "s", "weight", "exit", "event", "by_chirp"]
    f = pd.concat(out, ignore_index=True) if out else pd.DataFrame({c: [] for c in cols})
    f["_g"] = (f["group"] == CONTROL).astype(int)
    f = f.sort_values(["set", "_g", "interval"], kind="stable").drop(columns="_g").reset_index(drop=True)[cols]
    f.attrs["n_without_controls"] = n_without
    f.attrs["rule"] = rule
    return f


# ---------------------------------------------------------------------------
# estimators, per competition (resampling) and pooled
# ---------------------------------------------------------------------------
def time_grid(horizon: float = HORIZON_S, dt: float = GRID_DT_S) -> np.ndarray:
    return np.round(np.arange(0.0, horizon + dt / 2, dt), 10)


def reflected_kernel(grid: np.ndarray, times: np.ndarray, weights: np.ndarray,
                     sigma: float = KERNEL_SIGMA_S) -> np.ndarray:
    """Sum of weighted Gaussian kernels (Fig. 6's) at ``times`` on ``grid``, reflected at 0 so that no mass of
    an event just after the trigger is lost to u < 0 (the kernel integrates to the weight on u >= 0).

    The events are first summed into bins of sigma / 150 (0.002 s for Fig. 6's kernel; each event moves by
    at most 0.001 s), so that hundreds of thousands of control events cost no more than a few thousand."""
    times, weights = np.asarray(times, float), np.asarray(weights, float)
    if (times < 0).any():
        raise ValueError("event times before the trigger")
    near = times <= grid[-1] + 8 * sigma  # farther events add < 1e-14 of a kernel on the grid
    times, weights = times[near], weights[near]
    if len(times) == 0:
        return np.zeros_like(grid, dtype=float)
    width = sigma / 150
    k = np.floor(times / width).astype(int)
    wb = np.bincount(k, weights=weights)
    used = np.flatnonzero(wb)
    t, w = (used + 0.5) * width, wb[used]
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    out = np.zeros_like(grid, dtype=float)
    for chunk in range(0, len(t), 4000):
        tc, wc = t[chunk:chunk + 4000], w[chunk:chunk + 4000]
        d1 = (grid[:, None] - tc[None, :]) / sigma
        d2 = (grid[:, None] + tc[None, :]) / sigma
        out += norm * ((np.exp(-0.5 * d1 ** 2) + np.exp(-0.5 * d2 ** 2)) @ wc)
    return out


@dataclass
class MatchedSummary:
    """Per competition and group: what every estimator needs, so that competitions can be resampled.

    Arrays have shape (n_competitions, ...) and are indexed by ``competitions``; per group (trigger, control):
    ``num`` smoothed re-attacks on ``grid`` (per s), ``at_risk`` weighted intervals at risk on ``grid``,
    ``km_dn`` / ``km_y`` weighted events and at-risk at each of ``km_times[g]`` (the distinct event times
    up to the horizon), ``E`` / ``X`` events and time at risk in each window of ``windows``, ``n_sets``.
    """

    competitions: list[str]
    grid: np.ndarray
    windows: tuple[float, ...]
    num: dict[str, np.ndarray] = field(default_factory=dict)
    at_risk: dict[str, np.ndarray] = field(default_factory=dict)
    km_times: dict[str, np.ndarray] = field(default_factory=dict)
    km_dn: dict[str, np.ndarray] = field(default_factory=dict)
    km_y: dict[str, np.ndarray] = field(default_factory=dict)
    E: dict[str, np.ndarray] = field(default_factory=dict)
    X: dict[str, np.ndarray] = field(default_factory=dict)
    n_sets: np.ndarray = field(default_factory=lambda: np.zeros(0))


def summarise(f: pd.DataFrame, competitions: list[str] | None = None, grid: np.ndarray | None = None,
              windows: tuple[float, ...] = (10.0, 30.0), sigma: float = KERNEL_SIGMA_S) -> MatchedSummary:
    """Per-competition ingredients of the estimators from ``matched_followups`` rows."""
    grid = time_grid() if grid is None else np.asarray(grid, float)
    comps = sorted(f[COMP].unique()) if competitions is None else list(competitions)
    horizon = float(grid[-1])
    out = MatchedSummary(comps, grid, tuple(windows))
    out.n_sets = f.groupby(COMP)["set"].nunique().reindex(comps, fill_value=0).to_numpy(float)
    empty = {k: np.zeros(0) for k in ("exit", "weight", "event")}
    for g in GROUPS:
        fg = f[f["group"] == g]
        ev_t = np.unique(fg.loc[(fg["event"] == 1) & (fg["exit"] <= horizon), "exit"].to_numpy(float))
        out.km_times[g] = ev_t
        by_comp = {c: {k: d[k].to_numpy(float) for k in empty} for c, d in fg.groupby(COMP, sort=False)}
        num, risk, dn, y, E, X = [], [], [], [], [], []
        for c in comps:
            ex, w, e = (by_comp.get(c, empty)[k] for k in ("exit", "weight", "event"))
            evm = e == 1
            num.append(reflected_kernel(grid, ex[evm], w[evm], sigma))
            order = np.argsort(ex, kind="stable")
            exs, cw = ex[order], np.r_[0.0, np.cumsum(w[order])]
            risk.append(cw[-1] - cw[np.searchsorted(exs, grid, side="right")])  # exit > u
            # KM: weighted events at each distinct event time, at risk = exit >= t (left-continuous)
            sel = evm & (ex <= horizon)
            dn.append(np.bincount(np.searchsorted(ev_t, ex[sel]), weights=w[sel], minlength=len(ev_t)))
            y.append(cw[-1] - cw[np.searchsorted(exs, ev_t, side="left")])
            E.append([w[evm & (ex <= T)].sum() for T in windows])
            X.append([(w * np.minimum(ex, T)).sum() for T in windows])
        out.num[g], out.at_risk[g] = np.array(num), np.array(risk)
        out.km_dn[g] = np.array(dn).reshape(len(comps), len(ev_t))
        out.km_y[g] = np.array(y).reshape(len(comps), len(ev_t))
        out.E[g], out.X[g] = np.array(E, float), np.array(X, float)
    return out


def _combine(m: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.tensordot(m, a, axes=(0, 0))


def hazard_per_min(sm: MatchedSummary, m: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Kernel-smoothed hazard (re-attacks per min per interval at risk) on ``sm.grid`` per group; ``m``
    counts each competition (bootstrap multiplicities; default 1 each). NaN where nobody is at risk."""
    m = np.ones(len(sm.competitions)) if m is None else np.asarray(m, float)
    out = {}
    for g in GROUPS:
        num, y = _combine(m, sm.num[g]), _combine(m, sm.at_risk[g])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[g] = np.where(y > 0, 60.0 * num / y, np.nan)
    return out


def survival(sm: MatchedSummary, m: np.ndarray | None = None, at: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Weighted Kaplan-Meier fraction not yet re-attacked, at ``at`` (default ``sm.grid``), per group."""
    m = np.ones(len(sm.competitions)) if m is None else np.asarray(m, float)
    at = sm.grid if at is None else np.asarray(at, float)
    out = {}
    for g in GROUPS:
        t = sm.km_times[g]
        if len(t) == 0:
            out[g] = np.ones_like(at, dtype=float)
            continue
        dn, y = _combine(m, sm.km_dn[g]), _combine(m, sm.km_y[g])
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(y > 0, 1.0 - dn / y, 1.0)
        S = np.r_[1.0, np.cumprod(step)]
        out[g] = S[np.searchsorted(t, at, side="right")]
    return out


def window_rates(sm: MatchedSummary, m: np.ndarray | None = None) -> dict:
    """Per window T: events and time at risk in [0, T] per group, rates per min and their ratio."""
    m = np.ones(len(sm.competitions)) if m is None else np.asarray(m, float)
    out = {}
    for k, T in enumerate(sm.windows):
        E = {g: float(m @ sm.E[g][:, k]) for g in GROUPS}
        X = {g: float(m @ sm.X[g][:, k]) for g in GROUPS}
        rate = {g: 60.0 * E[g] / X[g] if X[g] > 0 else np.nan for g in GROUPS}
        ratio = rate[TRIGGER] / rate[CONTROL] if rate[CONTROL] > 0 else np.nan
        out[f"{T:g}"] = {"events": E, "exposure_s": X, "rate_per_min": rate, "ratio": ratio}
    return out


def resample_weights(n_comp: int, n: int, seed: int = SEED, mode: str = "bootstrap",
                     leave_out_frac: float = 0.1) -> np.ndarray:
    """(n, n_comp) multiplicities of the competitions: 'bootstrap' draws competitions with replacement;
    'leave_out' drops floor(leave_out_frac * n_comp) (>= 1) of them, as Fig. 6's resamples do."""
    rng = np.random.default_rng(seed)
    if mode == "bootstrap":
        return np.stack([np.bincount(rng.integers(0, n_comp, n_comp), minlength=n_comp) for _ in range(n)])
    if mode == "leave_out":
        k = max(1, int(n_comp * leave_out_frac))
        out = np.ones((n, n_comp))
        for i in range(n):
            out[i, rng.choice(n_comp, k, replace=False)] = 0
        return out
    raise ValueError(f"unknown resampling mode {mode!r}")


def resampled(sm: MatchedSummary, weights: np.ndarray, survival_at: np.ndarray | None = None) -> dict:
    """Every estimator for each row of ``weights``: hazard[g] (n, grid), survival[g] (n, grid or
    ``survival_at``), window ratios (n, windows), window rates[g] (n, windows)."""
    haz = {g: [] for g in GROUPS}
    surv = {g: [] for g in GROUPS}
    ratio, rate = [], {g: [] for g in GROUPS}
    for m in weights:
        h = hazard_per_min(sm, m)
        s = survival(sm, m, survival_at)
        w = window_rates(sm, m)
        for g in GROUPS:
            haz[g].append(h[g])
            surv[g].append(s[g])
            rate[g].append([w[k]["rate_per_min"][g] for k in w])
        ratio.append([w[k]["ratio"] for k in w])
    return {"hazard": {g: np.array(v) for g, v in haz.items()}, "survival": {g: np.array(v) for g, v in surv.items()},
            "ratio": np.array(ratio), "rate": {g: np.array(v) for g, v in rate.items()}}


# ---------------------------------------------------------------------------
# surrogates
# ---------------------------------------------------------------------------
def _with_first_rel(intervals: pd.DataFrame, rel: np.ndarray) -> pd.DataFrame:
    out = intervals.copy()
    out["first_chirp"] = np.where(np.isfinite(rel), out["offset"].to_numpy(float) + rel, np.nan)
    return out


def shuffle_first_chirps(intervals: pd.DataFrame, chirps: dict[str, np.ndarray], rng: np.random.Generator,
                         left: float = -5.0) -> pd.DataFrame:
    """No-effect surrogate: within each competition, the intervals' loser chirp trains (times since the
    offset) are permuted among the intervals; each interval keeps the chirps of its new train that fall in
    its own counting range [max(left, onset - offset), length). The first of them is its new first chirp.
    The intervals' lengths and events stay real, so a chirp now says nothing about when the winner attacks."""
    rel = np.full(len(intervals), np.nan)
    pos = {k: i for i, k in enumerate(intervals.index)}
    for rec, ivc in intervals.groupby(COMP, sort=False):
        lc = np.asarray(chirps[rec], float)
        trains = []
        for _, r in ivc.iterrows():
            a = lc - float(r["offset"])
            trains.append(a[(a >= left) & (a < float(r["length"]))])
        perm = rng.permutation(len(trains))
        for (idx, r), p in zip(ivc.iterrows(), perm):
            t = trains[p]
            lo = max(left, float(r["onset"]) - float(r["offset"]))
            t = t[(t >= lo) & (t < float(r["length"]))]
            rel[pos[idx]] = t.min() if len(t) else np.nan
    return _with_first_rel(intervals, rel)


def baseline_rates(intervals: pd.DataFrame, pieces=rh.PIECES_S) -> tuple[np.ndarray, pd.Series]:
    """Pooled re-attack rate per piece of time since the offset (events / s at risk, every interval) and each
    competition's frailty (observed / expected events under the pooled rates): the simulation's baseline.
    A piece beyond the longest interval (no time at risk) takes the rate of the last piece with time at risk;
    a piece with time at risk but no event keeps rate 0."""
    p = np.asarray(pieces, float)
    L = intervals["length"].to_numpy(float)
    ev = intervals["event"].to_numpy(int)
    lo, hi = p[:-1], p[1:]
    X = np.clip(L[:, None], lo[None, :], hi[None, :]) - lo[None, :]  # time at risk in each piece
    k = np.searchsorted(p, L, side="right") - 1
    E = np.zeros(len(lo))
    np.add.at(E, k[ev == 1], 1)
    Xs = X.sum(axis=0)
    if Xs[0] <= 0 or E.sum() == 0:
        raise ValueError("no time at risk or no re-attack: no baseline to simulate from")
    rate = np.zeros(len(lo))
    for i in range(len(lo)):
        rate[i] = E[i] / Xs[i] if Xs[i] > 0 else rate[i - 1]
    expected = pd.Series((X * rate[None, :]).sum(axis=1), index=intervals.index)
    frailty = (intervals.groupby(COMP)["event"].sum() / expected.groupby(intervals[COMP]).sum())
    return rate, frailty


def _cumhaz_inverse(target: float, rate: np.ndarray, pieces: np.ndarray, switch: float, hr: float,
                    scale: float) -> float:
    """Time u at which the cumulative hazard (piecewise-constant baseline x scale, x hr from ``switch`` on)
    reaches ``target``; inf if it never does."""
    cuts = np.unique(np.r_[pieces[:-1], [switch] if np.isfinite(switch) else []])
    cuts = cuts[np.isfinite(cuts)]
    H = 0.0
    for a, b in zip(cuts, np.r_[cuts[1:], np.inf]):
        k = np.searchsorted(pieces, a, side="right") - 1
        h = rate[k] * scale * (hr if a >= switch else 1.0)
        if h <= 0:
            continue
        if H + h * (b - a) >= target:
            return a + (target - H) / h
        H += h * (b - a)
    return np.inf


def simulate_reattacks(intervals: pd.DataFrame, hr: float, rng: np.random.Generator,
                       pieces=rh.PIECES_S) -> pd.DataFrame:
    """Planted-effect surrogate: every interval keeps its offset and its real first chirp time (the chirp
    process stays real), but the next onset is drawn anew from the pooled piecewise baseline x the
    competition's frailty, multiplied by ``hr`` from the first chirp on (the hazard model with a known
    effect). A chirp after the simulated onset does not happen (first_chirp NaN). A censored real interval
    stays censored at its real length; an event can fall later than the real onset."""
    p = np.asarray(pieces, float)
    rate, frailty = baseline_rates(intervals, p)
    sw = switch_times(intervals)
    L_real = intervals["length"].to_numpy(float)
    ev_real = intervals["event"].to_numpy(int)
    targets = rng.exponential(size=len(intervals))
    L, ev = np.empty(len(intervals)), np.empty(len(intervals), int)
    for i, rec in enumerate(intervals[COMP].to_numpy()):
        t = _cumhaz_inverse(targets[i], rate, p, sw[i], hr, float(frailty[rec]))
        if ev_real[i] == 0 and t >= L_real[i]:
            L[i], ev[i] = L_real[i], 0
        else:
            L[i], ev[i] = t, 1
    out = intervals.copy()
    out["length"], out["event"] = L, ev
    out["end"] = out["offset"] + L
    rel = (intervals["first_chirp"] - intervals["offset"]).to_numpy(float)
    rel = np.where(np.isfinite(rel) & (rel < L), rel, np.nan)
    return _with_first_rel(out, rel)
