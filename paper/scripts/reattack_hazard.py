"""Data side of the re-attack hazard: inter-chase intervals and their person-period table.

Question: after a chase offset, is the hazard of the winner's next chase onset lower (appeasement) or higher
once the loser has chirped? The windowed Gamma GLMMs (``stats_chirp_affects_chasing.py``) relate the next
inter-chase interval to a loser chirp in a window around the offset that is clipped at the next onset, so the
predictor depends on the response; their complete-window check removes that by conditioning on the outcome.
A hazard model with the chirp as a time-varying covariate uses every interval, and a chirp counts only once it
has happened. This module builds the data; ``stats_reattack_hazard.py`` fits the models,
``stats_reattack_matched.py`` makes the matched chirp-triggered comparison (``reattack_matched.py``) and
``plot_receiver_before_after.py`` (Fig. 8) and ``plot_receiver_matched_checks.py`` (supplement) draw them.

Data: the windowed models' own (``summarize_submissive_chirp_effect_on_dominant``, which writes their
``chirp_effect_on_dominant_obs.csv``), step by step: the same trials (``is_bad_trial``: group >= 5, LED files,
no draws, not the four close-EODf recordings), BORIS chases through the paper's loader
(``load_and_convert_boris_events``: each onset paired with the next later offset, video frames mapped to
recording time with ``led_alignment.fit_led_alignment``), the analysis window ``t_max`` = min(last event of the
recording, 3 h), chases with onset and offset before ``t_max``, overlapping or nested chases merged into one
(``_merge_overlapping_chases``: an onset at or before the running offset extends that chase), the loser's
chirps from ``chirps.load_chirps`` (``PAPER_DETECTIONS``) before ``t_max``. ``check_against_obs`` asserts that
the chases, intervals and window predictors equal the obs.csv rows.

Risk set: every interval from a chase offset to the next chase onset (the event); the last chase's interval is
censored at ``t_max`` (or ends with an event at the onset of a chase that straddles ``t_max`` and was dropped
by the windowed filter; none in these data, counted in ``n_straddling``). Time is time since the offset.

Person-period table (piecewise-exponential model): each interval is split into rows at the baseline pieces
``PIECES_S`` and at every moment a chirp covariate changes; a row carries its exposure (s), event (1 only in
the last row of an interval that ends in an onset) and the covariates' values, which are constant within it.

Covariates (time-varying, ``covariate_on_intervals``):

``chirped``
    1 from the loser's first chirp at or after max(offset + left, onset) on (left = the offset window's left
    edge, -5 s; the chase-tail half of the windowed predictor), 0 before. A chirp in the chase tail switches it
    on at the offset; one after the offset switches it on at the chirp. Never switches off.
``recent_1s``, ``recent_5s``
    >= 1 loser chirp in the preceding 1 s or 5 s (chirps anywhere, also before the offset or the chase onset).
``guard``
    >= 1 loser chirp 0.5-1.5 s before (the lag-guarded indicator of the hazard analysis, ``hz_hazard``).

Interval-level covariates: ``log_prev`` (log previous inter-chase interval: chasing density; missing for the
first chase of a trial), ``log_dur`` (log duration of the chase that just ended) and the placebo ``next_chirped``
(the windowed models' predictor of the *next* chase, a loser chirp in [max(next offset + left, next onset),
min(next offset + right, the onset after)]: it is measured after the event, so it cannot cause it; the hazard
analogue of the acausal model M2; missing for censored intervals).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Baseline pieces of time since the offset (s), log-spaced (1-2-5 steps): the re-attack hazard changes fastest
# just after an offset; the last piece is open. No interval is shorter than 1.66 s (chases closer than that were
# annotated as one), so 0-1 s would hold no event and 0-2 s is one piece; the long tail (99th percentile
# 600 s) is split at 500 s. Events and exposure per piece are in the stats (reattack_hazard.pieces).
PIECES_S = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, np.inf)

RECENT_S = {"recent_1s": 1.0, "recent_5s": 5.0}
GUARD_S = (0.5, 1.5)  # lag-guarded indicator: a chirp between 0.5 and 1.5 s before
TIME_VARYING = ("chirped", "recent_1s", "recent_5s", "guard")
COMP = "recording"


# ---------------------------------------------------------------------------
# data: competitions and their intervals
# ---------------------------------------------------------------------------
@dataclass
class Competition:
    """One analysed competition in recording time (s)."""

    recording: str
    chases: np.ndarray  # (m, 2) onsets, offsets: merged, sorted, both before t_max
    loser_chirps: np.ndarray  # sorted, before t_max
    t_max: float  # end of the analysis window
    straddling_onset: float = np.nan  # onset of a chase that starts before t_max and ends after it

    def __post_init__(self) -> None:
        self.chases = np.asarray(self.chases, float).reshape(-1, 2)
        self.loser_chirps = np.sort(np.asarray(self.loser_chirps, float))
        if len(self.chases):
            if np.any(self.chases[:, 1] < self.chases[:, 0]):
                raise ValueError(f"{self.recording}: chase with offset before onset")
            if np.any(self.chases[1:, 0] <= self.chases[:-1, 1]):
                raise ValueError(f"{self.recording}: chases overlap or touch (merge them first)")
            if self.chases[-1, 1] >= self.t_max:
                raise ValueError(f"{self.recording}: chase offset at or after t_max")
        if len(self.loser_chirps) and self.loser_chirps[-1] >= self.t_max:
            raise ValueError(f"{self.recording}: loser chirp at or after t_max")
        if np.isfinite(self.straddling_onset) and len(self.chases) and (
                self.straddling_onset <= self.chases[-1, 1] or self.straddling_onset >= self.t_max):
            raise ValueError(f"{self.recording}: straddling onset outside (last offset, t_max)")


def load_competitions(data_path: Path, t_max_cap: float | None = None) -> list[Competition]:
    """The windowed models' competitions, chases and loser chirps (see the module docstring)."""
    from analyze_event_time_rates_trial import load_and_convert_boris_events
    from chirps import load_chirps
    from summarize_submissive_chirp_effect_on_dominant import (
        DEFAULT_T_MAX_CAP,
        _merge_overlapping_chases,
        is_bad_trial,
    )

    cap = DEFAULT_T_MAX_CAP if t_max_cap is None else t_max_cap
    summary = pd.read_csv(Path(data_path) / "trial_summary.csv", index_col=0)
    comps = []
    for _, row in summary.iterrows():
        if is_bad_trial(row, Path(data_path)):
            continue
        rec = row["recording"]
        folder = Path(data_path) / rec
        contacts, on_off, _, _ = load_and_convert_boris_events(folder, rec, sr=20000)
        chirp_t, chirp_id = load_chirps(folder)
        if np.isnan(chirp_t).any():
            raise ValueError(f"{rec}: NaN chirp times")
        # t_max exactly as extract_chasing_data_for_trial
        observed_max = max(float(chirp_t.max()) if len(chirp_t) else 0.0,
                           float(on_off[:, 1].max()) if on_off.size else 0.0,
                           float(contacts.max()) if len(contacts) else 0.0)
        t_max = min(observed_max, cap) if observed_max > 0 else cap
        loser = chirp_t[chirp_id == row["lose_ID"]]
        loser = loser[loser < t_max]
        on, off = (on_off[:, 0], on_off[:, 1]) if on_off.size else (np.zeros(0), np.zeros(0))
        keep = (on < t_max) & (off < t_max)
        straddle = on[(on < t_max) & (off >= t_max)]
        m_on, m_off = _merge_overlapping_chases(on[keep], off[keep], rec)
        chases = np.column_stack([m_on, m_off]) if len(m_on) else np.zeros((0, 2))
        s_on = float(straddle.min()) if len(straddle) else np.nan
        if np.isfinite(s_on) and len(chases) and s_on <= chases[-1, 1]:
            s_on = np.nan  # starts inside the last kept chase: no extra event
        comps.append(Competition(rec, chases, loser, t_max, s_on))
    return comps


def _count(chirps: np.ndarray, start: float, end: float) -> int:
    """Chirps in [start, end] (as count_chirps_in_interval of the windowed analysis)."""
    if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
        return 0
    return int(np.sum((chirps >= start) & (chirps <= end)))


def intervals_table(comps: list[Competition], window: tuple[float, float]) -> pd.DataFrame:
    """One row per chase: the interval from its offset to the next onset (or to the censoring time).

    Columns: recording, chase (index in the competition), onset, offset, end (next onset or censoring time),
    event (1 if it ends in an onset), length (end - offset), duration (of the chase), prev_interval (previous
    offset to this onset; NaN for the first chase), first_chirp (the loser's first chirp at or after
    max(offset + left, onset); NaN if none before ``end``), and the windowed predictors of this chase
    (``chirped_window``: a loser chirp in [max(offset + left, onset), min(offset + right, next onset)]) and
    of the next one (``next_chirped``: NaN if there is no next chase).
    """
    left, right = window
    rows = []
    for c in comps:
        ch, lc = c.chases, c.loser_chirps
        m = len(ch)
        for i in range(m):
            on, off = ch[i]
            if i + 1 < m:
                end, event = ch[i + 1, 0], 1
            elif np.isfinite(c.straddling_onset):
                end, event = c.straddling_onset, 1
            else:
                end, event = c.t_max, 0
            next_on = ch[i + 1, 0] if i + 1 < m else np.inf
            start = max(off + left, on)
            later = lc[(lc >= start) & (lc < end)]
            first = float(later[0]) if len(later) else np.nan

            def window_chirped(j: int) -> float:
                on_j, off_j = ch[j]
                nxt = ch[j + 1, 0] if j + 1 < m else np.inf
                return float(_count(lc, max(off_j + left, on_j), min(off_j + right, nxt)) > 0)

            rows.append({
                COMP: c.recording, "chase": i, "onset": on, "offset": off, "end": end, "event": event,
                "length": end - off, "duration": off - on,
                "prev_interval": on - ch[i - 1, 1] if i > 0 else np.nan,
                "first_chirp": first,
                "next_onset": next_on,
                "chirped_window": window_chirped(i),
                "next_chirped": window_chirped(i + 1) if i + 1 < m else np.nan,
            })
    df = pd.DataFrame(rows)
    if len(df) and (df["length"] <= 0).any():
        raise ValueError("an interval of zero or negative length")
    return df


def check_against_obs(intervals: pd.DataFrame, obs: pd.DataFrame, atol: float = 1e-6) -> dict:
    """Assert that the intervals are the windowed models' chases (obs.csv rows, in the same order).

    Compared per row: recording, chase duration, previous and next interval (``time_to_next_chasing`` is
    NaN for the last chase, which this table censors) and the window predictor
    (``loser_chirped_offset_window``). Returns the counts that were compared.
    """
    if len(intervals) != len(obs):
        raise ValueError(f"{len(intervals)} intervals vs {len(obs)} obs.csv rows")
    a, b = intervals.reset_index(drop=True), obs.reset_index(drop=True)
    if not (a[COMP].astype(str) == b["recording"].astype(str)).all():
        raise ValueError("recordings differ from obs.csv (order or content)")
    nxt = np.where(a["next_onset"].notna() & np.isfinite(a["next_onset"]), a["next_onset"] - a["offset"], np.nan)
    pairs = {"chasing_duration": a["duration"], "time_since_previous_chasing": a["prev_interval"],
             "time_to_next_chasing": pd.Series(nxt)}
    for col, mine in pairs.items():
        theirs = b[col].to_numpy(float)
        mine = np.asarray(mine, float)
        if not np.array_equal(np.isnan(mine), np.isnan(theirs)):
            raise ValueError(f"{col}: missing values differ from obs.csv")
        ok = ~np.isnan(mine)
        if not np.allclose(mine[ok], theirs[ok], rtol=0, atol=atol):
            k = int(np.argmax(np.abs(mine[ok] - theirs[ok])))
            raise ValueError(f"{col} differs from obs.csv (max |diff| {np.abs(mine[ok] - theirs[ok])[k]:.3g} s)")
    if not (a["chirped_window"].astype(bool) == b["loser_chirped_offset_window"].astype(bool)).all():
        raise ValueError("loser_chirped_offset_window differs from obs.csv")
    return {"n_rows": int(len(a)), "n_recordings": int(a[COMP].nunique())}


# ---------------------------------------------------------------------------
# person-period table
# ---------------------------------------------------------------------------
def _union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of half-open intervals [a, b), sorted and merged (empty ones dropped)."""
    iv = sorted((a, b) for a, b in iv if b > a)
    out: list[list[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def covariate_on_intervals(name: str, row: pd.Series | dict, chirps: np.ndarray) -> list[tuple[float, float]]:
    """Where a time-varying covariate is 1 within one interval: sorted, disjoint [a, b) in time since offset.

    ``row`` is a row of ``intervals_table`` (offset, length, first_chirp); ``chirps`` are the loser's chirps
    of its competition (recording time).
    """
    off, length = float(row["offset"]), float(row["length"])
    if name == "chirped":
        fc = float(row["first_chirp"])
        if not np.isfinite(fc):
            return []
        a = max(fc - off, 0.0)
        return [(a, length)] if a < length else []
    if name in RECENT_S:
        lo, hi = 0.0, RECENT_S[name]  # on during [c, c + tau)
    elif name == "guard":
        lo, hi = GUARD_S  # on during [c + 0.5, c + 1.5)
    else:
        raise ValueError(f"unknown covariate {name!r} (known: {TIME_VARYING})")
    rel = np.asarray(chirps, float) - off
    rel = rel[(rel + hi > 0) & (rel + lo < length)]
    return _union([(max(r + lo, 0.0), min(r + hi, length)) for r in rel])


def _value_at(on: list[tuple[float, float]], u: float) -> int:
    return int(any(a <= u < b for a, b in on))


def person_period(intervals: pd.DataFrame, chirps: dict[str, np.ndarray],
                  covariates: tuple[str, ...] = TIME_VARYING,
                  pieces: tuple[float, ...] = PIECES_S) -> pd.DataFrame:
    """Split every interval at the baseline pieces and where any of ``covariates`` changes.

    ``chirps`` maps recording -> loser chirp times. One row per piece of constant covariates: recording,
    interval (row index of ``intervals``), piece (index into ``pieces``), t0 and t1 (time since the offset,
    s), exposure (t1 - t0), event (1 only in the last row of an interval with ``event`` = 1), and one 0/1
    column per covariate.
    """
    pieces = np.asarray(pieces, float)
    if pieces[0] != 0 or not np.all(np.diff(pieces) > 0) or not np.isinf(pieces[-1]):
        raise ValueError("pieces must start at 0, increase and end with inf")
    out: dict[str, list] = {k: [] for k in (COMP, "interval", "piece", "t0", "t1", "exposure", "event",
                                             *covariates)}
    for idx, row in intervals.iterrows():
        length = float(row["length"])
        on = {c: covariate_on_intervals(c, row, chirps[row[COMP]]) for c in covariates}
        cuts = set(pieces[(pieces > 0) & (pieces < length)].tolist())
        for iv in on.values():
            for a, b in iv:
                cuts.update(x for x in (a, b) if 0 < x < length)
        edges = np.array(sorted({0.0, length, *cuts}))
        t0, t1 = edges[:-1], edges[1:]
        n = len(t0)
        mid = 0.5 * (t0 + t1)
        out[COMP].extend([row[COMP]] * n)
        out["interval"].extend([idx] * n)
        out["piece"].extend((np.searchsorted(pieces, mid, side="right") - 1).tolist())
        out["t0"].extend(t0.tolist())
        out["t1"].extend(t1.tolist())
        out["exposure"].extend((t1 - t0).tolist())
        ev = np.zeros(n, int)
        ev[-1] = int(row["event"])
        out["event"].extend(ev.tolist())
        for c in covariates:
            out[c].extend(_value_at(on[c], u) for u in mid)
    pp = pd.DataFrame(out)
    pp["piece"] = pp["piece"].astype(int)
    return pp


def add_interval_covariates(pp: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Attach the interval-level covariates (log previous interval, log chase duration, the placebo)."""
    iv = intervals[["prev_interval", "duration", "next_chirped"]]
    out = pp.join(iv, on="interval")
    out["log_prev"] = np.log(out.pop("prev_interval"))
    out["log_dur"] = np.log(out.pop("duration"))
    return out


def aggregate(pp: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum events and exposure over rows that share ``keys`` (the Poisson likelihood depends on the rows
    only through these sums, so the posterior is unchanged; the table gets much smaller)."""
    return pp.groupby(keys, observed=True, sort=True, dropna=False)[["event", "exposure"]].sum().reset_index()


def piece_labels(pieces: tuple[float, ...] = PIECES_S) -> list[str]:
    """'0-1', '1-2', ..., '200+' (seconds)."""
    p = list(pieces)
    return [f"{a:g}-{b:g}" if np.isfinite(b) else f"{a:g}+" for a, b in zip(p[:-1], p[1:])]


# ---------------------------------------------------------------------------
# observed rates (figure and stats)
# ---------------------------------------------------------------------------
def cell_table(pp: pd.DataFrame, covariate: str = "chirped") -> pd.DataFrame:
    """Events and exposure per competition, piece and covariate value: index (recording, piece), columns
    E0, E1, X0, X1 (zero where a cell has no exposure)."""
    g = pp.groupby([COMP, "piece", covariate])[["event", "exposure"]].sum().unstack(covariate, fill_value=0)
    g = g.reindex(columns=pd.MultiIndex.from_product([["event", "exposure"], [0, 1]]), fill_value=0)
    return pd.DataFrame({"E0": g[("event", 0)], "E1": g[("event", 1)],
                         "X0": g[("exposure", 0)], "X1": g[("exposure", 1)]})


def mh_rates(cells: pd.DataFrame) -> pd.DataFrame:
    """Per piece, the rates with and without the covariate averaged over competitions with the same weight in
    both groups, w = X1 X0 / (X1 + X0) (the exposure analogue of the chase weights of panel A; 0 where a
    competition has no exposure in one group). The ratio of the two weighted rates is the Mantel-Haenszel rate
    ratio of the piece. Columns: rate0, rate1 (events per s), ratio, n_comp (competitions with weight > 0),
    E0, E1, X0, X1 (sums over those competitions), E0_all, E1_all, X0_all, X1_all (over all)."""
    rows = {}
    for piece, c in cells.groupby(level="piece"):
        T = c["X0"] + c["X1"]
        w = (c["X0"] * c["X1"] / T).where((c["X0"] > 0) & (c["X1"] > 0), 0.0)
        use = w > 0
        if w.sum() > 0:
            r1 = float((w[use] * c.loc[use, "E1"] / c.loc[use, "X1"]).sum() / w.sum())
            r0 = float((w[use] * c.loc[use, "E0"] / c.loc[use, "X0"]).sum() / w.sum())
        else:
            r1 = r0 = np.nan
        rows[piece] = {"rate0": r0, "rate1": r1, "ratio": r1 / r0 if r0 > 0 else np.nan,
                       "n_comp": int(use.sum()),
                       **{k: float(c.loc[use, k].sum()) for k in ("E0", "E1", "X0", "X1")},
                       **{f"{k}_all": float(c[k].sum()) for k in ("E0", "E1", "X0", "X1")}}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("piece")


def competition_rate_ratios(cells: pd.DataFrame) -> pd.DataFrame:
    """Per competition, the Mantel-Haenszel rate ratio over the pieces (chirped vs not, the baseline pieces as
    strata): sum_k E1 X0 / T over sum_k E0 X1 / T. Columns: ratio (NaN without events in one group), num,
    den, E0, E1, X0, X1."""
    T = cells["X0"] + cells["X1"]
    num = (cells["E1"] * cells["X0"] / T).where(T > 0, 0.0)
    den = (cells["E0"] * cells["X1"] / T).where(T > 0, 0.0)
    g = pd.DataFrame({"num": num, "den": den, **{k: cells[k] for k in ("E0", "E1", "X0", "X1")}})
    out = g.groupby(level=COMP).sum()
    out["ratio"] = (out["num"] / out["den"]).where((out["num"] > 0) & (out["den"] > 0))
    return out
