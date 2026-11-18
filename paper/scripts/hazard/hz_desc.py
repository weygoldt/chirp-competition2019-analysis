"""Descriptive event alignments with within-trial, within-state chirp nulls.

Units: seconds and Hz. Three families of alignments, each for subordinate and
(as a comparison) dominant chirps:

``chirp_to_event``
    Chase offsets aligned to chirps emitted inside chases, and chase onsets
    aligned to chirps emitted outside chases (event rate in Hz vs lag).
``event_to_chirp``
    Chirps (both states) aligned to chase offsets and to chase onsets (chirp
    rate in Hz vs lag).
``post_chirp_hazard``
    Hazard (1/s) that the *current* chase ends (after an in-chase chirp), or
    that the *current* inter-chase interval ends with a new onset (after an
    out-of-chase chirp), by time since the chirp. Right-censored at the end of
    the analysis window and at unknown-state intervals.

Rates are exposure-normalised: counts divided by the valid (analysed,
known-state) time covered by each lag bin, so window edges and excluded
intervals do not bias the curves.

Nulls (chase times are never moved; chirps stay in their chase state):

``shift`` (primary, as specified)
    Per trial and chase state, the in-chase time (and separately the
    out-of-chase time) is concatenated into one state clock; the chirp train of
    that state is circularly shifted by a uniform random amount on that clock
    and mapped back. Keeps chirp counts per state and inter-chirp intervals
    within a state (except at the wrap point and where the train crosses a
    state boundary); destroys all alignment of chirps to chase boundaries.
``matched`` (secondary, elapsed-time-matched)
    Each chirp in a chase (or in an inter-chase gap that starts at an offset)
    moves to a random interval of the same state and trial that lasted longer
    than the chirp's time since the interval start, keeping that elapsed time.
    Keeps each chirp's timing relative to the preceding onset/offset and
    destroys only its relation to how the interval ends: the descriptive
    analogue of conditioning on elapsed chase / inter-chase time in the hazard
    models. Needed because sub chirps cluster in the first seconds after an
    offset and accumulate late in long chases; when the chirp rate depends on
    elapsed state time and the chase/gap hazard is not constant, the ``shift``
    null is biased. Simulation without any chirp effect (10 trials of 3 h,
    chirp rate rising within chases and decaying after offsets, non-exponential
    chase durations, 60 datasets): ``shift`` gave post-chirp offset/onset
    ratios of 1.14/1.35 and 92%/98% false positives at alpha 0.05; ``matched``
    gave 0.99/1.00 and 10%/2% (trial-bootstrap CI excluding 1: 10%/7%). With a
    true 3x offset effect, ``matched`` detected it in 20/20 datasets and did
    not leak it into the onset test. Chirps move independently (a grouped move
    is biased, see :func:`swap_matched`), so the null ignores within-interval
    clustering and its p is somewhat anti-conservative. Chirps in intervals
    that do not start at a chase event stay fixed.
``matched_local``
    As ``matched`` but targets must start within +-``desc_local_window``
    (default 600 s) of the chirp's own interval, so slow changes of chirp and
    chase rates over the night (the time-in-trial control of the hazard
    models) cannot create an association.

Tests use pre-specified windows of length ``cfg.hist_max`` (2 s): after the
chirp ([0, 2) s) for chirp-triggered alignments; before ([-2, 0) s, primary)
and after ([0, 2) s) the chase event for event-triggered ones. Statistics
against each null: pooled mean rate in the window (two- and one-sided
permutation p) and max-|z| over the window bins (global envelope). The
observed / null-mean ratio gets a trial-bootstrap CI (dyads resampled).

Note: ``chirp_to_event`` [0, 2) s and ``event_to_chirp`` [-2, 0) s count
almost the same chirp-event pairs (one cross-correlogram read from either
side); they are not independent evidence. All of this is descriptive: an
excess of offsets after chirps is equally consistent with chirps causing
offsets and with chirps anticipating offsets or sharing a hidden trigger.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402

SENDERS = ("sub", "dom")
EVENTS = ("offset", "onset")
FAMILIES = ("chirp_to_event", "event_to_chirp", "post_chirp_hazard")
NULLS = ("shift", "matched", "matched_local")
STATES = ("in", "out")
# chirp state that is a valid trigger for each event type (chirp_to_event, hazard)
TRIGGER_STATE = {"offset": "in", "onset": "out"}
SENDER_NAME = {"sub": "subordinate", "dom": "dominant"}
WINDOW_SIGN = {"post": (0.0, 1.0), "pre": (-1.0, 0.0)}
NULL_TEXT = {
    "shift": "within-state circular-shift null",
    "matched": "elapsed-time-matched swap null",
    "matched_local": "elapsed-time-matched swap null, targets within +-{w:g} s",
}
NULL_NOTE = {
    "shift": "biased when chirp rate depends on elapsed time in the chase state and the "
             "chase/gap hazard is not constant (simulation: >90% false positives); see matched",
    "matched": "per-chirp swap ignores within-interval chirp clustering: permutation p "
               "somewhat anti-conservative (~10% at nominal 5% in simulation); prefer the "
               "trial-bootstrap CI",
    "matched_local": "as matched, plus targets restricted in time; chirps without other "
                     "eligible intervals nearby stay in place (pulls the ratio towards 1)",
}

# categorical slots 1/2 of the dataviz reference palette + neutral ink
COLORS = {"sub": "#2a78d6", "dom": "#eb6834"}
NULL_FILL = "#dcdbd5"
NULL_LINE = "#6f6e69"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WINDOW_FILL = "#f3f2ee"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DescParams:
    """Descriptive-analysis parameters (overridable via ``cfg.extra['desc_*']``)."""

    window: float = 5.0  # s, PETH half-width
    bin: float = 0.2  # s, PETH / hazard bin width
    haz_max: float = 5.0  # s, extent of the post-chirp hazard
    test_len: float = 2.0  # s, pre-specified test window length
    n_jitter: int = 200
    n_boot: int = 2000  # trial-bootstrap replicates
    level: float = 0.95
    seed: int = 42
    nulls: tuple[str, ...] = NULLS
    local_window: float = 600.0  # s, matched_local: max |start difference| of swap targets

    @classmethod
    def from_config(cls, cfg: Config) -> DescParams:
        ex = cfg.extra or {}
        n_jit, n_boot = int(cfg.n_jitter), int(ex.get("desc_n_boot", 2000))
        if cfg.quick:
            n_jit, n_boot = min(n_jit, 50), min(n_boot, 500)
        nulls = tuple(ex.get("desc_nulls", NULLS))
        bad = set(nulls) - set(NULLS)
        if bad:
            raise ValueError(f"unknown desc_nulls {sorted(bad)}; choose from {NULLS}")
        return cls(
            window=float(ex.get("desc_window", 5.0)),
            bin=float(ex.get("desc_bin", 0.2)),
            haz_max=float(ex.get("desc_haz_max", 5.0)),
            test_len=float(ex.get("desc_test_len", cfg.hist_max)),
            n_jitter=n_jit,
            n_boot=n_boot,
            seed=int(cfg.seed),
            nulls=nulls,
            local_window=float(ex.get("desc_local_window", 600.0)),
        )

    def null_text(self, null: str) -> str:
        return NULL_TEXT[null].format(w=self.local_window)

    def peth_edges(self) -> np.ndarray:
        k = int(round(self.window / self.bin))
        return np.round(np.arange(-k, k + 1) * self.bin, 10)

    def haz_edges(self) -> np.ndarray:
        k = int(round(self.haz_max / self.bin))
        return np.round(np.arange(0, k + 1) * self.bin, 10)


@dataclass(frozen=True)
class Spec:
    """One alignment: family x event type x chirp sender."""

    family: str
    event: str
    sender: str

    @property
    def key(self) -> str:
        return f"{self.family}:{self.event}:{self.sender}"

    @property
    def windows(self) -> tuple[str, ...]:
        """Test windows; the first is primary (plotted envelope, forest)."""
        return ("pre", "post") if self.family == "event_to_chirp" else ("post",)

    def window_bounds(self, name: str, length: float) -> tuple[float, float]:
        a, b = WINDOW_SIGN[name]
        return a * length + 0.0, b * length + 0.0

    def label(self) -> str:
        s = SENDER_NAME[self.sender]
        if self.family == "chirp_to_event":
            where = "inside" if self.event == "offset" else "outside"
            return f"Chase {self.event}s aligned to {s} chirps {where} chases"
        if self.family == "event_to_chirp":
            return f"{s.capitalize()} chirps aligned to chase {self.event}s"
        if self.event == "offset":
            return f"Hazard that the current chase ends after an in-chase {s} chirp"
        return f"Hazard that the next chase starts after an out-of-chase {s} chirp"

    def short(self) -> str:
        s = self.sender
        if self.family == "chirp_to_event":
            where = "in-chase" if self.event == "offset" else "out-of-chase"
            return f"Chase {self.event}s after {where} {s} chirps"
        if self.family == "event_to_chirp":
            return f"{s.capitalize()} chirps around chase {self.event}s"
        where = "in-chase" if self.event == "offset" else "out-of-chase"
        return f"{self.event.capitalize()} hazard after {where} {s} chirp"

    def ylabel(self) -> str:
        if self.family == "chirp_to_event":
            return f"chase {self.event} rate (Hz)"
        if self.family == "event_to_chirp":
            return f"{self.sender} chirp rate (Hz)"
        return f"{self.event} hazard (1/s)"

    def xlabel(self) -> str:
        if self.family == "event_to_chirp":
            return f"time relative to chase {self.event} (s)"
        return f"time since {self.sender} chirp (s)"


ALL_SPECS: tuple[Spec, ...] = tuple(
    Spec(f, e, s) for f in FAMILIES for s in SENDERS for e in EVENTS
)


# ---------------------------------------------------------------------------
# Interval algebra and chase states
# ---------------------------------------------------------------------------
def _merge(iv: np.ndarray) -> np.ndarray:
    """Sort and merge overlapping half-open intervals."""
    iv = np.asarray(iv, float).reshape(-1, 2)
    iv = iv[iv[:, 1] > iv[:, 0]]
    if len(iv) == 0:
        return iv
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, float)


def _complement(iv: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """[lo, hi) minus the union of iv."""
    out, cur = [], lo
    for a, b in _merge(iv):
        if b <= cur:
            continue
        if a >= hi:
            break
        if a > cur:
            out.append((cur, min(a, hi)))
        cur = max(cur, b)
        if cur >= hi:
            break
    if cur < hi:
        out.append((cur, hi))
    return np.array(out, float).reshape(-1, 2)


def _intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of two interval sets (two-pointer sweep)."""
    a, b = _merge(a), _merge(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            out.append((lo, hi))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.array(out, float).reshape(-1, 2)


def _in_iv(t: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """Membership in sorted, disjoint half-open intervals."""
    t = np.asarray(t, float)
    if len(iv) == 0:
        return np.zeros(t.shape, bool)
    j = np.searchsorted(iv[:, 0], t, side="right") - 1
    ok = j >= 0
    out = np.zeros(t.shape, bool)
    out[ok] = t[ok] < iv[j[ok], 1]
    return out


@dataclass
class States:
    """Chase-state partition of a trial's valid (analysed, known-state) time."""

    recording: str
    valid: np.ndarray  # (n, 2)
    iv: dict[str, np.ndarray]  # 'in' / 'out' -> (n, 2), sorted, disjoint
    ends_event: dict[str, np.ndarray]  # interval ends with offset ('in') / onset ('out')
    starts_event: dict[str, np.ndarray]  # interval starts at onset ('in') / offset ('out')
    events: dict[str, np.ndarray]  # 'offset' / 'onset' -> observed event times
    _vx: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    _vy: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        v = self.valid
        self._vx = v.reshape(-1)
        cum = np.concatenate([[0.0], np.cumsum(v[:, 1] - v[:, 0])])
        self._vy = np.repeat(cum, 2)[1:-1] if len(v) else np.zeros(0)

    def vcum(self, t: np.ndarray) -> np.ndarray:
        """Cumulative valid time up to t (piecewise linear)."""
        if len(self._vx) == 0:
            return np.zeros(np.shape(t))
        return np.interp(t, self._vx, self._vy)

    def state_time(self, state: str) -> float:
        iv = self.iv[state]
        return float(np.sum(iv[:, 1] - iv[:, 0]))


def trial_states(tr: Trial) -> States:
    """Split [obs_start, obs_end) minus exclude_intervals into in/out-of-chase time."""
    valid = _complement(tr.exclude_intervals, tr.obs_start, tr.obs_end)
    ch = tr.chases.reshape(-1, 2)
    iv_in = _intersect(valid, ch)
    iv_out = _intersect(valid, _complement(ch, tr.obs_start, tr.obs_end))
    return States(
        recording=tr.recording, valid=valid, iv={"in": iv_in, "out": iv_out},
        ends_event={"in": np.isin(iv_in[:, 1], ch[:, 1]), "out": np.isin(iv_out[:, 1], ch[:, 0])},
        starts_event={"in": np.isin(iv_in[:, 0], ch[:, 0]), "out": np.isin(iv_out[:, 0], ch[:, 1])},
        events={
            "offset": np.sort(iv_in[np.isin(iv_in[:, 1], ch[:, 1]), 1]),
            "onset": np.sort(iv_in[np.isin(iv_in[:, 0], ch[:, 0]), 0]),
        },
    )


# ---------------------------------------------------------------------------
# Null chirp trains
# ---------------------------------------------------------------------------
def jitter_within_state(
    t: np.ndarray, iv: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Circularly shift a chirp train on the concatenated state clock.

    ``t`` must lie inside the intervals ``iv``. Returns (n + 1, len(t)); row 0
    is the observed train, rows 1..n are shifted trains (each row sorted).
    """
    t = np.sort(np.asarray(t, float))
    out = np.empty((n + 1, len(t)))
    out[0] = t
    if len(t) == 0 or n == 0:
        return out
    lens = iv[:, 1] - iv[:, 0]
    starts = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
    total = float(lens.sum())
    i = np.searchsorted(iv[:, 0], t, side="right") - 1
    u = starts[i] + (t - iv[i, 0])
    shift = rng.uniform(0.0, total, size=(n, 1))
    u2 = np.mod(u[None, :] + shift, total)
    j = np.clip(np.searchsorted(starts, u2, side="right") - 1, 0, len(iv) - 1)
    t2 = np.minimum(iv[j, 0] + (u2 - starts[j]), iv[j, 1] - 1e-9)  # float guard
    out[1:] = np.sort(t2, axis=1)
    return out


def swap_matched(
    t: np.ndarray, iv: np.ndarray, movable: np.ndarray, n: int, rng: np.random.Generator,
    grouped: bool = False, local: float | None = None,
) -> np.ndarray:
    """Elapsed-time-matched swap (see module docstring). Returns (n + 1, len(t)).

    Each chirp in a movable interval (one that starts at a chase event) moves to
    a uniformly chosen movable interval of the same state that lasted longer
    than the chirp's elapsed time since the interval start, keeping that
    elapsed time; chirps in non-movable intervals stay. Chirps move
    independently: under H0 (interval ends do not depend on chirps, chirp rate
    depends on elapsed time) the interval containing a chirp at elapsed e is a
    draw from the risk set at e, so the per-chirp null mean is unbiased.

    ``local`` (s) restricts targets to intervals starting within +-local of
    the chirp's own interval start (controls slow, within-night changes in
    chirp and chase rates); if only the own interval qualifies the chirp stays.

    ``grouped=True`` moves all chirps of an interval together to an interval
    longer than the last of them. It keeps bursts intact but is BIASED: the
    observed interval is also selected on the absence of chirps after the last
    one, which shortens its residual, so it inflates post-chirp offset/onset
    rates under H0 (x1.6-1.7 in simulation). Diagnostics only; not used.
    """
    t = np.sort(np.asarray(t, float))
    out = np.empty((n + 1, len(t)))
    out[0] = t
    if len(t) == 0 or n == 0:
        return out
    out[1:] = t
    i = np.searchsorted(iv[:, 0], t, side="right") - 1
    e = t - iv[i, 0]
    lens = iv[:, 1] - iv[:, 0]
    mov_ids = np.where(movable)[0]
    order = np.argsort(lens[mov_ids], kind="stable")
    mov_sorted, len_sorted = mov_ids[order], lens[mov_ids][order]
    if grouped:
        for src in np.unique(i):
            if not movable[src]:
                continue
            sel = np.where(i == src)[0]
            k0 = np.searchsorted(len_sorted, e[sel].max(), side="right")
            tgt = mov_sorted[rng.integers(k0, len(mov_sorted), size=n)]  # includes src
            out[1:, sel] = np.minimum(iv[tgt, 0][:, None] + e[sel][None, :],
                                      iv[tgt, 1][:, None] - 1e-9)
    elif local is not None:
        starts = iv[:, 0]
        for c in np.where(movable[i])[0]:
            a = iv[i[c], 0]
            lo, hi = np.searchsorted(starts, [a - local, a + local], side="left")
            hi = max(hi, i[c] + 1)
            cand = np.arange(lo, hi)
            cand = cand[movable[cand] & (lens[cand] > e[c])]
            tgt = cand[rng.integers(0, len(cand), size=n)]  # own interval always qualifies
            out[1:, c] = np.minimum(iv[tgt, 0] + e[c], iv[tgt, 1] - 1e-9)
    else:
        sel = np.where(movable[i])[0]
        if len(sel):
            k0 = np.searchsorted(len_sorted, e[sel], side="right")  # own interval qualifies
            u = rng.uniform(size=(n, len(sel)))
            k = np.minimum(k0 + np.floor(u * (len(mov_sorted) - k0)).astype(int),
                           len(mov_sorted) - 1)
            tgt = mov_sorted[k]
            out[1:, sel] = np.minimum(iv[tgt, 0] + e[sel], iv[tgt, 1] - 1e-9)
    out[1:] = np.sort(out[1:], axis=1)
    return out


def _rng(seed: int, recording: str, sender: str, state: str, null: str) -> np.random.Generator:
    """Order-independent deterministic generator per trial/sender/state/null."""
    return np.random.default_rng([
        seed, zlib.crc32(recording.encode()), SENDERS.index(sender), STATES.index(state),
        NULLS.index(null),
    ])


def null_replicates(
    trials: list[Trial], states: list[States], p: DescParams, null: str
) -> dict[tuple[str, str, str], np.ndarray]:
    """Observed + null chirp trains per (recording, sender, state); row 0 = observed."""
    reps: dict[tuple[str, str, str], np.ndarray] = {}
    for tr, st in zip(trials, states):
        for who in SENDERS:
            x = tr.chirps_sub if who == "sub" else tr.chirps_dom
            for state in STATES:
                iv = st.iv[state]
                xs = x[_in_iv(x, iv)]
                rng = _rng(p.seed, tr.recording, who, state, null)
                if null == "shift":
                    r = jitter_within_state(xs, iv, p.n_jitter, rng)
                else:
                    loc = p.local_window if null == "matched_local" else None
                    r = swap_matched(xs, iv, st.starts_event[state], p.n_jitter, rng, local=loc)
                reps[(tr.recording, who, state)] = r
    return reps


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def _chunks(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _chunk_rows(m: int, k: int, budget: int = 4_000_000) -> int:
    return max(1, budget // max(1, m * k))


def _lag_counts(src: np.ndarray, dst: np.ndarray, edges: np.ndarray, side: str = "left") -> np.ndarray:
    """Counts of dst - src in lag bins, summed over src. src (R, m) -> (R, K)."""
    R, m = src.shape
    K = len(edges) - 1
    out = np.zeros((R, K))
    if m == 0 or len(dst) == 0:
        return out
    for sl in _chunks(R, _chunk_rows(m, K + 1)):
        pos = np.searchsorted(dst, src[sl, :, None] + edges, side=side)
        out[sl] = np.diff(pos, axis=-1).sum(axis=1)
    return out


def _lag_exposure(src: np.ndarray, st: States, edges: np.ndarray) -> np.ndarray:
    """Valid time covered by each lag bin, summed over triggers. (R, m) -> (R, K)."""
    R, m = src.shape
    K = len(edges) - 1
    out = np.zeros((R, K))
    if m == 0:
        return out
    for sl in _chunks(R, _chunk_rows(m, K + 1)):
        v = st.vcum(src[sl, :, None] + edges)
        out[sl] = np.diff(v, axis=-1).sum(axis=1)
    return out


def _residual_hazard(
    src: np.ndarray, iv: np.ndarray, ends_event: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Events and at-risk time of the current state interval ending, by time since src.

    src (R, m) chirp times inside ``iv``. Returns (counts, exposure), each (R, K).
    """
    R, m = src.shape
    K = len(edges) - 1
    counts, expo = np.zeros((R, K)), np.zeros((R, K))
    if m == 0 or len(iv) == 0:
        return counts, expo
    lo, w = edges[:-1], np.diff(edges)
    for sl in _chunks(R, _chunk_rows(m, K)):
        s = src[sl]
        i = np.clip(np.searchsorted(iv[:, 0], s, side="right") - 1, 0, len(iv) - 1)
        d = iv[i, 1] - s  # residual time to the interval end
        expo[sl] = np.clip(d[..., None] - lo, 0.0, w).sum(axis=1)
        k = np.searchsorted(edges, d, side="right") - 1
        k = np.where(ends_event[i] & (k >= 0) & (k < K), k, K)
        rows = np.repeat(np.arange(s.shape[0]), m)
        c = np.bincount(rows * (K + 1) + k.ravel(), minlength=s.shape[0] * (K + 1))
        counts[sl] = c.reshape(s.shape[0], K + 1)[:, :K]
    return counts, expo


# ---------------------------------------------------------------------------
# Per-alignment computation
# ---------------------------------------------------------------------------
@dataclass
class AlignResult:
    """Counts and exposures for one alignment; axis 1 = replicate (0 = observed)."""

    spec: Spec
    null: str
    edges: np.ndarray
    recordings: list[str]
    counts: np.ndarray  # (T, R, K)
    exposure: np.ndarray  # (T, R, K)
    n_trig: np.ndarray  # (T,) observed triggers per trial
    n_target: np.ndarray  # (T,) observed target events per trial

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def active(self) -> np.ndarray:
        """Trials that contribute (observed exposure > 0)."""
        return self.exposure[:, 0, :].sum(axis=1) > 0


def compute_alignment(
    spec: Spec,
    trials: list[Trial],
    states: list[States],
    reps: dict[tuple[str, str, str], np.ndarray],
    p: DescParams,
    null: str,
) -> AlignResult:
    """Counts/exposure of one alignment for every trial and replicate."""
    R = p.n_jitter + 1
    edges = p.haz_edges() if spec.family == "post_chirp_hazard" else p.peth_edges()
    K = len(edges) - 1
    T = len(trials)
    counts, expo = np.zeros((T, R, K)), np.zeros((T, R, K))
    n_trig, n_target = np.zeros(T, int), np.zeros(T, int)
    state = TRIGGER_STATE[spec.event]
    for ti, (tr, st) in enumerate(zip(trials, states)):
        ev = st.events[spec.event]
        if spec.family == "chirp_to_event":
            src = reps[(tr.recording, spec.sender, state)]
            counts[ti] = _lag_counts(src, ev, edges)
            expo[ti] = _lag_exposure(src, st, edges)
            n_trig[ti], n_target[ti] = src.shape[1], len(ev)
        elif spec.family == "event_to_chirp":
            src = np.concatenate(
                [reps[(tr.recording, spec.sender, s)] for s in STATES], axis=1
            )
            # chirp - event in [e_k, e_k+1)  <=>  event - chirp in (-e_k+1, -e_k]
            counts[ti] = _lag_counts(src, ev, -edges[::-1], side="right")[:, ::-1]
            expo[ti] = np.broadcast_to(_lag_exposure(ev[None, :], st, edges), (R, K))
            n_trig[ti], n_target[ti] = len(ev), src.shape[1]
        else:
            src = reps[(tr.recording, spec.sender, state)]
            c, e = _residual_hazard(src, st.iv[state], st.ends_event[state], edges)
            counts[ti], expo[ti] = c, e
            n_trig[ti], n_target[ti] = src.shape[1], int(c[0].sum())
    return AlignResult(spec, null, edges, [t.recording for t in trials], counts, expo,
                       n_trig, n_target)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _perm_p(obs: float, null: np.ndarray) -> tuple[float, float, float]:
    """(two-sided, greater, less) permutation p with +1 correction."""
    null = null[np.isfinite(null)]
    n = len(null)
    if n == 0 or not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    mu = null.mean()
    p2 = (1 + np.sum(np.abs(null - mu) >= abs(obs - mu) - 1e-12)) / (n + 1)
    pg = (1 + np.sum(null >= obs - 1e-12)) / (n + 1)
    pl = (1 + np.sum(null <= obs + 1e-12)) / (n + 1)
    return float(p2), float(pg), float(pl)


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence all-NaN / empty-slice RuntimeWarnings (handled explicitly)."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _pct(x: np.ndarray, q: Any, axis: int | None = None) -> Any:
    """nanpercentile that returns NaN instead of warning on all-NaN input."""
    x = np.asarray(x, float)
    if not np.isfinite(x).any():
        return np.full(np.shape(q), np.nan) if axis is None else np.full(
            (len(np.atleast_1d(q)),) + tuple(np.delete(x.shape, axis)), np.nan)
    with _quiet():
        return np.nanpercentile(x, q, axis=axis)


def _boot_weights(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """(B, n) multinomial resampling counts of n units."""
    idx = rng.integers(0, n, size=(B, n))
    w = np.zeros((B, n))
    np.add.at(w, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return w


def _ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def summarize_alignment(res: AlignResult, p: DescParams) -> dict[str, pd.DataFrame]:
    """Pooled and per-trial curves, null bands and window tests of one alignment."""
    spec, lv = res.spec, p.level
    qlo, qhi = 100 * (1 - lv) / 2, 100 * (1 + lv) / 2
    act = res.active
    C, E = res.counts[act], res.exposure[act]
    recs = [r for r, a in zip(res.recordings, act) if a]
    Ta, K, x = len(recs), len(res.centers), res.centers
    base = dict(analysis=spec.key, null=res.null, family=spec.family, event=spec.event,
                sender=spec.sender)
    rng = np.random.default_rng([p.seed, zlib.crc32(f"{spec.key}|{res.null}".encode())])
    W = _boot_weights(Ta, p.n_boot, rng) if Ta >= 2 else None

    pooled = _div(C.sum(0), E.sum(0)) if Ta else np.full((p.n_jitter + 1, K), np.nan)
    obs, null = pooled[0], pooled[1:]
    has_null = null.shape[0] > 0 and np.isfinite(null).any()
    with _quiet():
        n_mean = np.nanmean(null, 0) if has_null else np.full(K, np.nan)
        n_sd = np.nanstd(null, 0, ddof=1) if has_null and null.shape[0] > 1 else np.full(K, np.nan)
    n_lo, n_hi = _pct(null, [qlo, qhi], axis=0) if has_null else (np.full(K, np.nan),) * 2

    if W is not None:
        ci_lo, ci_hi = _pct(_div(W @ C[:, 0, :], W @ E[:, 0, :]), [qlo, qhi], axis=0)
    else:
        ci_lo = ci_hi = np.full(K, np.nan)
    rt = _div(C[:, 0, :], E[:, 0, :]) if Ta else np.full((1, K), np.nan)
    with _quiet():
        trial_mean = np.nanmean(rt, axis=0)

    env_lo, env_hi = np.full(K, np.nan), np.full(K, np.nan)
    win_col = np.full(K, "", dtype=object)
    tests, tests_pt = [], []
    for wi, wname in enumerate(spec.windows):
        if res.null.startswith("matched") and spec.family == "event_to_chirp" and wname == "post":
            continue  # degenerate: matched null keeps chirp timing after the preceding event
        lo_w, hi_w = spec.window_bounds(wname, p.test_len)
        wmask = (x > lo_w) & (x < hi_w)
        win_col[wmask] = wname
        wbase = dict(**base, window=wname, primary=wi == 0, window_lo=lo_w, window_hi=hi_w)

        # global envelope: max |z| over the window bins
        zmask = wmask & (n_sd > 0) & np.isfinite(obs)
        p_glob = q_env = z_obs = np.nan
        if has_null and zmask.any():
            z = np.abs(pooled[:, zmask] - n_mean[zmask]) / n_sd[zmask]
            with _quiet():
                zmax = np.nanmax(z, axis=1)
            z_obs = float(zmax[0])
            p_glob = float((1 + np.sum(zmax[1:] >= z_obs - 1e-12)) / len(zmax))
            q_env = float(_pct(zmax[1:], 100 * lv))
            env_lo[wmask] = n_mean[wmask] - q_env * n_sd[wmask]
            env_hi[wmask] = n_mean[wmask] + q_env * n_sd[wmask]

        cw, ew = C[:, :, wmask].sum(-1), E[:, :, wmask].sum(-1)  # (Ta, R)
        S = _div(cw.sum(0), ew.sum(0)) if Ta else np.full(p.n_jitter + 1, np.nan)
        s_obs = float(S[0])
        s_mu = float(np.nanmean(S[1:])) if has_null and np.isfinite(S[1:]).any() else np.nan
        p2, pg, pl = _perm_p(s_obs, S[1:])
        nr = S[1:] / s_mu if np.isfinite(s_mu) and s_mu > 0 else np.full(1, np.nan)
        r_lo = r_hi = np.nan
        if W is not None and np.isfinite(s_mu):
            bS = _div(W @ cw, W @ ew)
            with _quiet():
                b_ratio = bS[:, 0] / np.nanmean(bS[:, 1:], axis=1)
            r_lo, r_hi = _pct(b_ratio[np.isfinite(b_ratio)], [qlo, qhi])

        rows_t = []
        for i, rec in enumerate(recs):
            si = _div(cw[i], ew[i])
            mu = float(np.nanmean(si[1:])) if np.isfinite(si[1:]).any() else np.nan
            tp2, tpg, tpl = _perm_p(float(si[0]), si[1:])
            nri = si[1:] / mu if np.isfinite(mu) and mu > 0 else np.full(1, np.nan)
            j = res.recordings.index(rec)
            rows_t.append(dict(
                **wbase, recording=rec, n_triggers=int(res.n_trig[j]),
                n_targets=int(res.n_target[j]), counts_window=float(cw[i, 0]),
                rate_window_obs=float(si[0]), rate_window_null=mu, ratio=_ratio(si[0], mu),
                null_ratio_low=float(_pct(nri, qlo)), null_ratio_high=float(_pct(nri, qhi)),
                p_two_sided=tp2, p_greater=tpg, p_less=tpl,
            ))
        tests_pt += rows_t
        ratios = np.array([r["ratio"] for r in rows_t], float)
        fin = np.isfinite(ratios)
        tests.append(dict(
            **wbase, label=spec.label(), n_trials=Ta, n_triggers=int(res.n_trig[act].sum()),
            n_targets=int(res.n_target[act].sum()),
            counts_window=float(cw[:, 0].sum()) if Ta else 0.0,
            rate_window_obs=s_obs, rate_window_null=s_mu, ratio=_ratio(s_obs, s_mu),
            ratio_ci_low=r_lo, ratio_ci_high=r_hi,
            null_ratio_low=float(_pct(nr, qlo)), null_ratio_high=float(_pct(nr, qhi)),
            p_window_two_sided=p2, p_window_greater=pg, p_window_less=pl,
            maxz_obs=z_obs, envelope_q=q_env, p_global_maxz=p_glob,
            median_trial_ratio=float(np.median(ratios[fin])) if fin.any() else np.nan,
            n_trials_ratio_gt1=int(np.sum(ratios[fin] > 1)), n_trials_ratio_finite=int(fin.sum()),
            n_jitter=p.n_jitter, n_boot=p.n_boot if W is not None else 0,
            note=NULL_NOTE[res.null] + (
                f"; mirrors chirp_to_event:{spec.event}:{spec.sender} (same chirp-event pairs)"
                if spec.family == "event_to_chirp" and wname == "pre" else ""),
        ))

    peth = pd.DataFrame(dict(
        **base, lag=x, lag_lo=res.edges[:-1], lag_hi=res.edges[1:],
        rate_obs=obs, ci_low=ci_lo, ci_high=ci_hi, trial_mean_rate=trial_mean,
        null_mean=n_mean, null_sd=n_sd, null_low=n_lo, null_high=n_hi,
        env_low=env_lo, env_high=env_hi, test_window=win_col,
        counts_obs=C[:, 0, :].sum(0) if Ta else np.zeros(K),
        exposure_obs=E[:, 0, :].sum(0) if Ta else np.zeros(K),
        n_trials=Ta, n_triggers=int(res.n_trig[act].sum()),
    ))
    pt_rows = []
    for i, rec in enumerate(recs):
        r_all = _div(C[i], E[i])
        with _quiet():
            m_ = np.nanmean(r_all[1:], 0)
        l_, h_ = _pct(r_all[1:], [qlo, qhi], axis=0)
        pt_rows.append(pd.DataFrame(dict(
            **base, recording=rec, lag=x, rate_obs=r_all[0], null_mean=m_, null_low=l_,
            null_high=h_, counts_obs=C[i, 0], exposure_obs=E[i, 0],
        )))
    return dict(
        peth=peth,
        per_trial=pd.concat(pt_rows, ignore_index=True) if pt_rows else pd.DataFrame(),
        test=pd.DataFrame(tests), tests_per_trial=pd.DataFrame(tests_pt),
    )


def state_rates(trials: list[Trial], states: list[States]) -> pd.DataFrame:
    """Per-trial chirp counts and rates inside vs outside chases."""
    rows = []
    for tr, st in zip(trials, states):
        t_in, t_out = st.state_time("in"), st.state_time("out")
        r: dict[str, Any] = dict(
            recording=tr.recording, time_in_chase=t_in, time_out_chase=t_out,
            n_offsets=len(st.events["offset"]), n_onsets=len(st.events["onset"]),
        )
        for who in SENDERS:
            x = tr.chirps_sub if who == "sub" else tr.chirps_dom
            n_in, n_out = int(_in_iv(x, st.iv["in"]).sum()), int(_in_iv(x, st.iv["out"]).sum())
            ri = n_in / t_in if t_in > 0 else np.nan
            ro = n_out / t_out if t_out > 0 else np.nan
            r.update({f"n_{who}_in": n_in, f"n_{who}_out": n_out, f"rate_{who}_in": ri,
                      f"rate_{who}_out": ro, f"rate_ratio_{who}_in_out": _ratio(ri, ro)})
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 150,
        "savefig.dpi": 200, "axes.titlecolor": INK, "text.color": INK,
    })


def _save(fig: plt.Figure, out: Path, name: str, fmts: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.{x}" for x in fmts]
    for f in paths:
        fig.savefig(f, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def _nanmax(*arrs: Any) -> float:
    v = [np.nanmax(np.asarray(a, float)) for a in arrs if np.isfinite(np.asarray(a, float)).any()]
    return max(v) if v else 1.0


def _draw_panel(ax: plt.Axes, peth: pd.DataFrame, tests: pd.DataFrame, spec: Spec) -> None:
    """Observed pooled curve with trial-bootstrap CI over the null band."""
    x = peth["lag"].to_numpy()
    col = COLORS[spec.sender]
    for _, t in tests.iterrows():
        ax.axvspan(t["window_lo"], t["window_hi"], color=WINDOW_FILL, zorder=0, lw=0)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.fill_between(x, peth["null_low"], peth["null_high"], color=NULL_FILL, lw=0, zorder=2)
    ax.plot(x, peth["null_mean"], color=NULL_LINE, lw=1.0, ls=(0, (3, 2)), zorder=3)
    for wname in tests["window"]:
        m = (peth["test_window"] == wname).to_numpy()
        for c in ("env_low", "env_high"):
            ax.plot(x[m], peth[c].to_numpy()[m], color=NULL_LINE, lw=0.8, ls=":", zorder=3)
    ax.fill_between(x, peth["ci_low"], peth["ci_high"], color=col, alpha=0.18, lw=0, zorder=4)
    ax.plot(x, peth["rate_obs"], color=col, lw=1.8, zorder=5)
    dx = x[1] - x[0]
    ax.set_xlim(x[0] - dx / 2, x[-1] + dx / 2)
    core = _nanmax(peth["rate_obs"], peth["null_high"], peth["env_high"])
    top = min(_nanmax(peth["ci_high"], core), 1.6 * core)
    ax.set_ylim(0, top * (1.0 + 0.16 * len(tests) + 0.1))  # headroom for the stats text
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(spec.xlabel())
    ax.set_ylabel(spec.ylabel())
    t0 = tests.iloc[0]
    n_tr, n_trig, n_tg = int(t0["n_trials"]), int(t0["n_triggers"]), int(t0["n_targets"])
    if spec.family == "event_to_chirp":
        sub = f"n = {n_tr} trials, {n_trig:,} {spec.event}s, {n_tg:,} chirps"
    elif spec.family == "chirp_to_event":
        sub = f"n = {n_tr} trials, {n_trig:,} chirps, {n_tg:,} {spec.event}s"
    else:
        sub = f"n = {n_tr} trials, {n_trig:,} chirps, {n_tg:,} {spec.event}s in range"
    ax.set_title(f"{spec.short()}\n{sub}", loc="left", fontsize=8.3)
    lines = []
    for _, t in tests.iterrows():
        if np.isfinite(t["ratio"]):
            lines.append(
                f"[{t['window_lo']:g}, {t['window_hi']:g}) s: obs/null {t['ratio']:.2f} "
                f"[{t['ratio_ci_low']:.2f}, {t['ratio_ci_high']:.2f}], "
                f"p = {_fmt_p(t['p_window_two_sided'])} / {_fmt_p(t['p_global_maxz'])}")
        else:
            lines.append(f"[{t['window_lo']:g}, {t['window_hi']:g}) s: no data")
    ax.text(0.99, 0.99, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.6, color=INK2, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0))


def _legend_handles() -> list[Any]:
    return [
        Line2D([], [], color=COLORS["sub"], lw=1.8, label="observed, subordinate chirps"),
        Line2D([], [], color=COLORS["dom"], lw=1.8, label="observed, dominant chirps"),
        Patch(facecolor=MUTED, alpha=0.25, label="95% CI, trial bootstrap (colored)"),
        Line2D([], [], color=NULL_LINE, lw=1.0, ls=(0, (3, 2)), label="null mean"),
        Patch(facecolor=NULL_FILL, label="null 2.5-97.5% (pointwise)"),
        Line2D([], [], color=NULL_LINE, lw=0.8, ls=":", label="95% global envelope"),
        Patch(facecolor=WINDOW_FILL, edgecolor=AXIS, lw=0.5, label="test window"),
    ]


FAMILY_TITLE = {
    "chirp_to_event": "Chase events aligned to chirps (in-chase chirps -> offsets, "
                      "out-of-chase chirps -> onsets)",
    "event_to_chirp": "Chirps aligned to chase events (all chirps in the analysis window)",
    "post_chirp_hazard": "Hazard that the current chase state ends, by time since the chirp",
}


def plot_family(
    family: str, null: str, summ: dict[tuple[str, str], dict[str, pd.DataFrame]],
    p: DescParams, out: Path,
) -> list[Path]:
    """2 x 2 grid (rows: sub, dom; columns: offset, onset) of pooled curves."""
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), constrained_layout=True)
    for i, who in enumerate(SENDERS):
        for j, ev in enumerate(EVENTS):
            spec = Spec(family, ev, who)
            s = summ[(null, spec.key)]
            _draw_panel(axes[i, j], s["peth"], s["test"], spec)
    fig.suptitle(
        f"{FAMILY_TITLE[family]}\nObserved vs {p.null_text(null)} ({p.n_jitter} replicates).\n"
        "sub = subordinate (loser), dom = dominant (winner); p = window-mean / max-|z| "
        "permutation p.",
        fontsize=8.8, color=INK, x=0.01, ha="left")
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.075))
    return _save(fig, out, f"desc_{family}_{null}")


def plot_per_trial(spec: Spec, null: str, summ: dict[str, pd.DataFrame], p: DescParams,
                   out: Path) -> list[Path]:
    """Small multiples: one panel per trial, observed vs per-trial null band."""
    pt, tpt = summ["per_trial"], summ["tests_per_trial"]
    if not len(tpt):
        return []
    tpt = tpt[tpt["primary"]]
    recs = list(tpt["recording"])
    nc = 4
    nr = int(np.ceil(len(recs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(8.2, 1.8 * nr + 0.6), squeeze=False,
                             constrained_layout=True)
    col = COLORS[spec.sender]
    lo_w, hi_w = spec.window_bounds(spec.windows[0], p.test_len)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(recs):
            ax.set_visible(False)
            continue
        rec = recs[k]
        d = pt[pt["recording"] == rec]
        t = tpt[tpt["recording"] == rec].iloc[0]
        x = d["lag"].to_numpy()
        ax.axvspan(lo_w, hi_w, color=WINDOW_FILL, zorder=0, lw=0)
        ax.axvline(0, color=AXIS, lw=0.7)
        ax.fill_between(x, d["null_low"], d["null_high"], color=NULL_FILL, lw=0)
        ax.plot(x, d["null_mean"], color=NULL_LINE, lw=0.8, ls=(0, (3, 2)))
        ax.plot(x, d["rate_obs"], color=col, lw=1.2)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"{rec[5:10]}  n={int(t['n_triggers'])}, obs/null={t['ratio']:.2f}, "
                     f"p={_fmt_p(t['p_two_sided'])}", fontsize=6.8, loc="left")
        if k % nc == 0:
            ax.set_ylabel(spec.ylabel(), fontsize=7)
        if k >= len(recs) - nc:
            ax.set_xlabel(spec.xlabel(), fontsize=7)
    trig = "chase events" if spec.family == "event_to_chirp" else "chirps"
    fig.suptitle(f"{spec.label()}, per trial\nn = {trig}; ratio and p for [{lo_w:g}, {hi_w:g}) s "
                 f"(shaded); grey: {p.null_text(null)}, 2.5-97.5% of {p.n_jitter} replicates",
                 fontsize=8.3, x=0.01, ha="left")
    return _save(fig, out, f"desc_per_trial_{spec.family}_{spec.event}_{spec.sender}_{null}",
                 fmts=("png",))


def plot_forest(tests: pd.DataFrame, tests_pt: pd.DataFrame, null: str, p: DescParams,
                out: Path) -> list[Path]:
    """Per-trial observed / null-mean ratios in each test window."""
    tp = tests_pt[tests_pt["null"] == null]
    tt = tests[tests["null"] == null]
    have = set(zip(tt["family"], tt["event"], tt["window"]))
    panels = [(s.family, s.event, w) for s in ALL_SPECS if s.sender == "sub"
              for w in s.windows if (s.family, s.event, w) in have]
    recs = sorted(tp["recording"].unique()) if len(tp) else []
    ylab = [r[5:10] for r in recs] + ["pooled"]
    xlo, xhi = 1 / 8, 8.0
    fig, axes = plt.subplots(1, len(panels), figsize=(1.55 * len(panels) + 0.6,
                                                      0.24 * len(ylab) + 1.7),
                             sharey=True, constrained_layout=True)
    for ax, (fam, ev, win) in zip(np.atleast_1d(axes), panels):
        ax.axvline(1.0, color=AXIS, lw=0.8)
        for who in SENDERS:
            off = -0.18 if who == "sub" else 0.18
            key = Spec(fam, ev, who).key
            d = tp[(tp["analysis"] == key) & (tp["window"] == win)].set_index("recording")
            for yi, rec in enumerate(recs):
                if rec not in d.index or not np.isfinite(d.loc[rec, "ratio"]):
                    continue
                r = d.loc[rec]
                ax.plot(np.clip([r["null_ratio_low"], r["null_ratio_high"]], xlo, xhi),
                        [yi + off] * 2, color=NULL_FILL, lw=2.2, solid_capstyle="butt")
                v = float(r["ratio"])
                clipped = v < xlo or v > xhi
                ax.plot(np.clip(v, xlo, xhi), yi + off, "o", ms=3.6, zorder=3,
                        mfc="white" if clipped else COLORS[who], mec=COLORS[who], mew=0.9)
            t = tt[(tt["analysis"] == key) & (tt["window"] == win)]
            if len(t) and np.isfinite(t.iloc[0]["ratio"]):
                t = t.iloc[0]
                yi = len(recs)
                ax.plot(np.clip([t["null_ratio_low"], t["null_ratio_high"]], xlo, xhi),
                        [yi + off] * 2, color=NULL_FILL, lw=2.2, solid_capstyle="butt")
                ax.plot(np.clip([t["ratio_ci_low"], t["ratio_ci_high"]], xlo, xhi),
                        [yi + off] * 2, color=COLORS[who], lw=1.2)
                ax.plot(np.clip(t["ratio"], xlo, xhi), yi + off, "D", ms=4.2,
                        color=COLORS[who], mec="white", mew=0.4, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(xlo * 0.9, xhi * 1.1)
        ax.set_xticks([0.125, 0.5, 1, 2, 8])
        ax.set_xticklabels(["1/8", "1/2", "1", "2", "8"])
        ax.minorticks_off()
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        lo_w, hi_w = Spec(fam, ev, "sub").window_bounds(win, p.test_len)
        name = {"chirp_to_event": f"{ev}s after chirp", "event_to_chirp": f"chirps vs {ev}",
                "post_chirp_hazard": f"{ev} hazard after chirp"}[fam]
        ax.set_title(f"{name}\n[{lo_w:g}, {hi_w:g}) s", fontsize=7.6, loc="left")
        ax.set_xlabel("obs / null mean", fontsize=7.5)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(ylab)))
    ax0.set_yticklabels(ylab, fontsize=7)
    ax0.set_ylim(len(ylab) - 0.5, -0.5)
    handles = [
        Line2D([], [], marker="o", ls="", color=COLORS["sub"], label="subordinate chirps"),
        Line2D([], [], marker="o", ls="", color=COLORS["dom"], label="dominant chirps"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="clipped to axis"),
        Line2D([], [], color=NULL_FILL, lw=2.2, label="null 2.5-97.5%"),
        Line2D([], [], marker="D", ls="-", color=INK2, lw=1.2,
               label="pooled, 95% trial-bootstrap CI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Observed / null-mean rate in each test window, per trial\n"
                 f"{p.null_text(null)}, {p.n_jitter} replicates; trials without triggers blank",
                 fontsize=9, x=0.01, ha="left")
    return _save(fig, out, f"desc_forest_{null}")


def plot_null_comparison(tests: pd.DataFrame, p: DescParams, out: Path) -> list[Path]:
    """Pooled window ratio with trial-bootstrap CI under every null, per alignment."""
    t = tests[tests["primary"].astype(bool)]
    panels = [(f, e) for f in FAMILIES for e in EVENTS]
    nulls = [n for n in p.nulls if n in set(t["null"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(11.5, 0.5 * len(nulls) + 1.9),
                             sharey=True, constrained_layout=True)
    xlo, xhi = 1 / 4, 4.0
    for ax, (fam, ev) in zip(np.atleast_1d(axes), panels):
        ax.axvline(1.0, color=AXIS, lw=0.8)
        for yi, null in enumerate(nulls):
            for who in SENDERS:
                off = -0.16 if who == "sub" else 0.16
                r = t[(t["null"] == null) & (t["analysis"] == Spec(fam, ev, who).key)]
                if not len(r) or not np.isfinite(r.iloc[0]["ratio"]):
                    continue
                r = r.iloc[0]
                ax.plot(np.clip([r["null_ratio_low"], r["null_ratio_high"]], xlo, xhi),
                        [yi + off] * 2, color=NULL_FILL, lw=3.0, solid_capstyle="butt")
                ax.plot(np.clip([r["ratio_ci_low"], r["ratio_ci_high"]], xlo, xhi),
                        [yi + off] * 2, color=COLORS[who], lw=1.2)
                ax.plot(np.clip(r["ratio"], xlo, xhi), yi + off, "D", ms=4.2,
                        color=COLORS[who], mec="white", mew=0.4, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(xlo * 0.9, xhi * 1.1)
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
        ax.minorticks_off()
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        lo_w, hi_w = Spec(fam, ev, "sub").window_bounds(Spec(fam, ev, "sub").windows[0],
                                                       p.test_len)
        name = {"chirp_to_event": f"{ev}s after chirp", "event_to_chirp": f"chirps vs {ev}",
                "post_chirp_hazard": f"{ev} hazard after chirp"}[fam]
        ax.set_title(f"{name}\n[{lo_w:g}, {hi_w:g}) s", fontsize=7.6, loc="left")
        ax.set_xlabel("obs / null mean", fontsize=7.5)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(nulls)))
    ax0.set_yticklabels(nulls, fontsize=7.5)
    ax0.set_ylim(len(nulls) - 0.5, -0.5)
    handles = [
        Line2D([], [], marker="D", ls="-", color=COLORS["sub"], label="subordinate chirps"),
        Line2D([], [], marker="D", ls="-", color=COLORS["dom"], label="dominant chirps"),
        Line2D([], [], color=NULL_FILL, lw=3.0, label="null 2.5-97.5%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Pooled observed / null-mean ratio (95% trial-bootstrap CI) under each null; "
                 f"{p.n_jitter} replicates. shift: circular shift on the state clock; matched: "
                 "elapsed-time-matched swap;\nmatched_local: same, targets within "
                 f"+-{p.local_window:g} s. Values outside [1/4, 4] clipped.",
                 fontsize=8.5, x=0.01, ha="left")
    return _save(fig, out, "desc_null_comparison")


def plot_state_rates(sr: pd.DataFrame, out: Path) -> list[Path]:
    """Chirp rate inside vs outside chases, per trial (paired dots)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True, sharey=True)
    floor = 0.02
    for ax, who in zip(axes, SENDERS):
        for _, r in sr.iterrows():
            a, b = r[f"rate_{who}_out"] * 60, r[f"rate_{who}_in"] * 60
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            ax.plot([0, 1], [max(a, floor), max(b, floor)], color=COLORS[who], lw=0.9,
                    alpha=0.7, marker="o", ms=3)
        ax.axhline(floor, color=AXIS, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["outside chase", "inside chase"])
        ax.set_xlim(-0.3, 1.3)
        ax.grid(axis="y", color=GRID, lw=0.5)
        n_ok = int(np.sum((sr[f"n_{who}_in"] + sr[f"n_{who}_out"]) > 0))
        ax.set_title(f"{SENDER_NAME[who]} chirps ({n_ok} of {len(sr)} trials with chirps)",
                     fontsize=8.5, loc="left")
    axes[0].set_ylabel(f"chirp rate (1/min, log; 0 drawn at {floor})")
    fig.suptitle("Chirp rate by chase state, per trial (analysis window)", fontsize=9.5,
                 x=0.01, ha="left")
    return _save(fig, out, "desc_chirp_rate_by_state")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_descriptives(
    trials: list[Trial], cfg: Config, make_figures: bool = True, verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """Compute all alignments, nulls and tests; write CSVs and figures.

    Returns dict with keys peth_pooled, peth_per_trial, tests, tests_per_trial,
    state_rates, params (all with a ``null`` column where relevant). CSVs go to
    cfg.out_dir/'descriptives', figures to cfg.out_dir/'figures'.
    """
    p = DescParams.from_config(cfg)
    t0 = time.time()
    states = [trial_states(tr) for tr in trials]
    summ: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for null in p.nulls:
        reps = null_replicates(trials, states, p, null)
        for spec in ALL_SPECS:
            res = compute_alignment(spec, trials, states, reps, p, null)
            summ[(null, spec.key)] = summarize_alignment(res, p)
    if verbose:
        print(f"[hz_desc] {len(ALL_SPECS)} alignments x {len(p.nulls)} nulls x "
              f"{p.n_jitter} replicates on {len(trials)} trials: {time.time() - t0:.1f} s")

    def cat(name: str) -> pd.DataFrame:
        parts = [s[name] for s in summ.values() if len(s[name])]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    pr = asdict(p)
    pr["nulls"] = ",".join(p.nulls)
    out = dict(
        peth_pooled=cat("peth"), peth_per_trial=cat("per_trial"), tests=cat("test"),
        tests_per_trial=cat("tests_per_trial"), state_rates=state_rates(trials, states),
        params=pd.DataFrame([dict(**pr, max_time=cfg.max_time, n_trials=len(trials))]),
    )
    ddir = Path(cfg.out_dir) / "descriptives"
    ddir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(ddir / f"{name}.csv", index=False)

    if make_figures:
        _style()
        fdir = Path(cfg.out_dir) / "figures"
        for null in p.nulls:
            for fam in FAMILIES:
                plot_family(fam, null, summ, p, fdir)
            plot_forest(out["tests"], out["tests_per_trial"], null, p, fdir)
        plot_null_comparison(out["tests"], p, fdir)
        for null in p.nulls:
            for spec in ALL_SPECS:
                if spec.sender == "sub":
                    plot_per_trial(spec, null, summ[(null, spec.key)], p, fdir / "desc_per_trial")
        plot_state_rates(out["state_rates"], fdir)
        if verbose:
            print(f"[hz_desc] figures -> {fdir}; tables -> {ddir} "
                  f"({time.time() - t0:.1f} s total)")
    return out


def summary_text(out: dict[str, pd.DataFrame]) -> str:
    """Short text summary of the pre-specified window tests."""
    t = out["tests"]
    lines = ["Descriptive key results (subordinate chirps; hazard in the 2 s after a chirp, "
             "observed / null mean [95% trial-bootstrap CI], trials with ratio > 1):"]
    for ev, q in (("offset", "(a) chase offset after in-chase chirp"),
                  ("onset", "(b) chase onset after out-of-chase chirp")):
        parts = []
        for _, r in t[(t["analysis"] == f"post_chirp_hazard:{ev}:sub")].iterrows():
            if np.isfinite(r["ratio"]):
                parts.append(f"{r['null']} {r['ratio']:.2f} [{r['ratio_ci_low']:.2f}, "
                             f"{r['ratio_ci_high']:.2f}] {int(r['n_trials_ratio_gt1'])}/"
                             f"{int(r['n_trials_ratio_finite'])}")
        lines.append(f"  {q}: " + "; ".join(parts))
    lines += ["", "All alignments: observed / null-mean rate in the test window "
              "[95% trial-bootstrap CI], null 95% range, p (window mean, max-|z|)"]
    for null in t["null"].unique():
        lines.append(f" null = {null}:")
        for _, r in t[t["null"] == null].iterrows():
            head = f"  {r['analysis']:<30s} [{r['window_lo']:+g},{r['window_hi']:+g}) s"
            if not np.isfinite(r["ratio"]):
                lines.append(f"{head}  no data")
                continue
            lines.append(
                f"{head}  {r['ratio']:.2f} [{r['ratio_ci_low']:.2f}, {r['ratio_ci_high']:.2f}]"
                f"  median trial {r['median_trial_ratio']:.2f}"
                f"  null {r['null_ratio_low']:.2f}-{r['null_ratio_high']:.2f}"
                f"  p {_fmt_p(r['p_window_two_sided'])}, {_fmt_p(r['p_global_maxz'])}"
                f"  trials>1 {int(r['n_trials_ratio_gt1'])}/{int(r['n_trials_ratio_finite'])}"
                f"  ({int(r['n_trials'])} trials, {int(r['n_triggers'])} triggers)"
            )
    return "\n".join(lines)


def main() -> None:
    from hz_data import load_trials

    ap = argparse.ArgumentParser(description="Descriptive chirp/chase alignments.")
    ap.add_argument("data_root", type=Path, nargs="?", default=Config().data_root)
    ap.add_argument("--out", type=Path, default=Config().out_dir)
    ap.add_argument("--n-jitter", type=int, default=None)
    ap.add_argument("--max-time", type=float, default=None, help="s; <=0 disables the cap")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    cfg = Config(data_root=a.data_root, out_dir=a.out).updated(
        n_jitter=a.n_jitter, max_time=a.max_time, quick=a.quick or None
    )
    trials, _, _ = load_trials(cfg)
    out = run_descriptives(trials, cfg, make_figures=not a.no_figures)
    print(summary_text(out))


if __name__ == "__main__":
    main()
