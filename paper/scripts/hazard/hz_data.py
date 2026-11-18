# Ported from chirp_competition b65108e (src_hazard/hz_data.py).
# Changes: chirps of the v2 detections (cfg.chirp_detector = PAPER_DETECTIONS, rtdetr*) from
# chirps_<det>.parquet via chirps.load_chirp_table (rows with keep, box centre t_center_s,
# sender track_id); the sender QC uses that parquet's boxes instead of chirpdetector_bboxes.csv;
# detector names in the flag texts. v1 (rcnn) still reads the npy files as before.
"""Data layer: load, validate and tidy the 2019/2020 tube-competition trials.

Units: seconds (grid/recording time, t = grid sample index / 20 kHz) and Hz.

Per trial this module builds a :class:`Trial` holding chase intervals and
contacts (BORIS video annotations converted to grid time), dominant and
subordinate chirp times (automated detector, ``cfg.chirp_detector``), optional rise times, and a
smoothed baseline EODf per fish. Everything that contradicts the analysis
assumptions is written to a structured flags table instead of being silently
fixed. Where the data layer does repair something (unpaired BORIS events,
duplicate chirp detections, misaligned LED anchors) the repair is logged in the
flags table with ``action`` set accordingly.

BORIS -> grid conversion reuses the logic of ``load_boris``,
``converte_video_frames_to_grid_idx`` and ``load_and_convert_boris_events`` in
``src/analyze_event_time_rates_trial.py`` (frame = round(t_video * FPS), linear
map from video LED frames to grid LED sample indices). The reference maps with
the first and last LED only; this module adds an anchor check because in two
trials the first/last LEDs of video and grid are not the same pulse.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/scripts
from chirps import load_chirp_table  # noqa: E402
from hz_config import Config  # noqa: E402

# BORIS "Behavior" codes as found in the csv files. The dataset README.txt
# states 1/2/3 for onset/offset/contact; the files use 0/1/2 (verified: every
# 0 is followed by a 1 except one double onset, durations 1-71 s, and 2 is a
# point event that almost never falls inside a 0-1 interval).
CODE_ON, CODE_OFF, CODE_CONTACT = 0, 1, 2
BORIS_NOMINAL_FPS = 25.0

FLAG_COLUMNS = ["recording", "category", "severity", "t", "value", "action", "detail"]


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
class FlagLog:
    """Collects data-quality flags as rows of a tidy table."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        recording: str,
        category: str,
        detail: str,
        severity: str = "warning",
        t: float = np.nan,
        value: float = np.nan,
        action: str = "flagged",
    ) -> None:
        """Add one flag. severity: info|warning|error; action: what was done."""
        self.rows.append(
            dict(
                recording=recording,
                category=category,
                severity=severity,
                t=float(t),
                value=float(value),
                action=action,
                detail=detail,
            )
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=FLAG_COLUMNS)


# ---------------------------------------------------------------------------
# Trial container
# ---------------------------------------------------------------------------
def _empty2() -> np.ndarray:
    return np.zeros((0, 2), dtype=float)


def _empty1() -> np.ndarray:
    return np.zeros(0, dtype=float)


@dataclass
class Trial:
    """One dyadic competition trial in grid time (s) and Hz.

    ``chases`` are [onset, offset] pairs, sorted and non-overlapping. The
    analysis window is ``[obs_start, obs_end)``; chase state is unknown inside
    ``exclude_intervals`` (e.g. after an unpaired BORIS onset), so bins there
    belong to neither risk set. Chase/contact actor is not annotated in BORIS;
    chases are assumed dominant -> subordinate.
    """

    recording: str
    group: int
    dom_id: float
    sub_id: float
    dom_fish: int
    sub_fish: int
    t_start: float
    t_end: float
    chases: np.ndarray = field(default_factory=_empty2)
    contacts: np.ndarray = field(default_factory=_empty1)
    chirps_dom: np.ndarray = field(default_factory=_empty1)
    chirps_sub: np.ndarray = field(default_factory=_empty1)
    rises_dom: np.ndarray | None = None
    rises_sub: np.ndarray | None = None
    eodf_t: np.ndarray = field(default_factory=_empty1)
    eodf_dom: np.ndarray = field(default_factory=_empty1)
    eodf_sub: np.ndarray = field(default_factory=_empty1)
    size_dom: float = np.nan
    size_sub: float = np.nan
    sex_dom: str = ""
    sex_sub: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    obs_start: float = np.nan  # defaults to t_start
    obs_end: float = np.nan  # defaults to t_end
    exclude_intervals: np.ndarray = field(default_factory=_empty2)

    def __post_init__(self) -> None:
        self.chases = np.asarray(self.chases, dtype=float).reshape(-1, 2)
        if len(self.chases):
            self.chases = self.chases[np.argsort(self.chases[:, 0], kind="stable")]
            if np.any(self.chases[:, 1] <= self.chases[:, 0]):
                raise ValueError(f"{self.recording}: chase with offset <= onset")
            if np.any(self.chases[1:, 0] < self.chases[:-1, 1]):
                raise ValueError(f"{self.recording}: overlapping chases")
        for name in ("contacts", "chirps_dom", "chirps_sub"):
            setattr(self, name, np.sort(np.asarray(getattr(self, name), dtype=float)))
        for name in ("rises_dom", "rises_sub"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, np.sort(np.asarray(v, dtype=float)))
        self.exclude_intervals = np.asarray(self.exclude_intervals, float).reshape(-1, 2)
        if not np.isfinite(self.obs_start):
            self.obs_start = float(self.t_start)
        if not np.isfinite(self.obs_end):
            self.obs_end = float(self.t_end)

    # -- helpers -------------------------------------------------------------
    @property
    def duration(self) -> float:
        """Length of the analysis window (s)."""
        return float(self.obs_end - self.obs_start)

    @property
    def chase_durations(self) -> np.ndarray:
        return self.chases[:, 1] - self.chases[:, 0]

    def delta_eodf(self, t: np.ndarray) -> np.ndarray:
        """Baseline EODf dominant minus subordinate (Hz) at times t (s)."""
        t = np.asarray(t, dtype=float)
        if len(self.eodf_t) == 0:
            return np.full(t.shape, np.nan)
        return np.interp(t, self.eodf_t, self.eodf_dom - self.eodf_sub)

    def eodf(self, t: np.ndarray, who: str) -> np.ndarray:
        """Baseline EODf (Hz) of 'dom' or 'sub' at times t (s)."""
        t = np.asarray(t, dtype=float)
        y = self.eodf_dom if who == "dom" else self.eodf_sub
        if len(self.eodf_t) == 0:
            return np.full(t.shape, np.nan)
        return np.interp(t, self.eodf_t, y)

    def in_chase(self, t: np.ndarray) -> np.ndarray:
        """True where onset <= t < offset for some chase."""
        return _in_intervals(np.asarray(t, dtype=float), self.chases)

    def valid(self, t: np.ndarray) -> np.ndarray:
        """True where t is inside the analysis window and chase state is known."""
        t = np.asarray(t, dtype=float)
        ok = (t >= self.obs_start) & (t < self.obs_end)
        return ok & ~_in_intervals(t, self.exclude_intervals)


def _in_intervals(t: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """Membership of t in half-open, sorted, non-overlapping intervals."""
    if len(iv) == 0:
        return np.zeros(t.shape, dtype=bool)
    iv = iv[np.argsort(iv[:, 0])]
    j = np.searchsorted(iv[:, 0], t, side="right") - 1
    ok = j >= 0
    out = np.zeros(t.shape, dtype=bool)
    out[ok] = t[ok] < iv[j[ok], 1]
    return out


# ---------------------------------------------------------------------------
# BORIS / LED alignment
# ---------------------------------------------------------------------------
@dataclass
class LedAlignment:
    """Piecewise-linear video-frame -> grid-time map.

    Within segment k, grid_s = seg_slope[k] * frame + seg_icpt[k]. Segments are
    separated at detected steps in the LED residuals (dropped or duplicated video
    frames); ``seg_bounds`` holds the frame boundaries between segments.
    """

    seg_bounds: np.ndarray
    seg_slope: np.ndarray
    seg_icpt: np.ndarray
    ref_slope: float
    ref_intercept: float
    method: str
    anchors: tuple[int, int, int, int]  # LEDs dropped: video start, grid start, video end, grid end
    n_led_video: int
    n_led_grid: int
    n_matched: int  # video LEDs within led_inlier_tol of a grid LED
    n_unpaired: int  # video LEDs farther than led_match_tol from any grid LED
    frame_first: int
    frame_last: int
    ref_resid_med: float
    ref_resid_p95: float
    resid_med: float
    resid_p95: float
    steps: list[tuple[float, float, float, float]]  # (grid t, size s, frame lo, frame hi)
    period_shifts: list[int]  # whole-LED-period intercept shift per segment

    def to_grid(self, frames: np.ndarray) -> np.ndarray:
        f = np.asarray(frames, dtype=float)
        k = np.searchsorted(self.seg_bounds, f, side="right")
        return self.seg_slope[k] * f + self.seg_icpt[k]

    def ref_to_grid(self, frames: np.ndarray) -> np.ndarray:
        return self.ref_slope * np.asarray(frames, dtype=float) + self.ref_intercept

    @property
    def fps_effective(self) -> float:
        """Video frames per grid second in the first segment."""
        return float(1.0 / self.seg_slope[0])


def _anchor_map(f0: float, f1: float, g0: float, g1: float) -> tuple[float, float]:
    """Slope/intercept through (f0, g0) and (f1, g1); cf. converte_video_frames_to_grid_idx."""
    slope = (g1 - g0) / (f1 - f0)
    return slope, g0 - slope * f0


def _nn_resid(mapped: np.ndarray, grid_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed residual to the nearest grid LED and that LED's index."""
    j = np.searchsorted(grid_t, mapped).clip(1, len(grid_t) - 1)
    left, right = grid_t[j - 1], grid_t[j]
    use_left = np.abs(mapped - left) <= np.abs(right - mapped)
    nearest = np.where(use_left, j - 1, j)
    return mapped - grid_t[nearest], nearest


def _find_steps(r: np.ndarray, tol: float, w: int = 5) -> list[int]:
    """Indices i where median(r[i+1:i+1+w]) - median(r[i+1-w:i+1]) exceeds tol."""
    n = len(r)
    d = np.zeros(n)
    for i in range(w - 1, n - w):
        d[i] = np.median(r[i + 1 : i + 1 + w]) - np.median(r[i + 1 - w : i + 1])
    cand = np.where(np.abs(d) > tol)[0]
    steps: list[int] = []
    for i in cand:  # non-maximum suppression within w
        lo, hi = max(0, i - w), min(n, i + w + 1)
        if np.abs(d[i]) == np.abs(d[lo:hi]).max() and (not steps or i - steps[-1] > w):
            steps.append(int(i))
    return steps


def _best_line(
    f: np.ndarray, gt: np.ndarray, slope: float, icpt: float, period: float, tol: float,
    max_m: int = 3,
) -> tuple[float, float, int, int]:
    """Resolve the LED-period ambiguity of a line and refit it.

    A regular LED train matches equally well after shifting by whole periods;
    the irregular gaps (~every 5 min) break that symmetry, so the shift m with
    most pairs |residual| < tol is chosen. Returns (slope, icpt, n_inliers, m).
    """
    best: tuple[int, int, np.ndarray, np.ndarray] | None = None
    for m in sorted(range(-max_m, max_m + 1), key=abs):
        r, near = _nn_resid(slope * f + icpt + m * period, gt)
        inl = np.abs(r) < tol
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), m, inl, near)
    n, m, inl, near = best
    if n >= 3:
        slope, icpt = np.polyfit(f[inl], gt[near[inl]], 1)
    else:
        icpt = icpt + m * period
    return float(slope), float(icpt), n, m


def fit_led_alignment(
    led_frames: np.ndarray, led_idx: np.ndarray, sr: float, cfg: Config
) -> LedAlignment:
    """Map video frames to grid seconds from LED sync pulses.

    The reference map (first/last LED, as in converte_video_frames_to_grid_idx)
    is always computed for comparison. In "robust" mode: (1) anchors are searched
    over up to ``led_max_anchor_shift`` dropped LEDs at each end, minimising the
    median nearest-LED residual (fixes the slope; the reference is kept unless a
    candidate is > 5 ms better); (2) the whole-LED-period ambiguity of the
    intercept is resolved with the gap fingerprint of the LED train and the line
    is refit by least squares; (3) video LEDs are paired with their nearest grid
    LED (|residual| < ``led_match_tol``) and steps in the residuals larger than
    ``led_step_tol`` (dropped/duplicated video frames) split the map into
    segments; (4) each segment repeats (2) on its own LEDs.
    """
    lf = np.asarray(led_frames, dtype=float)
    gt = np.asarray(led_idx, dtype=float) / sr
    ref = _anchor_map(lf[0], lf[-1], gt[0], gt[-1])

    def score(slope: float, icpt: float) -> np.ndarray:
        return np.abs(_nn_resid(slope * lf + icpt, gt)[0])

    ref_res = score(*ref)
    ref_med, ref_p95 = float(np.median(ref_res)), float(np.percentile(ref_res, 95))
    common = dict(ref_slope=ref[0], ref_intercept=ref[1], n_led_video=len(lf),
                  n_led_grid=len(gt), frame_first=int(lf[0]), frame_last=int(lf[-1]),
                  ref_resid_med=ref_med, ref_resid_p95=ref_p95)

    if cfg.led_alignment == "reference":
        return LedAlignment(
            seg_bounds=np.zeros(0), seg_slope=np.array([ref[0]]),
            seg_icpt=np.array([ref[1]]), method="reference", anchors=(0, 0, 0, 0),
            n_matched=int(np.sum(ref_res < cfg.led_inlier_tol)),
            n_unpaired=int(np.sum(ref_res > cfg.led_match_tol)), resid_med=ref_med,
            resid_p95=ref_p95, steps=[], period_shifts=[0], **common,
        )

    best, best_med, best_anchor = ref, ref_med, (0, 0, 0, 0)
    k = cfg.led_max_anchor_shift
    for i in range(k + 1):
        for j in range(k + 1):
            for a in range(k + 1):
                for b in range(k + 1):
                    cand = _anchor_map(lf[i], lf[-1 - a], gt[j], gt[-1 - b])
                    med = float(np.median(score(*cand)))
                    if med < best_med - 0.005:
                        best, best_med, best_anchor = cand, med, (i, j, a, b)

    period = float(np.median(np.diff(gt)))
    tol = cfg.led_inlier_tol

    def nn_lsq(f: np.ndarray, slope: float, icpt: float) -> tuple[float, float]:
        """Least squares on nearest-LED pairs (period-agnostic)."""
        r, near = _nn_resid(slope * f + icpt, gt)
        ok = np.abs(r) < cfg.led_match_tol
        if ok.sum() < 3:
            return slope, icpt
        p1 = np.polyfit(f[ok], gt[near[ok]], 1)
        return float(p1[0]), float(p1[1])

    slope, icpt = nn_lsq(lf, *best)

    # segments at residual steps (frame drops), each with its own line
    r, _ = _nn_resid(slope * lf + icpt, gt)
    pm = np.abs(r) < cfg.led_match_tol
    fm, rm = lf[pm], r[pm]
    cuts = _find_steps(rm, cfg.led_step_tol)
    bounds = np.array([0.5 * (fm[c] + fm[c + 1]) for c in cuts])
    seg = np.searchsorted(bounds, lf, side="right")
    slopes, icpts, shifts = [], [], []
    for kseg in range(len(cuts) + 1):
        sel = seg == kseg
        s_k, b_k = nn_lsq(lf[sel], slope, icpt)
        s_k, b_k, _, m_k = _best_line(lf[sel], gt, s_k, b_k, period, tol)
        s_k, b_k, _, m2 = _best_line(lf[sel], gt, s_k, b_k, period, tol, max_m=1)
        slopes.append(s_k)
        icpts.append(b_k)
        shifts.append(m_k + m2)
    slopes_a, icpts_a = np.array(slopes), np.array(icpts)
    steps = []
    for kseg, c in enumerate(cuts):
        x = bounds[kseg]
        size = float((slopes_a[kseg + 1] * x + icpts_a[kseg + 1]) - (slopes_a[kseg] * x + icpts_a[kseg]))
        steps.append((float(slope * fm[c] + icpt), size, float(fm[c]), float(fm[c + 1])))

    method = "robust"
    if best_anchor != (0, 0, 0, 0):
        method += "_anchor"
    if any(shifts):
        method += "_period"
    if cuts:
        method += f"_{len(cuts) + 1}seg"
    fin = np.abs(_nn_resid(slopes_a[seg] * lf + icpts_a[seg], gt)[0])
    return LedAlignment(
        seg_bounds=bounds, seg_slope=slopes_a, seg_icpt=icpts_a, method=method,
        anchors=best_anchor, n_matched=int(np.sum(fin < tol)),
        n_unpaired=int(np.sum(fin > cfg.led_match_tol)), resid_med=float(np.median(fin)),
        resid_p95=float(np.percentile(fin, 95)), steps=steps,
        period_shifts=shifts, **common,
    )


def pair_chases(t: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Pair chase onsets with the next offset if no other onset comes first.

    Returns (chases (n, 2) in input time units, issues). Each issue is a dict
    with kind ('unmatched_onset' | 'unmatched_offset' | 'zero_length'), t, and
    the ambiguous interval (lo, hi) in which chase state is unknown.
    """
    m = np.isin(codes, [CODE_ON, CODE_OFF])
    tt, cc = np.asarray(t, float)[m], np.asarray(codes)[m]
    order = np.lexsort((cc, tt))  # by time; at ties onset (0) before offset (1)
    tt, cc = tt[order], cc[order]
    chases: list[tuple[float, float]] = []
    issues: list[dict[str, Any]] = []
    i, n = 0, len(tt)
    while i < n:
        if cc[i] == CODE_ON:
            if i + 1 < n and cc[i + 1] == CODE_OFF:
                if tt[i + 1] > tt[i]:
                    chases.append((tt[i], tt[i + 1]))
                else:
                    issues.append(dict(kind="zero_length", t=tt[i], lo=tt[i], hi=tt[i + 1]))
                i += 2
                continue
            hi = tt[i + 1] if i + 1 < n else np.inf
            issues.append(dict(kind="unmatched_onset", t=tt[i], lo=tt[i], hi=hi))
        else:
            lo = tt[i - 1] if i > 0 else -np.inf
            issues.append(dict(kind="unmatched_offset", t=tt[i], lo=lo, hi=tt[i]))
        i += 1
    return np.array(chases, dtype=float).reshape(-1, 2), issues


def boris_file(trial_dir: Path, recording: str) -> Path:
    """BORIS csv path, '<YYYY-MM-DD>.csv' (cf. load_boris)."""
    return trial_dir / ("-".join(recording.split("-")[:3]) + ".csv")


def _load_led(trial_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    f_frames, f_idx = trial_dir / "LED_frames.npy", trial_dir / "led_idxs.csv"
    if not (f_frames.exists() and f_idx.exists()):
        return None
    frames = np.load(f_frames)
    idx = pd.read_csv(f_idx, header=None).iloc[:, 0].to_numpy()
    return frames, idx


# ---------------------------------------------------------------------------
# EODf, chirps, rises
# ---------------------------------------------------------------------------
def _raw_track(
    times: np.ndarray, fund: np.ndarray, ident: np.ndarray, idx: np.ndarray, fid: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Raw EODf track of one fish (median over duplicate frames)."""
    m = ident == fid
    if not np.any(m):
        return _empty1(), _empty1(), 0
    s = pd.Series(fund[m]).groupby(idx[m]).median()
    ii = s.index.to_numpy().astype(int)
    n_dup = int(m.sum() - len(ii))
    return times[ii], s.to_numpy(dtype=float), n_dup


def _baseline(t: np.ndarray, f: np.ndarray, window: float, grid: np.ndarray) -> np.ndarray:
    """Trailing rolling median over (t - window, t] of an irregular track, on a grid.

    Step-hold: grid point g takes the trailing median at the last raw sample <= g,
    so it uses no EODf sample after g (no interpolation towards the next sample
    across track gaps). Grid points before the first raw sample are NaN because no
    past EODf exists; the hazard design codes them with an indicator (hz_hazard).
    """
    if len(t) == 0:
        return np.full(grid.shape, np.nan)
    ser = pd.Series(f, index=pd.to_timedelta(t, unit="s"))
    base = ser.rolling(pd.Timedelta(seconds=window), min_periods=1).median().to_numpy()
    i = np.searchsorted(t, grid, side="right") - 1  # last raw sample <= g
    return np.where(i >= 0, base[np.maximum(i, 0)], np.nan)


def _dedup(t: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop detections closer than window to the last kept one. Returns (kept, dropped)."""
    if len(t) == 0:
        return t, t
    keep = np.ones(len(t), dtype=bool)
    last = t[0]
    for i in range(1, len(t)):
        if t[i] - last < window:
            keep[i] = False
        else:
            last = t[i]
    return t[keep], t[~keep]


def _load_chirps(d: Path, det: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    """Chirp times (s), sender ids and the detector's boxes of one recording.

    v2 (``det`` = ``rtdetr*``): ``chirps_<det>.parquet`` via
    ``chirps.load_chirp_table``. The chirps are the rows with ``keep`` (score
    at or above the operating threshold, assigned to a track, not dropped by
    post-processing); time = box centre ``t_center_s`` (the v1 times are box
    centres too), sender = ``track_id``. The boxes are the chirps before
    track assignment (``chirp`` and not dropped by post-processing), so
    chirps without a sender can be counted.
    Otherwise (v1 ``rcnn``): ``chirp_times_<det>.npy`` / ``chirp_ids_<det>.npy``
    and the boxes of ``chirpdetector_bboxes.csv`` (None if it is missing).
    Box columns: t1, t2 (s), f1, f2 (Hz), track_id.
    """
    cols = ["t1", "t2", "f1", "f2", "track_id"]
    if det.startswith("rtdetr"):
        tab = load_chirp_table(d, det)
        kept = tab[tab["keep"]]
        boxes = tab[tab["chirp"] & (tab["pp_drop_reason"] == "")].rename(columns=dict(
            t_start_s="t1", t_end_s="t2", f_low_hz="f1", f_high_hz="f2"))
        return (kept["t_center_s"].to_numpy(float), kept["track_id"].to_numpy(float),
                boxes[cols].reset_index(drop=True))
    ct = np.load(d / f"chirp_times_{det}.npy").astype(float)
    ci = np.load(d / f"chirp_ids_{det}.npy").astype(float)
    f = d / "chirpdetector_bboxes.csv"
    return ct, ci, (pd.read_csv(f, usecols=cols) if f.exists() else None)


def _chirp_files(det: str) -> str:
    """The chirp file(s) of a detection set, for flag texts."""
    return f"chirps_{det}.parquet" if det.startswith("rtdetr") else f"chirp_times_{det}.npy"


def _eodf_at(track: tuple[np.ndarray, np.ndarray], t: np.ndarray, max_dt: float = 2.0
             ) -> np.ndarray:
    """Raw EODf (Hz) of the track sample nearest to each t (NaN if > max_dt s away)."""
    tt, ff = track
    t = np.asarray(t, float)
    if len(tt) == 0:
        return np.full(len(t), np.nan)
    j = np.searchsorted(tt, t).clip(0, len(tt) - 1)
    jl = np.maximum(j - 1, 0)
    j = np.where(np.abs(tt[jl] - t) < np.abs(tt[j] - t), jl, j)
    out = ff[j].astype(float)
    out[np.abs(tt[j] - t) > max_dt] = np.nan
    return out


def _nearest_dist(ref: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Distance from each t to the nearest element of sorted ref (inf if ref empty)."""
    if len(ref) == 0:
        return np.full(len(t), np.inf)
    j = np.searchsorted(ref, t).clip(0, len(ref) - 1)
    jl = np.maximum(j - 1, 0)
    return np.minimum(np.abs(ref[j] - t), np.abs(ref[jl] - t))


def _bbox_qc(
    bb: pd.DataFrame | None, rec: str, ct_assigned: np.ndarray, chirps: dict[str, np.ndarray],
    raw: dict[str, tuple[np.ndarray, np.ndarray]], chases: np.ndarray, t_lo: float,
    t_hi: float, cfg: Config, log: FlagLog,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Checks against the detector's own bounding boxes ``bb`` (see _load_chirps:
    v1 chirpdetector_bboxes.csv, v2 the chirps of chirps_<det>.parquet).

    1. Unassigned detections: the v1 chirp detector drops boxes without a track
       id before writing chirp_times_rcnn.npy, so the npy files always look
       fully assigned; in v2 they are the parquet's chirps without a track_id,
       which are not in the analysed selection (keep). Boxes with track_id NaN
       farther than chirp_dedup_window from every assigned chirp are distinct
       chirps of unknown sender.
    2. Sender ambiguity: for each kept chirp, both fish's EODf inside the box's
       frequency range means the box alone cannot tell the sender; the id then
       rests on the detector's emitter assignment only.
    Returns (summary columns, meta arrays).
    """
    det = cfg.chirp_detector
    if bb is None:
        log.add(rec, "chirp_bboxes_missing", "chirpdetector_bboxes.csv missing; unassigned "
                "detections and sender ambiguity not checked", "info")
        return {}, {}
    tc = 0.5 * (bb["t1"].to_numpy(float) + bb["t2"].to_numpy(float))
    f1, f2 = bb["f1"].to_numpy(float), bb["f2"].to_numpy(float)
    tid = bb["track_id"].to_numpy(float)
    s: dict[str, Any] = dict(bbox_height_median=round(float(np.median(f2 - f1)), 1)
                             if len(bb) else np.nan)
    meta: dict[str, Any] = {}
    # 1. unassigned detections
    na = ~np.isfinite(tid)
    tna = np.sort(tc[na])
    distinct = tna[_nearest_dist(np.sort(ct_assigned), tna) >= cfg.chirp_dedup_window]
    dw = distinct[(distinct >= t_lo) & (distinct < t_hi)]
    n_in = int(_in_intervals(dw, chases).sum()) if len(dw) else 0
    s.update(n_unassigned_upstream=int(na.sum()), n_unassigned_distinct=len(distinct),
             n_unassigned_distinct_window=len(dw), n_unassigned_distinct_in_chase_window=n_in)
    meta["unassigned_distinct"] = distinct
    if na.any():
        src = (f"{_chirp_files(det)} has {int(na.sum())} chirps without a track_id that are "
               "not in the analysed selection (keep)" if det.startswith("rtdetr") else
               f"chirpdetector_bboxes.csv has {int(na.sum())} detections with track_id NaN that "
               f"the detector dropped before writing {_chirp_files(det)} (so the npy files "
               "always look fully assigned)")
        log.add(rec, "chirp_unassigned_upstream",
                f"{src}; {len(distinct)} are >= "
                f"{cfg.chirp_dedup_window * 1e3:.0f} ms from any assigned chirp (distinct chirps "
                f"of unknown sender), {len(dw)} of them in the analysis window ({n_in} inside "
                f"chases, {len(dw) - n_in} outside); the rest duplicate an assigned chirp. "
                "Not used", "warning" if len(dw) else "info", value=len(dw))
    # 2. sender ambiguity of the kept chirps
    order = np.argsort(tc)
    tcs = tc[order]
    for who in ("dom", "sub"):
        x = chirps[who]
        amb = np.zeros(len(x), bool)
        if len(x) and len(tcs):
            j = np.searchsorted(tcs, x).clip(0, len(tcs) - 1)
            jl = np.maximum(j - 1, 0)
            j = np.where(np.abs(tcs[jl] - x) < np.abs(tcs[j] - x), jl, j)
            row = order[j]
            hit = np.abs(tc[row] - x) < 1e-3
            fd = _eodf_at(raw.get("dom", (_empty1(), _empty1())), x)
            fs = _eodf_at(raw.get("sub", (_empty1(), _empty1())), x)
            lo, hi = f1[row], f2[row]
            amb = hit & (fd >= lo) & (fd <= hi) & (fs >= lo) & (fs <= hi)
            s[f"n_{who}_chirps_without_bbox"] = int((~hit).sum())
        meta[f"chirps_{who}_ambiguous"] = amb
        w = (x >= t_lo) & (x < t_hi)
        s[f"frac_{who}_ambiguous_window"] = round(float(amb[w].mean()), 3) if w.any() else np.nan
        s[f"n_{who}_ambiguous_window"] = int(amb[w].sum())
    n_w = sum(int(((chirps[w] >= t_lo) & (chirps[w] < t_hi)).sum()) for w in ("dom", "sub"))
    n_a = s["n_dom_ambiguous_window"] + s["n_sub_ambiguous_window"]
    share = n_a / n_w if n_w else np.nan
    s["frac_all_ambiguous_window"] = round(share, 3) if n_w else np.nan
    if n_w and share > 0.5:
        log.add(rec, "sender_ambiguous_bbox",
                f"for {n_a}/{n_w} chirps in the window ({s['n_sub_ambiguous_window']} sub, "
                f"{s['n_dom_ambiguous_window']} dom) both fish's EODf lie inside the chirp's "
                f"bounding box (median box height {s['bbox_height_median']:.0f} Hz): the sender "
                f"id rests on the {det} emitter assignment only", "warning", value=share)
    return s, meta


def _led_index_offsets(align: LedAlignment, lf: np.ndarray, gt: np.ndarray, tol: float
                       ) -> tuple[int, int]:
    """(grid LED index - video LED index) at the start and at the end of the recording."""
    r, near = _nn_resid(align.to_grid(lf), gt)
    ok = np.abs(r) < tol
    off = (near - np.arange(len(lf)))[ok]
    if len(off) == 0:
        return 0, 0
    return int(np.median(off[:20])), int(np.median(off[-20:]))


def _wallclock_gaps(d: Path, rec: str, min_gap: float = 1.0) -> list[tuple[int, float]]:
    """(frame, gap s) where the video's per-frame wall clock (<date>_0_times.dat) jumps."""
    f = d / ("-".join(rec.split("-")[:3]) + "_0_times.dat")
    if not f.exists():
        return []
    try:
        t = pd.to_timedelta(pd.read_csv(f, header=None, dtype=str).iloc[:, 0])
    except (ValueError, pd.errors.ParserError):
        return []
    dt = np.diff(t.dt.total_seconds().to_numpy())
    return [(int(k + 1), float(dt[k])) for k in np.flatnonzero(dt > min_gap)]


# ---------------------------------------------------------------------------
# Per-trial loading
# ---------------------------------------------------------------------------
def _fish_meta(data_root: Path) -> pd.DataFrame | None:
    f = data_root / "id_meta.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    m["length_mean"] = m[["l1", "l2", "l3"]].mean(axis=1)
    return m.set_index(["group", "fish"])


def _cnn_notes(data_root: Path) -> pd.DataFrame | None:
    f = data_root / "chirp_notes.csv"
    return pd.read_csv(f, index_col=0).set_index("recording") if f.exists() else None


def _build_trial(
    row: pd.Series,
    cfg: Config,
    log: FlagLog,
    fish_meta: pd.DataFrame | None,
    cnn_notes: pd.DataFrame | None,
) -> tuple[Trial | None, dict[str, Any], list[float]]:
    """Load one trial. Returns (Trial or None, summary row, raw BORIS chase durations)."""
    rec = str(row["recording"])
    d = cfg.data_root / rec
    grp = int(row["group"])
    dom_id, sub_id = float(row["win_ID"]), float(row["lose_ID"])
    s: dict[str, Any] = dict(
        recording=rec, group=grp, dom_fish=int(row["win_fish"]), sub_fish=int(row["lose_fish"]),
        dom_id=dom_id, sub_id=sub_id, sex_dom=row["sex_win"], sex_sub=row["sex_lose"],
        size_dom=float(row["size_win"]), size_sub=float(row["size_lose"]),
        dsize=float(row["size_win"] - row["size_lose"]), draw=int(row["draw"]),
        exp_dom=int(row["exp_win"]), exp_sub=int(row["exp_lose"]),
    )
    reasons: list[str] = []
    raw_durs: list[float] = []

    # --- recording extent ----------------------------------------------------
    times = np.load(d / "times.npy")
    if (d / "meta.npy").exists():
        meta_arr = np.load(d / "meta.npy", allow_pickle=True)
        t_start, t_end = float(meta_arr[0]), float(meta_arr[1])
    else:
        t_start, t_end = float(times[0]), float(times[-1])
        log.add(rec, "missing_meta", "meta.npy missing; extent from times.npy", "info")
    s.update(t_start=t_start, t_end=t_end, spec_dt=float(np.median(np.diff(times))))

    if not np.isfinite(dom_id) or not np.isfinite(sub_id) or dom_id == sub_id:
        log.add(rec, "role_ids_invalid", f"win_ID={dom_id}, lose_ID={sub_id}", "error",
                action="excluded_trial")
        reasons.append("invalid dominance ids")

    # --- EODf tracks ---------------------------------------------------------
    fund = np.load(d / "fund_v.npy")
    ident = np.load(d / "ident_v.npy")
    idxv = np.load(d / "idx_v.npy")
    track_ids = np.unique(ident[np.isfinite(ident)])
    foreign = sorted(set(track_ids.tolist()) - {dom_id, sub_id})
    if foreign:
        log.add(rec, "eodf_extra_track_ids", f"ident_v has ids {foreign} besides dom/sub",
                "warning")
    grid = np.arange(t_start, t_end + cfg.eodf_grid_dt / 2, cfg.eodf_grid_dt)
    eodf: dict[str, np.ndarray] = {}
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_frames_win = np.sum((times >= t_start) & (times < (cfg.max_time or np.inf)))
    for who, fid in (("dom", dom_id), ("sub", sub_id)):
        tt, ff, ndup = _raw_track(times, fund, ident, idxv, fid)
        raw[who] = (tt, ff)
        if len(tt) == 0:
            log.add(rec, "eodf_track_missing", f"no EODf track for {who} id {fid}", "error",
                    action="excluded_trial")
            reasons.append(f"no EODf track ({who})")
            eodf[who] = np.full(grid.shape, np.nan)
            s[f"eodf_cov_{who}"] = 0.0
            continue
        cov = len(tt) / len(times)
        cov_win = np.sum(tt < (cfg.max_time or np.inf)) / max(n_frames_win, 1)
        gaps = np.diff(np.r_[t_start, tt, t_end])
        big = gaps > cfg.eodf_max_gap
        s.update({
            f"eodf_cov_{who}": round(cov, 3), f"eodf_cov_win_{who}": round(cov_win, 3),
            f"eodf_maxgap_{who}": round(float(gaps.max()), 1),
            f"eodf_raw_median_{who}": round(float(np.median(ff)), 2),
        })
        if ndup:
            log.add(rec, "eodf_duplicate_frames",
                    f"{who}: {ndup} spectrogram frames with >1 value (median used)", "info",
                    value=ndup)
        if big.any():
            log.add(rec, "eodf_gap",
                    f"{who}: {big.sum()} gaps > {cfg.eodf_max_gap:.0f} s (longest "
                    f"{gaps.max():.0f} s); baseline held at the last sample before each gap", "info",
                    value=float(gaps.max()))
        if cov_win < cfg.eodf_min_coverage:
            log.add(rec, "eodf_low_coverage",
                    f"{who}: EODf present in {cov_win:.0%} of frames in analysis window",
                    "warning", value=cov_win)
        eodf[who] = _baseline(tt, ff, cfg.eodf_smooth_window, grid)
    summ_eodf = {"dom": float(row["EODf_win"]), "sub": float(row["EODf_lose"])}
    for who in ("dom", "sub"):
        med = s.get(f"eodf_raw_median_{who}", np.nan)
        s[f"eodf_trialsummary_{who}"] = summ_eodf[who]
        if np.isfinite(summ_eodf[who]) and abs(med - summ_eodf[who]) > 10:
            log.add(rec, "eodf_vs_trial_summary",
                    f"{who}: trial_summary EODf {summ_eodf[who]:.1f} Hz vs track median "
                    f"{med:.1f} Hz (definition of summary EODf unknown; track used)", "info",
                    value=summ_eodf[who] - med)
    s["delta_eodf_median"] = round(float(np.nanmedian(eodf["dom"] - eodf["sub"])), 2)
    wg = grid < (cfg.max_time or np.inf)
    ad = np.abs(eodf["dom"] - eodf["sub"])[wg]
    if np.isfinite(ad).any():
        close = float(np.nanmean(ad < cfg.close_eodf_hz))
        s.update(abs_delta_eodf_min=round(float(np.nanmin(ad)), 1),
                 frac_abs_delta_eodf_lt_close=round(close, 3))
        if close > cfg.close_eodf_frac and rec not in cfg.bad_recordings:
            log.add(rec, "close_eodfs_not_in_bad_list",
                    f"|dEODf| < {cfg.close_eodf_hz:.0f} Hz in {close:.0%} of the window (min "
                    f"{np.nanmin(ad):.1f} Hz), comparable to recordings excluded as "
                    "bad_recordings; chirp sender assignment may be unreliable", "warning",
                    value=close)

    # --- chirps --------------------------------------------------------------
    det = cfg.chirp_detector
    ct, ci, boxes = _load_chirps(d, det)
    s.update(n_chirps_raw=len(ct), chirps_win_summary=int(row["chirps_win"]),
             chirps_lose_summary=int(row["chirps_lose"]))
    if len(ct) != len(ci):
        log.add(rec, "chirp_arrays_mismatch", f"{len(ct)} times vs {len(ci)} ids", "error",
                action="excluded_trial")
        reasons.append("chirp arrays mismatch")
        ct, ci = ct[: min(len(ct), len(ci))], ci[: min(len(ct), len(ci))]
    if np.any(np.diff(ct) < 0):
        log.add(rec, "chirps_unsorted", "chirp_times not sorted in file (sorted on load)",
                "info")
    nan_id = ~np.isfinite(ci)
    if nan_id.any():
        log.add(rec, "chirp_unassigned", f"{nan_id.sum()} chirps without sender id",
                "warning", value=nan_id.sum(), action="dropped")
    other = np.isfinite(ci) & ~np.isin(ci, [dom_id, sub_id])
    if other.any():
        log.add(rec, "chirp_foreign_id",
                f"{other.sum()} chirps with ids {sorted(set(ci[other].tolist()))} "
                "matching neither fish", "warning", value=other.sum(), action="dropped")
    outside = (ct < t_start) | (ct > t_end)
    if outside.any():
        log.add(rec, "chirp_outside_recording", f"{outside.sum()} chirps outside "
                f"[{t_start:.1f}, {t_end:.1f}] s", "warning", value=outside.sum(),
                action="dropped")
    chirps: dict[str, np.ndarray] = {}
    for who, fid in (("dom", dom_id), ("sub", sub_id)):
        x = np.sort(ct[(ci == fid) & ~outside])
        kept, dropped = _dedup(x, cfg.chirp_dedup_window)
        for tdrop in dropped:
            prev = kept[np.searchsorted(kept, tdrop) - 1]
            log.add(rec, "chirp_duplicate",
                    f"{who} chirp {tdrop - prev:.3f} s after previous {who} chirp; merged",
                    "info", t=tdrop, value=tdrop - prev, action="merged")
        chirps[who] = kept
        s[f"n_chirps_{who}_raw"] = len(x)
        s[f"n_chirps_{who}"] = len(kept)
    for who, key in (("dom", "chirps_win_summary"), ("sub", "chirps_lose_summary")):
        if s[f"n_chirps_{who}_raw"] != s[key]:
            log.add(rec, "chirp_count_vs_trial_summary",
                    f"{who}: {s[f'n_chirps_{who}_raw']} in files vs {s[key]} in trial_summary",
                    "info")
    a, b = chirps["dom"], chirps["sub"]
    if len(a) and len(b):
        j = np.searchsorted(a, b).clip(1, len(a) - 1)
        dd = np.minimum(np.abs(a[j] - b), np.abs(a[j - 1] - b))
        for tb, dv in zip(b[dd < cfg.cross_sender_window], dd[dd < cfg.cross_sender_window]):
            log.add(rec, "chirp_cross_sender_coincidence",
                    f"sub chirp within {dv * 1000:.0f} ms of a dom chirp (possible double "
                    "assignment of one chirp); kept", "info", t=tb, value=dv)
    if len(chirps["dom"]) == 0:
        log.add(rec, "no_dom_chirps", "dominant produced no detected chirps", "info")
    if cnn_notes is not None and rec in cnn_notes.index:
        s["cnn_chirp_qc_good"] = cnn_notes.loc[rec, "good"]

    # --- rises (analysis/rise_idx.npy rows ordered like analysis/ids.npy) ---
    rises: dict[str, np.ndarray | None] = {"dom": None, "sub": None}
    f_ids, f_rise = d / "analysis/ids.npy", d / "analysis/rise_idx.npy"
    if f_ids.exists() and f_rise.exists():
        ids = np.load(f_ids).astype(float).tolist()
        ri = np.load(f_rise)
        if set(ids) != {dom_id, sub_id}:
            log.add(rec, "rise_ids_mismatch", f"analysis/ids.npy {ids} != dom/sub ids",
                    "warning", action="dropped")
        else:
            for who, fid in (("dom", dom_id), ("sub", sub_id)):
                r = ri[ids.index(fid)]
                r = r[np.isfinite(r)].astype(int)
                s[f"n_rises_{who}"] = len(r)
                if cfg.use_rises:
                    rises[who] = times[r]
    else:
        log.add(rec, "rises_missing", "analysis/ids.npy or rise_idx.npy missing", "info")

    # --- BORIS chases / contacts --------------------------------------------
    chases_g, contacts_g, excl_g = _empty2(), _empty1(), _empty2()
    fb = boris_file(d, rec)
    led = _load_led(d)
    s.update(has_boris=fb.exists(), has_led_sync=led is not None,
             chase_count_summary=int(row["chase_count"]),
             contact_count_summary=int(row["contact_count"]))
    last_annot = np.nan
    if not fb.exists():
        reasons.append("no BORIS annotation")
    elif led is None:
        reasons.append("BORIS csv present but no LED sync files (LED_frames.npy/led_idxs.csv)")
        log.add(rec, "boris_without_sync",
                "BORIS csv exists but LED_frames.npy/led_idxs.csv missing; video time cannot "
                "be mapped to grid time with the reference converter", "warning",
                action="excluded_trial")
    else:
        bd = pd.read_csv(fb)
        bt = bd["Start (s)"].to_numpy(dtype=float)
        code = bd["Behavior"].to_numpy()
        fps = float(bd["FPS"].iloc[0])
        subj = sorted(bd["Subject"].astype(str).unique().tolist())
        s.update(boris_fps=fps, boris_subject="|".join(subj))
        if subj != ["No focal subject"]:
            log.add(rec, "boris_subject_present", f"Subject values {subj}; chaser identity "
                    "may be annotated but is not used", "info")
        if fps != BORIS_NOMINAL_FPS:
            log.add(rec, "boris_fps", f"FPS column = {fps}", "info")
        unknown = ~np.isin(code, [CODE_ON, CODE_OFF, CODE_CONTACT])
        if unknown.any():
            log.add(rec, "boris_unknown_code", f"codes {sorted(set(code[unknown]))}",
                    "warning", action="dropped")
        if (bd["Behavior type"].astype(str) != "POINT").any():
            log.add(rec, "boris_state_events", "non-POINT behaviour types present", "warning")

        align = fit_led_alignment(led[0], led[1], cfg.sr, cfg)
        frames = np.round(bt * fps)
        s.update(
            led_n_video=align.n_led_video, led_n_grid=align.n_led_grid,
            led_matched=align.n_matched, led_unpaired=align.n_unpaired,
            led_method=align.method, led_anchor_drop=str(align.anchors),
            led_period_shifts=str(align.period_shifts),
            led_n_segments=len(align.seg_slope),
            led_fps_effective=round(align.fps_effective, 4),
            led_ref_resid_med_ms=round(align.ref_resid_med * 1e3, 1),
            led_resid_med_ms=round(align.resid_med * 1e3, 1),
            led_resid_p95_ms=round(align.resid_p95 * 1e3, 1),
        )
        lf_ = np.asarray(led[0], float)
        gt_ = np.asarray(led[1], float) / cfg.sr
        off0, off1 = _led_index_offsets(align, lf_, gt_, cfg.led_inlier_tol)
        s.update(led_index_offset_start=off0, led_index_offset_end=off1)
        wall = _wallclock_gaps(d, rec)
        s["video_wallclock_gaps"] = "; ".join(f"frame {k}: {g:.2f} s" for k, g in wall)
        for k, g in wall:
            log.add(rec, "video_wallclock_gap",
                    f"per-frame wall clock ({'-'.join(rec.split('-')[:3])}_0_times.dat) jumps "
                    f"{g:.2f} s at video frame {k} (grid t={float(align.to_grid(np.array([k]))[0]):.0f}"
                    " s): frames were dropped there", "info", t=float(align.to_grid(np.array([k]))[0]),
                    value=g)
        if align.n_led_video != align.n_led_grid or off0 or off1:
            parts = []
            if off0 > 0:
                parts.append(f"the video starts {off0} grid LED(s) late (first video LED = "
                             f"grid LED {off0 + 1})")
            elif off0 < 0:
                parts.append(f"the video has {-off0} LED(s) before the first grid LED")
            if off1 != off0:
                where = (f" at the video frame discontinuity near grid t={align.steps[0][0]:.0f} s"
                         if align.steps else "")
                if wall:
                    where += f" (wall clock jumps {wall[0][1]:.2f} s at frame {wall[0][0]})"
                parts.append(f"{off1 - off0} grid LED(s) missing from the video mid-recording"
                             f"{where}" if off1 > off0 else
                             f"{off0 - off1} extra video LED(s) mid-recording{where}")
            tail = (align.n_led_grid - 1) - (align.n_led_video - 1 + off1)
            if tail:
                parts.append(f"the video ends {tail} grid LED(s) early" if tail > 0 else
                             f"the video has {-tail} LED(s) after the last grid LED")
            log.add(rec, "led_count_mismatch",
                    f"{align.n_led_video} video LED frames vs {align.n_led_grid} grid LED "
                    f"indices: {'; '.join(parts) or 'unresolved'} (from the fitted LED pairing). "
                    "The reference converter pairs first with first and last with last LED, "
                    "which is wrong here", "warning", value=align.n_led_grid - align.n_led_video)
        if align.ref_resid_med > cfg.led_bad_resid:
            act = "corrected" if cfg.led_alignment == "robust" else "flagged"
            log.add(rec, "led_reference_misaligned",
                    f"reference first/last-LED map: median nearest-LED residual "
                    f"{align.ref_resid_med * 1e3:.0f} ms (p95 {align.ref_resid_p95 * 1e3:.0f} "
                    f"ms); {align.method} map (anchors dropped {align.anchors}, period shifts "
                    f"{align.period_shifts}): median {align.resid_med * 1e3:.0f} ms, "
                    f"{align.n_unpaired} unpaired video LEDs", "warning",
                    value=align.ref_resid_med, action=act)
        if align.resid_med > cfg.led_bad_resid:
            log.add(rec, "led_alignment_poor",
                    f"final map median residual {align.resid_med * 1e3:.0f} ms", "error")
            reasons.append("LED alignment failed")
        if align.n_unpaired > 10:
            log.add(rec, "led_unpaired",
                    f"{align.n_unpaired} video LEDs have no grid LED within "
                    f"{cfg.led_match_tol} s (possible LED-period aliasing)", "warning",
                    value=align.n_unpaired)
        for t_step, size, f_lo, f_hi in align.steps:
            in_gap = int(np.sum((frames > f_lo) & (frames < f_hi)))
            log.add(rec, "video_frame_discontinuity",
                    f"LED residual step of {size * 1e3:+.0f} ms near grid t={t_step:.0f} s "
                    f"(video frames dropped/duplicated); map split into segments; "
                    f"{in_gap} BORIS events inside the {(f_hi - f_lo) / fps:.1f} s LED gap "
                    "around the step are uncertain by up to the step size", "warning",
                    t=t_step, value=size, action="corrected")
        extrap = (frames < align.frame_first) | (frames > align.frame_last)
        if extrap.any():
            log.add(rec, "boris_extrapolated",
                    f"{extrap.sum()} BORIS events outside the LED sync range (map "
                    "extrapolated)", "warning", value=extrap.sum())
        tg = align.to_grid(frames)
        shift = np.abs(tg - align.ref_to_grid(frames))
        s["led_max_shift_vs_reference_s"] = round(float(shift.max()), 3) if len(shift) else 0.0
        s["led_shift_vs_ref"] = (shift, tg)
        last_annot = float(tg.max()) if len(tg) else np.nan

        # pair in video time (as annotated), then map to grid time
        def v2g(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = x.copy()
            fin = np.isfinite(x)
            out[fin] = align.to_grid(np.round(x[fin] * fps))
            return out

        chv, issues = pair_chases(bt, code)
        raw_durs = list(chv[:, 1] - chv[:, 0])
        ch = np.column_stack([v2g(chv[:, 0]), v2g(chv[:, 1])]) if len(chv) else _empty2()
        excl = []
        for it in issues:
            lo, hi, t0 = (float(v2g(np.array([it[k]]))[0]) for k in ("lo", "hi", "t"))
            log.add(rec, f"chase_{it['kind']}",
                    f"{it['kind'].replace('_', ' ')} at {t0:.2f} s grid ({it['t']:.3f} s "
                    f"video); chase state unknown in [{lo:.2f}, {hi:.2f}] s", "warning",
                    t=t0, action="dropped")
            excl.append((max(lo, t_start), min(hi, t_end)))
        excl_g = np.array(excl, dtype=float).reshape(-1, 2)
        s.update(n_boris_onsets=int(np.sum(code == CODE_ON)),
                 n_boris_offsets=int(np.sum(code == CODE_OFF)))

        # recording extent
        start_out = ch[:, 0] >= t_end
        end_out = (ch[:, 1] > t_end) & ~start_out
        pre = ch[:, 0] < t_start
        if start_out.any() or pre.any():
            log.add(rec, "chase_outside_recording",
                    f"{(start_out | pre).sum()} chases start outside recording", "warning",
                    action="dropped")
        if end_out.any():
            log.add(rec, "chase_past_recording_end",
                    f"{end_out.sum()} chases end after recording end; kept (censored at "
                    "window end)", "warning", t=float(ch[end_out, 0][0]))
        ch = ch[~(start_out | pre)]
        dur = ch[:, 1] - ch[:, 0]
        for c0, dv in zip(ch[dur > cfg.max_plausible_chase, 0], dur[dur > cfg.max_plausible_chase]):
            log.add(rec, "chase_implausibly_long", f"chase of {dv:.1f} s; kept", "warning",
                    t=c0, value=dv)
        # overlaps are impossible after pairing unless a negative LED step reorders time
        keep = np.ones(len(ch), dtype=bool)
        for i in range(1, len(ch)):
            prev = np.where(keep[:i])[0]
            if len(prev) and ch[i, 0] < ch[prev[-1], 1]:
                keep[i] = False
                log.add(rec, "chase_overlap", f"chase at {ch[i, 0]:.2f} s overlaps previous",
                        "error", t=ch[i, 0], action="dropped")
        chases_g = ch[keep]

        ct_ = tg[code == CODE_CONTACT]
        cout = (ct_ < t_start) | (ct_ > t_end)
        if cout.any():
            log.add(rec, "contact_outside_recording", f"{cout.sum()} contacts", "warning",
                    action="dropped")
        contacts_g = np.sort(ct_[~cout])
        n_in = int(_in_intervals(contacts_g, chases_g).sum())
        s["n_contacts_in_chase"] = n_in
        if grp >= 5 and int(row["chase_count"]) != len(chases_g):
            log.add(rec, "chase_count_vs_trial_summary",
                    f"{len(chases_g)} validated chases vs chase_count={int(row['chase_count'])}"
                    " in trial_summary", "info")
        if int(row["draw"]) == 1 and len(chases_g):
            log.add(rec, "draw_with_chases",
                    f"draw trial has {len(chases_g)} annotated chases; roles undefined",
                    "warning")

    # --- analysis window -----------------------------------------------------
    obs_end = t_end
    if cfg.max_time is not None:
        obs_end = min(obs_end, cfg.max_time)
    if cfg.clip_to_last_annotation and np.isfinite(last_annot):
        obs_end = min(obs_end, last_annot)
    s.update(obs_start=t_start, obs_end=obs_end, last_boris_event=last_annot)
    s["n_chases_after_window"] = int(np.sum(chases_g[:, 0] >= obs_end))
    s["n_chases_straddling_window_end"] = int(
        np.sum((chases_g[:, 0] < obs_end) & (chases_g[:, 1] > obs_end)))
    if "led_shift_vs_ref" in s:
        sh, tg_ = s.pop("led_shift_vs_ref")
        w = tg_ < obs_end
        s["led_max_shift_vs_reference_window_s"] = round(float(sh[w].max()), 3) if w.any() else 0.0
    if np.isfinite(last_annot) and last_annot > obs_end:
        n_after = s["n_chases_after_window"]
        if n_after:
            log.add(rec, "chases_after_window",
                    f"{n_after} chases start after window end {obs_end:.0f} s (last BORIS "
                    f"event {last_annot:.0f} s); outside analysis", "info", value=n_after)
    elif np.isfinite(last_annot) and last_annot < obs_end - 900:
        log.add(rec, "annotation_ends_early",
                f"last BORIS event at {last_annot:.0f} s, window ends {obs_end:.0f} s",
                "info", value=obs_end - last_annot)

    # --- window-level summary -----------------------------------------------
    win_ch = chases_g[chases_g[:, 0] < obs_end] if len(chases_g) else chases_g
    s.update(
        duration_window=round(obs_end - t_start, 1),
        n_chases=len(chases_g), n_chases_window=len(win_ch),
        chase_dur_median=round(float(np.median(win_ch[:, 1] - win_ch[:, 0])), 2)
        if len(win_ch) else np.nan,
        chase_dur_median_all=round(float(np.median(chases_g[:, 1] - chases_g[:, 0])), 2)
        if len(chases_g) else np.nan,
        n_contacts=len(contacts_g), n_contacts_window=int(np.sum(contacts_g < obs_end)),
        excluded_time_s=round(float(np.sum(excl_g[:, 1] - excl_g[:, 0])), 1),
    )
    for who in ("dom", "sub"):
        x = chirps[who]
        xw = x[(x >= t_start) & (x < obs_end)]
        s[f"n_chirps_{who}_window"] = len(xw)
        s[f"n_chirps_{who}_after_window"] = int(np.sum(x >= obs_end))
        s[f"frac_{who}_chirps_in_chase"] = (
            round(float(_in_intervals(xw, chases_g).mean()), 3)
            if len(xw) and len(chases_g) else np.nan
        )
    if len(chases_g):
        in_frac = float(np.sum(np.clip(np.minimum(win_ch[:, 1], obs_end) - win_ch[:, 0], 0, None)))
        s["frac_time_in_chase"] = round(in_frac / (obs_end - t_start), 3)

    # detector bounding boxes: unassigned detections, sender ambiguity
    s_bb, meta_bb = _bbox_qc(boxes, rec, ct[np.isfinite(ci)], chirps, raw, chases_g, t_start,
                             obs_end, cfg, log)
    s.update(s_bb)

    # size cross-check
    if fish_meta is not None:
        for who, fish in (("dom", s["dom_fish"]), ("sub", s["sub_fish"])):
            if (grp, fish) in fish_meta.index:
                lm = float(fish_meta.loc[(grp, fish), "length_mean"])
                s[f"fish_uid_{who}"] = f"g{grp}f{fish}:{fish_meta.loc[(grp, fish), 'ID']}"
                if abs(lm - s[f"size_{who}"]) > 0.05:
                    log.add(rec, "size_vs_id_meta",
                            f"{who} size {s[f'size_{who}']:.2f} vs id_meta mean length {lm:.2f}",
                            "info")

    # --- inclusion -------------------------------------------------------------
    if cfg.exclude_draws and int(row["draw"]) == 1:
        reasons.append("draw")
    if cfg.exclude_bad_recordings and rec in cfg.bad_recordings:
        reasons.append("bad_recording (EODfs too close, chirp assignment unreliable)")
    s["included"] = len(reasons) == 0
    s["exclusion_reason"] = "; ".join(reasons)

    trial = Trial(
        recording=rec, group=grp, dom_id=dom_id, sub_id=sub_id, dom_fish=s["dom_fish"],
        sub_fish=s["sub_fish"], t_start=t_start, t_end=t_end, chases=chases_g,
        contacts=contacts_g, chirps_dom=chirps["dom"], chirps_sub=chirps["sub"],
        rises_dom=rises["dom"], rises_sub=rises["sub"], eodf_t=grid, eodf_dom=eodf["dom"],
        eodf_sub=eodf["sub"], size_dom=s["size_dom"], size_sub=s["size_sub"],
        sex_dom=str(s["sex_dom"]), sex_sub=str(s["sex_sub"]),
        meta=dict(
            draw=s["draw"], exp_dom=s["exp_dom"], exp_sub=s["exp_sub"],
            fish_uid_dom=s.get("fish_uid_dom", f"g{grp}f{s['dom_fish']}"),
            fish_uid_sub=s.get("fish_uid_sub", f"g{grp}f{s['sub_fish']}"),
            last_boris_event=last_annot, led_method=s.get("led_method"),
            led_max_shift_vs_reference_window_s=s.get("led_max_shift_vs_reference_window_s"),
            led_resid_med_ms=s.get("led_resid_med_ms"), has_chases=len(chases_g) > 0,
            chirp_detector=det, **meta_bb,
        ),
        obs_start=t_start, obs_end=obs_end, exclude_intervals=excl_g,
    )
    return trial, s, raw_durs


def _boris_project(data_root: Path) -> str:
    """Behaviour codes and subjects defined in the BORIS project file, as text."""
    import json

    f = data_root / "Competition_experiment.boris"
    if not f.exists():
        return ""
    try:
        pj = json.loads(f.read_text())
    except (OSError, ValueError):
        return ""
    beh = pj.get("behaviors_conf", {})
    codes = ", ".join(f"{v.get('code')} = '{v.get('description')}' (key {v.get('key')})"
                      for _, v in sorted(beh.items()))
    subj = pj.get("subjects_conf", {})
    obs = ", ".join(sorted(pj.get("observations", {})))
    return (f"The BORIS project file {f.name} defines {codes} and "
            f"{'no subjects (subjects_conf is empty)' if not subj else f'subjects {subj}'}; "
            f"its own observations cover {obs} only")


def _global_flags(log: FlagLog, durs: np.ndarray, cfg: Config) -> None:
    """Dataset-wide flags that are not tied to a single trial."""
    proj = _boris_project(cfg.data_root)
    log.add("ALL", "chase_direction_unverified",
            "BORIS Subject is 'No focal subject' in every file: chaser identity is not "
            "annotated, so 'dominant chases subordinate' is assumed, not verified; "
            "reversed chases cannot be detected or dropped"
            + (f". {proj}: no subject was ever configured" if proj else ""), "warning")
    log.add("ALL", "boris_code_contradiction",
            "dataset README.txt says 1=onset, 2=offset, 3=contact; csv files contain codes "
            "0/1/2. Used 0=onset, 1=offset, 2=contact (0 always precedes 1)"
            + (f". {proj}, which confirms codes 0/1/2 ('aggression' = chase onset, 'passiv' "
               "= chase offset)" if proj else ""), "warning")
    log.add("ALL", "boris_readme_conversion_outdated",
            "README.txt describes a LED_on_time/factor=1.034141 conversion; the reference "
            "converter (LED_frames.npy + led_idxs.csv) is used instead", "info")
    if len(durs):
        short = durs[durs < 3.0]
        q = np.mean(np.abs(short - 0.5 * np.round(short / 0.5)) < 0.1) if len(short) else np.nan
        log.add("ALL", "chase_boundary_precision",
                f"min chase duration {durs.min():.2f} s; {q:.0%} of chases < 3 s have "
                "durations within 0.1 s of a multiple of 0.5 s (1.0/1.5/2.0/2.5 s clusters): "
                "offsets were likely set by stepping the video in ~0.5 s jumps, so chase "
                "boundary timing is uncertain on a ~0.5 s scale", "warning", value=q)
    log.add("ALL", "chirp_notes_refer_to_cnn",
            "chirp_notes.csv counts match chirp_times_cnn.npy (older CNN detector), not the "
            f"{cfg.chirp_detector} detections used here; its 'good' column is reported only",
            "info")
    log.add("ALL", "rises_available",
            "analysis/rise_idx.npy holds rise indices (spectrogram frames, ~0.33 s) per fish; "
            f"use_rises={cfg.use_rises}", "info")


def load_trials(
    cfg: Config, recordings: list[str] | None = None
) -> tuple[list[Trial], pd.DataFrame, pd.DataFrame]:
    """Load all trials listed in trial_summary.csv.

    Returns (included trials, flags table, per-trial summary for all trials with
    ``included`` and ``exclusion_reason`` columns).
    """
    ts = pd.read_csv(cfg.data_root / "trial_summary.csv", index_col=0)
    if recordings is not None:
        ts = ts[ts["recording"].isin(recordings)]
    log = FlagLog()
    fish_meta, cnn = _fish_meta(cfg.data_root), _cnn_notes(cfg.data_root)
    trials, rows, durs = [], [], []
    for _, row in ts.iterrows():
        rec = str(row["recording"])
        try:
            tr, s, rd = _build_trial(row, cfg, log, fish_meta, cnn)
        except FileNotFoundError as e:
            log.add(rec, "missing_file", str(e), "error", action="excluded_trial")
            rows.append(dict(recording=rec, group=int(row["group"]), included=False,
                             exclusion_reason=f"missing file: {Path(e.filename).name}"))
            continue
        rows.append(s)
        if int(row["group"]) >= 5 and int(row["draw"]) != 1:
            durs.extend(rd)
        if s["included"]:
            trials.append(tr)
        else:
            log.add(rec, "trial_excluded", s["exclusion_reason"], "info",
                    action="excluded_trial")
    _global_flags(log, np.asarray(durs), cfg)
    summary = pd.DataFrame(rows)
    inc = summary[summary.get("included", pd.Series(False, index=summary.index)).astype(bool)]
    if "n_contacts_in_chase" in inc and len(inc):
        n_c = int(inc["n_contacts"].sum())
        n_ic = int(inc["n_contacts_in_chase"].sum())
        log.add("ALL", "contacts_outside_chases",
                f"'contact' (BORIS code 2) is annotated outside chases: {n_ic} of {n_c} contacts "
                "in included trials fall inside a chase. It is a behaviour category disjoint "
                "from chasing, not contact during pursuit; inside chases the 'time since "
                "contact' control is time since a contact before the chase", "info",
                value=n_ic)
    return trials, log.frame(), summary


def events_table(trials: list[Trial]) -> pd.DataFrame:
    """Tidy event table: recording, group, event, sender, t, in_chase.

    event in {'chase_on','chase_off','contact','chirp','rise'}; sender in
    {'dom','sub','none'}. Chase/contact actor is not annotated -> 'none'.
    """
    parts = []
    for tr in trials:
        spec = [
            ("chase_on", "none", tr.chases[:, 0]),
            ("chase_off", "none", tr.chases[:, 1]),
            ("contact", "none", tr.contacts),
            ("chirp", "dom", tr.chirps_dom),
            ("chirp", "sub", tr.chirps_sub),
        ]
        if tr.rises_dom is not None:
            spec.append(("rise", "dom", tr.rises_dom))
        if tr.rises_sub is not None:
            spec.append(("rise", "sub", tr.rises_sub))
        for ev, snd, t in spec:
            if len(t) == 0:
                continue
            parts.append(pd.DataFrame(dict(
                recording=tr.recording, group=tr.group, event=ev, sender=snd, t=t,
                in_chase=tr.in_chase(t),  # onset <= t < offset; chase_off is False
            )))
    if not parts:
        return pd.DataFrame(columns=["recording", "group", "event", "sender", "t", "in_chase"])
    return pd.concat(parts, ignore_index=True).sort_values(["recording", "t"], kind="stable")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for v in r.values:
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else f"{v:g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_validation_report(
    trials: list[Trial], flags: pd.DataFrame, summary: pd.DataFrame, out_dir: Path,
    cfg: Config | None = None,
) -> Path:
    """Write flags.csv, trial_summary.csv and data_report.md into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags.to_csv(out_dir / "flags.csv", index=False)
    summary.to_csv(out_dir / "trial_summary.csv", index=False)

    L: list[str] = ["# Data validation report", ""]
    if cfg is not None:
        L += [(f"Data root: `{cfg.data_root}`. Chirp detector: `{cfg.chirp_detector}`. "
               f"LED alignment: `{cfg.led_alignment}`. Analysis window: "
               f"[recording start, {cfg.max_time if cfg.max_time else 'recording end'} s). "
               f"Duplicate-chirp merge window {cfg.chirp_dedup_window} s. "
               f"use_rises={cfg.use_rises}."), ""]
    inc = summary[summary["included"].astype(bool)]
    L += [(f"Trials in trial_summary.csv: {len(summary)}; included in hazard models: "
           f"{len(inc)} ({', '.join(inc['recording'])})."), ""]
    L += ["## Units and sampling", "",
          ("- Time: grid seconds (sample index / 20 kHz). Spectrogram frames (EODf tracks, "
           "rise indices) every ~0.328 s (groups 3, 5, 6) or ~0.123 s (group 4)."),
          (f"- Chirps: {cfg.chirp_detector if cfg is not None else 'detector'} times (s, box "
           "centre) with float sender ids equal to track ids."),
          ("- Chases/contacts: BORIS point events in video seconds at nominal 25 FPS; the "
           "video actually runs at ~25.86 frames per grid second, mapped to grid time via "
           "LED sync pulses every ~5 s (see led_* columns in trial_summary.csv)."),
          ("- Baseline EODf: per-fish trailing 60 s rolling median of the raw track, "
           "held (step-hold, last sample at or before each point) on a 1 s grid (Hz); "
           "NaN before the first tracked sample."),
          ("- Chirp sender QC from the detector's boxes (v1 chirpdetector_bboxes.csv, v2 the "
           "chirps of chirps_<det>.parquet): 'n_unassigned_distinct_window' = "
           "detections without track id (not analysed) >= 50 ms from any assigned "
           "chirp; 'frac_*_ambiguous_window' = share of chirps whose bounding box contains both "
           "fish's EODf."), ""]

    L += ["## Exclusions", ""]
    exc = summary[~summary["included"].astype(bool)][["recording", "group", "exclusion_reason"]]
    L += [_md_table(exc), ""]

    L += ["## Per-trial summary", ""]
    cols = [c for c in [
        "recording", "group", "included", "duration_window", "n_chases_window",
        "n_chases_straddling_window_end", "chase_dur_median", "n_contacts_window",
        "n_contacts_in_chase", "n_chirps_dom_window", "n_chirps_sub_window",
        "frac_sub_chirps_in_chase", "frac_time_in_chase", "n_unassigned_distinct_window",
        "frac_sub_ambiguous_window", "frac_dom_ambiguous_window", "eodf_cov_win_dom",
        "eodf_cov_win_sub", "delta_eodf_median", "led_resid_med_ms", "last_boris_event",
    ] if c in summary.columns]
    tab = summary[cols].copy()
    wf = flags[(flags["severity"] != "info") & (flags["recording"] != "ALL")]
    cats = wf.groupby("recording")["category"].agg(lambda x: ", ".join(sorted(set(x))))
    tab["warning_flags"] = tab["recording"].map(cats).fillna("")
    L += [_md_table(tab), ""]

    L += ["## Dataset-level flags", ""]
    for _, r in flags[flags["recording"] == "ALL"].iterrows():
        L.append(f"- `{r['category']}` ({r['severity']}): {r['detail']}")
    L += [""]

    L += ["## Flags by category", ""]
    agg = (flags.groupby(["category", "severity", "action"])
           .agg(n=("recording", "size"), recordings=("recording", lambda x: ", ".join(sorted(set(x)))))
           .reset_index().sort_values(["severity", "category"]))
    L += [_md_table(agg), ""]

    L += ["## Warnings and errors (detail)", ""]
    w = flags[(flags["severity"] != "info") & (flags["recording"] != "ALL")]
    for _, r in w.iterrows():
        tstr = f" (t={r['t']:.1f} s)" if np.isfinite(r["t"]) else ""
        L.append(f"- **{r['recording']}** `{r['category']}` [{r['action']}]{tstr}: {r['detail']}")
    L.append("")
    f = out_dir / "data_report.md"
    f.write_text("\n".join(L))
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Load and validate tube-competition data.")
    ap.add_argument("data_root", type=Path, nargs="?", default=Config().data_root)
    ap.add_argument("--out", type=Path, default=Config().out_dir)
    ap.add_argument("--max-time", type=float, default=None, help="s; <=0 disables the cap")
    ap.add_argument("--use-rises", action="store_true")
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--include-draws", action="store_true")
    ap.add_argument("--led-alignment", choices=["robust", "reference"], default=None)
    a = ap.parse_args()
    cfg = Config(data_root=a.data_root, out_dir=a.out).updated(
        max_time=a.max_time, led_alignment=a.led_alignment,
        use_rises=a.use_rises or None,
        exclude_bad_recordings=False if a.include_bad else None,
        exclude_draws=False if a.include_draws else None,
    )
    trials, flags, summary = load_trials(cfg)
    f = write_validation_report(trials, flags, summary, cfg.out_dir / "validation", cfg)
    ev = events_table(trials)
    print(f"{len(trials)} trials included; {len(ev)} events; "
          f"{(flags['severity'] != 'info').sum()} warning/error flags -> {f}")


if __name__ == "__main__":
    main()
