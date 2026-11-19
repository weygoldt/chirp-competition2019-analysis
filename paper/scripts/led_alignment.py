"""Video frame -> grid (recording) time from the LED sync pulses.

The one place that converts video frames to recording time. Chases and
contacts were annotated in the video (BORIS); an LED, switched by the grid
recording, flashes every 5 s with a 10 s pause about every 5 min. Its flashes
are found in the video (``LED_frames.npy``, video frames) and in the grid
recording (``led_idxs.csv``, sample indices). This module fits the map from
video frames to grid seconds on these pairs. The paper's BORIS loader
(``analyze_event_time_rates_trial.load_and_convert_boris_events``) and the
hazard data layer (``hazard/hz_data.py``) both use :func:`fit_led_alignment`.

Moved from ``hazard/hz_data.py`` (2026-11-19) without changes to the method;
only the hazard ``Config`` argument became keyword arguments with the same
defaults (``hz_config`` reads its defaults from here).

Why not a line through the first and the last LED (the paper's converter
until 2026-11-19, ``converte_video_frames_to_grid_idx``): it assumes that the
first and the last video LED are the first and the last grid LED and that no
video frames were lost in between. In 2020-06-09 the LED was moved in the
first seconds of the recording, the video LED detector watched the new spot
and missed the first two flashes, so the first video LED is the third grid
LED (events up to 10 s early). In 2020-05-13 the video lost 4.6 s of frames
at 4.9 h, together with one flash; the line spreads that loss over the whole
recording (events up to 2.3 s late in the first 3 h). See
``plot_led_alignment.py`` and ``paper/output/led_alignment/``.

Residuals are in seconds: mapped video LED minus the nearest grid LED. One
video frame is 1/25.86 s = 39 ms, so a correct map has residuals within
about +-20 ms (median ~10 ms, 95th percentile ~19 ms).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Defaults of fit_led_alignment (hz_config.Config uses the same values).
MATCH_TOL_S = 1.0  # max |residual| to pair a video LED with a grid LED
INLIER_TOL_S = 0.1  # max |residual| for a pair to enter the fit
STEP_TOL_S = 0.1  # residual step that splits the map (frame drop)
MAX_ANCHOR_SHIFT = 3  # LEDs dropped at either end in the anchor search

# Bounds of check_led_alignment. A correct map has a median residual of ~10 ms
# and a 95th percentile of ~19 ms (frame quantisation); the first/last-LED
# line of the two misaligned trials had medians of 1.27 and 1.35 s.
MAX_RESID_MED_S = 0.05  # = hz_config.led_bad_resid, at which hz_data drops a trial
MAX_RESID_P95_S = 0.1  # = INLIER_TOL_S: 95 % of the video LEDs within 0.1 s


class LedAlignmentError(ValueError):
    """The LED map does not fit the LED pairs; converted times would be wrong."""


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
    n_matched: int  # video LEDs within inlier_tol of a grid LED
    n_unpaired: int  # video LEDs farther than match_tol from any grid LED
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
    led_frames: np.ndarray,
    led_idx: np.ndarray,
    sr: float,
    *,
    method: str = "robust",
    match_tol: float = MATCH_TOL_S,
    inlier_tol: float = INLIER_TOL_S,
    step_tol: float = STEP_TOL_S,
    max_anchor_shift: int = MAX_ANCHOR_SHIFT,
) -> LedAlignment:
    """Map video frames to grid seconds from LED sync pulses.

    ``led_frames``: video frames of the LED flashes; ``led_idx``: grid sample
    indices of the LED flashes; ``sr``: grid sampling rate (Hz).

    The reference map (first/last LED, as in converte_video_frames_to_grid_idx)
    is always computed for comparison; ``method="reference"`` returns it. In
    "robust" mode: (1) anchors are searched over up to ``max_anchor_shift``
    dropped LEDs at each end, minimising the median nearest-LED residual (fixes
    the slope; the reference is kept unless a candidate is > 5 ms better); (2)
    the whole-LED-period ambiguity of the intercept is resolved with the gap
    fingerprint of the LED train and the line is refit by least squares; (3)
    video LEDs are paired with their nearest grid LED (|residual| <
    ``match_tol``) and steps in the residuals larger than ``step_tol``
    (dropped/duplicated video frames) split the map into segments; (4) each
    segment repeats (2) on its own LEDs.
    """
    if method not in ("robust", "reference"):
        raise ValueError("method must be 'robust' or 'reference'")
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

    if method == "reference":
        return LedAlignment(
            seg_bounds=np.zeros(0), seg_slope=np.array([ref[0]]),
            seg_icpt=np.array([ref[1]]), method="reference", anchors=(0, 0, 0, 0),
            n_matched=int(np.sum(ref_res < inlier_tol)),
            n_unpaired=int(np.sum(ref_res > match_tol)), resid_med=ref_med,
            resid_p95=ref_p95, steps=[], period_shifts=[0], **common,
        )

    best, best_med, best_anchor = ref, ref_med, (0, 0, 0, 0)
    k = max_anchor_shift
    for i in range(k + 1):
        for j in range(k + 1):
            for a in range(k + 1):
                for b in range(k + 1):
                    cand = _anchor_map(lf[i], lf[-1 - a], gt[j], gt[-1 - b])
                    med = float(np.median(score(*cand)))
                    if med < best_med - 0.005:
                        best, best_med, best_anchor = cand, med, (i, j, a, b)

    period = float(np.median(np.diff(gt)))
    tol = inlier_tol

    def nn_lsq(f: np.ndarray, slope: float, icpt: float) -> tuple[float, float]:
        """Least squares on nearest-LED pairs (period-agnostic)."""
        r, near = _nn_resid(slope * f + icpt, gt)
        ok = np.abs(r) < match_tol
        if ok.sum() < 3:
            return slope, icpt
        p1 = np.polyfit(f[ok], gt[near[ok]], 1)
        return float(p1[0]), float(p1[1])

    slope, icpt = nn_lsq(lf, *best)

    # segments at residual steps (frame drops), each with its own line
    r, _ = _nn_resid(slope * lf + icpt, gt)
    pm = np.abs(r) < match_tol
    fm, rm = lf[pm], r[pm]
    cuts = _find_steps(rm, step_tol)
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

    name = "robust"
    if best_anchor != (0, 0, 0, 0):
        name += "_anchor"
    if any(shifts):
        name += "_period"
    if cuts:
        name += f"_{len(cuts) + 1}seg"
    fin = np.abs(_nn_resid(slopes_a[seg] * lf + icpts_a[seg], gt)[0])
    return LedAlignment(
        seg_bounds=bounds, seg_slope=slopes_a, seg_icpt=icpts_a, method=name,
        anchors=best_anchor, n_matched=int(np.sum(fin < tol)),
        n_unpaired=int(np.sum(fin > match_tol)), resid_med=float(np.median(fin)),
        resid_p95=float(np.percentile(fin, 95)), steps=steps,
        period_shifts=shifts, **common,
    )


def check_led_alignment(
    align: LedAlignment, recording: str = "",
    max_resid_med: float = MAX_RESID_MED_S, max_resid_p95: float = MAX_RESID_P95_S,
) -> None:
    """Raise :class:`LedAlignmentError` if the map does not fit its LEDs.

    The residual of a video LED is its mapped time minus the time of the
    nearest grid LED. A correct map has a median |residual| of ~10 ms and a
    95th percentile of ~19 ms (one frame = 39 ms); the bounds are 50 ms and
    100 ms.
    """
    if align.resid_med > max_resid_med or align.resid_p95 > max_resid_p95:
        raise LedAlignmentError(
            f"{recording}: the LED map ({align.method}) does not fit the LED pairs: median "
            f"|residual| {align.resid_med * 1e3:.0f} ms (bound {max_resid_med * 1e3:.0f} ms), "
            f"95th percentile {align.resid_p95 * 1e3:.0f} ms (bound {max_resid_p95 * 1e3:.0f} "
            f"ms); {align.n_led_video} video LEDs, {align.n_led_grid} grid LEDs. Converted "
            "BORIS times would be shifted; check LED_frames.npy and led_idxs.csv."
        )
