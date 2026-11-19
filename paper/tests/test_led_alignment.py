"""The video -> grid LED map (led_alignment) on synthetic LED trains and on the real trials.

Synthetic trains mimic the recordings: the grid switches the LED every 5 s
with a 10 s pause after every 59 flashes; the video runs at 25.862 frames per
grid second and starts 90 s before the grid. The two failure cases are the
ones found in the data: the video LED list misses the first two flashes
(2020-06-09) and the video loses 4.6 s of frames with one flash in them
(2020-05-13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyze_event_time_rates_trial import (
    converte_video_frames_to_grid_idx,
    load_and_convert_boris_events,
)
from led_alignment import LedAlignmentError, check_led_alignment, fit_led_alignment
from paper_config import DATA_ROOT

FPS = 25.862  # video frames per grid second
SR = 20_000  # grid sampling rate (Hz)
VIDEO_LEAD_S = 90.0  # the video starts this long before the grid
FRAME_S = 1 / FPS


def led_train(duration_s: float = 6 * 3600) -> np.ndarray:
    """Grid times (s) of the LED flashes: every 5 s, a 10 s pause after every 59."""
    t, out = 5.0, []
    while t < duration_s:
        out.append(t)
        t += 10.0 if len(out) % 59 == 0 else 5.0
    return np.array(out)


def video_frame(t_grid: np.ndarray, lost: tuple[float, float] | None = None) -> np.ndarray:
    """Video frame showing grid time t; ``lost`` = (start, length) of lost video, s."""
    t = np.asarray(t_grid, float) + VIDEO_LEAD_S
    if lost is not None:
        start, length = lost
        t = np.where(t_grid >= start + length, t - length, t)
    return np.round(t * FPS).astype(int)


def leds(lost: tuple[float, float] | None = None, missed_first: int = 0):
    """(video LED frames, grid LED sample indices); flashes inside a lost stretch are missing."""
    gt = led_train()
    seen = np.ones(len(gt), bool)
    if lost is not None:
        seen &= ~((gt > lost[0]) & (gt < lost[0] + lost[1]))
    seen[:missed_first] = False
    return video_frame(gt[seen], lost), np.round(gt * SR).astype(int)


def event_error(align, lost=None) -> np.ndarray:
    """Mapped minus true grid time (s) of events spread over the LED range."""
    t_true = np.linspace(100, 6 * 3600 - 100, 500)
    if lost is not None:  # events in the lost video stretch were never annotated
        t_true = t_true[(t_true < lost[0]) | (t_true > lost[0] + lost[1])]
    return align.to_grid(video_frame(t_true, lost)) - t_true


def test_clean_train_is_reproduced():
    frames, idx = leds()
    align = fit_led_alignment(frames, idx, SR)
    assert align.method == "robust"
    assert np.abs(event_error(align)).max() < FRAME_S
    assert align.resid_med < 0.015 and align.resid_p95 < 0.025
    check_led_alignment(align)
    # here the old first/last-LED line is right too
    old = converte_video_frames_to_grid_idx(video_frame(np.array([1000.0, 20000.0])), frames, idx)
    np.testing.assert_allclose(old / SR, [1000.0, 20000.0], atol=FRAME_S)


def test_video_misses_first_two_flashes():
    """2020-06-09: the first video LED is the third grid LED."""
    frames, idx = leds(missed_first=2)
    align = fit_led_alignment(frames, idx, SR)
    assert np.abs(event_error(align)).max() < FRAME_S
    check_led_alignment(align)
    # the first/last-LED line puts the first video LED on the first grid LED: 10 s early
    ref = fit_led_alignment(frames, idx, SR, method="reference")
    err = event_error(ref)
    assert err[0] < -9.0 and abs(err[-1]) < 0.1
    with pytest.raises(LedAlignmentError):
        check_led_alignment(ref)


def test_video_loses_frames_and_a_flash():
    """2020-05-13: 4.6 s of video (and one flash) lost at 4.9 h; first and last LED pair correctly."""
    gt = led_train()
    flash = gt[np.searchsorted(gt, 17_600.0)]
    lost = (flash - 2.0, 4.6)  # holds this flash and no other
    frames, idx = leds(lost=lost)
    assert len(idx) - len(frames) == 1
    align = fit_led_alignment(frames, idx, SR)
    assert len(align.seg_slope) == 2
    assert align.steps[0][1] == pytest.approx(4.6, abs=FRAME_S)
    assert np.abs(event_error(align, lost)).max() < FRAME_S
    check_led_alignment(align)
    # the line spreads the 4.6 s over the recording: late before the loss, early after
    ref = fit_led_alignment(frames, idx, SR, method="reference")
    err = event_error(ref, lost)
    assert err.max() > 3.0 and err.min() < -0.5
    with pytest.raises(LedAlignmentError):
        check_led_alignment(ref)


def test_extra_video_flash_at_the_start():
    """A spurious detection before the first flash (e.g. a reflection) does not shift the map."""
    frames, idx = leds()
    frames = np.concatenate([[frames[0] - 50], frames])
    align = fit_led_alignment(frames, idx, SR)
    assert np.abs(event_error(align)).max() < FRAME_S
    check_led_alignment(align)


def test_unrelated_led_lists_raise():
    """LED lists that do not belong together (e.g. files of another recording) fail loudly."""
    frames, idx = leds()
    rng = np.random.default_rng(1)
    other = np.cumsum(rng.uniform(4.0, 6.0, len(idx))) * SR
    with pytest.raises(LedAlignmentError, match="does not fit"):
        check_led_alignment(fit_led_alignment(frames, other.astype(int), SR), "synthetic")


def write_trial(path, frames, idx, events_true: dict, lost=None):
    """A trial directory as the loader expects it (BORIS at 25 FPS nominal)."""
    np.save(path / "LED_frames.npy", frames)
    pd.DataFrame(idx).to_csv(path / "led_idxs.csv", header=False, index=False)
    rows = []
    for code, t in events_true.items():
        rows += [dict(Behavior=code, **{"Start (s)": f / 25.0}) for f in video_frame(t, lost)]
    df = pd.DataFrame(rows).sort_values("Start (s)")
    df["FPS"] = 25.0
    df.to_csv(path / "2020-01-01.csv", index=False)


def test_loader_returns_corrected_times(tmp_path):
    """load_and_convert_boris_events keeps its API and returns the LED-map times."""
    frames, idx = leds(missed_first=2)
    on = np.array([200.0, 5000.0, 10_000.0])
    off, contact = on + 4.0, np.array([300.0, 9000.0])
    write_trial(tmp_path, frames, idx, {0: on, 1: off, 2: contact})
    contact_t, on_off_t, led_idx, led_frames = load_and_convert_boris_events(
        tmp_path, "2020-01-01-10_00", sr=SR)
    np.testing.assert_allclose(on_off_t, np.column_stack([on, off]), atol=FRAME_S)
    np.testing.assert_allclose(contact_t, contact, atol=FRAME_S)
    np.testing.assert_array_equal(led_idx, idx)
    np.testing.assert_array_equal(led_frames, frames)


def test_loader_raises_on_implausible_alignment(tmp_path):
    frames, idx = leds()
    write_trial(tmp_path, frames, idx, {0: np.array([200.0]), 1: np.array([204.0])})
    rng = np.random.default_rng(2)
    other = (np.cumsum(rng.uniform(4.0, 6.0, len(idx))) * SR).astype(int)
    pd.DataFrame(other).to_csv(tmp_path / "led_idxs.csv", header=False, index=False)
    with pytest.raises(LedAlignmentError):
        load_and_convert_boris_events(tmp_path, "2020-01-01-10_00", sr=SR)


# --- the real trials ----------------------------------------------------------
# trials of trial_summary.csv with LED sync files (2020-05-20 has them too but is not a trial)
TRIAL_SUMMARY = DATA_ROOT / "trial_summary.csv"
ANNOTATED = sorted(
    r for r in pd.read_csv(TRIAL_SUMMARY)["recording"]
    if (DATA_ROOT / r / "LED_frames.npy").exists() and (DATA_ROOT / r / "led_idxs.csv").exists()
) if TRIAL_SUMMARY.exists() else []
MISALIGNED = {"2020-05-13-10_00", "2020-06-09-10_00"}
needs_data = pytest.mark.skipif(not TRIAL_SUMMARY.exists(), reason="recordings not mounted")


@needs_data
def test_annotated_trials_found():
    assert len(ANNOTATED) == 20 and MISALIGNED <= set(ANNOTATED)


@pytest.mark.parametrize("rec", ANNOTATED)
def test_real_trial_map_fits_and_differs_from_the_line_only_where_expected(rec):
    d = DATA_ROOT / rec
    frames = np.load(d / "LED_frames.npy")
    idx = pd.read_csv(d / "led_idxs.csv", header=None).iloc[:, 0].to_numpy()
    align = fit_led_alignment(frames, idx, SR)
    check_led_alignment(align, rec)
    assert align.resid_med < 0.012
    f = np.linspace(frames[0], frames[-1], 2000)
    shift = np.abs(align.to_grid(f) - converte_video_frames_to_grid_idx(f, frames, idx) / SR)
    if rec in MISALIGNED:
        assert shift.max() > 3.0
    else:
        assert shift.max() < 0.03
