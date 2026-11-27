"""Diagnostic plots of the BORIS/LED fix: the old first/last-LED line vs the LED map.

Until 2026-11-19 the paper converted video frames to recording time with a
straight line through the first and the last LED flash
(``converte_video_frames_to_grid_idx``). Since then
``load_and_convert_boris_events`` uses ``led_alignment.fit_led_alignment``.
This script shows, for the 20 trials with BORIS annotations and LED files,
where the two differ and why, and checks the result against the loser's
chirping around chases.

Writes numbered PNGs, ``README.txt`` and ``led_alignment_trials.csv`` (the
per-trial numbers) to ``paper/output/led_alignment/`` (not a run folder; the
chirp check reads the primary detections, ``paper_config.PRIMARY``). Figure 03
reads the start of the 2020-06-09 video with ``ffmpeg``.

    paper/run.sh plot_led_alignment.py          # all figures
    paper/run.sh plot_led_alignment.py 03 07    # some figures
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_event_time_rates_trial import (  # noqa: E402
    bad_recordings,
    converte_video_frames_to_grid_idx,
    load_and_convert_boris_events,
    load_boris,
)
from chirps import load_chirp_table  # noqa: E402
from led_alignment import LedAlignment, _nn_resid, fit_led_alignment  # noqa: E402
from paper_config import DATA_ROOT, PAPER, PRIMARY  # noqa: E402

OUT = PAPER / "output" / "led_alignment"
SR = 20_000  # grid sampling rate (Hz), as the paper scripts pass it
WINDOW_S = 3 * 3600.0  # the analysed part of each trial (first 3 h)
DETECTIONS = PRIMARY  # primary detections, for the chirp check
BAD = ("2020-05-13-10_00", "2020-06-09-10_00")
OLD_COLOR, NEW_COLOR = "tab:red", "tab:blue"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10,
                     "axes.titlesize": 10, "text.usetex": False})


@dataclass
class TrialLeds:
    """LED pairs, both maps and the converted BORIS events of one trial."""

    recording: str
    analysed: bool  # used by the paper's chase/contact analyses
    lose_id: float
    led_frames: np.ndarray
    led_idx: np.ndarray
    align: LedAlignment
    events: pd.DataFrame  # event, t_video_s, t_old_s, t_new_s

    @property
    def date(self) -> str:
        return self.recording[:10]

    @property
    def grid_t(self) -> np.ndarray:
        return self.led_idx / SR

    def old_map(self, frames: np.ndarray) -> np.ndarray:
        return converte_video_frames_to_grid_idx(frames, self.led_frames, self.led_idx) / SR

    def led_residuals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(grid time of each flash, old residual, new residual), s.

        Residual = mapped time of a video LED minus the time of the same flash
        in the grid recording. The flash is the grid LED nearest to the new
        map (its residual is below 0.1 s for every video LED here), so the
        old residual is not wrapped at half the LED period as a nearest-LED
        residual would be.
        """
        new, partner = _nn_resid(self.align.to_grid(self.led_frames), self.grid_t)
        old = self.old_map(self.led_frames) - self.grid_t[partner]
        return self.grid_t[partner], old, new


def load_trial(row: pd.Series) -> TrialLeds:
    rec = row["recording"]
    d = DATA_ROOT / rec
    led_frames = np.load(d / "LED_frames.npy")
    led_idx = pd.read_csv(d / "led_idxs.csv", header=None).iloc[:, 0].to_numpy()
    _, _, video_on_off, video_contact, fps = load_boris(d, rec)
    contact_new, on_off_new, _, _ = load_and_convert_boris_events(d, rec, SR)
    video_on_off = np.asarray(video_on_off, float).reshape(-1, 2)
    on_off_new = np.asarray(on_off_new, float).reshape(-1, 2)
    tr = TrialLeds(rec, bool(row["group"] >= 5 and row["draw"] != 1 and rec not in bad_recordings),
                   float(row["lose_ID"]), led_frames, led_idx,
                   fit_led_alignment(led_frames, led_idx, SR), pd.DataFrame())
    parts = []
    for event, video, new in (("chase onset", video_on_off[:, 0], on_off_new[:, 0]),
                              ("chase offset", video_on_off[:, 1], on_off_new[:, 1]),
                              ("contact", np.asarray(video_contact, float), np.asarray(contact_new))):
        frames = np.array(np.round(video * fps), dtype=int)
        parts.append(pd.DataFrame(dict(event=event, t_video_s=video, t_old_s=tr.old_map(frames),
                                       t_new_s=new)))
    tr.events = pd.concat(parts, ignore_index=True)
    tr.events["shift_s"] = tr.events["t_new_s"] - tr.events["t_old_s"]
    return tr


def load_trials() -> list[TrialLeds]:
    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    has_led = [(DATA_ROOT / r / "LED_frames.npy").exists() and (DATA_ROOT / r / "led_idxs.csv").exists()
               for r in ts["recording"]]
    return [load_trial(row) for _, row in ts[has_led].iterrows()]


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def fmt_s(x: float) -> str:
    """Seconds as '9.91 s' or '12 ms'."""
    return f"{x:.2f} s" if abs(x) >= 0.1 else f"{x * 1e3:.0f} ms"


def trial_grid(n: int, ncols: int = 5, w: float = 3.2, h: float = 2.2):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.9), squeeze=False,
                             layout="constrained")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)


def label_trial(ax, tr: TrialLeds) -> str:
    tag = "" if tr.analysed else " (not analysed)"
    return f"{tr.date}{tag}"


# --- 01 ----------------------------------------------------------------------
def fig01_led_residuals(trials: list[TrialLeds]) -> None:
    fig, axes = trial_grid(len(trials))
    for ax, tr in zip(axes, trials):
        t, old, new = tr.led_residuals()
        ax.axhline(0, color="0.7", lw=0.8)
        ax.plot(t / 3600, old, ".", color=OLD_COLOR, ms=2.5, label="old: line through first and last LED")
        ax.plot(t / 3600, new, ".", color=NEW_COLOR, ms=1.5, label="new: LED map (led_alignment)")
        ax.set_ylim(-11, 5)
        bad = tr.recording in BAD
        ax.set_title(f"{label_trial(ax, tr)}\nmedian: old {fmt_s(np.median(np.abs(old)))}, "
                     f"new {fmt_s(np.median(np.abs(new)))}", color="tab:red" if bad else "black")
    for ax in axes[::5]:
        ax.set_ylabel("LED residual (s)")
    for ax in axes[-5:]:
        ax.set_xlabel("recording time (h)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=2, frameon=False, markerscale=5)
    fig.suptitle("The old map misses the LED flashes by seconds in 2020-05-13 and 2020-06-09; the new map "
                 "fits all 20 trials\n(residual = mapped video LED minus the same flash in the recording; "
                 "panel titles: median |residual|). "
                 "Isolated dots, the same under both maps (about 0.6 % of flashes):\nmostly single flashes "
                 "whose grid index (led_idxs.csv) is 0.1-0.6 s early while the video agrees with the logged "
                 "LED time (led_times.csv). The fit leaves them out.", x=0.01, ha="left", fontsize=11)
    save(fig, "01_led_residual_old_vs_new.png")


# --- 02 ----------------------------------------------------------------------
def fig02_gap_pattern_0609(trials: list[TrialLeds]) -> None:
    tr = next(t for t in trials if t.recording == "2020-06-09-10_00")
    fps = tr.align.fps_effective
    grid_iv = np.diff(tr.grid_t)
    video_iv = np.diff(tr.led_frames) / fps
    n = 200
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, layout="constrained")
    k = np.arange(1, n + 1)
    a1.plot(k, grid_iv[:n], "o-", color="black", ms=3, lw=0.8)
    a2.plot(k, video_iv[:n], "o-", color="tab:orange", ms=3, lw=0.8)
    pauses_g = np.flatnonzero(grid_iv[:n] > 7.5) + 1
    pauses_v = np.flatnonzero(video_iv[:n] > 7.5) + 1
    for p in pauses_g:
        a1.annotate(f"flash {p}", (p, 10.1), ha="center", va="bottom", fontsize=9)
        a2.axvline(p, color="black", ls=":", lw=0.8)
    for p in pauses_v:
        a2.annotate(f"flash {p}", (p, 10.1), ha="center", va="bottom", fontsize=9, color="tab:orange")
    a1.set_ylabel("interval to next\nflash (s)")
    a2.set_ylabel("interval to next\nflash (s)")
    a1.set_title("Grid recording (led_idxs.csv): flashes every 5 s, a 10 s pause after every 59th")
    a2.set_title("Video (LED_frames.npy, frames / 25.86 fps): the same pauses, but 2 flashes earlier "
                 "(dotted: grid pauses)")
    for a in (a1, a2):
        a.set_ylim(3, 12)
    a2.set_xlabel("flash number in the list")
    fig.suptitle("2020-06-09: the video's LED list starts with the 3rd flash of the recording, so the old map "
                 "pairs\nvideo flash 1 with grid flash 1 (10 s apart) at the start", x=0.01, ha="left",
                 fontsize=11)
    save(fig, "02_led_gap_pattern_2020-06-09.png")


# --- 03 ----------------------------------------------------------------------
def read_gray_frames(video: Path, n: int) -> np.ndarray:
    """The first n frames of a video as uint8 grey images (ffmpeg)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True, check=True).stdout
    w, h = (int(x) for x in probe.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", str(n), "-f",
                          "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def image_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Whole-pixel shift (dy, dx) of image b against image a (phase correlation)."""
    cross = np.fft.fft2(a.astype(float)) * np.conj(np.fft.fft2(b.astype(float)))
    corr = np.abs(np.fft.ifft2(cross / (np.abs(cross) + 1e-12)))
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = a.shape
    return int((dy + h // 2) % h - h // 2), int((dx + w // 2) % w - w // 2)


def fig03_video_0609(trials: list[TrialLeds]) -> None:
    tr = next(t for t in trials if t.recording == "2020-06-09-10_00")
    first_detected = int(tr.led_frames[0])
    # video frames of grid flashes 1 and 2, from the new map
    expected = (tr.grid_t[:2] - tr.align.seg_icpt[0]) / tr.align.seg_slope[0]
    frames = read_gray_frames(DATA_ROOT / tr.recording / "2020-06-09_0.avi", first_detected + 200)
    lo = int(expected[0]) - 60
    on1 = int(np.round(expected[0]))

    def spot(f_on: int, r: int = 20) -> tuple[int, int]:
        """Centre of the pixels near the strongest change that brighten by > 100 when the LED switches on."""
        d = frames[f_on + 4].astype(int) - frames[f_on - 4].astype(int)
        y, x = np.unravel_index(np.argmax(d), d.shape)
        ys, xs = np.nonzero(d[y - r:y + r + 1, x - r:x + r + 1] > 100)
        return int(round(y - r + ys.mean())), int(round(x - r + xs.mean()))

    s_early, s_late = spot(on1), spot(first_detected + 1)
    # the scene itself (LED off: before flash 1, before flash 3) did not move
    cam_dy, cam_dx = image_shift(frames[on1 - 5], frames[first_detected - 5])

    def brightness(s: tuple[int, int]) -> np.ndarray:
        y, x = s
        return frames[lo:, y - 3:y + 4, x - 3:x + 4].mean(axis=(1, 2))

    fig = plt.figure(figsize=(11, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1))
    h, w = frames.shape[1:]
    y0, y1 = max(0, s_late[0] - 45), min(h, s_late[0] + 45)
    x0, x1 = max(0, s_late[1] - 110), min(w, s_late[1] + 50)
    for col, (f, what) in enumerate(((on1 + 4, f"frame {on1 + 4}: grid flash 1 (not in the LED list)"),
                                     (first_detected + 5, f"frame {first_detected + 5}: grid flash 3 "
                                      "(first entry of the LED list)"))):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frames[f, y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255,
                  extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        ax.plot(s_early[1], s_early[0], "o", mfc="none", mec="tab:green", ms=18, mew=1.5)
        ax.plot(s_late[1], s_late[0], "s", mfc="none", mec="tab:orange", ms=18, mew=1.5)
        ax.set_title(what)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
    ax = fig.add_subplot(gs[1, :])
    t = np.arange(lo, lo + len(frames) - lo)
    ax.plot(t, brightness(s_early), color="tab:green", label=f"early LED spot (circle, x={s_early[1]})")
    ax.plot(t, brightness(s_late), color="tab:orange", label=f"later LED spot (square, x={s_late[1]})")
    for i, f in enumerate(expected):
        ax.axvline(f, color="black", ls=":", lw=1)
        ax.annotate(f"grid flash {i + 1}", (f, 262), ha="center", fontsize=9)
    for f in tr.led_frames[:2]:
        ax.annotate("in LED list", (f, -30), ha="center", fontsize=9, color="tab:orange")
    ax.set_ylim(-45, 285)
    ax.set_xlabel("video frame")
    ax.set_ylabel("brightness (0-255)")
    ax.legend(loc="center right", frameon=False)
    fig.suptitle("2020-06-09: the first two flashes are in the video, but the LED was then "
                 f"{s_late[1] - s_early[1]} pixels further left;\nit was moved before flash 3 "
                 f"(shift of the whole image: {cam_dx} x, {cam_dy} y pixels), and the LED detector only "
                 "watched the later spot", x=0.01, ha="left", fontsize=11)
    save(fig, "03_video_led_moved_2020-06-09.png")


# --- 04 ----------------------------------------------------------------------
def frame_clock(rec: str) -> np.ndarray:
    """The camera's per-frame clock (<date>_0_times.dat), s."""
    f = DATA_ROOT / rec / f"{rec[:10]}_0_times.dat"
    return pd.to_timedelta(pd.read_csv(f, header=None, dtype=str).iloc[:, 0]).dt.total_seconds().to_numpy()


def fig04_lost_frames_0513(trials: list[TrialLeds]) -> None:
    tr = next(t for t in trials if t.recording == "2020-05-13-10_00")
    fps = tr.align.fps_effective
    t_grid, _, _ = tr.led_residuals()
    lag = tr.led_frames / fps - t_grid
    lag = lag - lag[0]
    clock = frame_clock(tr.recording)
    jumps = np.diff(clock)
    k = int(np.argmax(jumps))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=(1.7, 1), layout="constrained")
    a1.plot(t_grid / 3600, lag, ".", color="black", ms=2, label="LED pairs")
    a1.plot(t_grid[[0, -1]] / 3600, lag[[0, -1]], "--", color=OLD_COLOR, lw=1.5,
            label="what the old map assumes:\na straight line from first to last pair")
    a1.axvline(WINDOW_S / 3600, color="0.6", lw=0.8)
    a1.annotate("analysed: first 3 h ", (WINDOW_S / 3600, 0.95), ha="right", fontsize=9, color="0.4")
    # the old map's error at the end of the analysed window
    line_3h = np.interp(WINDOW_S, t_grid[[0, -1]], lag[[0, -1]])
    a1.annotate("", xy=(WINDOW_S / 3600, line_3h), xytext=(WINDOW_S / 3600, 0),
                arrowprops=dict(arrowstyle="<->", color=OLD_COLOR))
    a1.annotate(f" old map {-line_3h:.1f} s late here", (WINDOW_S / 3600, line_3h / 2),
                color=OLD_COLOR, fontsize=9, va="center")
    a1.set_ylim(-5.4, 1.2)
    a1.set_xlabel("recording time (h)")
    a1.set_ylabel("video clock minus recording clock (s)\n(frame / 25.86 fps; 0 at the first flash)")
    step = tr.align.steps[0][1] if tr.align.steps else np.nan
    a1.set_title(f"video falls {step:.1f} s behind at {t_grid[np.argmax(np.diff(lag) < -1) + 1] / 3600:.1f} h")
    a1.legend(loc="lower left", frameon=False)
    sl = slice(k - 150, k + 150)
    a2.plot(np.arange(len(jumps))[sl], jumps[sl], ".-", color="black", ms=3, lw=0.8)
    a2.set_xlabel("video frame")
    a2.set_ylabel("time to the next frame (s)")
    a2.set_title(f"camera's frame clock: one {jumps[k]:.2f} s gap at frame {k + 1}")
    a2.ticklabel_format(axis="x", style="plain", useOffset=False)
    a2.tick_params(axis="x", labelrotation=20)
    fig.suptitle("2020-05-13: the video lost 4.6 s of frames (and one LED flash) at 4.9 h; the old map "
                 "spreads this\nloss over the whole recording, so its times are up to 2.3 s late in "
                 "the first 3 h", x=0.01, ha="left", fontsize=11)
    save(fig, "04_video_lost_frames_2020-05-13.png")


# --- 05 ----------------------------------------------------------------------
def fig05_event_shifts(trials: list[TrialLeds]) -> None:
    fig, axes = trial_grid(len(trials))
    for ax, tr in zip(axes, trials):
        ev = tr.events
        ax.plot(ev["t_new_s"] / 3600, ev["shift_s"], ".", ms=3, color="black")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(WINDOW_S / 3600, color="0.8", lw=0.8)
        mx = ev["shift_s"].abs().max()
        ax.set_title(f"{label_trial(ax, tr)}\nmax |shift| {fmt_s(mx)}",
                     color="tab:red" if tr.recording in BAD else "black")
        if mx < 0.1:
            ax.set_ylim(-0.03, 0.03)
    for ax in axes[::5]:
        ax.set_ylabel("new minus old (s)")
    for ax in axes[-5:]:
        ax.set_xlabel("recording time (h)")
    others = max(t.events["shift_s"].abs().max() for t in trials if t.recording not in BAD)
    fig.suptitle("Every annotated event (chase onsets, offsets, contacts) moves by seconds in 2020-05-13 "
                 "(earlier) and 2020-06-09 (later),\nand by at most "
                 f"{fmt_s(others)} (less than one video frame, 39 ms) in all other trials "
                 "(their y axis: ±30 ms)", x=0.01, ha="left", fontsize=11)
    save(fig, "05_event_shift_per_trial.png")


# --- 06 ----------------------------------------------------------------------
def fig06_max_shift_bars(trials: list[TrialLeds]) -> None:
    rows = []
    for tr in trials:
        w = tr.events["t_new_s"] < WINDOW_S
        rows.append((tr.date, tr.analysed, tr.recording in BAD, tr.events.loc[w, "shift_s"].abs().max()))
    df = pd.DataFrame(rows, columns=["date", "analysed", "bad", "max_shift"])
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    x = np.arange(len(df))
    colors = [OLD_COLOR if b else ("0.45" if a else "0.75") for a, b in zip(df["analysed"], df["bad"])]
    ax.bar(x, df["max_shift"], color=colors)
    for xi, v in zip(x, df["max_shift"]):
        ax.annotate(fmt_s(v), (xi, v * 1.15), ha="center", fontsize=8)
    ax.axhline(1 / 25.86, color="black", ls="--", lw=0.8)
    ax.annotate("one video frame (39 ms)", (-0.5, 1 / 25.86 * 1.15), ha="left", fontsize=9)
    legend = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OLD_COLOR, "0.45", "0.75")]
    ax.legend(legend, ["first/last-LED line was wrong", "other analysed trials",
                       "not analysed by the paper (draw or EODfs too close)"],
              loc="upper left", frameon=False, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 40)
    ax.set_xticks(x, [d + ("" if a else "\n(not analysed)") for d, a in zip(df["date"], df["analysed"])],
                  rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("max |new minus old| of the events\nin the first 3 h (s, log scale)")
    ax.set_title("The fix moves events by seconds in two trials and by less than one video frame "
                 "in the other 18", loc="left")
    save(fig, "06_max_shift_per_trial.png")


# --- 07, 08 ------------------------------------------------------------------
PETH_HALF_S, PETH_BIN_S = 30.0, 1.0


def loser_chirps(tr: TrialLeds) -> np.ndarray:
    df = load_chirp_table(DATA_ROOT / tr.recording, DETECTIONS)
    return np.sort(df.loc[df["keep"] & (df["track_id"] == tr.lose_id), "t_center_s"].to_numpy())


def peth(chirps: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chirp rate (Hz) around centres: chirps per bin per centre / bin width."""
    edges = np.arange(-PETH_HALF_S, PETH_HALF_S + PETH_BIN_S / 2, PETH_BIN_S)
    counts = np.zeros(len(edges) - 1)
    for c in centres:
        counts += np.histogram(chirps - c, edges)[0]
    return 0.5 * (edges[1:] + edges[:-1]), counts / (len(centres) * PETH_BIN_S)


def fig_chirps_around(trials: list[TrialLeds], event: str, name: str, title: str) -> None:
    analysed = [t for t in trials if t.analysed]
    others = [t for t in analysed if t.recording not in BAD]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True, layout="constrained")
    for ax, tr in zip(axes, [next(t for t in analysed if t.recording == r) for r in BAD]):
        ch = loser_chirps(tr)
        ev = tr.events[(tr.events["event"] == event) & (tr.events["t_new_s"] < WINDOW_S)]
        for col, color, lab in (("t_old_s", OLD_COLOR, "old times"), ("t_new_s", NEW_COLOR, "new times")):
            x, r = peth(ch, ev[col].to_numpy())
            ax.step(x, r, where="mid", color=color, lw=1.5, label=lab)
        ax.set_title(f"{tr.date}: {len(ev)} {event}s, {len(ch)} loser chirps")
    xs, rates, n_ev = None, [], 0
    for tr in others:
        ev = tr.events[(tr.events["event"] == event) & (tr.events["t_new_s"] < WINDOW_S)]
        if len(ev) == 0:
            continue
        xs, r = peth(loser_chirps(tr), ev["t_new_s"].to_numpy())
        rates.append(r * len(ev))
        n_ev += len(ev)
    axes[2].step(xs, np.sum(rates, axis=0) / n_ev, where="mid", color="black", lw=1.5)
    max_other = max(t.events["shift_s"].abs().max() for t in others)
    axes[2].set_title(f"other {len(others)} analysed trials pooled: {n_ev} {event}s\n"
                      f"(new times; old ones differ by at most {fmt_s(max_other)})")
    for ax in axes:
        ax.axvline(0, color="0.5", ls=":", lw=1)
        ax.set_xlabel(f"time relative to {event} (s)")
        ax.set_ylim(bottom=0)
    for ax in axes[:2]:
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylabel("loser chirp rate (Hz)")
    fig.suptitle(f"{title}\n(loser chirps, {DETECTIONS}; 1 s bins; events of the first 3 h; "
                 "each panel has its own y axis)", x=0.01, ha="left", fontsize=11)
    save(fig, name)


def fig07_chirps_offsets(trials):
    fig_chirps_around(trials, "chase offset", "07_loser_chirps_around_chase_offsets.png",
                      "Independent check: with the new times the loser's chirp peak sits at the chase "
                      "end in both trials, as in the other trials;\nwith the old times the peak moves: "
                      "1-3 s before the end in 2020-05-13 (old times up to 2.3 s late), 5-10 s after "
                      "it in 2020-06-09 (old times 5-10 s early)")


def fig08_chirps_onsets(trials):
    fig_chirps_around(trials, "chase onset", "08_loser_chirps_around_chase_onsets.png",
                      "Around chase starts: with the new times 2020-06-09 shows the dip before and the "
                      "rise after the start like the other trials\n(with the old, 5-10 s early times, "
                      "they come 5-10 s after the start). In 2020-05-13 old and new look alike:\na shift "
                      "of 0-2.3 s is small against this slow response, so this panel is inconclusive")


FIGURES = {"01": fig01_led_residuals, "02": fig02_gap_pattern_0609, "03": fig03_video_0609,
           "04": fig04_lost_frames_0513, "05": fig05_event_shifts, "06": fig06_max_shift_bars,
           "07": fig07_chirps_offsets, "08": fig08_chirps_onsets}


# --- table and README --------------------------------------------------------
def trial_table(trials: list[TrialLeds]) -> pd.DataFrame:
    """Per trial: the LED maps and how far the events move (new minus old)."""
    rows = []
    for tr in trials:
        _, old, new = tr.led_residuals()
        ev = tr.events
        w = ev["t_new_s"] < WINDOW_S
        rows.append(dict(
            recording=tr.recording, analysed=tr.analysed, led_map=tr.align.method,
            n_led_video=tr.align.n_led_video, n_led_grid=tr.align.n_led_grid,
            led_resid_median_old_ms=1e3 * np.median(np.abs(old)),
            led_resid_median_new_ms=1e3 * np.median(np.abs(new)),
            led_resid_p95_new_ms=1e3 * tr.align.resid_p95,
            n_events=len(ev), n_events_3h=int(w.sum()),
            max_abs_shift_s=ev["shift_s"].abs().max(),
            max_abs_shift_3h_s=ev.loc[w, "shift_s"].abs().max(),
            median_shift_3h_s=ev.loc[w, "shift_s"].median(),
        ))
    return pd.DataFrame(rows)


README = """\
LED alignment fix (BORIS chase/contact times -> recording time), diagnostic plots
Made by paper/scripts/plot_led_alignment.py; numbers per trial in led_alignment_trials.csv.

What was wrong
  The paper converted video frames to recording time with a straight line through
  the first and the last LED flash. That is only right if the first and last video
  flash are the first and last flash of the recording and no video was lost between.
  - 2020-06-09: the LED was moved by about 30 pixels before its 3rd flash (the camera
    did not move). The LED detector watched the later spot, so the video's LED list
    (LED_frames.npy) starts with the 3rd flash. The line therefore paired video flash
    1 with recording flash 1, 10 s apart: event times were 10 s early at the start,
    shrinking to 0 at the end (5-10 s early in the analysed first 3 h).
  - 2020-05-13: the video lost 4.6 s of frames at 4.9 h (the camera's frame clock
    jumps 4.87 s at frame 458116), and one LED flash with them. First and last flash
    pair correctly, but the line spreads the 4.6 s over the whole recording: events
    were 0-2.3 s late in the first 3 h.
  Since the fix, load_and_convert_boris_events uses led_alignment.fit_led_alignment
  (the reviewed map of the hazard analysis, moved to paper/scripts/led_alignment.py):
  it pairs video and recording flashes by the pattern of the 10 s pauses and splits
  the map where video was lost. It raises an error if the map does not fit the LEDs.

Figures
  01  LED residual (mapped video flash minus the same flash in the recording) over
      the recording, old line vs new map, all 20 trials. Old: off by seconds in
      2020-05-13 and 2020-06-09; new: ~10 ms median (one frame = 39 ms) everywhere.
      Isolated dots in every trial (about 0.6 % of flashes, the same under both maps)
      are mostly single flashes whose recording index (led_idxs.csv) is 0.1-0.6 s early
      while the video agrees with the logged LED time (led_times.csv); a few are
      spurious video detections. The fit leaves them out.
  02  2020-06-09, raw LED lists: the 10 s pauses come 2 flashes earlier in the video
      list than in the recording, i.e. the video list misses the first 2 flashes.
  03  2020-06-09, video frames: the first two flashes are in the video, ~30 pixels
      left of where the LED sits from flash 3 on; brightness at both spots.
  04  2020-05-13, raw LED pairs: the video clock falls 4.6 s behind at 4.9 h (lost
      frames, with one flash); the camera's frame clock shows the gap. The dashed line
      is what the old map assumed.
  05  Shift of every annotated event (new minus old) over time, per trial.
  06  Maximum |shift| of the events in the first 3 h per trial (log axis): seconds in
      the two trials, below one video frame (max 24 ms) in the other 18.
  07  Independent check: loser chirp rate around chase ends, old vs new times, in the
      two trials, next to the other 13 analysed trials pooled. With the new times the
      peak sits at the chase end as in the other trials; with the old times it moves
      (2020-05-13: 1-3 s before the end; 2020-06-09: 5-10 s after it).
  08  The same around chase starts: supports the fix in 2020-06-09; inconclusive in
      2020-05-13 (old and new look alike, the response is too slow for a 0-2.3 s shift).

Per trial (max_abs_shift_3h_s = max |new minus old| of the events in the first 3 h):
"""


def write_readme(table: pd.DataFrame) -> None:
    cols = ["recording", "analysed", "led_map", "led_resid_median_old_ms", "led_resid_median_new_ms",
            "n_events_3h", "max_abs_shift_3h_s", "median_shift_3h_s"]
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "led_alignment_trials.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        text = README + table[cols].round(3).to_string(index=False) + "\n"
    (OUT / "README.txt").write_text(text)
    print("wrote", OUT / "README.txt", "and led_alignment_trials.csv")


def main() -> None:
    which = sys.argv[1:] or list(FIGURES)
    trials = load_trials()
    for k in which:
        FIGURES[k](trials)
    write_readme(trial_table(trials))


if __name__ == "__main__":
    main()
