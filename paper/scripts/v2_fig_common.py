"""Shared helpers of the three pipeline-v2 figures (paper environment).

Paths of the v2 production run and its evaluations, the cached spectrogram
snippets written by ``v2_fig_extract.py``, and one visual language for all
three figures (see "Figure style" in ``paper/v2_figures.md``):

- one palette from seaborn's ``mako`` (cool: spectrograms, detectors,
  densities) and its companion ``flare`` (warm: the two fish, the human
  expert), plus greys for references;
- spectrograms in ``mako`` (capped below white, so that white annotations
  stand out): power in dB re 1 V^2 (the snippets' unclipped ``power_db``),
  display levels the 2nd and 98th percentile of each panel's crop
  (``CLIP_PCT``; no colour bars since round 3, 2026-11-25); a frequency axis
  offset from the image, no time axis but white scale bars; the detector's
  RGB input keeps its own 0-1 levels;
- on the spectrograms no colour for fish identity (the manuscript's figure
  standard): the frequency tracks as thin black lines on their bands,
  labelled EOD$f_1$ (higher-frequency fish of the panel) and EOD$f_2$,
  dashed where interpolated (gap-filled samples, and gaps of more than 1 s
  that the assignment bridges by linear interpolation); detections as solid
  white boxes, all in one line width (``BOX_LW``), with the predicted emitter
  EOD$f$ as a white circle: the circle on a fish's track shows the
  assignment (round 3: no line style per fish); light grey solid,
  unassigned; white dashed and translucent, removed by post-processing;
  light grey dotted, below the operating threshold; black triangles above
  the image: hand labels;
- annotation arrows with one gentle curvature and a gap at both ends.

Environment: ``paper/.venv`` (imports ``style``, which needs seaborn).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import paper_config

CM = 1 / 2.54
FIG_WIDTH = 17.4 * CM  # width of the manuscript's full-width figures

CHIRP_DATA = paper_config.CHIRP_DATA  # env CHIRP_DATA (default /mnt/data1/chirp_v2)
# the detections follow PAPER_DETECTIONS (paper_config): the production run,
# its dataset patch with the trial fish's track gaps filled (interior gaps
# 10-300 s with an EODf jump <= 15 Hz, linear interpolation), and that patch
# at another operating threshold. Per run: directory, chirp table, and the run
# whose tracks_gapfill.npz holds the patched trial-fish tracks (None: the
# wavetracker tracks as they are)
# Both pipelines' runs (paper_config.PIPELINES: v2 = the released three-window
# detector, its chirp tables those of the *_recal1 runs; sw = the single-window
# detector since 2026-11-27); PAPER_DETECTIONS selects one.
RUNS = paper_config.RUNS
DETECTIONS = paper_config.DETECTIONS
# the production run itself (track_gaps/, qc/, windows/ live only there)
SOURCE_RUN = paper_config.pipeline_path("source_run")
RUN = CHIRP_DATA / "detections" / RUNS[DETECTIONS][0]
PARQUET = RUNS[DETECTIONS][1]
TRACKS_RUN = (CHIRP_DATA / "detections" / RUNS[DETECTIONS][2]) if RUNS[DETECTIONS][2] else None
GAPFILL = TRACKS_RUN is not None


def _operating_threshold() -> float:
    """The run's operating threshold (meta.json of its recordings; one value)."""
    values = {json.loads(m.read_text())["operating_threshold"]
              for m in sorted(RUN.glob("*/meta.json"))}
    if len(values) != 1:
        msg = f"{RUN}: operating thresholds {sorted(values)}"
        raise ValueError(msg)
    return float(values.pop())


OPERATING = _operating_threshold()  # the run's operating threshold


def _assign_max_gap_s() -> float:
    """The assignment's ``max_gap_s`` (meta.json of the run's recordings; one
    value): inside gaps up to this long a track is live and linearly
    interpolated (gridtools ``Track.frequency_at``)."""
    values = {json.loads(m.read_text())["assignment"]["config"]["max_gap_s"]
              for m in sorted(RUN.glob("*/meta.json"))}
    if len(values) != 1:
        msg = f"{RUN}: assignment max_gap_s {sorted(values)}"
        raise ValueError(msg)
    return float(values.pop())


ASSIGN_MAX_GAP_S = _assign_max_gap_s()


def tile_duration_s() -> float:
    """The detector's tile length of the run (meta.json of its recordings; one value)."""
    values = {json.loads(m.read_text())["models"]["detector"]["tiling"]["tile_duration_s"]
              for m in sorted(RUN.glob("*/meta.json"))}
    if len(values) != 1:
        msg = f"{RUN}: tile durations {sorted(values)}"
        raise ValueError(msg)
    return float(values.pop())
REAL_EVAL = CHIRP_DATA / "eval" / "real_labels_365"
REAL_LABELS = REAL_EVAL / "labels"  # the hand labels (detector-independent)
# the selected pipeline's detector ("dualhead": the production detector, either
# pipeline) and the YOLO26s reference: their simulated-test evaluations
PP_EVAL = {
    "dualhead": paper_config.pipeline_path("sim_eval"),
    "yolo26s": CHIRP_DATA / "runs" / "yolo26s_sim_v4_r1" / "eval_postprocess",
}
BENCHMARK = paper_config.pipeline_path("benchmark")
BENCHMARK_MODEL = paper_config.PIPELINE["benchmark_model"]  # the detector's name in it
REAL_SESSION = paper_config.pipeline_path("real_session")
# The chirp-parameter decoder's evaluation (the pipeline's decoder; both are
# calibrated without the octave-error template of the validation split,
# chirpdetector train params-recalibrate, 2026-11-27).
DECODER_EVAL = paper_config.pipeline_path("decoder") / "eval"
# Template library v4: chirp-chamber templates whose zero-crossing frequency
# trace reads twice the baseline EODf during the chirp (octave error; plateau /
# baseline 1.006-1.014, every other big template 0.2-0.78): 61 (train split),
# 1598 (val; left out of the decoder's calibration) and 391 (test; left out of
# the decoder panels of Fig. 10). The library fix follows after submission.
OCTAVE_TEMPLATES = (61, 1598, 391)
TRIAL_SUMMARY = paper_config.DATA_ROOT / "trial_summary.csv"

SNIPPETS = paper_config.PROCESSED / "v2_figures"
FIGURES = paper_config.FIGURES

TOLERANCE_HZ = 20.0  # assignment tolerance of the run

# ---------------------------------------------------------------------------
# palette: mako (cool) + flare (warm), greys
# ---------------------------------------------------------------------------
_MAKO = sns.color_palette("mako", as_cmap=True)
_FLARE = sns.color_palette("flare", as_cmap=True)
BLUE = "#3e4d93"  # mako 0.35
TEAL = "#3eb4ad"  # mako 0.70
PEACH = "#eca479"  # flare 0.05
CORAL = "#e2655c"  # flare 0.30
ROSE = "#c2456a"  # flare ~0.5
GREY = "#a0a0a0"
INK = "k"
MUTED = "#6e6e6e"  # secondary text

# spectrograms: mako without its near-white top, so that white marks stand out
SPEC_CMAP = LinearSegmentedColormap.from_list("mako_capped", _MAKO(np.linspace(0, 0.93, 256)))
SPEC_BG = SPEC_CMAP(0.0)  # the spectrogram's black, for key swatches
# display levels of every spectrogram panel (power in dB re 1 V^2): the
# CLIP_PCT-th and (100 - CLIP_PCT)-th percentile of the panel's crop
CLIP_PCT = 2.0
# densities: light mint -> the detector blue (the same mako, reversed)
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "mako_density", _MAKO(np.linspace(0.92, 0.35, 256)))

# the fish of a panel: never told apart by colour or line style (round 3,
# 2026-11-25). Boxes and emitter markers white (contrast on mako), every
# assigned box solid: its emitter marker on a fish's track shows the
# assignment; the tracks thin black lines on their (saturated) bands
BOX_COLOR = "white"
BOX_LS = "-"
TRACK_COLOR = "k"
TRACK_LW = 0.6
# interpolated track samples (gap-filled run, and gaps the assignment bridges)
FILLED_LS = (0, (2.5, 1.5))
TRACK_GAP_S = 1.0  # samples further apart than this are a gap in the track
# boxes that are not assigned output
GREY_BOX = "#c8c8c8"
UNASSIGNED_COLOR = GREY_BOX  # solid, with its emitter marker
DROPPED_COLOR = BOX_COLOR  # removed by post-processing: dashed, translucent
DROPPED_LS = (0, (3.0, 2.0))
DROPPED_ALPHA = 0.55
MISSED_COLOR = GREY_BOX  # below the operating threshold (dotted)
MISSED_LS = (0, (1.0, 1.2))
BOX_LW = 0.6  # one box line width in every panel, thumbnail and key
ANNOT_FS = "small"  # 6.7 pt at the 8-pt base; nothing in the figures is smaller
ANNOT = {"color": "white", "fontsize": ANNOT_FS}
DARK_BACK = {"bbox": dict(boxstyle="square,pad=0.12", fc=SPEC_BG, ec="none", alpha=0.8)}

# performance figure: one colour per model, the same in every panel
MODEL_COLORS = {
    "dualhead": BLUE,  # RT-DETR with a second head for the emitter's EODf
    "yolo26s": TEAL,  # YOLO26s, boxes only (reference)
    "human": CORAL,  # human expert (blind benchmark)
    "edge": GREY,  # assignment from the box position only
}
MODEL_LABELS = {
    "dualhead": "RT-DETR dual-head",
    "yolo26s": "YOLO26s (boxes only)",
    "human": "human expert",
}

# annotation arrows: one gentle curvature, a gap at both ends
ARC = 0.2


def arrowprops(color="white", rad: float = ARC, shrink_a: float = 2.0,
               shrink_b: float = 2.5) -> dict:
    return dict(arrowstyle="-|>", color=color, lw=0.7, mutation_scale=5,
                shrinkA=shrink_a, shrinkB=shrink_b, connectionstyle=f"arc3,rad={rad}")


ARROW = arrowprops()


def annotate(ax, text, xy, xytext, *, color="white", rad: float = ARC, ha="center",
             va="center", fontsize=ANNOT_FS, shrink_a=2.0, shrink_b=2.5, **kw):
    """Text with a gently curved arrow that stops short of its target."""
    return ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, color=color,
                       fontsize=fontsize,
                       arrowprops=arrowprops(color, rad, shrink_a, shrink_b), **kw)


def rc_tweaks() -> None:
    """Settings on top of ``style.set_light_style`` (paper defaults otherwise).

    ``set_light_style`` sets a marker edge width of 5 pt (for its scatter
    plots); reset it, and use the slightly shorter ticks of the newer paper
    figures.
    """
    mpl.rcParams.update(
        {
            "lines.markeredgewidth": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.linewidth": 0.8,
            "legend.handlelength": 1.4,
            "legend.fontsize": ANNOT_FS,
            "axes.titlesize": "medium",
            "axes.titlepad": 3,
        }
    )


def axes_cm(fig, x: float, y: float, w: float, h: float):
    """Axes at (x, y) cm from the figure's lower left corner, w x h cm."""
    W, H = fig.get_size_inches() / CM
    return fig.add_axes((x / W, y / H, w / W, h / H))


def fig_cm(fig):
    """Transform: figure centimetres -> display (for annotations in cm)."""
    W, H = fig.get_size_inches() / CM
    return mpl.transforms.Affine2D().scale(1 / W, 1 / H) + fig.transFigure


def letters_cm(fig, axes_and_offsets):
    """``style.letter_subplots`` with offsets in cm from each axes' upper left corner."""
    from style import letter_subplots

    W, H = fig.get_size_inches() / CM
    axs, xo, yo, lets = [], [], [], []
    for ax, let, dx, dy in axes_and_offsets:
        bb = ax.get_position()
        axs.append(ax)
        lets.append(let)
        xo.append(-dx / (bb.width * W))
        yo.append(1 + dy / (bb.height * H))
    letter_subplots(axs, letters=lets, xoffset=xo, yoffset=yo)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@cache
def detections(recording: str) -> pd.DataFrame:
    """All stored rows of one recording of the run (PAPER_DETECTIONS)."""
    return pd.read_parquet(RUN / recording / PARQUET)


@cache
def analysed_recordings() -> frozenset[str]:
    """The 33 competitions of the manuscript (no draws, 4 excluded trials)."""
    s = pd.read_csv(TRIAL_SUMMARY, index_col=0)
    bad = ["2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00"]
    return frozenset(s[(s.draw != 1) & ~s.recording.isin(bad)].recording)


def load_snippet(name: str) -> dict:
    """A cached snippet: ``img`` (3, F, T) in [0, 1], times, freqs, tracks, meta."""
    path = SNIPPETS / f"{name}.npz"
    if not path.exists():
        msg = f"{path} missing: run scripts/v2_fig_extract.py snippets (upgrade env)"
        raise FileNotFoundError(msg)
    z = np.load(path)
    snip = {k: z[k] for k in z.files if k != "meta"}
    snip["meta"] = json.loads(str(z["meta"]))
    off = snip["track_offsets"]
    snip["tracks"] = {
        int(tid): (snip["track_times_s"][a:b], snip["track_freqs_hz"][a:b])
        for tid, a, b in zip(snip["track_ids"], off[:-1], off[1:], strict=True)
    }
    # interpolated samples of the gap-filled tracks (absent in older caches)
    filled = snip.get("track_filled", np.zeros(len(snip["track_times_s"]), bool))
    snip["filled"] = {
        int(tid): filled[a:b].astype(bool)
        for tid, a, b in zip(snip["track_ids"], off[:-1], off[1:], strict=True)
    }
    return snip


def window(snip: dict, t0: float, t1: float) -> pd.DataFrame:
    """Detections of the snippet's recording whose box overlaps [t0, t1]."""
    d = detections(snip["meta"]["recording"])
    return d[(d.t_end_s > t0) & (d.t_start_s < t1)].copy()


def track_median(snip: dict, tid: int, t0: float, t1: float) -> float:
    tt, ff = snip["tracks"][tid]
    m = (tt >= t0) & (tt <= t1)
    return float(np.median(ff[m]))


def fish_order(snip: dict, t0: float, t1: float) -> list[int]:
    """Track ids live in [t0, t1], highest median frequency first (EOD$f_1$)."""
    med = {}
    for tid, (tt, ff) in snip["tracks"].items():
        m = (tt >= t0) & (tt <= t1)
        if m.sum() >= 2:
            med[tid] = float(np.median(ff[m]))
    return sorted(med, key=lambda k: -med[k])


def fish_styles(snip: dict, t0: float, t1: float) -> tuple[dict, list[int]]:
    """The panel's fish: {track id: k} (k = 0 for EOD$f_1$, the highest
    median frequency in [t0, t1], 1 for EOD$f_2$, ...) and the track ids in
    that order. Boxes no longer differ by fish (round 3); the dict says which
    tracks are drawn and which boxes count as assigned in the panel."""
    order = fish_order(snip, t0, t1)
    return {tid: k for k, tid in enumerate(order)}, order


# ---------------------------------------------------------------------------
# spectrograms and detections
# ---------------------------------------------------------------------------
def spec_crop(snip: dict, t0: float, t1: float, f0: float, f1: float, channel,
              key: str = "img"):
    """Crop of ``snip[key]`` (``img``: the detector's 0-1 image; ``power_db``:
    the unclipped spectrogram in dB), with a margin of 50 ms and 10 Hz."""
    t, f = snip["times_s"], snip["freqs_hz"]
    it = (t >= t0 - 0.05) & (t <= t1 + 0.05)
    jf = (f >= f0 - 10) & (f <= f1 + 10)
    dt, df = t[1] - t[0], f[1] - f[0]
    ext = (t[it][0] - dt / 2, t[it][-1] + dt / 2, f[jf][0] - df / 2, f[jf][-1] + df / 2)
    if channel == "rgb":
        img = np.stack([snip[key][c][np.ix_(jf, it)].astype(float) for c in range(3)], -1)
    else:
        img = snip[key][channel][np.ix_(jf, it)].astype(float)
    return img, ext


def power_crop_db(snip: dict, t0: float, t1: float, f0: float, f1: float,
                  channel: int = 0) -> np.ndarray:
    """The unclipped power [dB re 1 V^2] inside [t0, t1] x [f0, f1] (no margin)."""
    if "power_db" not in snip:
        msg = (f"{snip['meta']['recording']}: snippet has no power_db; "
               "rerun scripts/v2_fig_extract.py snippets")
        raise KeyError(msg)
    t, f = snip["times_s"], snip["freqs_hz"]
    it, jf = (t >= t0) & (t <= t1), (f >= f0) & (f <= f1)
    return snip["power_db"][channel][np.ix_(jf, it)].astype(float)


def clip_levels(values: np.ndarray, q: float = CLIP_PCT) -> tuple[float, float]:
    """Display levels: the q-th and (100 - q)-th percentile of ``values``."""
    if not 0 <= q < 50:
        msg = f"clip percentile {q} not in [0, 50)"
        raise ValueError(msg)
    lo, hi = (float(v) for v in np.percentile(values, [q, 100 - q]))
    if not lo < hi:
        msg = f"display levels {lo:.2f}..{hi:.2f}: the crop is flat"
        raise ValueError(msg)
    return lo, hi


def show_power(ax, snip: dict, t0: float, t1: float, f0: float, f1: float, *,
               channel: int = 0, levels: tuple[float, float] | None = None,
               q: float = CLIP_PCT, t_ref: float | None = None):
    """Draw one channel's power [dB re 1 V^2], time relative to ``t_ref`` (default t0).

    ``levels`` (dB); None: the q-th / (100 - q)-th percentile of this crop
    (``clip_levels``). Returns the image (for a colour bar) and the levels.
    """
    t_ref = t0 if t_ref is None else t_ref
    if levels is None:
        levels = clip_levels(power_crop_db(snip, t0, t1, f0, f1, channel), q)
    img, ext = spec_crop(snip, t0, t1, f0, f1, channel, key="power_db")
    im = ax.imshow(img, extent=(ext[0] - t_ref, ext[1] - t_ref, ext[2], ext[3]),
                   aspect="auto", origin="lower", cmap=SPEC_CMAP, vmin=levels[0],
                   vmax=levels[1], interpolation="antialiased", rasterized=True)
    ax.set_xlim(t0 - t_ref, t1 - t_ref)
    ax.set_ylim(f0, f1)
    return im, levels


def axes_extents(fig, named_axes: list[tuple[str, object]]) -> str:
    """A table of each named axes' extent in figure cm (to check alignment):
    left, right, bottom, top, width, height."""
    W, H = fig.get_size_inches() / CM
    lines = [f"{'axes':<14}{'left':>7}{'right':>7}{'bottom':>8}{'top':>7}{'width':>7}"
             f"{'height':>8}  [cm; figure {W:.2f} x {H:.2f}]"]
    for name, ax in named_axes:
        bb = ax.get_position()
        lines.append(f"{name:<14}{bb.x0 * W:7.3f}{bb.x1 * W:7.3f}{bb.y0 * H:8.3f}"
                     f"{bb.y1 * H:7.3f}{bb.width * W:7.3f}{bb.height * H:8.3f}")
    return "\n".join(lines)


def show_rgb(ax, snip: dict, t0: float, t1: float, f0: float, f1: float,
             t_ref: float | None = None) -> None:
    """The detector's input: channels 205/51/13 ms as R/G/B on its own 0-1 levels."""
    t_ref = t0 if t_ref is None else t_ref
    img, ext = spec_crop(snip, t0, t1, f0, f1, "rgb")
    ax.imshow(np.clip(img, 0, 1), extent=(ext[0] - t_ref, ext[1] - t_ref, ext[2], ext[3]),
              aspect="auto", origin="lower", interpolation="antialiased", rasterized=True)
    ax.set_xlim(t0 - t_ref, t1 - t_ref)
    ax.set_ylim(f0, f1)


def track_segments(tt: np.ndarray, ff: np.ndarray, filled: np.ndarray, *,
                   gap_s: float = TRACK_GAP_S, bridge_s: float | None = None):
    """Split a track into its detected (solid) and interpolated (dashed) parts.

    Detected samples further apart than ``gap_s`` break the solid line. The
    ``filled`` samples of a gap-filled track are drawn dashed, joined to the
    neighbouring detections; with ``bridge_s``, so is every gap of at most
    ``bridge_s`` between two detections (the straight line between them: the
    linear interpolation the assignment uses inside gaps up to its
    ``max_gap_s``). Returns (t_solid, f_solid, t_dashed, f_dashed) with NaN
    between pieces, and the dashed pieces as a list of (t_start, t_end).
    """
    tt, ff, fl = np.asarray(tt, float), np.asarray(ff, float), np.asarray(filled, bool)
    ts, fs = np.where(fl, np.nan, tt), np.where(fl, np.nan, ff)
    step = np.diff(tt)
    gaps = np.flatnonzero(step > gap_s)
    t_solid = np.insert(ts, gaps + 1, np.nan)
    f_solid = np.insert(fs, gaps + 1, np.nan)
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    # filled runs, each joined to the detections on either side
    if fl.any():
        edges = np.flatnonzero(np.diff(np.r_[0, fl.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            lo, hi = max(a - 1, 0), min(b, len(tt) - 1)
            pieces.append((tt[lo:hi + 1], ff[lo:hi + 1]))
    if bridge_s is not None:
        for i in gaps:
            if step[i] <= bridge_s and not (fl[i] or fl[i + 1]):
                pieces.append((tt[i:i + 2], ff[i:i + 2]))
    pieces.sort(key=lambda p: p[0][0])
    t_dash = np.concatenate([np.r_[p[0], np.nan] for p in pieces]) if pieces else np.zeros(0)
    f_dash = np.concatenate([np.r_[p[1], np.nan] for p in pieces]) if pieces else np.zeros(0)
    return t_solid, f_solid, t_dash, f_dash, [(float(p[0][0]), float(p[0][-1])) for p in pieces]


def draw_tracks(
    ax, snip: dict, fish: dict, t0: float, t1: float, *, t_ref: float | None = None,
    lw: float = TRACK_LW, color=TRACK_COLOR, alpha: float = 1.0,
    bridge_s: float | None = None,
) -> list[tuple[int, float, float]]:
    """The tracks of the fish in ``fish`` (track ids), thin black lines.

    Solid where detected, dashed where interpolated (``track_segments``;
    ``bridge_s``: also gaps between detections of up to that length).
    Returns the dashed pieces that reach into [t0, t1]: (track id, start, end).
    """
    t_ref = t0 if t_ref is None else t_ref
    dashed = []
    margin = 0.5 if bridge_s is None else max(0.5, bridge_s)
    for tid, (tt, ff) in snip["tracks"].items():
        if tid not in fish:
            continue
        # with bridges: the detections just outside the view, for the gaps
        # that reach into it
        m = (tt >= t0 - margin) & (tt <= t1 + margin)
        if m.sum() < 2:
            continue
        fl = snip.get("filled", {}).get(tid, np.zeros(len(tt), bool))[m]
        ts, fs, td, fd, pieces = track_segments(tt[m], ff[m], fl, bridge_s=bridge_s)
        ax.plot(ts - t_ref, fs, color=color, lw=lw, alpha=alpha, zorder=2,
                solid_capstyle="butt")
        if len(td):
            ax.plot(td - t_ref, fd, color=color, lw=lw, alpha=alpha, zorder=2,
                    ls=FILLED_LS, dash_capstyle="butt")
        dashed += [(int(tid), a, b) for a, b in pieces if b >= t0 and a <= t1]
    return dashed


def box_style(row: pd.Series, fish: dict) -> dict | None:
    """Edge colour, line style and alpha of a detection (None: not drawn).

    ``fish``: the panel's fish (``fish_styles``; only its keys are used): a
    box assigned to one of them is a solid white box.
    """
    if row.pp_drop_reason in ("harmonic", "duplicate") and row.operating:
        return {"ec": DROPPED_COLOR, "ls": DROPPED_LS, "alpha": DROPPED_ALPHA,
                "kind": row.pp_drop_reason}
    if not row.operating:
        return None
    tid = row.track_id
    if np.isfinite(tid) and int(tid) in fish:
        return {"ec": BOX_COLOR, "ls": BOX_LS, "alpha": 1.0, "kind": "assigned"}
    return {"ec": UNASSIGNED_COLOR, "ls": "-", "alpha": 1.0, "kind": "unassigned"}


def fish_box(ax, row: pd.Series, t_ref: float, fish: dict, **kw) -> None:
    """A detection assigned to one of ``fish``: a solid white box."""
    if not (np.isfinite(row.track_id) and int(row.track_id) in fish):
        msg = f"chirp {row.name}: track {row.track_id} is not a fish of the panel"
        raise ValueError(msg)
    draw_box(ax, row, t_ref, BOX_COLOR, ls=BOX_LS, **kw)


def dropped_box(ax, row: pd.Series, t_ref: float, **kw) -> None:
    """A detection removed by post-processing: dashed, translucent, no marker."""
    draw_box(ax, row, t_ref, DROPPED_COLOR, ls=DROPPED_LS, alpha=DROPPED_ALPHA,
             marker=False, **kw)


def draw_box(ax, row: pd.Series, t_ref: float, ec=BOX_COLOR, ls="-", lw: float = BOX_LW,
             marker: bool = True, zorder: int = 5, ms: float = 2.8,
             alpha: float = 1.0) -> None:
    ax.add_patch(
        Rectangle(
            (row.t_start_s - t_ref, row.f_low_hz),
            row.t_end_s - row.t_start_s,
            row.f_high_hz - row.f_low_hz,
            fill=False, ec=ec, ls=ls, lw=lw, zorder=zorder, alpha=alpha,
        )
    )
    if marker and np.isfinite(row.emitter_hz):
        # markers are never cut at the border: draw unclipped, but only if
        # their centre lies inside the visible image
        x, y = row.t_center_s - t_ref, row.emitter_hz
        (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
        if xa <= x <= xb and ya <= y <= yb:
            ax.plot(x, y, "o", ms=ms, mfc=ec, mec="k", mew=0.45, zorder=zorder + 1,
                    clip_on=False)


def draw_missed(ax, row: pd.Series, t_ref: float, lw: float = BOX_LW) -> None:
    """A detection below the operating threshold (not part of the output)."""
    draw_box(ax, row, t_ref, MISSED_COLOR, ls=MISSED_LS, lw=lw, marker=False)


def detection_kinds(rows: pd.DataFrame, fish: dict, *,
                    show_dropped: bool = True) -> pd.DataFrame:
    """The rows (by score) with ``drawn_as``: how ``draw_detections`` draws
    each ("assigned", "unassigned", "harmonic", "duplicate"; None: not drawn)."""
    kinds = []
    for _, row in rows.sort_values("score").iterrows():
        st = box_style(row, fish)
        drop = st is not None and st["kind"] in ("harmonic", "duplicate")
        kinds.append(None if st is None or (drop and not show_dropped) else st["kind"])
    out = rows.sort_values("score").copy()
    out["drawn_as"] = kinds
    return out


def draw_detections(ax, rows: pd.DataFrame, fish: dict, t_ref: float, *,
                    lw: float = BOX_LW, show_dropped: bool = True,
                    ms: float = 2.8) -> pd.DataFrame:
    """Draw every operating detection; returns the rows with their style kind."""
    out = detection_kinds(rows, fish, show_dropped=show_dropped)
    for _, row in out.iterrows():
        if row.drawn_as is None:
            continue
        st = box_style(row, fish)
        draw_box(ax, row, t_ref, st["ec"], st["ls"], lw=lw, ms=ms, alpha=st["alpha"],
                 marker=st["kind"] not in ("harmonic", "duplicate"))
    return out


def label_marks(ax, times, ms: float = 3.4, pad_pt: float = 3.5) -> None:
    """Hand labels: black downward triangles just above the image."""
    tr = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    tr = tr + mpl.transforms.ScaledTranslation(0, pad_pt / 72, ax.figure.dpi_scale_trans)
    ax.plot(times, np.ones(len(times)), "v", ms=ms, mfc="k", mec="k", mew=0.4,
            zorder=20, clip_on=False, transform=tr)


def scalebar(ax, x1: float, y: float, length: float, label: str, *,
             color: str = "white", dy_pt: float = 1.5, above: bool = True,
             fontsize=ANNOT_FS, lw: float = 1.5) -> None:
    """Horizontal scale bar ending at data x ``x1``, label centred above (below)."""
    ax.plot([x1 - length, x1], [y, y], color=color, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=20)
    off = mpl.transforms.offset_copy(ax.transData, fig=ax.figure, x=0,
                                     y=dy_pt if above else -dy_pt, units="points")
    ax.text(x1 - length / 2, y, label, color=color, ha="center",
            va="bottom" if above else "top", fontsize=fontsize, zorder=20,
            transform=off)


def eod_label(ax, x: float, y: float, k: int | None, color: str = "white",
              ha: str = "left", va: str = "center", text: str | None = None,
              **kw) -> None:
    s = text if text is not None else rf"EOD$f_{{{k}}}$"
    ax.text(x, y, s, color=color, fontsize=ANNOT_FS, va=va, ha=ha, zorder=20, **kw)


def spec_yaxis(ax, ticks=None, ylabel: bool = True, labels: bool = True) -> None:
    """The paper's spectrogram axis: frequency spine offset from the image, no time axis."""
    ax.set_xticks([])
    if ticks is not None:
        ax.set_yticks(ticks)
    if not labels:
        ax.set_yticklabels([])
    sns.despine(ax=ax, bottom=True, offset=3, trim=False)
    if ylabel:
        ax.set_ylabel("Frequency [Hz]")


def no_yaxis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def zoom_span(ax_from, x0: float, x1: float, ax_to, *, color="white", lw: float = 0.7,
              ls=(0, (2.5, 1.5))):
    """Mark the time span [x0, x1] of ``ax_from`` that ``ax_to`` enlarges over
    the same frequency range: two dashed vertical lines over the whole image
    (never a rectangle, which could be read as a detection box), repeated at
    the left and right edge of the enlargement. With panels of one height,
    connector lines would run along the images' borders."""
    for x in (x0, x1):
        ax_from.axvline(x, color=color, lw=lw, ls=ls, zorder=15)
    # just inside the enlargement's edges (on the edge, half the line would
    # fall on the white page)
    for x, dx in ((0, lw / 2), (1, -lw / 2)):
        tr = mpl.transforms.offset_copy(ax_to.transAxes, fig=ax_to.figure, x=dx,
                                        units="points")
        ax_to.plot([x, x], [0, 1], color=color, lw=lw, ls=ls, zorder=15, transform=tr,
                   clip_on=False)


# ---------------------------------------------------------------------------
# keys and connectors (figure centimetres)
# ---------------------------------------------------------------------------
def text_width_cm(fig, text: str, fontsize=ANNOT_FS) -> float:
    t = fig.text(0, 0, text, fontsize=fontsize)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 2.54


def key_strip(fig, x: float, y: float, items: list[tuple[str, dict, str]], *,
              gap: float = 0.35, sw_w: float = 0.5, sw_h: float = 0.32,
              measure: bool = False) -> float:
    """A one-line key of spectrogram symbols at (x, y) cm (vertical centre).

    Each item: (kind, style, text) with kind in "box", "boxdot", "dot",
    "track", "label" (hand label, drawn on white as above the images),
    "qmark"; the other symbols are drawn on a swatch of the spectrogram's
    black so that they look as in the panels. Returns the x after the last
    item (cm); ``measure``: only measure.
    """
    W, H = fig.get_size_inches() / CM
    for kind, st, text in items:
        if not measure:
            ax = fig.add_axes((x / W, (y - sw_h / 2) / H, sw_w / W, sw_h / H))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor("white" if kind == "label" else SPEC_BG)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            c = st.get("color", "white")
            if kind in ("box", "boxdot"):
                ax.add_patch(Rectangle((0.22, 0.16), 0.56, 0.68, fill=False, ec=c,
                                       ls=st.get("ls", "-"), lw=BOX_LW,
                                       alpha=st.get("alpha", 1.0)))
                if kind == "boxdot":
                    ax.plot(0.5, 0.36, "o", ms=2.6, mfc=c, mec="k", mew=0.45)
            elif kind == "dot":
                ax.plot(0.5, 0.5, "o", ms=2.8, mfc=c, mec="k", mew=0.45)
            elif kind == "track":
                # a black track on its band (the band's saturated colour);
                # st["ls"]: dashed for interpolated samples
                ax.axhspan(0.3, 0.7, color=SPEC_CMAP(1.0), lw=0)
                ax.plot([0.0, 1.0], [0.5, 0.5], color=st.get("color", TRACK_COLOR),
                        lw=TRACK_LW, ls=st.get("ls", "-"), dash_capstyle="butt")
            elif kind == "label":
                ax.plot(0.5, 0.5, "v", ms=3.4, mfc="k", mec="k", mew=0.4)
            elif kind == "qmark":
                ax.text(0.5, 0.5, "?", ha="center", va="center", color="white",
                        fontsize=ANNOT_FS, fontweight="bold")
            fig.text((x + sw_w + 0.08) / W, y / H, text, ha="left", va="center_baseline",
                     fontsize=ANNOT_FS)
        x = x + sw_w + 0.08 + text_width_cm(fig, text) + gap
    return x - gap


def flow_arrow(fig, x0: float, y0: float, x1: float, y1: float, *, color=INK,
               lw: float = 0.7, rad: float = 0.0, ms: float = 6) -> None:
    """Arrow between two points in figure cm."""
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig_cm(fig),
                                   arrowstyle="-|>", mutation_scale=ms, color=color,
                                   lw=lw, connectionstyle=f"arc3,rad={rad}",
                                   shrinkA=0, shrinkB=0))


def rounded_clip(ax, radius: float = 0.12, ec="#bdbdbd", lw: float = 0.6):
    """Clip everything in ``ax`` to a rounded rectangle (thumbnails); returns the frame."""
    W, H = ax.figure.get_size_inches() / CM
    bb = ax.get_position()
    w_cm, h_cm = bb.width * W, bb.height * H
    patch = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                           boxstyle=f"round,pad=0,rounding_size={radius / max(w_cm, h_cm)}",
                           mutation_aspect=w_cm / h_cm, fc="none", ec="none")
    ax.add_patch(patch)
    for art in list(ax.images) + list(ax.lines) + list(ax.patches) + list(ax.collections):
        if art is not patch:
            art.set_clip_path(patch)
    frame = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                           boxstyle=f"round,pad=0,rounding_size={radius / max(w_cm, h_cm)}",
                           mutation_aspect=w_cm / h_cm, fc="none", ec=ec, lw=lw,
                           zorder=30, clip_on=False)
    ax.add_patch(frame)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return frame


# ---------------------------------------------------------------------------
# output and statistics
# ---------------------------------------------------------------------------
def save(fig, name: str) -> list[Path]:
    """PDF (vector, images rasterised at 300 dpi) and a 300-dpi PNG."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def wilson(k: np.ndarray | int, n: np.ndarray | int, z: float = 1.96):
    """Wilson score interval of a binomial proportion (lower, upper)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half
