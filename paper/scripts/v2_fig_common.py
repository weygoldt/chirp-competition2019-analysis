"""Shared helpers of the three pipeline-v2 figures (paper environment).

Paths of the v2 production run and its evaluations, the cached spectrogram
snippets written by ``v2_fig_extract.py``, and one visual language for all
three figures (see "Figure style" in ``paper/v2_figures.md``):

- one palette from seaborn's ``mako`` (cool: spectrograms, detectors,
  densities) and its companion ``flare`` (warm: the two fish, the human
  expert), plus greys for references;
- spectrograms in ``mako`` (capped below white, so that white annotations
  stand out), display levels from the crop (background percentile ->
  chirp-pixel percentile), a frequency axis offset from the image, no time
  axis but white scale bars;
- detections as boxes in the colour of the fish they were assigned to
  (EOD$f_1$ = higher-frequency fish of the panel, peach; EOD$f_2$, rose),
  the predicted emitter EOD$f$ as a circle; white boxes: unassigned; dashed
  grey: removed by post-processing; dotted white: below the operating
  threshold; black triangles above the image: hand labels;
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
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle

import paper_config

CM = 1 / 2.54
FIG_WIDTH = 17.4 * CM  # width of the manuscript's full-width figures

CHIRP_DATA = Path("/mnt/data1/chirp_v2")
RUN = CHIRP_DATA / "detections" / "competition2019_v2_20261113"
REAL_EVAL = CHIRP_DATA / "eval" / "real_labels_365"
REAL_LABELS = REAL_EVAL / "labels"
PP_EVAL = {
    "dualhead": CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_r1" / "eval_postprocess",
    "yolo26s": CHIRP_DATA / "runs" / "yolo26s_sim_v4_r1" / "eval_postprocess",
}
BENCHMARK = CHIRP_DATA / "labels" / "benchmark" / "report_v2"
DECODER_EVAL = CHIRP_DATA / "params" / "runs" / "v4-dualhead-sim-v4-r2" / "bundle" / "eval"
TRIAL_SUMMARY = Path("/mnt/data2/2019_tube_competition/trial_summary.csv")

SNIPPETS = paper_config.PROCESSED / "v2_figures"
FIGURES = paper_config.FIGURES

OPERATING = 0.5  # the run's operating threshold
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
# densities: light mint -> the detector blue (the same mako, reversed)
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "mako_density", _MAKO(np.linspace(0.92, 0.35, 256)))

# the two fish of a panel (warm, distinct from the cool spectrogram; checked
# with the dataviz validator on the spectrogram's black: CVD separation
# 21 dE, contrast >= 3:1)
FISH_COLORS = (PEACH, ROSE)
DROPPED_COLOR = "#c8c8c8"  # boxes removed by post-processing (dashed)
DROPPED_LS = (0, (3, 1.6))
UNASSIGNED_COLOR = "white"
MISSED_LS = (0, (1, 1.2))  # detections below the operating threshold (dotted)
BOX_LW = 0.9
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
    """All stored rows of one recording of the production run."""
    return pd.read_parquet(RUN / recording / "chirps_rtdetr.parquet")


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


def fish_colors(snip: dict, t0: float, t1: float) -> tuple[dict, list[int]]:
    """{track id: colour} (EOD$f_1$ peach, EOD$f_2$ rose) and the order."""
    order = fish_order(snip, t0, t1)
    return {tid: FISH_COLORS[i % 2] for i, tid in enumerate(order)}, order


# ---------------------------------------------------------------------------
# spectrograms and detections
# ---------------------------------------------------------------------------
def spec_crop(snip: dict, t0: float, t1: float, f0: float, f1: float, channel):
    t, f = snip["times_s"], snip["freqs_hz"]
    it = (t >= t0 - 0.05) & (t <= t1 + 0.05)
    jf = (f >= f0 - 10) & (f <= f1 + 10)
    dt, df = t[1] - t[0], f[1] - f[0]
    ext = (t[it][0] - dt / 2, t[it][-1] + dt / 2, f[jf][0] - df / 2, f[jf][-1] + df / 2)
    if channel == "rgb":
        img = np.stack([snip["img"][c][np.ix_(jf, it)].astype(float) for c in range(3)], -1)
    else:
        img = snip["img"][channel][np.ix_(jf, it)].astype(float)
    return img, ext


def chirp_levels(snip: dict, t0: float, t1: float, f0: float, f1: float,
                 rows: pd.DataFrame, *, channel: int = 0, q_bg: float = 85,
                 q_chirp: float = 99, guard_hz: float = 15.0) -> tuple[float, float]:
    """Display levels that make the chirps visible.

    vmin: the ``q_bg`` percentile of the whole crop (mostly background);
    vmax: the ``q_chirp`` percentile of the pixels inside the crop's
    operating chirp boxes, excluding rows within ``guard_hz`` of a live
    track (the baselines, which then saturate). Falls back to (0.3, 0.95).
    """
    img, _ = spec_crop(snip, t0, t1, f0, f1, channel)
    t, f = snip["times_s"], snip["freqs_hz"]
    lo = float(np.percentile(img, q_bg))
    vals = []
    tracks = [track_median(snip, tid, t0 - 5, t1 + 5) for tid in fish_order(snip, t0 - 5, t1 + 5)]
    for r in rows[rows.operating & rows.chirp].itertuples():
        it = (t >= max(r.t_start_s, t0)) & (t <= min(r.t_end_s, t1))
        jf = (f >= max(r.f_low_hz, f0)) & (f <= min(r.f_high_hz, f1))
        for hz in tracks:
            jf &= np.abs(f - hz) > guard_hz
        if it.any() and jf.any():
            vals.append(snip["img"][channel][np.ix_(jf, it)].astype(float).ravel())
    if not vals:
        return 0.3, 0.95
    hi = float(np.percentile(np.concatenate(vals), q_chirp))
    return lo, max(hi, lo + 0.1)


def background_levels(snip: dict, t0: float, t1: float, f0: float, f1: float,
                      rows: pd.DataFrame, *, channel: int = 0, q_bg: float = 60,
                      q_chirp: float = 97, guard_hz: float = 15.0) -> tuple[float, float]:
    """Display levels for crops where the baselines fill much of the image.

    vmin: the ``q_bg`` percentile of the background pixels only (outside every
    stored box with score >= 0.2 and farther than ``guard_hz`` from a live
    track); vmax: the ``q_chirp`` percentile of the pixels inside the
    operating chirp boxes away from the tracks. Unlike ``chirp_levels`` the
    floor is not pulled up by the baselines' glow (in a 300-Hz crop around two
    close fish the glow fills > 15 % of the pixels).
    """
    t, f = snip["times_s"], snip["freqs_hz"]
    it, jf = (t >= t0) & (t <= t1), (f >= f0) & (f <= f1)
    tt, ff = t[it], f[jf]
    crop = snip["img"][channel][np.ix_(jf, it)].astype(float)
    tracks = [track_median(snip, tid, t0 - 5, t1 + 5) for tid in fish_order(snip, t0 - 5, t1 + 5)]
    off_track = np.ones(len(ff), bool)
    for hz in tracks:
        off_track &= np.abs(ff - hz) > guard_hz
    bg = np.repeat(off_track[:, None], len(tt), axis=1)
    vals = []
    for r in rows.itertuples():
        mt = (tt >= r.t_start_s) & (tt <= r.t_end_s)
        mf = (ff >= r.f_low_hz) & (ff <= r.f_high_hz)
        if r.score >= 0.2:
            bg[np.ix_(mf, mt)] = False
        if r.operating and r.chirp and mt.any() and (mf & off_track).any():
            vals.append(crop[np.ix_(mf & off_track, mt)].ravel())
    lo = float(np.percentile(crop[bg], q_bg))
    hi = float(np.percentile(np.concatenate(vals), q_chirp)) if vals else lo + 0.3
    return lo, max(hi, lo + 0.1)


def show_spec(
    ax,
    snip: dict,
    t0: float,
    t1: float,
    f0: float,
    f1: float,
    *,
    channel: int = 0,
    vmin: float | None = 0.3,
    vmax: float | None = 0.95,
    t_ref: float | None = None,
) -> tuple[float, float]:
    """Draw one channel of a snippet, time relative to ``t_ref`` (default t0).

    ``vmin``/``vmax`` are levels of the normalised image (0-1 = the front
    end's 60-dB range); None: the 40th / 99.8th percentile of the crop.
    Returns the levels used.
    """
    t_ref = t0 if t_ref is None else t_ref
    img, ext = spec_crop(snip, t0, t1, f0, f1, channel)
    if vmin is None:
        vmin = float(np.percentile(img, 40))
    if vmax is None:
        vmax = float(np.percentile(img, 99.8))
    ax.imshow(
        img,
        extent=(ext[0] - t_ref, ext[1] - t_ref, ext[2], ext[3]),
        aspect="auto",
        origin="lower",
        cmap=SPEC_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="antialiased",
        rasterized=True,
    )
    ax.set_xlim(t0 - t_ref, t1 - t_ref)
    ax.set_ylim(f0, f1)
    return vmin, vmax


def show_rgb(ax, snip: dict, t0: float, t1: float, f0: float, f1: float,
             t_ref: float | None = None) -> None:
    """The detector's input: channels 205/51/13 ms as R/G/B on its own 0-1 levels."""
    t_ref = t0 if t_ref is None else t_ref
    img, ext = spec_crop(snip, t0, t1, f0, f1, "rgb")
    ax.imshow(np.clip(img, 0, 1), extent=(ext[0] - t_ref, ext[1] - t_ref, ext[2], ext[3]),
              aspect="auto", origin="lower", interpolation="antialiased", rasterized=True)
    ax.set_xlim(t0 - t_ref, t1 - t_ref)
    ax.set_ylim(f0, f1)


def draw_tracks(
    ax, snip: dict, colors: dict, t0: float, t1: float, *, t_ref: float | None = None,
    lw: float = 0.6, alpha: float = 0.95,
) -> None:
    t_ref = t0 if t_ref is None else t_ref
    for tid, (tt, ff) in snip["tracks"].items():
        if tid not in colors:
            continue
        m = (tt >= t0 - 0.5) & (tt <= t1 + 0.5)
        if m.sum() < 2:
            continue
        # break the line at gaps longer than 1 s
        tm, fm = tt[m], ff[m].astype(float)
        gaps = np.flatnonzero(np.diff(tm) > 1.0)
        tm = np.insert(tm, gaps + 1, np.nan)
        fm = np.insert(fm, gaps + 1, np.nan)
        ax.plot(tm - t_ref, fm, color=colors[tid], lw=lw, alpha=alpha, zorder=2,
                solid_capstyle="butt")


def box_style(row: pd.Series, colors: dict) -> dict | None:
    """Edge colour and line style of a detection (None: not drawn)."""
    if row.pp_drop_reason in ("harmonic", "duplicate") and row.operating:
        return {"ec": DROPPED_COLOR, "ls": DROPPED_LS, "kind": row.pp_drop_reason}
    if not row.operating:
        return None
    tid = row.track_id
    if np.isfinite(tid) and int(tid) in colors:
        return {"ec": colors[int(tid)], "ls": "-", "kind": "assigned"}
    return {"ec": UNASSIGNED_COLOR, "ls": "-", "kind": "unassigned"}


def draw_box(ax, row: pd.Series, t_ref: float, ec, ls="-", lw: float = BOX_LW,
             marker: bool = True, zorder: int = 5, ms: float = 2.8) -> None:
    ax.add_patch(
        Rectangle(
            (row.t_start_s - t_ref, row.f_low_hz),
            row.t_end_s - row.t_start_s,
            row.f_high_hz - row.f_low_hz,
            fill=False, ec=ec, ls=ls, lw=lw, zorder=zorder,
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
    draw_box(ax, row, t_ref, "white", ls=MISSED_LS, lw=lw, marker=False)


def draw_detections(ax, rows: pd.DataFrame, colors: dict, t_ref: float, *,
                    lw: float = BOX_LW, show_dropped: bool = True) -> pd.DataFrame:
    """Draw every operating detection; returns the rows with their style kind."""
    kinds = []
    for _, row in rows.sort_values("score").iterrows():
        st = box_style(row, colors)
        if st is None or (not show_dropped and st["kind"] in ("harmonic", "duplicate")):
            kinds.append(None)
            continue
        draw_box(ax, row, t_ref, st["ec"], st["ls"], lw=lw,
                 marker=st["kind"] not in ("harmonic", "duplicate"))
        kinds.append(st["kind"])
    out = rows.sort_values("score").copy()
    out["drawn_as"] = kinds
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


def zoom_connect(fig, ax_from, rect, ax_to, *, color="white", to_color="k",
                 corners=("right", "left"), lw: float = 0.6, ls=(0, (2, 1.5))):
    """Mark ``rect`` = (x0, y0, x1, y1) in ``ax_from`` and connect it to ``ax_to``."""
    x0, y0, x1, y1 = rect
    ax_from.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw,
                                ls=ls, zorder=15))
    if corners == ("right", "left"):
        pairs = [((x1, y1), (0, 1)), ((x1, y0), (0, 0))]
    elif corners == ("bottom", "top"):
        pairs = [((x0, y0), (0, 1)), ((x1, y0), (1, 1))]
    else:
        raise ValueError(corners)
    for xy_a, xy_b in pairs:
        con = ConnectionPatch(xyA=xy_a, coordsA=ax_from.transData, xyB=xy_b,
                              coordsB=ax_to.transAxes, color=to_color, lw=lw, ls=ls,
                              zorder=0)
        fig.add_artist(con)


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
                                       ls=st.get("ls", "-"), lw=BOX_LW))
                if kind == "boxdot":
                    ax.plot(0.5, 0.36, "o", ms=2.6, mfc=c, mec="k", mew=0.45)
            elif kind == "dot":
                ax.plot(0.5, 0.5, "o", ms=2.8, mfc=c, mec="k", mew=0.45)
            elif kind == "track":
                ax.plot([0.08, 0.92], [0.5, 0.5], color=c, lw=0.8)
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
