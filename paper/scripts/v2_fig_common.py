"""Shared helpers of the three pipeline-v2 figures (paper environment).

Paths of the v2 production run and its evaluations, the cached spectrogram
snippets written by ``v2_fig_extract.py``, and one visual language for all
three figures (see "Figure style" in ``paper/v2_figures.md``):

- spectrograms in seaborn's ``mako`` (the paper's spectrogram colormap), a
  frequency axis offset from the image, no time axis but white scale bars;
- detections drawn as boxes in the colour of the fish they were assigned to
  (EOD$f_1$ = higher-frequency fish of the panel, orange; EOD$f_2$, reddish
  purple), with the predicted emitter EOD$f$ as a circle; white boxes:
  unassigned; dashed grey: removed by post-processing; dotted white: below
  the operating threshold; white triangles: hand labels;
- one colour per model in the performance figure (dual-head = production);
- pipeline stage boxes, brackets and arrows drawn in figure centimetres.

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
# colours and styles
# ---------------------------------------------------------------------------
SPEC_CMAP = sns.color_palette("mako", as_cmap=True)
# fish colours: Okabe-Ito orange and reddish purple, both visible on mako
# (checked: CVD separation 19 dE, contrast >= 3:1 on the spectrogram's black)
FISH_COLORS = ("#E69F00", "#CC79A7")
DROPPED_COLOR = "#c8c8c8"  # boxes removed by post-processing (dashed)
DROPPED_LS = (0, (3, 1.6))
UNASSIGNED_COLOR = "white"
MISSED_LS = (0, (1, 1.2))  # detections below the operating threshold (dotted)
BOX_LW = 0.9
ANNOT_FS = "small"  # 6.7 pt at the 8-pt base; nothing in the figures is smaller
ANNOT = {"color": "white", "fontsize": ANNOT_FS}
ARROW = {"arrowstyle": "-|>", "color": "white", "lw": 0.7, "mutation_scale": 5,
         "shrinkA": 1.5, "shrinkB": 1.5}
INK = "k"
MUTED = "#6e6e6e"  # secondary text (n per bin, notes)
SPEC_BG = SPEC_CMAP(0.0)  # the spectrogram's black, for legend swatches

# performance figure: one colour per model, the same in every panel
# (mako blue/teal of the paper's chirp/baseline colours, flare coral, grey)
MODEL_COLORS = {
    "dualhead": "#395d9c",  # v2 dual-head RT-DETR, the production detector
    "yolo26s": "#3497a9",  # v2 YOLO26s, box-only reference
    "v1": "#a0a0a0",  # v1, the detector of the original analysis
    "human": "#e5715e",  # expert labeller (blind benchmark)
    "edge": "#a0a0a0",  # box-only assignment rule
}
MODEL_LABELS = {
    "dualhead": "v2 dual-head (production)",
    "yolo26s": "v2 YOLO26s (boxes only)",
    "v1": "v1 (original analysis)",
    "human": "expert labeller",
}
# light fills of the paper (obliterate figure): stage boxes, highlights
FILL_GREY = "#F1EFE8"
FILL_TEAL = "#E1F5EE"


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
    """{track id: colour} (EOD$f_1$ orange, EOD$f_2$ purple) and the order."""
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
) -> None:
    """Draw one channel of a snippet, time relative to ``t_ref`` (default t0).

    ``vmin``/``vmax`` are levels of the normalised image (0-1 = the front
    end's 60-dB range); None: the 40th / 99.8th percentile of the crop.
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
        ax.plot(row.t_center_s - t_ref, row.emitter_hz, "o", ms=ms, mfc=ec,
                mec="k", mew=0.45, zorder=zorder + 1)


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


def label_marks(ax, times, y, ms: float = 3.6, **kw) -> None:
    """Hand labels: white downward triangles at the chirps' centre times."""
    ax.plot(times, np.full(len(times), y), "v", ms=ms, mfc="white", mec="k",
            mew=0.4, zorder=20, clip_on=False, **kw)


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


# kept for the old call sites
def clean_spec_axes(ax, ylabel: bool = True) -> None:
    spec_yaxis(ax, ylabel=ylabel)


def zoom_connect(fig, ax_from, rect, ax_to, *, color="white", to_color="k",
                 corners=("right", "left"), lw: float = 0.6, ls=(0, (2, 1.5))):
    """Mark ``rect`` = (x0, y0, x1, y1) in ``ax_from`` and connect it to ``ax_to``.

    corners: which sides to connect ("right"->"left": rect's right edge to the
    target's left edge; "bottom"->"top").
    """
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
# keys, stage boxes, brackets, arrows (figure centimetres)
# ---------------------------------------------------------------------------
def text_width_cm(fig, text: str, fontsize=ANNOT_FS) -> float:
    t = fig.text(0, 0, text, fontsize=fontsize)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 2.54


def key_strip(fig, x: float, y: float, items: list[tuple[str, dict, str]], *,
              gap: float = 0.35, sw_w: float = 0.5, sw_h: float = 0.32) -> float:
    """Draw a one-line key of spectrogram symbols at (x, y) cm (vertical centre).

    Each item: (kind, style, text) with kind in "box", "dot", "track",
    "label", "band", "boxdot"; the symbol is drawn on a small swatch of the
    spectrogram's black so that it looks as in the panels. Returns the x
    after the last item (cm).
    """
    W, H = fig.get_size_inches() / CM
    for kind, st, text in items:
        ax = fig.add_axes((x / W, (y - sw_h / 2) / H, sw_w / W, sw_h / H))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(SPEC_BG)
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
            ax.plot(0.5, 0.55, "v", ms=3.6, mfc="white", mec="k", mew=0.4)
        elif kind == "band":
            ax.axhspan(0.2, 0.8, color=c, alpha=0.35, lw=0)
            ax.plot([0.08, 0.92], [0.5, 0.5], color=c, lw=0.8)
        fig.text((x + sw_w + 0.08) / W, y / H, text, ha="left", va="center_baseline",
                 fontsize=ANNOT_FS)
        x = x + sw_w + 0.08 + text_width_cm(fig, text) + gap
    return x


def bracket(fig, x0: float, x1: float, y: float, text: str, *, tick: float = 0.12,
            fontsize="medium", color=INK, lw: float = 0.6, pad: float = 0.06) -> None:
    """Horizontal bracket from x0 to x1 cm at y cm, ticks down, text centred above."""
    tr = fig_cm(fig)
    fig.add_artist(plt.Line2D([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=color,
                              lw=lw, transform=tr, solid_joinstyle="miter"))
    fig.text((x0 + x1) / 2, y + pad, text, ha="center", va="bottom",
             fontsize=fontsize, color=color, transform=tr)


def stage_box(fig, x: float, y: float, w: float, h: float, letter: str | None,
              title: str, detail: str, *, fc=FILL_GREY, ec=INK) -> None:
    """Rounded stage box (the paper's schematic style) with a bold letter + text."""
    tr = fig_cm(fig)
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                  fc=fc, ec=ec, lw=0.8, transform=tr))
    head = (rf"\textbf{{{letter}}}\enspace " if letter else "") + title
    fig.text(x + w / 2, y + h * 0.68, head, ha="center", va="center", fontsize="small",
             transform=tr)
    fig.text(x + w / 2, y + h * 0.3, detail, ha="center", va="center", fontsize="small",
             color=MUTED, transform=tr)


def flow_arrow(fig, x0: float, y0: float, x1: float, y1: float, *, color=INK,
               lw: float = 0.8, rad: float = 0.0, ms: float = 7) -> None:
    """Arrow between two points in figure cm."""
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig_cm(fig),
                                   arrowstyle="-|>", mutation_scale=ms, color=color,
                                   lw=lw, connectionstyle=f"arc3,rad={rad}",
                                   shrinkA=0, shrinkB=0))


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
