"""Shared helpers of the three pipeline-v2 figures (paper environment).

Paths of the v2 production run and its evaluations, the cached spectrogram
snippets written by ``v2_fig_extract.py``, and the drawing conventions of
the old pipeline figures (mako spectrograms, white annotations, scale bars
instead of a time axis, "EOD$f_1$" labels at the fish's bands), with boxes
coloured by the fish they were assigned to.

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
from matplotlib.patches import Rectangle

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

SPEC_CMAP = sns.color_palette("mako", as_cmap=True)
# fish colours: Okabe-Ito orange and reddish purple, both visible on mako
FISH_COLORS = ("#E69F00", "#CC79A7")
DROPPED_COLOR = "#bdbdbd"  # boxes removed by post-processing
UNASSIGNED_COLOR = "white"
ANNOT = {"color": "white", "fontsize": "small"}
ARROW = {"arrowstyle": "->", "color": "white", "lw": 0.75}

# model colours of the performance figure
MODEL_COLORS = {
    "dualhead": SPEC_CMAP(0.25),
    "yolo26s": SPEC_CMAP(0.62),
    "v1": "#9e9e9e",
    "human": "#E69F00",
    "edge": "#9e9e9e",
}


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
    t, f = snip["times_s"], snip["freqs_hz"]
    it = (t >= t0 - 0.05) & (t <= t1 + 0.05)
    jf = (f >= f0 - 10) & (f <= f1 + 10)
    img = snip["img"][channel][np.ix_(jf, it)].astype(float)
    dt, df = t[1] - t[0], f[1] - f[0]
    if vmin is None:
        vmin = float(np.percentile(img, 40))
    if vmax is None:
        vmax = float(np.percentile(img, 99.8))
    ax.imshow(
        img,
        extent=(
            t[it][0] - t_ref - dt / 2,
            t[it][-1] - t_ref + dt / 2,
            f[jf][0] - df / 2,
            f[jf][-1] + df / 2,
        ),
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


def fish_order(snip: dict, t0: float, t1: float) -> list[int]:
    """Track ids live in [t0, t1], highest median frequency first (EOD$f_1$)."""
    med = {}
    for tid, (tt, ff) in snip["tracks"].items():
        m = (tt >= t0) & (tt <= t1)
        if m.sum() >= 2:
            med[tid] = float(np.median(ff[m]))
    return sorted(med, key=lambda k: -med[k])


def draw_tracks(
    ax, snip: dict, colors: dict, t0: float, t1: float, *, t_ref: float | None = None,
    lw: float = 0.6, alpha: float = 0.9,
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
        return {"ec": DROPPED_COLOR, "ls": (0, (3, 2)), "kind": row.pp_drop_reason}
    if not row.operating:
        return None
    tid = row.track_id
    if np.isfinite(tid) and int(tid) in colors:
        return {"ec": colors[int(tid)], "ls": "-", "kind": "assigned"}
    return {"ec": UNASSIGNED_COLOR, "ls": "-", "kind": "unassigned"}


def draw_box(ax, row: pd.Series, t_ref: float, ec, ls="-", lw: float = 0.8,
             marker: bool = True, zorder: int = 5) -> None:
    ax.add_patch(
        Rectangle(
            (row.t_start_s - t_ref, row.f_low_hz),
            row.t_end_s - row.t_start_s,
            row.f_high_hz - row.f_low_hz,
            fill=False, ec=ec, ls=ls, lw=lw, zorder=zorder,
        )
    )
    if marker and np.isfinite(row.emitter_hz):
        ax.plot(row.t_center_s - t_ref, row.emitter_hz, "o", ms=2.6, mfc=ec,
                mec="k", mew=0.4, zorder=zorder + 1)


def draw_detections(ax, rows: pd.DataFrame, colors: dict, t_ref: float, *,
                    lw: float = 0.8, show_dropped: bool = True) -> pd.DataFrame:
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


def scalebar(ax, x1: float, y: float, length: float, label: str, *,
             color: str = "white", dy: float = 0.0, fontsize="small") -> None:
    """Horizontal scale bar ending at data x ``x1``; label centred below it."""
    ax.plot([x1 - length, x1], [y, y], color=color, lw=1.5, solid_capstyle="butt",
            clip_on=False, zorder=20)
    ax.text(x1 - length / 2, y - dy, label, color=color, ha="center", va="top",
            fontsize=fontsize, zorder=20)


def eod_label(ax, x: float, y: float, k: int, color: str = "white", **kw) -> None:
    ax.text(x, y, rf"EOD$f_{{{k}}}$", color=color, fontsize="small", va="center",
            ha="left", zorder=20, **kw)


def clean_spec_axes(ax, ylabel: bool = True) -> None:
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    if ylabel:
        ax.set_ylabel("Frequency [Hz]")


def save(fig, name: str) -> list[Path]:
    """PDF (vector, rasterised images at 300 dpi) and a 150-dpi PNG preview."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1], dpi=150)
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


def rc_tweaks() -> None:
    """Thinner lines and ticks for 8-pt figures (on top of style.set_light_style)."""
    mpl.rcParams.update(
        {
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.markeredgewidth": 0.5,
            "legend.handlelength": 1.2,
        }
    )
