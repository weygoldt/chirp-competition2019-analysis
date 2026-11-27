"""Figure: chirp detection pipeline v2, from the grid recording to a parameterised chirp.

Replaces ``plot_assignment_feature_extraction.pdf`` (the v1 assignment
features). A rail of thumbnails on top previews the stages (recording ->
A detection -> A assignment -> B post-processing -> C parameters); below,
one real example per stage, all from the production run of the selected
pipeline (``paper_config.PIPELINES``; the single-window detector since
2026-11-27, whose input is the plain 205-ms spectrogram the panels show).
The former panel A (the three analysis windows of pipeline v2's front end
and their RGB stack) is dropped with the switch (author, 2026-11-27). The
examples are named by their pipeline-v2 chirp ids (the author chose them
there); ``v2_examples.example_id`` finds the selected run's box at the same
place and fails if there is none. Drawn to the manuscript's figure standard
(17.4 cm wide; no colour or line style for fish identity on the
spectrograms: tracks as thin black lines labelled EOD$f_1$ / EOD$f_2$, every
detection a solid white box whose emitter marker (white circle) sits on the
track of the fish it was assigned to; see ``v2_fig_common``):

A   detection and assignment on 8.2 s of one recording (2020-06-02-10_00,
    2245.2-2253.4 s, ``DETECT``; the author's choice among three candidates,
    2026-11-24): every detection at the operating threshold as a box with the
    predicted emitter EOD$f$ (circle), assigned to the track nearest the
    predicted emitter (if within 20 Hz; not drawn since round 3);
B   post-processing removes a box on the second harmonic (2020-05-29-10_00,
    one detector tile: v2 chirp 2084 on the second harmonic of v2 chirp 2085)
    and a duplicate box on one chirp (2019-12-02-10_00, v2 chirps 244 kept /
    245 removed, both above the operating threshold of 0.8): kept boxes
    solid, removed ones dashed and translucent;
C   the decoder's estimates (calibrated) of a small and a big chirp of the
    2020-05-29 tile (v2 chirps 2085, the fundamental of B's harmonic copy,
    and 2086) as double-headed arrows, next to their instantaneous-frequency
    traces extracted from the raw recording: two independent estimates. The
    decoder outputs four scalars per box (height, FWHM, 10 % width,
    undershoot), each with a calibrated SD (``chirpdetector.params``:
    ``ParamNet`` predicts a mean and a log-variance per target); the SDs are
    in the caption, not drawn.

Round 3 (author, 2026-11-25): no colour bars, no keys in B and C, no
tolerance brackets (B full width), all boxes solid in one line width,
black straight rail arrows, channel dots instead of R/G/B chips, a
schematic of the scalar parameters as the "Parameters" thumbnail, the
parameters as double-headed arrows with a gap to the trace, and every row
on one left and right edge (``L``, ``R``; the script prints the axes'
extents).

Spectrograms (A, B and the thumbnails; the 205-ms window): power in dB re
1 V^2 (summed over electrodes), display levels the 2nd and 98th percentile
of each panel's crop (``C.CLIP_PCT``).

Inputs: the snippet/trace/raw cache of ``v2_fig_extract.py`` (run it first,
in the upgrade env) and the run's chirp table (``C.RUN``/``C.PARQUET``,
chosen by PAPER_DETECTIONS).

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_pipeline_explainer.py

Runtime: ~20 s cold, ~4 s warm (LaTeX text rendering). Prints the numbers it
shows; those of the caption also go to write_stats ("v2_pipeline_explainer").
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import v2_fig_common as C
from manuscript_stats import write_stats
from v2_examples import example_id, example_ids
from style import other_color, set_light_style

NAME = "plot_v2_pipeline_explainer"

# A: detection and assignment, 8.2 s of one recording (snippet ``f1_detect``
# of v2_fig_extract). Chirps by their pipeline-v2 ids (v2_examples).
# ``example``: the chirp of the detection thumbnail and of
# the "box" (above the box, pointing down at its top edge) and "predicted
# emitter" (left of the box) annotations, text positions in s from the window
# start and Hz; ``pair``: a chirp of each fish for the assignment thumbnail;
# ``scalebar``: end (s) and height (Hz)
DETECT = {
    "snippet": "f1_detect",
    "window": (2245.2, 2253.4),
    "band": (620.0, 1270.0),
    "ticks": (700, 900, 1100),
    "example": 2014,
    "box_text": 1085.0,
    "emitter_text": (7.27, 912.0),
    "pair": (2004, 2003),
    "scalebar": (7.6, 662.0),
}

# B and C: one detector tile (2020-05-29-10_00, 10850.4-10853.6 s); v2 ids
TILE_SNIPPET = "f1_tile"
TILE = (10850.4, 10853.6)
SMALL, BIG, HARMONIC = 2085, 2086, 2084
# duplicate example (author's choice 2026-11-16): both boxes >= 0.8; v2 ids
DUP_SNIPPET, DUP_KEPT, DUP_DROPPED = "f1_dup", 244, 245

TRACE_COLOR = other_color  # the measured frequency trace (indigo of Figs. 5-7)
TRACE_LW = 0.9
DOT_COLOR = "#8c8c8c"  # single EOD periods
DOT_MS = 2.4
SCHEMA_COLOR = "#6e6e6e"  # the schematic chirp of the "Parameters" thumbnail
SCHEMA_FILL = "#e6e6e6"
# C: the height arrow this far [ms] after the trace's peak (the big chirp's
# peak sits on its steep rising flank, which would hide an arrow there)
HEIGHT_ARROW_DT_MS = {"small": 0.0, "big": 7.0}
ARROW_GAP_PT = 1.2  # white gap between an arrowhead's tip and the trace [pt]
DIM_LW, DIM_MS = 0.7, 5.5  # double-headed arrows of C: line width, head size

# ---------------------------------------------------------------------------
# layout [cm]: every row from L to R; groups separated by GROUP_GAP
# ---------------------------------------------------------------------------
W = 17.4  # figure width; the height follows from the rows (H)
L, R = 1.5, 17.0  # left edge of the leftmost, right edge of the rightmost axes
GROUP_GAP = 0.85  # free space between the rows (rail, A, B/C)
# rail: five thumbnails from L to R
RAIL_W, RAIL_H = 1.6, 0.95
# B, C: two narrow spectrograms (from L) and two trace panels in the right two
# columns of a four-column grid (the former front-end row's: gaps COL_GAP,
# and COL_ARROW_GAP before the last column, where C right's y-label sits)
COL_GAP, COL_ARROW_GAP = 0.25, 1.3
COL_W = (R - L - 2 * COL_GAP - COL_ARROW_GAP) / 4
COL_X = [L, L + COL_W + COL_GAP, L + 2 * (COL_W + COL_GAP), R - COL_W]
CD_H, CD_BOTTOM = 3.3, 1.12
C_W, C_GAP = 2.65, 0.25
C_X = [L, L + C_W + C_GAP]
D_X, D_W = [COL_X[2], COL_X[3]], COL_W
# vertical positions from the bottom: room for titles above B and the labels
# under the rail
CD_TOP = CD_BOTTOM + CD_H
B_H = 3.8
B_BOTTOM = CD_TOP + 0.45 + GROUP_GAP
B_TOP = B_BOTTOM + B_H
RAIL_LABELS = 0.65  # the stage labels under the rail
RAIL_Y0 = B_TOP + GROUP_GAP + RAIL_LABELS
RAIL_TOP = RAIL_Y0 + RAIL_H
H = RAIL_TOP + 0.2  # figure height
LETTER_DX = 1.2  # letters centred this far left of their axes


def dim_arrow(ax, xy0, xy1, *, shrink_a=0.0, shrink_b=0.0, zorder=1.5, transform=None,
              ms=DIM_MS):
    """A double-headed arrow between two points (data coordinates); the
    shrinks [pt] set the arrowheads' tips back from the points."""
    kw = {} if transform is None else {"transform": transform}
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="<|-|>", mutation_scale=ms,
                                 lw=DIM_LW, color="k", shrinkA=shrink_a, shrinkB=shrink_b,
                                 zorder=zorder, clip_on=False, joinstyle="miter",
                                 capstyle="butt", **kw))


class DoubleArrowHandler(HandlerBase):
    """Legend handle: a double-headed arrow as drawn in C."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        y = 0.5 * height - ydescent
        arrow = FancyArrowPatch((-xdescent, y), (-xdescent + width, y), arrowstyle="<|-|>",
                                mutation_scale=DIM_MS, lw=DIM_LW, color="k", shrinkA=0,
                                shrinkB=0, transform=trans)
        return [arrow]


class DoubleArrowKey:
    """Placeholder handle for the legend entry of the estimates."""


def detect_data() -> dict:
    """A's snippet, its operating rows, the panel's fish and their order, and
    the selected run's ids of the example chirps (``ids``)."""
    snip = C.load_snippet(DETECT["snippet"])
    t0, t1 = DETECT["window"]
    rows = C.window(snip, t0, t1)
    cut = rows[rows.operating & ((rows.t_start_s < t0) | (rows.t_end_s > t1))]
    if len(cut):
        msg = f"A window {DETECT['window']} cuts boxes {cut.chirp_id.tolist()}"
        raise ValueError(msg)
    fish, order = C.fish_styles(snip, t0, t1)
    rec = snip["meta"]["recording"]
    example, *pair = example_ids(rows, rec, [DETECT["example"], *DETECT["pair"]])
    return {"snip": snip, "rows": rows, "fish": fish, "order": order,
            "ids": {"example": example, "pair": tuple(pair)}}


# ---------------------------------------------------------------------------
# rail of thumbnails
# ---------------------------------------------------------------------------
def parameter_schematic(ax):
    """The "Parameters" thumbnail: a schematic chirp (grey) with its height
    and width as double-headed arrows: the decoder returns scalars, not a
    trace."""
    x = np.linspace(0, 1, 200)
    base, peak, xc, sd = 0.16, 0.8, 0.5, 0.115
    y = base + (peak - base) * np.exp(-0.5 * ((x - xc) / sd) ** 2)
    ax.fill_between(x, base, y, color=SCHEMA_FILL, lw=0, zorder=0.5)
    ax.plot(x, y, color=SCHEMA_COLOR, lw=0.8, solid_capstyle="round", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    half = base + 0.5 * (peak - base)
    hw = sd * np.sqrt(2 * np.log(2))
    dim_arrow(ax, (xc, base), (xc, peak), shrink_a=0.6, shrink_b=1.2, ms=4.2)
    dim_arrow(ax, (xc - hw, half), (xc + hw, half), shrink_a=1.2, shrink_b=1.2, ms=4.2)


def rail(fig, det, tile):
    xs = np.linspace(L, R - RAIL_W, 5)
    axs = [C.axes_cm(fig, x, RAIL_Y0, RAIL_W, RAIL_H) for x in xs]

    # recording: raw voltage of one electrode (60 ms)
    z = np.load(C.SNIPPETS / "raw.npz")
    sig, fs = z["signal"].astype(float), float(z["samplerate_hz"])
    i0 = int(0.74 * fs)
    seg = sig[i0:i0 + int(0.06 * fs)]
    axs[0].plot(np.arange(len(seg)) / fs, seg, color="k", lw=0.35)
    axs[0].set_xlim(0, len(seg) / fs)
    pad = 0.15 * np.ptp(seg)
    axs[0].set_ylim(seg.min() - pad, seg.max() + pad)
    # A detection: one box and its emitter
    snip, by, fish = det["snip"], det["rows"].set_index("chirp_id"), det["fish"]
    ex = by.loc[det["ids"]["example"]]
    t0, t1 = ex.t_center_s - 0.32, ex.t_center_s + 0.32
    f0, f1 = ex.f_low_hz - 30, ex.f_high_hz + 30
    # thumbnails: the display rule of the panels, on each thumbnail's own crop
    C.show_power(axs[1], snip, t0, t1, f0, f1)
    C.draw_tracks(axs[1], snip, fish, t0, t1, lw=0.5)
    C.fish_box(axs[1], ex, t0, fish, ms=2.4)
    # A assignment: a chirp of each fish, each emitter marker on its track
    a, b = (by.loc[c] for c in det["ids"]["pair"])
    t0 = min(a.t_start_s, b.t_start_s) - 0.06
    t1 = max(a.t_end_s, b.t_end_s) + 0.06
    f0 = min(a.f_low_hz, b.f_low_hz) - 20
    f1 = max(a.f_high_hz, b.f_high_hz) + 20
    C.show_power(axs[2], snip, t0, t1, f0, f1)
    C.draw_tracks(axs[2], snip, fish, t0, t1, lw=0.5)
    for r in (a, b):
        C.fish_box(axs[2], r, t0, fish, ms=2.4)
    # B post-processing: the harmonic copy, removed
    tby, tfish = tile["rows"].set_index("chirp_id"), tile["fish"]
    s, h = tby.loc[tile["ids"]["small"]], tby.loc[tile["ids"]["harmonic"]]
    t0, t1 = s.t_center_s - 0.3, s.t_center_s + 0.3
    C.show_power(axs[3], tile["snip"], t0, t1, 590, 1510)
    C.fish_box(axs[3], s, t0, tfish, marker=False)
    C.dropped_box(axs[3], h, t0)
    # C parameters: scalars (height, widths), drawn as a schematic
    parameter_schematic(axs[4])
    for ax in axs:
        C.rounded_clip(ax)
    labels = ["Recording", r"\textbf{A}\, Detection", r"\textbf{A}\, Assignment",
              r"\textbf{B}\, Post-processing", r"\textbf{C}\, Parameters"]
    tr = C.fig_cm(fig)
    for x, lab in zip(xs, labels, strict=True):
        fig.text(x + RAIL_W / 2, RAIL_Y0 - 0.1, lab, ha="center", va="top", transform=tr)
    # short, black, straight arrows between the stages
    ym = RAIL_Y0 + RAIL_H / 2
    gap = 0.5 * (xs[1] - xs[0] - RAIL_W - 0.55)
    for xa, xb in zip(xs[:-1], xs[1:], strict=True):
        C.flow_arrow(fig, xa + RAIL_W + gap, ym, xb - gap, ym, color="k", lw=0.7, ms=5)
    return axs


# ---------------------------------------------------------------------------
# A: detection and assignment
# ---------------------------------------------------------------------------
def panel_detection(ax, det, report) -> dict:
    snip, rows, fish, order = det["snip"], det["rows"], det["fish"], det["order"]
    t0, t1 = DETECT["window"]
    f0, f1 = DETECT["band"]
    _, lv = C.show_power(ax, snip, t0, t1, f0, f1)
    dashed = C.draw_tracks(ax, snip, fish, t0, t1, bridge_s=C.ASSIGN_MAX_GAP_S)
    drawn = C.draw_detections(ax, rows, fish, t0, ms=3.0)
    C.spec_yaxis(ax, ticks=list(DETECT["ticks"]))
    hz = [C.track_median(snip, tid, t0, t1) for tid in order]
    C.eod_label(ax, 0.06, hz[0] + 26, 1)
    C.eod_label(ax, 0.06, hz[1] - 26, 2)
    # what the detector returns for each chirp: a box (arrow down onto its top
    # edge, at right angles, so that the tip stays visible) and the emitter
    # EODf (arrow from the left onto the marker)
    ex = rows.set_index("chirp_id").loc[det["ids"]["example"]]
    xc = ex.t_center_s - t0
    C.annotate(ax, "box", xy=(xc, ex.f_high_hz), xytext=(xc, DETECT["box_text"]),
               ha="center", va="bottom", rad=0.0, shrink_a=1.5, shrink_b=3.0)
    C.annotate(ax, "predicted\nemitter EOD$f$", xy=(xc, ex.emitter_hz),
               xytext=DETECT["emitter_text"], ha="right", va="center", rad=-0.25,
               shrink_b=3.5)
    C.scalebar(ax, *DETECT["scalebar"], 1.0, "1 s")

    kept = drawn[drawn.drawn_as == "assigned"]
    if (drawn.drawn_as == "unassigned").any() or len(kept) != int(rows.chirp.sum()):
        msg = "A: every chirp of the window must be drawn as assigned"
        raise ValueError(msg)
    per_fish = [int((kept.track_id == tid).sum()) for tid in order]
    spans_both = int(((kept.f_low_hz < min(hz)) & (kept.f_high_hz > max(hz))).sum())
    dist = np.abs(kept.emitter_hz - kept.track_hz)
    report["A"] = {
        "recording": snip["meta"]["recording"], "window_s": [t0, t1], "levels_db": lv,
        "example_ids": det["ids"],
        "tracks_hz": [round(h, 1) for h in hz], "track_ids": order,
        "interpolated_track_pieces": dashed,
        "chirps": kept[["chirp_id", "t_center_s", "score", "track_id", "emitter_hz",
                        "track_hz", "assign_confidence", "height_hz", "fwhm_s"]]
        .sort_values("t_center_s").round(3).to_dict(orient="records"),
        "other_operating": drawn[drawn.drawn_as.isin(["harmonic", "duplicate"])]
        .chirp_id.tolist(),
    }
    return {
        "duration_s": t1 - t0,
        "tracks_hz": hz,
        "df_hz": hz[0] - hz[1],
        "n_chirps": len(kept),
        "n_chirps_per_fish": per_fish,
        "height_hz": {"min": kept.height_hz.min(), "max": kept.height_hz.max()},
        "fwhm_ms": {"min": 1e3 * kept.fwhm_s.min(), "max": 1e3 * kept.fwhm_s.max()},
        "n_boxes_spanning_both_tracks": spans_both,
        "emitter_track_df_hz": {"median": float(np.median(dist)), "max": float(dist.max())},
        "assign_confidence_min": kept.assign_confidence.min(),
    }


# ---------------------------------------------------------------------------
# B: post-processing
# ---------------------------------------------------------------------------
def panel_harmonic(ax, tile):
    snip, by, fish = tile["snip"], tile["rows"].set_index("chirp_id"), tile["fish"]
    h, f = by.loc[tile["ids"]["harmonic"]], by.loc[tile["ids"]["small"]]
    t0, t1 = f.t_center_s - 0.3, f.t_center_s + 0.3
    _, lv = C.show_power(ax, snip, t0, t1, 580, 1520)
    C.draw_tracks(ax, snip, fish, t0, t1)
    C.fish_box(ax, f, t0, fish)
    C.dropped_box(ax, h, t0)
    x = f.t_center_s - t0
    ax.annotate("", xy=(x, h.f_low_hz), xytext=(x, f.f_high_hz),
                arrowprops=C.arrowprops(rad=0.0, shrink_a=3, shrink_b=3))
    ax.text(x - 0.025, 0.5 * (f.f_high_hz + h.f_low_hz), r"2$\times$", ha="right",
            va="center", **C.ANNOT)
    C.spec_yaxis(ax, ticks=[600, 800, 1000, 1200, 1400])
    C.scalebar(ax, 0.51, 960, 0.1, "100 ms")
    return {"harmonic": int(h.name), "harmonic_box": [float(h.f_low_hz), float(h.f_high_hz)],
            "harmonic_emitter_hz": float(h.emitter_hz),
            "fundamental_track_hz": float(f.track_hz), "score": float(h.score),
            "levels_db": lv}


def panel_duplicate(ax):
    snip = C.load_snippet(DUP_SNIPPET)
    rows = C.window(snip, snip["meta"]["t_start_s"], snip["meta"]["t_end_s"])
    by = rows.set_index("chirp_id")
    kept, dropped = example_ids(rows, snip["meta"]["recording"], [DUP_KEPT, DUP_DROPPED])
    k, d = by.loc[kept], by.loc[dropped]
    t0, t1 = k.t_center_s - 0.3, k.t_center_s + 0.3
    fish, _ = C.fish_styles(snip, t0 - 2, t1 + 2)
    _, lv = C.show_power(ax, snip, t0, t1, 580, 1520)
    C.draw_tracks(ax, snip, fish, t0, t1)
    C.fish_box(ax, k, t0, fish)
    C.dropped_box(ax, d, t0)
    C.no_yaxis(ax)
    return {"recording": snip["meta"]["recording"], "kept": kept,
            "kept_score": float(k.score), "kept_box": [float(k.f_low_hz), float(k.f_high_hz)],
            "dropped": dropped, "dropped_score": float(d.score),
            "dropped_box": [float(d.f_low_hz), float(d.f_high_hz)],
            "separation_ms": 1e3 * abs(float(k.t_center_s - d.t_center_s)),
            "levels_db": lv}


# ---------------------------------------------------------------------------
# C: decoder estimates next to the frequency trace
# ---------------------------------------------------------------------------
def crossings(t, df, ipk, level):
    """Trace times where it falls below ``level`` on either side of the peak."""
    above = df >= level
    a = ipk
    while a > 0 and above[a - 1]:
        a -= 1
    b = ipk
    while b < len(df) - 1 and above[b + 1]:
        b += 1
    return t[a], t[b]


def points_per_unit(ax) -> tuple[float, float]:
    """Points per data unit along x and y (linear axes, limits set)."""
    fig_w, fig_h = ax.figure.get_size_inches()
    bb = ax.get_position()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return bb.width * fig_w * 72 / (x1 - x0), bb.height * fig_h * 72 / (y1 - y0)


def tip_shrink(clearance_pt: float) -> float:
    """How far [pt] to set an arrowhead's tip back so that it ends at least
    ``ARROW_GAP_PT`` from the trace's edge, given the distance between the
    estimate's end and the trace's centre line (positive: the tip stays clear
    of the trace)."""
    return max(0.0, ARROW_GAP_PT + TRACE_LW / 2 - clearance_pt)


def panel_trace(ax, traces, meta, key, row, xlim_ms, report):
    t = traces[f"{key}_t_s"] * 1e3
    base = meta[key]["trace_baseline_hz"]
    df = traces[f"{key}_freq_hz"] - base
    tp = traces[f"{key}_period_t_s"] * 1e3
    fp = traces[f"{key}_period_freq_hz"] - base
    m = (tp > xlim_ms[0]) & (tp < xlim_ms[1])
    ax.plot(t, df, color=TRACE_COLOR, lw=TRACE_LW, zorder=3)
    # the single EOD periods on top of the trace (where they coincide with
    # it, as in the small chirp, they would otherwise be hidden)
    ax.plot(tp[m], fp[m], ".", ms=DOT_MS, color=DOT_COLOR, mew=0, zorder=3.5,
            rasterized=True)
    ax.axhline(0, color="k", lw=0.5, ls=":", zorder=0)
    ax.set_xlim(*xlim_ms)
    ppx, ppy = points_per_unit(ax)

    height, hsd = row.height_hz, row.height_sd_hz
    fwhm, fsd = row.fwhm_s * 1e3, row.fwhm_sd_s * 1e3
    w10, wsd = row.width10_s * 1e3, row.width10_sd_s * 1e3
    under, usd = row.undershoot_hz, row.undershoot_sd_hz
    inside = (t > xlim_ms[0]) & (t < xlim_ms[1])
    ipk = np.flatnonzero(inside)[np.nanargmax(df[inside])]
    t_pk, trace_peak = float(t[ipk]), float(df[ipk])
    span = np.diff(xlim_ms)[0]
    yr = np.diff(ax.get_ylim())[0]
    out = {"trace_peak_hz": trace_peak, "trace_peak_t_ms": t_pk,
           "decoder": {"height_hz": [height, hsd], "fwhm_ms": [fwhm, fsd],
                       "width10_ms": [w10, wsd], "undershoot_hz": [under, usd]},
           "arrow_shrink_pt": {}}
    base_clear = ARROW_GAP_PT + 0.25  # from the dotted baseline (0.5 pt)
    # widths at 50 % and 10 % of the decoded height, centred on the trace's
    # own crossings of that level (the decoder estimates no timing); the
    # arrowheads stop ARROW_GAP_PT short of the trace
    for frac, w, name in ((0.5, fwhm, "FWHM"), (0.1, w10, r"10\,\% width")):
        level = frac * height
        a, b = crossings(t, df, ipk, level)
        c = 0.5 * (a + b)
        lo, hi = c - w / 2, c + w / 2
        sa, sb = tip_shrink((lo - a) * ppx), tip_shrink((b - hi) * ppx)
        dim_arrow(ax, (lo, level), (hi, level), shrink_a=sa, shrink_b=sb)
        ax.text(b + 0.05 * span, level, name, ha="left", va="center", fontsize="small")
        out[f"trace_crossings_ms_{name}"] = [float(a), float(b)]
        out["arrow_shrink_pt"][name] = [round(sa, 2), round(sb, 2)]
    # height: from the baseline to the decoded height at (just after) the
    # trace's peak, behind the trace, which crosses it
    th = t_pk + HEIGHT_ARROW_DT_MS[key]
    trace_at = float(np.interp(th, t, df))
    st = tip_shrink((height - trace_at) * ppy)
    dim_arrow(ax, (th, 0.0), (th, height), shrink_a=base_clear, shrink_b=st)
    out["arrow_shrink_pt"]["height"] = [round(base_clear, 2), round(st, 2)]
    out["height_arrow_t_ms"] = th
    if key == "big":  # above the per-period dots, right of the arrow
        ax.text(th + 0.04 * span, height + 0.06 * yr, "height", ha="left",
                va="center", fontsize="small")
    else:
        ax.text(th + 0.05 * span, height, "height", ha="left", va="center",
                fontsize="small")
    if under > 0.05 * height:
        # undershoot: an arrow from the baseline down to the decoded
        # undershoot, where the trace has come back up to a quarter of it
        # after its trough (the trough itself is too narrow for an arrow; a
        # second head at the baseline would sit on the trace); label below
        after = (t > t_pk) & (t < xlim_ms[1])
        i_tr = np.flatnonzero(after)[np.nanargmin(df[after])]
        j = i_tr + np.flatnonzero(df[i_tr:] >= -under / 4)[0]
        tu = float(t[j])
        ax.add_patch(FancyArrowPatch((tu, 0.0), (tu, -under), arrowstyle="-|>",
                                     mutation_scale=DIM_MS, lw=DIM_LW, color="k",
                                     shrinkA=0, shrinkB=0, zorder=1.5, clip_on=False))
        ax.text(tu - 0.01 * span, -under - 0.012 * yr, "undershoot", ha="left",
                va="top", fontsize="small")
        out["undershoot_drawn"] = True
        out["undershoot_arrow_t_ms"] = tu
    else:
        out["undershoot_drawn"] = False
    ax.set_xlabel("Time [ms]")
    report[key] = out


def tile_data() -> dict:
    """The 2020-05-29 tile of B (harmonic) and C (decoder examples), with the
    selected run's ids of its example chirps (``ids``)."""
    snip = C.load_snippet(TILE_SNIPPET)
    rows = C.window(snip, *TILE)
    rows = rows[rows.operating]
    fish, order = C.fish_styles(snip, *TILE)
    rec = snip["meta"]["recording"]
    small, big, harmonic = example_ids(rows, rec, [SMALL, BIG, HARMONIC])
    return {"snip": snip, "rows": rows, "fish": fish, "order": order,
            "ids": {"small": small, "big": big, "harmonic": harmonic}}


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {}
    det = detect_data()
    tile = tile_data()
    by_tile = tile["rows"].set_index("chirp_id")
    z = np.load(C.SNIPPETS / "traces.npz")
    traces = {k: z[k] for k in z.files if k != "meta"}
    tmeta = json.loads(str(z["meta"]))
    # the traces must be those of the chirps drawn (v2_fig_extract traces resolves
    # the same v2 ids in the same run)
    for key in ("small", "big"):
        if int(tmeta[key]["chirp_id"]) != tile["ids"][key]:
            msg = (f"traces.npz {key}: chirp {tmeta[key]['chirp_id']}, the run's is "
                   f"{tile['ids'][key]}: re-run v2_fig_extract.py traces")
            raise ValueError(msg)

    fig = plt.figure(figsize=(W * C.CM, H * C.CM))
    ax_rail = rail(fig, det, tile)

    # --- A: detection and assignment (full width) ----------------------------
    ax_b = C.axes_cm(fig, L, B_BOTTOM, R - L, B_H)
    detection = panel_detection(ax_b, det, report)

    # --- B, C: post-processing and decoder -----------------------------------
    ax_c1 = C.axes_cm(fig, C_X[0], CD_BOTTOM, C_W, CD_H)
    ax_c2 = C.axes_cm(fig, C_X[1], CD_BOTTOM, C_W, CD_H)
    ax_d1 = C.axes_cm(fig, D_X[0], CD_BOTTOM, D_W, CD_H)
    ax_d2 = C.axes_cm(fig, D_X[1], CD_BOTTOM, D_W, CD_H)
    report["C_harmonic"] = panel_harmonic(ax_c1, tile)
    report["C_duplicate"] = panel_duplicate(ax_c2)
    # panel titles at the 8-pt base
    for ax, title in ((ax_c1, "harmonic"), (ax_c2, "duplicate")):
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom")
    ax_d1.set_ylim(-14, 86)
    ax_d2.set_ylim(-90, 475)
    panel_trace(ax_d1, traces, tmeta, "small", by_tile.loc[tile["ids"]["small"]], (-40, 40),
                report)
    panel_trace(ax_d2, traces, tmeta, "big", by_tile.loc[tile["ids"]["big"]], (-100, 150),
                report)
    ax_d1.set_xticks([-40, -20, 0, 20, 40])
    ax_d2.set_xticks([-100, -50, 0, 50, 100, 150])
    for ax in (ax_d1, ax_d2):
        ax.set_ylabel(r"EOD$f$ $-$ baseline [Hz]")
        sns.despine(ax=ax, offset=3, trim=True)
    key = DoubleArrowKey()
    handles = [
        Line2D([], [], color=TRACE_COLOR, lw=TRACE_LW),
        Line2D([], [], ls="none", marker=".", ms=DOT_MS + 1.2, mfc=DOT_COLOR, mec="none"),
        key,
    ]
    # one line above both C panels, at the height of B's titles
    ax_d1.legend(handles=handles, labels=["trace", "single EOD periods", "estimate"],
                 handler_map={DoubleArrowKey: DoubleArrowHandler()},
                 loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, ncol=3,
                 handlelength=1.6, columnspacing=1.6, borderaxespad=0.15,
                 borderpad=0.0)

    C.letters_cm(fig, [(ax_b, "A", LETTER_DX, 0.25), (ax_c1, "B", LETTER_DX, 0.4),
                       (ax_d1, "C", LETTER_DX, 0.4)])

    named = ([(f"rail{k}", ax) for k, ax in enumerate(ax_rail)]
             + [("A", ax_b), ("B1", ax_c1), ("B2", ax_c2), ("C1", ax_d1), ("C2", ax_d2)])
    extents = C.axes_extents(fig, named)
    paths = C.save(fig, NAME)
    # every stored row of the tile at score >= 0.5 (the operating column says
    # which ones the run keeps at its threshold)
    allrows = C.window(tile["snip"], *TILE)
    shown = allrows[allrows.score >= 0.5].set_index("chirp_id")
    cols = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score", "operating",
            "emitter_hz", "track_id", "track_hz", "emitter_track_df_hz", "assign_confidence",
            "pp_drop_reason", "pp_harmonic_of_track", "height_hz", "height_sd_hz",
            "fwhm_s", "fwhm_sd_s", "width10_s", "width10_sd_s", "undershoot_hz",
            "undershoot_sd_hz"]
    print(f"{tile['snip']['meta']['recording']} tile {TILE}, operating threshold "
          f"{C.OPERATING}:")
    print(shown[cols].round(4).to_string())
    print("trace baselines:", {k: round(v["trace_baseline_hz"], 2) for k, v in tmeta.items()})
    print(json.dumps(report, indent=1, default=float))
    print(extents)
    print("\n".join(map(str, paths)))
    write_stats("v2_pipeline_explainer", caption_stats(detection, report))


def caption_stats(detection, report) -> dict:
    """The numbers of the caption: snippets, boxes, scores and estimates of the panels."""
    harm, dup = report["C_harmonic"], report["C_duplicate"]

    def decoder(key):
        d = report[key]["decoder"]
        return {**{p: {"value": v, "sd": s} for p, (v, s) in d.items()},
                "trace_peak_hz": report[key]["trace_peak_hz"]}

    return {
        "operating_threshold": C.OPERATING,
        "tile": {"duration_s": C.tile_duration_s()},
        "detection": detection,
        "harmonic": {"box_hz": harm["harmonic_box"], "emitter_hz": harm["harmonic_emitter_hz"],
                     "fundamental_track_hz": harm["fundamental_track_hz"],
                     "score": harm["score"]},
        "duplicate": {"kept_score": dup["kept_score"], "dropped_score": dup["dropped_score"],
                      "separation_ms": dup["separation_ms"]},
        "decoder_small": decoder("small"),
        "decoder_big": decoder("big"),
    }


if __name__ == "__main__":
    main()
