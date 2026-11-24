"""Figure: chirp detection pipeline v2, from the grid recording to a parameterised chirp.

Replaces ``plot_assignment_feature_extraction.pdf`` (the v1 assignment
features). A rail of thumbnails on top previews the stages (recording ->
A spectrograms -> B detection -> B assignment -> C post-processing -> D
parameters); below, one real example per stage, all from the production
run ``competition2019_v2_20261113`` (dual-head RT-DETR detector, weights
9fcdef3c; chirp-parameter decoder c0801c56). Drawn to the manuscript's
figure standard (17.4 cm wide; no colour for fish identity on the
spectrograms: tracks as thin black lines labelled EOD$f_1$ / EOD$f_2$,
white boxes, solid for the fish EOD$f_1$ and dashed for EOD$f_2$, white
emitter markers; see ``v2_fig_common``):

A   one snippet (2020-06-05-10_00, 89.9-91.1 s: two fish 29 Hz apart, a
    small chirp of the lower and a big, short chirp of the upper fish) in the
    three analysis windows of the front end (205, 51 and 13 ms, on the long
    window's 4.88-Hz grid; the window drawn to scale in the corner), and the
    three stacked as the R, G, B channels of the image the detector receives
    (the detector's own 0-1 levels, a fixed 60-dB range; the R/G/B chips are
    the image's channels). The multi-window input is a design choice: a
    front-end ablation found it comparable to the long window alone (see
    v2_figures.md);
B   detection and assignment on 8.2 s of one recording (2020-06-02-10_00,
    2245.2-2253.4 s, ``DETECT``; the author's choice among three candidates,
    2026-11-24): every detection at the operating threshold as a box with the
    predicted emitter EOD$f$ (circle), in the line style of the fish it was
    assigned to (the track nearest the predicted emitter, if within 20 Hz:
    brackets at the right);
C   post-processing removes a box on the second harmonic (2020-05-29-10_00,
    the tile of the former panel B: chirp 2084 on the second harmonic of
    chirp 2085) and a duplicate box on one chirp (2019-12-02-10_00, chirps
    244 kept / 245 removed, both above the operating threshold of 0.8);
D   decoder estimates (calibrated, +- 1 calibrated SD) of a small and a big
    chirp of the 2020-05-29 tile (2085, the fundamental of C's harmonic copy,
    and 2086), next to their instantaneous-frequency traces extracted from
    the raw recording: two independent estimates.

Spectrograms (A's three windows, B, C and the thumbnails): power in dB re
1 V^2 (summed over electrodes), display levels the 2nd and 98th percentile
of each panel's crop (``C.CLIP_PCT``), with a small colour bar per panel.

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import v2_fig_common as C
from manuscript_stats import write_stats
from style import other_color, set_light_style

NAME = "plot_v2_pipeline_explainer"

# B: detection and assignment, 8.2 s of one recording (snippet ``f1_detect``
# of v2_fig_extract). ``example``: the chirp of the detection thumbnail and of
# the "box" / "predicted emitter" arrows (text positions, s from the window
# start and Hz; ``box_side``: the box edge the arrow points at); ``pair``: a
# chirp of each fish for the assignment thumbnail; ``scalebar``: end (s) and
# height (Hz)
DETECT = {
    "snippet": "f1_detect",
    "window": (2245.2, 2253.4),
    "band": (620.0, 1270.0),
    "ticks": (700, 900, 1100),
    "example": 2014,
    "box_text": (7.3, 1150.0),
    "box_side": "left",
    "emitter_text": (7.5, 985.0),
    "pair": (2004, 2003),
    "scalebar": (8.1, 655.0),
}

# C and D: the tile of the former panel B (2020-05-29-10_00, 10850.4-10853.6 s)
TILE_SNIPPET = "f1_tile"
TILE = (10850.4, 10853.6)
SMALL, BIG, HARMONIC = 2085, 2086, 2084
# duplicate example (author's choice 2026-11-16): both boxes >= 0.8
DUP_SNIPPET, DUP_KEPT, DUP_DROPPED = "f1_dup", 244, 245
FRONT_END = "f1_fe_e"  # 2020-06-05-10_00, 89.9-91.1 s: fish 29 Hz apart

# channel colours of the detector's RGB input (R, G, B = 205, 51, 13 ms)
CHANNEL_CHIPS = ("#d62728", "#2ca02c", "#1f5fd6")
TRACE_COLOR = other_color  # the measured frequency trace (indigo of Figs. 5-7)
DOT_COLOR = "#a0a0a0"  # single EOD periods
SD_COLOR = "#b5b5b5"  # +- SD patches of the decoder widths
ARROW_GREY = "#8c8c8c"  # flow arrows of the rail and of A

W, H = 17.4, 16.4  # figure size [cm]


def window_glyph(ax, x0, y0, dur_s, height, color="white"):
    """The analysis window (Hann) drawn to scale on the time axis."""
    x = np.linspace(0, dur_s, 60)
    y = 0.5 * (1 - np.cos(2 * np.pi * x / dur_s))
    ax.plot(x0 + x, y0 + height * y, color=color, lw=0.8, zorder=20,
            solid_capstyle="round")
    ax.plot([x0 - 0.01, x0 + dur_s + 0.01], [y0, y0], color=color, lw=0.5, zorder=20)


def detect_data() -> dict:
    """B's snippet, its operating rows, the fish line styles and the order."""
    snip = C.load_snippet(DETECT["snippet"])
    t0, t1 = DETECT["window"]
    rows = C.window(snip, t0, t1)
    cut = rows[rows.operating & ((rows.t_start_s < t0) | (rows.t_end_s > t1))]
    if len(cut):
        msg = f"B window {DETECT['window']} cuts boxes {cut.chirp_id.tolist()}"
        raise ValueError(msg)
    fish, order = C.fish_styles(snip, t0, t1)
    return {"snip": snip, "rows": rows, "fish": fish, "order": order}


# ---------------------------------------------------------------------------
# rail of thumbnails
# ---------------------------------------------------------------------------
def rail(fig, det, tile, traces, tmeta):
    tw, th, y0 = 1.55, 0.95, H - 1.3
    x_first, x_last = 0.6, W - 0.6 - tw
    xs = np.linspace(x_first, x_last, 6)
    axs = [C.axes_cm(fig, x, y0, tw, th) for x in xs]

    # recording: raw voltage of one electrode (60 ms)
    z = np.load(C.SNIPPETS / "raw.npz")
    sig, fs = z["signal"].astype(float), float(z["samplerate_hz"])
    i0 = int(0.74 * fs)
    seg = sig[i0:i0 + int(0.06 * fs)]
    axs[0].plot(np.arange(len(seg)) / fs, seg, color="k", lw=0.35)
    axs[0].set_xlim(0, len(seg) / fs)
    pad = 0.15 * np.ptp(seg)
    axs[0].set_ylim(seg.min() - pad, seg.max() + pad)
    # A: the detector's RGB input
    fe = C.load_snippet(FRONT_END)
    C.show_rgb(axs[1], fe, fe["meta"]["t_start_s"] + 0.15, fe["meta"]["t_end_s"] - 0.15,
               770, 1250)
    # B detection: one box and its emitter
    snip, by, fish = det["snip"], det["rows"].set_index("chirp_id"), det["fish"]
    ex = by.loc[DETECT["example"]]
    t0, t1 = ex.t_center_s - 0.32, ex.t_center_s + 0.32
    f0, f1 = ex.f_low_hz - 30, ex.f_high_hz + 30
    # thumbnails: the display rule of the panels, on each thumbnail's own crop
    C.show_power(axs[2], snip, t0, t1, f0, f1)
    C.draw_tracks(axs[2], snip, fish, t0, t1, lw=0.5)
    C.fish_box(axs[2], ex, t0, fish, lw=0.8, ms=2.4)
    # B assignment: a chirp of each fish, each box in its fish's line style
    a, b = (by.loc[c] for c in DETECT["pair"])
    t0 = min(a.t_start_s, b.t_start_s) - 0.06
    t1 = max(a.t_end_s, b.t_end_s) + 0.06
    f0 = min(a.f_low_hz, b.f_low_hz) - 20
    f1 = max(a.f_high_hz, b.f_high_hz) + 20
    C.show_power(axs[3], snip, t0, t1, f0, f1)
    C.draw_tracks(axs[3], snip, fish, t0, t1, lw=0.5)
    for r in (a, b):
        C.fish_box(axs[3], r, t0, fish, lw=0.8, ms=2.4)
    # C post-processing: the harmonic copy, removed
    tby, tfish = tile["rows"].set_index("chirp_id"), tile["fish"]
    s, h = tby.loc[SMALL], tby.loc[HARMONIC]
    t0, t1 = s.t_center_s - 0.3, s.t_center_s + 0.3
    C.show_power(axs[4], tile["snip"], t0, t1, 590, 1510)
    C.fish_box(axs[4], s, t0, tfish, lw=0.8, marker=False)
    C.draw_box(axs[4], h, t0, C.DROPPED_COLOR, ls=C.DROPPED_LS, lw=0.8, marker=False)
    # D parameters: the big chirp's frequency trace
    t = traces["big_t_s"] * 1e3
    df = traces["big_freq_hz"] - tmeta["big"]["trace_baseline_hz"]
    m = (t > -100) & (t < 120)
    axs[5].plot(t[m], df[m], color=TRACE_COLOR, lw=0.9)
    axs[5].set_xlim(-100, 120)
    axs[5].set_ylim(-90, 470)
    for ax in axs:
        C.rounded_clip(ax)
    labels = ["Recording", r"\textbf{A}\, Spectrograms", r"\textbf{B}\, Detection",
              r"\textbf{B}\, Assignment", r"\textbf{C}\, Post-processing",
              r"\textbf{D}\, Parameters"]
    tr = C.fig_cm(fig)
    for x, lab in zip(xs, labels, strict=True):
        fig.text(x + tw / 2, y0 - 0.1, lab, ha="center", va="top", transform=tr)
    for xa, xb in zip(xs[:-1], xs[1:], strict=True):
        C.flow_arrow(fig, xa + tw + 0.14, y0 + th / 2, xb - 0.14, y0 + th / 2,
                     color=ARROW_GREY, lw=0.7, rad=-0.25, ms=5)


# ---------------------------------------------------------------------------
# A: front end
# ---------------------------------------------------------------------------
def panel_front_end(fig, axs, report):
    snip = C.load_snippet(FRONT_END)
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 740, 1380
    wins = m["windows_s"]
    levels = []
    for ch, ax in enumerate(axs[:3]):
        # power [dB], levels of this panel's crop (2nd-98th percentile); a
        # colour bar under each window
        im, lv = C.show_power(ax, snip, t0, t1, f0, f1, channel=ch)
        levels.append(lv)
        window_glyph(ax, 0.04, 1255, wins[ch], 85)
        bb = ax.get_position()
        C.power_bar(fig, im, bb.x0 * W + 0.5, bb.y0 * H - 0.27, bb.width * W - 1.15)
    # the detector's input: its own 0-1 levels (a fixed 60-dB range)
    C.show_rgb(axs[3], snip, t0, t1, f0, f1)
    C.spec_yaxis(axs[0], ticks=[800, 1000, 1200])
    for ax in axs[1:]:
        C.no_yaxis(ax)
    # titles: window length with the channel chip of the RGB image
    titles = [f"{round(w * 1e3):d} ms" for w in wins] + ["detector input"]
    for k, (ax, title) in enumerate(zip(axs, titles, strict=True)):
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom")
        if k < 3:
            w = C.text_width_cm(fig, title, fontsize="medium")
            ax_w = ax.get_position().width * W
            xc = 0.5 - (w / 2 + 0.12) / ax_w
            ax.text(xc, 1.045, "RGB"[k], transform=ax.transAxes, ha="right",
                    va="bottom", fontsize="small", color="white", fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", fc=CHANNEL_CHIPS[k], ec="none"))
    # the trade-off, under the two extreme windows
    for ax, note in ((axs[0], "fish resolved,\nchirps blurred"),
                     (axs[2], "fish merged,\nchirps sharp")):
        off = mpl.transforms.offset_copy(ax.transAxes, fig=fig, y=-0.74 / 2.54, units="inches")
        ax.text(0.5, 0, note, transform=off, ha="center", va="top", fontsize="small",
                linespacing=1.15)
    order = C.fish_order(snip, t0, t1)
    hz = [C.track_median(snip, tid, t0, t1) for tid in order]
    C.eod_label(axs[0], 0.02, hz[0] + 40, 1)
    C.eod_label(axs[0], 0.02, hz[1] - 46, 2, **C.DARK_BACK)
    axs[0].text(0.04 + wins[0] + 0.03, 1300, "window", ha="left", va="center", **C.ANNOT)
    C.scalebar(axs[2], t1 - t0 - 0.04, 1290, 0.2, "200 ms")
    rows = C.window(snip, t0, t1)
    rows = rows[rows.chirp]
    report["A"] = {
        "recording": m["recording"], "window_s": [t0, t1], "windows_s": wins,
        "levels_db": [[round(a, 1), round(b, 1)] for a, b in levels],
        "tracks_hz": [round(h, 1) for h in hz], "df_hz": round(hz[0] - hz[1], 1),
        "chirps": rows[["chirp_id", "t_center_s", "score", "track_id", "track_hz",
                        "height_hz", "fwhm_s"]].round(3).to_dict(orient="records"),
    }
    return {"tracks_hz": hz, "df_hz": hz[0] - hz[1], "windows_ms": [1e3 * w for w in wins],
            "snippet_s": t1 - t0}


# ---------------------------------------------------------------------------
# B: detection and assignment
# ---------------------------------------------------------------------------
def tolerance_brackets(ax, hz, *, dx_cm=0.12, tick_cm=0.07, label=r"$\pm$20 Hz"):
    """The assignment tolerance around each track: black brackets right of the image."""
    w_cm = ax.get_position().width * ax.figure.get_size_inches()[0] / C.CM
    tr = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    x, xt = 1 + dx_cm / w_cm, 1 + (dx_cm - tick_cm) / w_cm
    for f in hz:
        lo, hi = f - C.TOLERANCE_HZ, f + C.TOLERANCE_HZ
        ax.plot([xt, x, x, xt], [lo, lo, hi, hi], color="k", lw=0.6, transform=tr,
                clip_on=False, solid_joinstyle="miter")
        ax.text(x + 0.08 / w_cm, f, label, transform=tr, ha="left", va="center",
                fontsize=C.ANNOT_FS)


def panel_detection(ax, det, report) -> dict:
    snip, rows, fish, order = det["snip"], det["rows"], det["fish"], det["order"]
    t0, t1 = DETECT["window"]
    f0, f1 = DETECT["band"]
    im, lv = C.show_power(ax, snip, t0, t1, f0, f1)
    det["image"] = im
    C.draw_tracks(ax, snip, fish, t0, t1)
    drawn = C.draw_detections(ax, rows, fish, t0, ms=3.0)
    C.spec_yaxis(ax, ticks=list(DETECT["ticks"]))
    hz = [C.track_median(snip, tid, t0, t1) for tid in order]
    C.eod_label(ax, 0.06, hz[0] + 26, 1)
    C.eod_label(ax, 0.06, hz[1] - 26, 2)
    tolerance_brackets(ax, hz)
    # what the detector returns for each chirp: a box and the emitter EODf
    ex = rows.set_index("chirp_id").loc[DETECT["example"]]
    left = DETECT["box_side"] == "left"
    edge = ex.t_start_s if left else ex.t_end_s
    ha, rad = ("right", -0.25) if left else ("left", 0.25)
    C.annotate(ax, "box", xy=(edge - t0, ex.f_high_hz - 0.2 * (ex.f_high_hz - ex.f_low_hz)),
               xytext=DETECT["box_text"], ha=ha, rad=rad)
    C.annotate(ax, "predicted\nemitter EOD$f$", xy=(ex.t_center_s - t0, ex.emitter_hz),
               xytext=DETECT["emitter_text"], ha=ha, rad=rad, shrink_b=3.5)
    C.scalebar(ax, *DETECT["scalebar"], 1.0, "1 s")

    kept = drawn[drawn.drawn_as == "assigned"]
    if (drawn.drawn_as == "unassigned").any() or len(kept) != int(rows.chirp.sum()):
        msg = "B: every chirp of the window must be drawn as assigned"
        raise ValueError(msg)
    per_fish = [int((kept.track_id == tid).sum()) for tid in order]
    spans_both = int(((kept.f_low_hz < min(hz)) & (kept.f_high_hz > max(hz))).sum())
    dist = np.abs(kept.emitter_hz - kept.track_hz)
    report["B"] = {
        "recording": snip["meta"]["recording"], "window_s": [t0, t1], "levels_db": lv,
        "tracks_hz": [round(h, 1) for h in hz], "track_ids": order,
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
# C: post-processing
# ---------------------------------------------------------------------------
def panel_harmonic(ax, tile):
    snip, by, fish = tile["snip"], tile["rows"].set_index("chirp_id"), tile["fish"]
    h, f = by.loc[HARMONIC], by.loc[SMALL]
    t0, t1 = f.t_center_s - 0.3, f.t_center_s + 0.3
    im, lv = C.show_power(ax, snip, t0, t1, 580, 1520)
    C.draw_tracks(ax, snip, fish, t0, t1)
    C.fish_box(ax, f, t0, fish, lw=1.0)
    C.draw_box(ax, h, t0, C.DROPPED_COLOR, ls=C.DROPPED_LS, lw=1.0, marker=False)
    x = f.t_center_s - t0
    ax.annotate("", xy=(x, h.f_low_hz), xytext=(x, f.f_high_hz),
                arrowprops=C.arrowprops(rad=0.0, shrink_a=3, shrink_b=3))
    ax.text(x - 0.025, 0.5 * (f.f_high_hz + h.f_low_hz), r"2$\times$", ha="right",
            va="center", **C.ANNOT)
    C.spec_yaxis(ax, ticks=[600, 800, 1000, 1200, 1400])
    C.scalebar(ax, 0.51, 960, 0.1, "100 ms")
    return {"harmonic": HARMONIC, "harmonic_box": [float(h.f_low_hz), float(h.f_high_hz)],
            "harmonic_emitter_hz": float(h.emitter_hz),
            "fundamental_track_hz": float(f.track_hz), "score": float(h.score),
            "levels_db": lv, "image": im}


def panel_duplicate(ax):
    snip = C.load_snippet(DUP_SNIPPET)
    rows = C.window(snip, snip["meta"]["t_start_s"], snip["meta"]["t_end_s"])
    by = rows.set_index("chirp_id")
    k, d = by.loc[DUP_KEPT], by.loc[DUP_DROPPED]
    t0, t1 = k.t_center_s - 0.3, k.t_center_s + 0.3
    fish, _ = C.fish_styles(snip, t0 - 2, t1 + 2)
    im, lv = C.show_power(ax, snip, t0, t1, 580, 1520)
    C.draw_tracks(ax, snip, fish, t0, t1)
    C.fish_box(ax, k, t0, fish, lw=1.0)
    C.draw_box(ax, d, t0, C.DROPPED_COLOR, ls=C.DROPPED_LS, lw=1.0, marker=False)
    C.no_yaxis(ax)
    return {"recording": snip["meta"]["recording"], "kept": DUP_KEPT,
            "kept_score": float(k.score), "kept_box": [float(k.f_low_hz), float(k.f_high_hz)],
            "dropped": DUP_DROPPED, "dropped_score": float(d.score),
            "dropped_box": [float(d.f_low_hz), float(d.f_high_hz)],
            "separation_ms": 1e3 * abs(float(k.t_center_s - d.t_center_s)),
            "levels_db": lv, "image": im}


# ---------------------------------------------------------------------------
# D: decoder estimates next to the frequency trace
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


def width_bar(ax, c, w, sd, y, cap):
    """Decoder width w +- sd centred on c: black bar with caps, grey SD at the ends."""
    for e in (c - w / 2, c + w / 2):
        ax.add_patch(Rectangle((e - sd / 2, y - cap), sd, 2 * cap, fc=SD_COLOR,
                               ec="none", zorder=3))
    ax.plot([c - w / 2, c + w / 2], [y, y], color="k", lw=0.9, zorder=4,
            solid_capstyle="butt")
    for e in (c - w / 2, c + w / 2):
        ax.plot([e, e], [y - cap, y + cap], color="k", lw=0.9, zorder=4)


def est_marker(ax, x, y, sd):
    ax.errorbar([x], [y], yerr=[sd], fmt="o", ms=3.4, mfc="white", mec="k", mew=0.7,
                ecolor="k", elinewidth=0.8, capsize=0, zorder=6, clip_on=False)


def panel_trace(ax, traces, meta, key, row, xlim_ms, report):
    t = traces[f"{key}_t_s"] * 1e3
    base = meta[key]["trace_baseline_hz"]
    df = traces[f"{key}_freq_hz"] - base
    tp = traces[f"{key}_period_t_s"] * 1e3
    fp = traces[f"{key}_period_freq_hz"] - base
    m = (tp > xlim_ms[0]) & (tp < xlim_ms[1])
    ax.plot(t, df, color=TRACE_COLOR, lw=1.2, zorder=2)
    ax.plot(tp[m], fp[m], ".", ms=1.7, color=DOT_COLOR, mew=0, zorder=2.5, rasterized=True)
    ax.axhline(0, color="k", lw=0.5, ls=":", zorder=0)

    height, hsd = row.height_hz, row.height_sd_hz
    fwhm, fsd = row.fwhm_s * 1e3, row.fwhm_sd_s * 1e3
    w10, wsd = row.width10_s * 1e3, row.width10_sd_s * 1e3
    under, usd = row.undershoot_hz, row.undershoot_sd_hz
    inside = (t > xlim_ms[0]) & (t < xlim_ms[1])
    ipk = np.flatnonzero(inside)[np.nanargmax(df[inside])]
    t_pk, trace_peak = float(t[ipk]), float(df[ipk])
    span = np.diff(xlim_ms)[0]
    yr = np.diff(ax.get_ylim())[0]
    cap = 0.025 * yr
    out = {"trace_peak_hz": trace_peak, "trace_peak_t_ms": t_pk,
           "decoder": {"height_hz": [height, hsd], "fwhm_ms": [fwhm, fsd],
                       "width10_ms": [w10, wsd], "undershoot_hz": [under, usd]}}
    # widths at 50 % and 10 % of the decoded height, centred on the trace's
    # own crossings of that level (the decoder estimates no timing)
    for frac, w, sd, name in ((0.5, fwhm, fsd, "FWHM"), (0.1, w10, wsd, r"10\,\% width")):
        level = frac * height
        a, b = crossings(t, df, ipk, level)
        c = 0.5 * (a + b)
        width_bar(ax, c, w, sd, level, cap)
        ax.text(c + w / 2 + sd / 2 + 0.05 * span, level, name, ha="left", va="center",
                fontsize="small")
        out[f"trace_crossings_ms_{name}"] = [float(a), float(b)]
    # height: marker +- SD at the trace's peak time
    est_marker(ax, t_pk, height, hsd)
    if key == "big":  # above the per-period dots, right of the marker
        ax.text(t_pk + 0.05 * span, height + 0.075 * yr, "height", ha="left",
                va="center", fontsize="small")
    else:
        ax.text(t_pk + 0.05 * span, height, "height", ha="left", va="center",
                fontsize="small")
    if under > 0.05 * height:
        after = (t > t_pk) & (t < xlim_ms[1])
        tu = float(t[after][np.nanargmin(df[after])])
        est_marker(ax, tu + 0.04 * span, -under, usd)
        ax.text(tu + 0.09 * span, -under, "undershoot", ha="left", va="center",
                fontsize="small")
        out["undershoot_drawn"] = True
    else:
        out["undershoot_drawn"] = False
    ax.set_xlim(*xlim_ms)
    ax.set_xlabel("Time [ms]")
    report[key] = out


def tile_data() -> dict:
    """The 2020-05-29 tile of C (harmonic) and D (decoder examples)."""
    snip = C.load_snippet(TILE_SNIPPET)
    rows = C.window(snip, *TILE)
    rows = rows[rows.operating]
    fish, order = C.fish_styles(snip, *TILE)
    return {"snip": snip, "rows": rows, "fish": fish, "order": order}


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

    fig = plt.figure(figsize=(W * C.CM, H * C.CM))
    rail(fig, det, tile, traces, tmeta)

    # --- A: front end ------------------------------------------------------
    ya, ha, wa = 11.45, 2.7, 3.45
    xs = [1.55, 5.25, 8.95, 13.75]
    ax_a = [C.axes_cm(fig, xx, ya, wa, ha) for xx in xs]
    front_end = panel_front_end(fig, ax_a, report)
    C.flow_arrow(fig, xs[2] + wa + 0.3, ya + ha / 2, xs[3] - 0.3, ya + ha / 2,
                 color=ARROW_GREY, lw=0.7, ms=5)

    # --- B: detection and assignment -----------------------------------------
    ax_b = C.axes_cm(fig, 1.55, 6.0, 14.6, 3.9)
    detection = panel_detection(ax_b, det, report)
    # key of B's symbols (as in the failure-mode figure) and the colour bar,
    # under B
    C.key_strip(fig, 1.55, 5.6, [("boxdot", {"ls": C.FISH_LS[0]}, r"EOD$f_1$"),
                                  ("boxdot", {"ls": C.FISH_LS[1]}, r"EOD$f_2$"),
                                  ("track", {}, "track")])
    C.power_bar(fig, det["image"], 12.35, 5.6 - 0.065, 2.4, label=r"dB re 1 V$^2$",
                ticks_before=True)

    # --- C, D: post-processing and decoder -----------------------------------
    yd, hd = 1.45, 3.4
    ax_c1 = C.axes_cm(fig, 1.55, yd, 2.15, hd)
    ax_c2 = C.axes_cm(fig, 3.85, yd, 2.15, hd)
    ax_d1 = C.axes_cm(fig, 7.75, yd, 3.85, hd)
    ax_d2 = C.axes_cm(fig, 13.2, yd, 3.85, hd)
    report["C_harmonic"] = panel_harmonic(ax_c1, tile)
    report["C_duplicate"] = panel_duplicate(ax_c2)
    # panel titles at the 8-pt base, as the window titles of A
    for ax, title in ((ax_c1, "harmonic"), (ax_c2, "duplicate")):
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom")
    # a colour bar under each (levels of its own crop)
    for ax, key in ((ax_c1, "C_harmonic"), (ax_c2, "C_duplicate")):
        bb = ax.get_position()
        C.power_bar(fig, report[key].pop("image"), bb.x0 * W + 0.1, yd - 0.27,
                    bb.width * W - 0.6)
    C.key_strip(fig, 1.55, 0.45, [("box", {"color": C.DROPPED_COLOR, "ls": C.DROPPED_LS},
                                   "removed")])
    ax_d1.set_ylim(-14, 86)
    ax_d2.set_ylim(-75, 475)
    panel_trace(ax_d1, traces, tmeta, "small", by_tile.loc[SMALL], (-40, 40), report)
    panel_trace(ax_d2, traces, tmeta, "big", by_tile.loc[BIG], (-100, 150), report)
    ax_d1.set_xticks([-40, -20, 0, 20, 40])
    ax_d2.set_xticks([-100, -50, 0, 50, 100, 150])
    for ax in (ax_d1, ax_d2):
        ax.set_ylabel(r"EOD$f$ $-$ baseline [Hz]")
        sns.despine(ax=ax, offset=3, trim=True)
    handles = [
        Line2D([], [], color=TRACE_COLOR, lw=1.2, marker=".", ms=3, mfc=DOT_COLOR,
               mec="none", label="trace"),
        Line2D([], [], color="k", lw=0.8, marker="o", ms=3.4, mfc="white", mec="k",
               mew=0.7, label=r"estimate $\pm$ SD"),
    ]
    ax_d1.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.04, 0.74),
                 frameon=False, handlelength=1.4, borderaxespad=0.1)

    C.letters_cm(fig, [(ax_a[0], "A", 1.25, 0.35), (ax_b, "B", 1.25, 0.25),
                       (ax_c1, "C", 1.25, 0.25), (ax_d1, "D", 1.25, 0.25)])

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
    print("\n".join(map(str, paths)))
    write_stats("v2_pipeline_explainer", caption_stats(front_end, detection, report))


def caption_stats(front_end, detection, report) -> dict:
    """The numbers of the caption: snippets, boxes, scores and estimates of the panels."""
    harm, dup = report["C_harmonic"], report["C_duplicate"]

    def decoder(key):
        d = report[key]["decoder"]
        return {**{p: {"value": v, "sd": s} for p, (v, s) in d.items()},
                "trace_peak_hz": report[key]["trace_peak_hz"]}

    return {
        "operating_threshold": C.OPERATING,
        "front_end": front_end,
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
