"""Figure: chirp detection pipeline v2, from the grid recording to a parameterised chirp.

Replaces ``plot_assignment_feature_extraction.pdf`` (the v1 assignment
features). A rail of thumbnails on top previews the stages (recording ->
A spectrograms -> B detection -> C assignment -> D post-processing -> E
parameters); below, one real example per stage, all from the production
run ``competition2019_v2_20261113`` (dual-head RT-DETR detector, weights
9fcdef3c; chirp-parameter decoder c0801c56):

A   one snippet (2020-06-05-10_00, 89.9-91.1 s: two fish 29 Hz apart, a
    small chirp of the lower and a big, short chirp of the upper fish) in the
    three analysis windows of the front end (205, 51 and 13 ms, on the long
    window's 4.88-Hz grid; display contrast per panel; the window drawn to
    scale in the corner), and the three stacked as the R, G, B channels of
    the image the detector receives (front end's own 0-1 levels). The
    multi-window input is a design choice: a front-end ablation found it
    comparable to the long window alone (see v2_figures.md);
B   one 3.2-s detector tile (2020-05-29-10_00, 10850.4 s), cropped to the
    band of the two fish's fundamentals, with every detection at the
    operating threshold: boxes in the colour of the fish they were assigned
    to, the predicted emitter EODf of each box (circle), the wavetracker
    tracks;
C   assignment of two chirps 61 ms apart whose boxes overlap the other
    fish's baseline: each goes to the live track nearest its predicted
    emitter EODf, within 20 Hz (shaded);
D   post-processing removes a box on the second harmonic (the first chirp of
    B) and a duplicate box on one chirp (2020-03-18-10_34, a draw not among
    the 33 analysed trials);
E   decoder estimates (calibrated, +- 1 calibrated SD) of the first small
    chirp of B and the big chirp of C, next to their instantaneous-frequency
    traces extracted from the raw recording: two independent estimates.

Display levels of B-D: chirp_levels() (85th percentile of the crop -> 99th
percentile of the chirp pixels away from the baselines).

Inputs: the snippet/trace/raw cache of ``v2_fig_extract.py`` (run it first,
in the upgrade env) and the run's chirp table (``C.RUN``/``C.PARQUET``,
chosen by PAPER_DETECTIONS).

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_pipeline_explainer.py

Runtime: ~20 s cold, ~4 s warm (LaTeX text rendering). Prints the numbers it shows.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import v2_fig_common as C
from style import set_light_style

NAME = "plot_v2_pipeline_explainer"

REC = "2020-05-29-10_00"
TILE = (10850.4, 10853.6)
SMALL, BIG, PAIR_SMALL, HARMONIC = 2085, 2086, 2087, 2084
DUP_KEPT, DUP_DROPPED = 49386, 49385
FRONT_END = "f1_fe_e"  # 2020-06-05-10_00, 89.9-91.1 s: fish 29 Hz apart
BAND = (590.0, 1190.0)  # B and C: the fundamentals of the two fish

# channel colours of the detector's RGB input (R, G, B = 205, 51, 13 ms)
CHANNEL_CHIPS = ("#d62728", "#2ca02c", "#1f5fd6")

W, H = 17.4, 15.3  # figure size [cm]


def window_glyph(ax, x0, y0, dur_s, height, color="white"):
    """The analysis window (Hann) drawn to scale on the time axis."""
    x = np.linspace(0, dur_s, 60)
    y = 0.5 * (1 - np.cos(2 * np.pi * x / dur_s))
    ax.plot(x0 + x, y0 + height * y, color=color, lw=0.8, zorder=20,
            solid_capstyle="round")
    ax.plot([x0 - 0.01, x0 + dur_s + 0.01], [y0, y0], color=color, lw=0.5, zorder=20)


# ---------------------------------------------------------------------------
# rail of thumbnails
# ---------------------------------------------------------------------------
def rail(fig, snip, rows, colors, order, rect_c, traces, tmeta, levels):
    tw, th, y0 = 1.55, 0.95, H - 1.3
    x_first, x_last = 0.6, W - 0.6 - tw
    xs = np.linspace(x_first, x_last, 6)
    axs = [C.axes_cm(fig, x, y0, tw, th) for x in xs]
    by = rows.set_index("chirp_id")

    # recording: raw voltage of one electrode (60 ms)
    z = np.load(C.SNIPPETS / "raw.npz")
    sig, fs = z["signal"].astype(float), float(z["samplerate_hz"])
    i0 = int(0.74 * fs)
    seg = sig[i0:i0 + int(0.06 * fs)]
    axs[0].plot(np.arange(len(seg)) / fs, seg, color=C.BLUE, lw=0.35)
    axs[0].set_xlim(0, len(seg) / fs)
    pad = 0.15 * np.ptp(seg)
    axs[0].set_ylim(seg.min() - pad, seg.max() + pad)
    # A: the detector's RGB input
    fe = C.load_snippet(FRONT_END)
    C.show_rgb(axs[1], fe, fe["meta"]["t_start_s"] + 0.15, fe["meta"]["t_end_s"] - 0.15,
               770, 1250)
    # B: one detection (box + emitter)
    s = by.loc[SMALL]
    t0, t1 = s.t_center_s - 0.32, s.t_center_s + 0.32
    C.show_spec(axs[2], snip, t0, t1, 600, 830, vmin=levels[0], vmax=levels[1])
    C.draw_box(axs[2], s, t0, colors[int(s.track_id)], lw=0.8, ms=2.4)
    # C: the two chirps with the tolerance bands
    t0, f0, t1, f1 = rect_c
    C.show_spec(axs[3], snip, t0, t1, f0, f1, vmin=levels[0], vmax=levels[1])
    for tid in order:
        hz = C.track_median(snip, tid, t0, t1)
        axs[3].axhspan(hz - C.TOLERANCE_HZ, hz + C.TOLERANCE_HZ, color=colors[tid],
                       alpha=0.35, lw=0)
    for cid in (BIG, PAIR_SMALL):
        r = by.loc[cid]
        C.draw_box(axs[3], r, t0, colors[int(r.track_id)], lw=0.8, ms=2.4)
    # D: the harmonic copy, removed
    h = by.loc[HARMONIC]
    t0, t1 = s.t_center_s - 0.3, s.t_center_s + 0.3
    C.show_spec(axs[4], snip, t0, t1, 590, 1510, vmin=levels[0], vmax=levels[1])
    C.draw_box(axs[4], s, t0, colors[int(s.track_id)], lw=0.8, marker=False)
    C.draw_box(axs[4], h, t0, C.DROPPED_COLOR, ls=C.DROPPED_LS, lw=0.8, marker=False)
    # E: the big chirp's frequency trace
    t = traces["big_t_s"] * 1e3
    df = traces["big_freq_hz"] - tmeta["big"]["trace_baseline_hz"]
    m = (t > -100) & (t < 120)
    axs[5].plot(t[m], df[m], color=colors[int(by.loc[BIG].track_id)], lw=0.9)
    axs[5].set_xlim(-100, 120)
    axs[5].set_ylim(-90, 470)
    for ax in axs:
        C.rounded_clip(ax)
    labels = ["Recording", r"\textbf{A}\, Spectrograms", r"\textbf{B}\, Detection",
              r"\textbf{C}\, Assignment", r"\textbf{D}\, Post-processing",
              r"\textbf{E}\, Parameters"]
    tr = C.fig_cm(fig)
    for x, lab in zip(xs, labels, strict=True):
        fig.text(x + tw / 2, y0 - 0.1, lab, ha="center", va="top", transform=tr)
    for xa, xb in zip(xs[:-1], xs[1:], strict=True):
        C.flow_arrow(fig, xa + tw + 0.14, y0 + th / 2, xb - 0.14, y0 + th / 2,
                     color="#8c8c8c", lw=0.7, rad=-0.25, ms=5)


# ---------------------------------------------------------------------------
# A: front end
# ---------------------------------------------------------------------------
def panel_front_end(fig, axs, report):
    snip = C.load_snippet(FRONT_END)
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 740, 1380
    wins = m["windows_s"]
    for ch, ax in enumerate(axs[:3]):
        # display contrast per panel (40th-99.8th percentile of the crop)
        C.show_spec(ax, snip, t0, t1, f0, f1, channel=ch, vmin=None, vmax=None)
        window_glyph(ax, 0.04, 1255, wins[ch], 85)
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
        ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center", va="top",
                fontsize="small", linespacing=1.15)
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
        "tracks_hz": [round(h, 1) for h in hz], "df_hz": round(hz[0] - hz[1], 1),
        "chirps": rows[["chirp_id", "t_center_s", "score", "track_id", "track_hz",
                        "height_hz", "fwhm_s"]].round(3).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# B: detector output on one tile; C: assignment
# ---------------------------------------------------------------------------
def panel_tile(ax, snip, rows, colors, order, rect_c, levels):
    t0, t1 = TILE
    f0, f1 = BAND
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=levels[0], vmax=levels[1])
    C.draw_tracks(ax, snip, colors, t0, t1)
    C.draw_detections(ax, rows, colors, t0)
    C.spec_yaxis(ax, ticks=[600, 800, 1000])
    for k, tid in enumerate(order):
        C.eod_label(ax, 0.03, C.track_median(snip, tid, t0, t1) + 36, k + 1,
                    color=colors[tid])
    s = rows.set_index("chirp_id").loc[SMALL]
    # what the detector returns for each chirp: a box and the emitter EODf
    # the box's right edge between the two bands (its top lies on EODf1)
    C.annotate(ax, "box", xy=(s.t_end_s - t0, 735), rad=0.2,
               xytext=(s.t_end_s - t0 + 0.3, 905), ha="left")
    C.annotate(ax, r"predicted emitter EOD$f$", xy=(s.t_center_s - t0, s.emitter_hz),
               xytext=(s.t_end_s - t0 + 0.3, 708), ha="left", rad=0.2, shrink_b=3.5)
    # time range shown in C
    x0, _, x1, _ = rect_c
    ax.add_patch(Rectangle((x0 - t0, f0 + 4), x1 - x0, f1 - f0 - 8, fill=False, ec="white",
                           lw=0.6, ls=(0, (2, 1.5)), zorder=15))
    ax.text(x0 - t0 - 0.03, f1 - 10, r"\textbf{C}", ha="right", va="top", **C.ANNOT)
    C.scalebar(ax, t1 - t0 - 0.06, 1110, 0.5, "500 ms")


def panel_assignment(ax, snip, rows, colors, order, rect_c, levels):
    by = rows.set_index("chirp_id")
    r_big, r_small = by.loc[BIG], by.loc[PAIR_SMALL]
    t0, f0, t1, f1 = rect_c
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=levels[0], vmax=levels[1])
    for tid in order:
        hz = C.track_median(snip, tid, t0, t1)
        ax.axhspan(hz - C.TOLERANCE_HZ, hz + C.TOLERANCE_HZ, color=colors[tid],
                   alpha=0.3, lw=0, zorder=1)
    C.draw_tracks(ax, snip, colors, t0, t1, lw=0.8)
    for r in (r_big, r_small):
        C.draw_box(ax, r, t0, colors[int(r.track_id)], lw=1.0, ms=3.2)
    C.no_yaxis(ax)
    hz1 = C.track_median(snip, order[0], t0, t1)
    hz2 = C.track_median(snip, order[1], t0, t1)
    xr = t1 - t0 - 0.008
    C.eod_label(ax, xr, hz1 + C.TOLERANCE_HZ + 9, None, color=colors[order[0]],
                ha="right", va="bottom", text=r"EOD$f_1 \pm$20 Hz")
    C.eod_label(ax, xr, hz2 + C.TOLERANCE_HZ + 9, None, color=colors[order[1]],
                ha="right", va="bottom", text=r"EOD$f_2 \pm$20 Hz")
    C.scalebar(ax, xr, 1110, 0.1, "100 ms")
    return {
        "big": {"emitter_hz": float(r_big.emitter_hz), "track_hz": float(r_big.track_hz),
                "df_hz": float(r_big.emitter_track_df_hz),
                "box": [float(r_big.f_low_hz), float(r_big.f_high_hz)]},
        "small": {"emitter_hz": float(r_small.emitter_hz),
                  "track_hz": float(r_small.track_hz),
                  "df_hz": float(r_small.emitter_track_df_hz),
                  "box": [float(r_small.f_low_hz), float(r_small.f_high_hz)]},
        "onset_difference_ms": 1e3 * float(r_small.t_start_s - r_big.t_start_s),
        "tracks_hz": [hz1, hz2],
    }


# ---------------------------------------------------------------------------
# D: post-processing
# ---------------------------------------------------------------------------
def panel_harmonic(ax, snip, rows, colors):
    by = rows.set_index("chirp_id")
    h, f = by.loc[HARMONIC], by.loc[SMALL]
    t0, t1 = f.t_center_s - 0.3, f.t_center_s + 0.3
    lv = C.chirp_levels(snip, t0, t1, 580, 1520, rows)
    C.show_spec(ax, snip, t0, t1, 580, 1520, vmin=lv[0], vmax=lv[1])
    C.draw_tracks(ax, snip, colors, t0, t1)
    C.draw_box(ax, f, t0, colors[int(f.track_id)], lw=1.0)
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
            "levels": lv}


def panel_duplicate(ax):
    snip = C.load_snippet("f1_dup")
    rows = C.window(snip, snip["meta"]["t_start_s"], snip["meta"]["t_end_s"])
    by = rows.set_index("chirp_id")
    k, d = by.loc[DUP_KEPT], by.loc[DUP_DROPPED]
    t0, t1 = k.t_center_s - 0.3, k.t_center_s + 0.3
    colors, _ = C.fish_colors(snip, t0 - 2, t1 + 2)
    lv = C.chirp_levels(snip, t0, t1, 580, 1520, rows)
    C.show_spec(ax, snip, t0, t1, 580, 1520, vmin=lv[0], vmax=lv[1])
    C.draw_tracks(ax, snip, colors, t0, t1)
    C.draw_box(ax, k, t0, colors[int(k.track_id)], lw=1.0)
    C.draw_box(ax, d, t0, C.DROPPED_COLOR, ls=C.DROPPED_LS, lw=1.0, marker=False)
    C.no_yaxis(ax)
    return {"recording": snip["meta"]["recording"], "kept": DUP_KEPT,
            "kept_score": float(k.score), "kept_box": [float(k.f_low_hz), float(k.f_high_hz)],
            "dropped": DUP_DROPPED, "dropped_score": float(d.score),
            "dropped_box": [float(d.f_low_hz), float(d.f_high_hz)], "levels": lv}


# ---------------------------------------------------------------------------
# E: decoder estimates next to the frequency trace
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
        ax.add_patch(Rectangle((e - sd / 2, y - cap), sd, 2 * cap, fc="#b5b5b5",
                               ec="none", zorder=3))
    ax.plot([c - w / 2, c + w / 2], [y, y], color="k", lw=0.9, zorder=4,
            solid_capstyle="butt")
    for e in (c - w / 2, c + w / 2):
        ax.plot([e, e], [y - cap, y + cap], color="k", lw=0.9, zorder=4)


def est_marker(ax, x, y, sd):
    ax.errorbar([x], [y], yerr=[sd], fmt="o", ms=3.4, mfc="white", mec="k", mew=0.7,
                ecolor="k", elinewidth=0.8, capsize=0, zorder=6, clip_on=False)


def panel_trace(ax, traces, meta, key, row, xlim_ms, color, report):
    t = traces[f"{key}_t_s"] * 1e3
    base = meta[key]["trace_baseline_hz"]
    df = traces[f"{key}_freq_hz"] - base
    tp = traces[f"{key}_period_t_s"] * 1e3
    fp = traces[f"{key}_period_freq_hz"] - base
    m = (tp > xlim_ms[0]) & (tp < xlim_ms[1])
    ax.plot(t, df, color=color, lw=1.2, zorder=2)
    ax.plot(tp[m], fp[m], ".", ms=1.7, color="#8c8c8c", mew=0, zorder=2.5, rasterized=True)
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


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {}
    snip = C.load_snippet("f1_tile")
    t0, t1 = TILE
    rows = C.window(snip, t0, t1)
    rows = rows[rows.operating]
    colors, order = C.fish_colors(snip, t0, t1)
    by_id = rows.set_index("chirp_id")
    z = np.load(C.SNIPPETS / "traces.npz")
    traces = {k: z[k] for k in z.files if k != "meta"}
    tmeta = json.loads(str(z["meta"]))
    r_big, r_small = by_id.loc[BIG], by_id.loc[PAIR_SMALL]
    rect_c = (min(r_big.t_start_s, r_small.t_start_s) - 0.05, BAND[0],
              max(r_big.t_end_s, r_small.t_end_s) + 0.22, BAND[1])
    # one set of display levels for B and C (the same data)
    levels = C.chirp_levels(snip, t0, t1, *BAND, rows)
    report["levels_BC"] = levels

    fig = plt.figure(figsize=(W * C.CM, H * C.CM))
    rail(fig, snip, rows, colors, order, rect_c, traces, tmeta, levels)

    # --- A: front end ------------------------------------------------------
    ya, ha, wa = 9.85, 2.7, 3.45
    xs = [1.55, 5.25, 8.95, 13.75]
    ax_a = [C.axes_cm(fig, xx, ya, wa, ha) for xx in xs]
    panel_front_end(fig, ax_a, report)
    C.flow_arrow(fig, xs[2] + wa + 0.3, ya + ha / 2, xs[3] - 0.3, ya + ha / 2,
                 color="#8c8c8c", lw=0.7, ms=5)

    # --- B, C: detection and assignment -------------------------------------
    yb, hb = 5.6, 3.1
    ax_b = C.axes_cm(fig, 1.55, yb, 9.3, hb)
    ax_c = C.axes_cm(fig, 11.2, yb, 6.0, hb)
    panel_tile(ax_b, snip, rows, colors, order, rect_c, levels)
    report["C"] = panel_assignment(ax_c, snip, rows, colors, order, rect_c, levels)

    # --- D, E: post-processing and decoder -----------------------------------
    yd, hd = 1.05, 3.7
    ax_d1 = C.axes_cm(fig, 1.55, yd, 2.15, hd)
    ax_d2 = C.axes_cm(fig, 3.85, yd, 2.15, hd)
    ax_e1 = C.axes_cm(fig, 7.75, yd, 3.85, hd)
    ax_e2 = C.axes_cm(fig, 13.35, yd, 3.85, hd)
    report["D_harmonic"] = panel_harmonic(ax_d1, snip, rows, colors)
    report["D_duplicate"] = panel_duplicate(ax_d2)
    for ax, title in ((ax_d1, "harmonic"), (ax_d2, "duplicate")):
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom",
                fontsize="small")
    C.key_strip(fig, 1.55, 0.45, [("box", {"color": C.DROPPED_COLOR, "ls": C.DROPPED_LS},
                                   "removed")])
    ax_e1.set_ylim(-14, 86)
    ax_e2.set_ylim(-75, 475)
    panel_trace(ax_e1, traces, tmeta, "small", by_id.loc[SMALL], (-40, 40),
                colors[int(by_id.loc[SMALL].track_id)], report)
    panel_trace(ax_e2, traces, tmeta, "big", by_id.loc[BIG], (-100, 150),
                colors[int(by_id.loc[BIG].track_id)], report)
    ax_e1.set_xticks([-40, -20, 0, 20, 40])
    ax_e2.set_xticks([-100, -50, 0, 50, 100, 150])
    for ax in (ax_e1, ax_e2):
        ax.set_ylabel(r"EOD$f$ $-$ baseline [Hz]")
        sns.despine(ax=ax, offset=3, trim=True)
    handles = [
        Line2D([], [], color=colors[int(by_id.loc[SMALL].track_id)], lw=1.2, marker=".",
               ms=3, mfc="#8c8c8c", mec="none", label="trace"),
        Line2D([], [], color="k", lw=0.8, marker="o", ms=3.4, mfc="white", mec="k",
               mew=0.7, label=r"estimate $\pm$ SD"),
    ]
    ax_e1.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.04, 0.74),
                 frameon=False, handlelength=1.4, borderaxespad=0.1)

    C.letters_cm(fig, [(ax_a[0], "A", 1.25, 0.35), (ax_b, "B", 1.25, 0.25),
                       (ax_c, "C", 0.35, 0.25), (ax_d1, "D", 1.25, 0.25),
                       (ax_e1, "E", 1.25, 0.25)])

    paths = C.save(fig, NAME)
    shown = by_id.loc[[SMALL, BIG, PAIR_SMALL, HARMONIC, 2088]]
    cols = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score", "emitter_hz",
            "track_id", "track_hz", "emitter_track_df_hz", "assign_confidence",
            "pp_drop_reason", "pp_harmonic_of_track", "height_hz", "height_sd_hz",
            "fwhm_s", "fwhm_sd_s", "width10_s", "width10_sd_s", "undershoot_hz",
            "undershoot_sd_hz"]
    print(f"{REC} tile {TILE}:")
    print(shown[cols].round(4).to_string())
    print("trace baselines:", {k: round(v["trace_baseline_hz"], 2) for k, v in tmeta.items()})
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
