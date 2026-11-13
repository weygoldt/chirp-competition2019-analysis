"""Figure: chirp detection pipeline v2, from the grid recording to a parameterised chirp.

Replaces ``plot_assignment_feature_extraction.pdf`` (the v1 assignment
features). Every panel shows real data of the production run
``competition2019_v2_20261113`` (released dual-head RT-DETR detector, weights
9fcdef3c; chirp-parameter decoder c0801c56):

A   one real snippet (2020-06-05-10_00, 89.9-91.1 s: two fish 29 Hz apart,
    a small chirp of the lower and a big, short chirp of the upper fish) in
    the three analysis windows of the front end (205, 51 and 13 ms, all on
    the long window's 4.88-Hz grid, 500-1500 Hz; display contrast per
    panel) and as the RGB image the detector receives (R, G, B = 205, 51,
    13 ms, on the front end's own 0-1 levels). Design, not a measured
    benefit: no front-end ablation was run;
B   one 3.2-s detector tile (2020-05-29-10_00, 10850.4 s) with every
    detection at the operating threshold: boxes in the colour of the fish
    they were assigned to, the predicted emitter baseline EODf of each box
    (circle), the wavetracker tracks, and a second-harmonic copy removed by
    post-processing (dashed);
C   assignment of two near-simultaneous chirps of the two fish whose boxes
    overlap both fish's bands: each box goes to the live track nearest its
    predicted emitter EODf, within 20 Hz (shaded);
D   post-processing: a harmonic copy (same tile) and a duplicate box on one
    chirp (2020-03-18-10_34, a draw not among the 33 analysed trials) are
    dropped;
E   decoder estimates (calibrated height, FWHM, 10 % width, undershoot) of
    the small chirp of A and the long chirp of C, drawn on their
    instantaneous-frequency traces extracted from the raw recording.

Inputs: the snippet/trace cache of ``v2_fig_extract.py`` (run it first, in
the upgrade env) and the run's ``chirps_rtdetr.parquet``.

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_pipeline_explainer.py

Runtime: ~15 s (LaTeX text rendering).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import v2_fig_common as C
from style import set_light_style

NAME = "plot_v2_pipeline_explainer"

REC = "2020-05-29-10_00"
TILE = (10850.4, 10853.6)
SMALL, BIG, PAIR_SMALL, HARMONIC = 2085, 2086, 2087, 2084
DUP_KEPT, DUP_DROPPED = 49386, 49385
FRONT_END = "f1_fe_e"  # 2020-06-05-10_00, 89.9-91.1 s: fish 29 Hz apart


def axes_cm(fig, x, y, w, h):
    W, H = fig.get_size_inches() / C.CM
    return fig.add_axes((x / W, y / H, w / W, h / H))


def fish_colors(snip, t0, t1):
    order = C.fish_order(snip, t0, t1)
    return {tid: C.FISH_COLORS[i % 2] for i, tid in enumerate(order)}, order


def track_median(snip, tid, t0, t1):
    tt, ff = snip["tracks"][tid]
    m = (tt >= t0) & (tt <= t1)
    return float(np.median(ff[m]))


def panel_front_end(axs):
    """The same snippet in the three windows and as the detector's RGB image."""
    snip = C.load_snippet(FRONT_END)
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 770, 1370
    titles = ["205-ms window", "51-ms window", "13-ms window", "detector input (RGB)"]
    for ch, (ax, title) in enumerate(zip(axs[:3], titles, strict=False)):
        # display contrast per panel (40th-99.8th percentile of the crop)
        C.show_spec(ax, snip, t0, t1, f0, f1, channel=ch, vmin=None, vmax=None)
    # the image the detector receives: channels 205/51/13 ms as R/G/B, on the
    # front end's own levels (0-1 = the 60-dB range), no display scaling
    t, f = snip["times_s"], snip["freqs_hz"]
    it = (t >= t0 - 0.05) & (t <= t1 + 0.05)
    jf = (f >= f0 - 10) & (f <= f1 + 10)
    rgb = np.stack([snip["img"][c][np.ix_(jf, it)].astype(float) for c in range(3)], -1)
    dt, df = t[1] - t[0], f[1] - f[0]
    axs[3].imshow(np.clip(rgb, 0, 1), aspect="auto", origin="lower",
                  extent=(t[it][0] - t0 - dt / 2, t[it][-1] - t0 + dt / 2,
                          f[jf][0] - df / 2, f[jf][-1] + df / 2),
                  interpolation="antialiased", rasterized=True)
    axs[3].set_xlim(0, t1 - t0)
    axs[3].set_ylim(f0, f1)
    for ch, (ax, title) in enumerate(zip(axs, titles, strict=True)):
        C.clean_spec_axes(ax, ylabel=ch == 0)
        ax.set_title(title, loc="center", fontsize="small", pad=2)
        if ch:
            ax.set_yticklabels([])
    rows = C.window(snip, t0, t1)
    rows = rows[rows.chirp]
    order = C.fish_order(snip, t0, t1)
    for k, tid in enumerate(order):
        tt, ff = snip["tracks"][tid]
        hz = float(np.median(ff[(tt >= t0) & (tt <= t1)]))
        C.eod_label(axs[0], 0.02, hz + (28 if k == 0 else -30), k + 1)
    C.scalebar(axs[2], t1 - t0 - 0.04, 1300, 0.2, "", dy=4)
    axs[2].text(t1 - t0 - 0.14, 1312, "200 ms", ha="center", va="bottom", **C.ANNOT)
    return m["recording"], rows


def panel_tile(ax, snip, rows, colors, order):
    t0, t1 = TILE
    C.show_spec(ax, snip, t0, t1, 500, 1500, vmin=0.3, vmax=0.95)
    C.draw_tracks(ax, snip, colors, t0, t1)
    C.draw_detections(ax, rows, colors, t0)
    C.clean_spec_axes(ax)
    for k, tid in enumerate(order):
        C.eod_label(ax, 0.03, track_median(snip, tid, t0, t1) + 22, k + 1,
                    color=colors[tid])
    h = rows.set_index("chirp_id").loc[HARMONIC]
    ax.annotate(
        "harmonic copy\n(removed)",
        xy=(h.t_start_s - t0, 0.5 * (h.f_low_hz + h.f_high_hz)),
        xytext=(h.t_end_s - t0 + 0.45, 1400),
        arrowprops=dict(**C.ARROW, connectionstyle="arc3,rad=0.3"),
        ha="center", va="center", **C.ANNOT,
    )
    C.scalebar(ax, 3.15, 540, 0.5, "", dy=8)
    ax.text(2.9, 552, "500 ms", ha="center", va="bottom", **C.ANNOT)


def panel_assignment(ax, snip, rows, colors, order):
    r_big = rows.set_index("chirp_id").loc[BIG]
    r_small = rows.set_index("chirp_id").loc[PAIR_SMALL]
    t0 = min(r_big.t_start_s, r_small.t_start_s) - 0.07
    t1 = max(r_big.t_end_s, r_small.t_end_s) + 0.07
    C.show_spec(ax, snip, t0, t1, 590, 1230, vmin=0.3, vmax=0.95)
    for tid in order:
        hz = track_median(snip, tid, t0, t1)
        ax.axhspan(hz - C.TOLERANCE_HZ, hz + C.TOLERANCE_HZ, color=colors[tid],
                   alpha=0.25, lw=0, zorder=1)
    C.draw_tracks(ax, snip, colors, t0, t1, lw=0.8)
    for r in (r_big, r_small):
        C.draw_box(ax, r, t0, colors[int(r.track_id)], lw=1.0)
    C.clean_spec_axes(ax)
    for k, tid in enumerate(order):
        C.eod_label(ax, 0.012, track_median(snip, tid, t0, t1) + 33, k + 1,
                    color=colors[tid])
    ax.annotate(
        "predicted\nemitter EOD$f$",
        xy=(r_big.t_center_s - t0 + 0.006, r_big.emitter_hz + 6),
        xytext=(r_big.t_center_s - t0 + 0.12, 1000),
        arrowprops=dict(**C.ARROW, connectionstyle="arc3,rad=-0.2"),
        ha="center", va="center", **C.ANNOT,
    )
    ax.annotate(
        "",
        xy=(r_small.t_center_s - t0 + 0.006, r_small.emitter_hz + 4),
        xytext=(r_big.t_center_s - t0 + 0.12, 950),
        arrowprops=dict(**C.ARROW, connectionstyle="arc3,rad=-0.3"),
    )
    C.scalebar(ax, t1 - t0 - 0.015, 612, 0.1, "", dy=10)
    ax.text(t1 - t0 - 0.065, 622, "100 ms", ha="center", va="bottom", **C.ANNOT)


def panel_harmonic(ax, snip, rows, colors):
    h = rows.set_index("chirp_id").loc[HARMONIC]
    f = rows.set_index("chirp_id").loc[SMALL]
    t0, t1 = f.t_center_s - 0.3, f.t_center_s + 0.3
    C.show_spec(ax, snip, t0, t1, 560, 1500, vmin=0.3, vmax=0.95)
    C.draw_tracks(ax, snip, colors, t0, t1)
    C.draw_box(ax, f, t0, colors[int(f.track_id)], lw=1.0)
    C.draw_box(ax, h, t0, C.DROPPED_COLOR, ls=(0, (3, 2)), lw=1.0, marker=False)
    ax.annotate(
        "", xy=(h.t_center_s - t0, h.f_low_hz - 8),
        xytext=(f.t_center_s - t0, f.f_high_hz + 8),
        arrowprops=dict(arrowstyle="->", color="white", lw=0.75, ls=(0, (2, 1.5))),
    )
    ax.text(f.t_center_s - t0 - 0.03, 0.5 * (f.f_high_hz + h.f_low_hz), r"2$\times$",
            ha="right", va="center", **C.ANNOT)
    C.clean_spec_axes(ax)
    ax.set_title("harmonic", loc="center", fontsize="small", pad=2)
    C.scalebar(ax, 0.58, 1000, 0.1, "", dy=14)
    ax.text(0.58, 1018, "100 ms", ha="right", va="bottom", **C.ANNOT)


def panel_duplicate(ax):
    snip = C.load_snippet("f1_dup")
    rows = C.window(snip, snip["meta"]["t_start_s"], snip["meta"]["t_end_s"])
    rows = rows.set_index("chirp_id")
    k, d = rows.loc[DUP_KEPT], rows.loc[DUP_DROPPED]
    t0, t1 = k.t_center_s - 0.3, k.t_center_s + 0.3
    colors, _ = fish_colors(snip, t0, t1)
    C.show_spec(ax, snip, t0, t1, 560, 1500, vmin=0.3, vmax=0.95)
    C.draw_tracks(ax, snip, colors, t0, t1)
    C.draw_box(ax, k, t0, colors[int(k.track_id)], lw=1.0)
    C.draw_box(ax, d, t0, C.DROPPED_COLOR, ls=(0, (3, 2)), lw=1.0, marker=False)
    C.clean_spec_axes(ax, ylabel=False)
    ax.set_yticklabels([])
    ax.set_title("duplicate", loc="center", fontsize="small", pad=2)
    return snip["meta"]["recording"]


def panel_trace(ax, traces, meta, key, row, xlim_ms, color):
    t = traces[f"{key}_t_s"] * 1e3
    base = meta[key]["trace_baseline_hz"]
    df = traces[f"{key}_freq_hz"] - base
    tp = traces[f"{key}_period_t_s"] * 1e3
    fp = traces[f"{key}_period_freq_hz"] - base
    m = (tp > xlim_ms[0]) & (tp < xlim_ms[1])
    ax.plot(tp[m], fp[m], ".", ms=1.6, color="#9e9e9e", mew=0, zorder=1,
            rasterized=True)
    ax.plot(t, df, color=color, lw=1.0, zorder=2)
    ax.axhline(0, color="k", lw=0.5, ls=":", zorder=0)

    height = row.height_hz
    fwhm, w10, under = row.fwhm_s * 1e3, row.width10_s * 1e3, row.undershoot_hz
    # centre the width bars on the trace's own crossings of 50 % / 10 %
    inside = (t > xlim_ms[0]) & (t < xlim_ms[1])
    ipk = np.flatnonzero(inside)[np.nanargmax(df[inside])]

    def centre(level):
        above = df >= level
        a = ipk
        while a > 0 and above[a - 1]:
            a -= 1
        b = ipk
        while b < len(df) - 1 and above[b + 1]:
            b += 1
        return 0.5 * (t[a] + t[b]), t[b]

    c50, _ = centre(0.5 * height)
    c10, end10 = centre(0.1 * height)
    col = "k"
    ax.axhline(height, color=col, lw=0.6, ls=(0, (4, 2)), zorder=3)
    for level, width, c, name in ((0.5 * height, fwhm, c50, "FWHM"),
                                  (0.1 * height, w10, c10, r"10\,\% width")):
        ax.annotate("", xy=(c - width / 2, level), xytext=(c + width / 2, level),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=0.7,
                                    shrinkA=0, shrinkB=0), zorder=4)
        x = c + width / 2 + 0.03 * np.diff(xlim_ms)[0]
        ax.text(x, level, name, ha="left", va="center", fontsize="small")
    ax.text(xlim_ms[1], height, "height", ha="right", va="bottom", fontsize="small")
    if under > 0.05 * height:
        # undershoot: deepest point of the trace after the peak
        after = (t > t[ipk]) & (t < xlim_ms[1])
        tu = t[after][np.nanargmin(df[after])]
        ax.annotate("", xy=(tu, -under), xytext=(tu, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.7,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.text(tu + 0.03 * np.diff(xlim_ms)[0], -under, "undershoot", ha="left",
                va="center", fontsize="small")
    ax.set_xlim(*xlim_ms)
    ax.set_xlabel("Time [ms]")


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    snip = C.load_snippet("f1_tile")
    t0, t1 = TILE
    rows = C.window(snip, t0, t1)
    rows = rows[rows.operating]
    colors, order = fish_colors(snip, t0, t1)
    by_id = rows.set_index("chirp_id")
    z = np.load(C.SNIPPETS / "traces.npz")
    traces = {k: z[k] for k in z.files if k != "meta"}
    tmeta = json.loads(str(z["meta"]))

    fig = plt.figure(figsize=(C.FIG_WIDTH, 14.2 * C.CM))
    # row 1: the front end (A)
    ax_a = [axes_cm(fig, 1.35 + i * 4.05, 10.45, 3.7, 3.2) for i in range(4)]
    # row 2: one tile with the detector's output (B) and assignment (C)
    ax_b = axes_cm(fig, 1.35, 5.95, 9.3, 3.45)
    ax_c = axes_cm(fig, 12.05, 5.95, 5.15, 3.45)
    ax_key = axes_cm(fig, 1.35, 4.55, 15.85, 0.6)
    # row 3: post-processing (D) and decoder (E)
    ax_d1 = axes_cm(fig, 1.35, 0.95, 2.45, 2.9)
    ax_d2 = axes_cm(fig, 3.95, 0.95, 2.45, 2.9)
    ax_e1 = axes_cm(fig, 7.75, 0.95, 4.1, 2.9)
    ax_e2 = axes_cm(fig, 13.1, 0.95, 4.1, 2.9)

    fe_rec, fe_rows = panel_front_end(ax_a)
    panel_tile(ax_b, snip, rows, colors, order)
    panel_assignment(ax_c, snip, rows, colors, order)
    panel_harmonic(ax_d1, snip, rows, colors)
    dup_rec = panel_duplicate(ax_d2)
    panel_trace(ax_e1, traces, tmeta, "small", by_id.loc[SMALL], (-40, 40),
                colors[int(by_id.loc[SMALL].track_id)])
    panel_trace(ax_e2, traces, tmeta, "big", by_id.loc[BIG], (-120, 120),
                colors[int(by_id.loc[BIG].track_id)])
    for ax in (ax_e1, ax_e2):
        ax.set_ylabel(r"EOD$f$ above" "\n" "baseline [Hz]")

    # key to the box styles, one line
    ax_key.set_axis_off()
    ax_key.set_xlim(0, 15.85)
    ax_key.set_ylim(0, 1)
    items = [
        ("box", C.FISH_COLORS[0], "-", r"chirp of EOD$f_1$"),
        ("box", C.FISH_COLORS[1], "-", r"chirp of EOD$f_2$"),
        ("box", "#6e6e6e", (0, (3, 2)), "removed"),
        ("dot", "#6e6e6e", None, "predicted emitter EOD$f$"),
        ("band", "#6e6e6e", None, r"$\pm$20 Hz tolerance"),
    ]
    for x, (kind, col, ls, text) in zip((0.0, 3.0, 6.0, 8.3, 11.9), items, strict=True):
        if kind == "box":
            ax_key.add_patch(plt.Rectangle((x, 0.2), 0.45, 0.6, fill=False, ec=col,
                                           ls=ls, lw=1.0))
        elif kind == "dot":
            ax_key.plot(x + 0.22, 0.5, "o", ms=3, mfc=col, mec="k", mew=0.4)
        else:
            ax_key.add_patch(plt.Rectangle((x, 0.25), 0.45, 0.5, color=col,
                                           alpha=0.35, lw=0))
        ax_key.text(x + 0.6, 0.5, text, va="center", ha="left", fontsize="small")

    ax_e1.set_ylim(-12, 85)
    ax_e2.set_ylim(-60, 470)

    from style import letter_subplots

    letter_subplots(
        [ax_a[0], ax_b, ax_c, ax_d1, ax_e1, ax_e2],
        letters=["A", "B", "C", "D", "E", ""],
        xoffset=[-0.28, -0.11, -0.2, -0.42, -0.25, -0.25],
        yoffset=[1.1, 1.05, 1.05, 1.1, 1.1, 1.1],
    )

    paths = C.save(fig, NAME)
    # numbers shown in or used by the figure
    shown = by_id.loc[[SMALL, BIG, PAIR_SMALL, HARMONIC, 2088]]
    cols = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score", "emitter_hz",
            "track_id", "track_hz", "emitter_track_df_hz", "second_track_hz",
            "assign_confidence", "pp_drop_reason", "pp_harmonic_of_track",
            "height_hz", "fwhm_s", "width10_s", "undershoot_hz", "params_ok"]
    print(f"{REC} tile {TILE}:")
    print(shown[cols].round(4).to_string())
    print("duplicate example:", dup_rec, DUP_KEPT, "kept,", DUP_DROPPED, "dropped")
    print("front end example:", fe_rec)
    print(fe_rows[["chirp_id", "t_center_s", "score", "track_id", "track_hz",
                   "track_df_other_hz", "height_hz", "fwhm_s"]].round(3).to_string())
    print("trace baselines:", {k: round(v["trace_baseline_hz"], 2) for k, v in tmeta.items()})
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
