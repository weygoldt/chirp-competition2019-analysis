"""Figure: where chirp detection pipeline v2 fails on real recordings.

Replaces ``plot_pipeline_performances_manuscript_supplement.pdf`` (two v1
snippets with annotated failures). Real spectrograms (detector's 205-ms
channel unless stated, click-cleaned, power summed over electrodes) of the
production run ``competition2019_v2_20261113`` with its boxes at the
operating threshold, coloured by the fish they were assigned to, and the
wavetracker tracks; one key for all panels (the symbols of the pipeline
figure). Only failures that can be checked by eye, or against hand labels,
are annotated:

A   2019-12-09-10_00, 4312-4328 s, inside a hand-labelled test region:
    hand labels (triangles) against detections. One faint labelled chirp
    scored 0.33 and is missed at the 0.5 threshold (zoom at right, with the
    crop's own display contrast); two detections have no label ("?").
B   2019-12-04-10_00, 7468-7488 s: the wavetracker track of the lower fish
    has a 140-s gap while the fish keeps chirping; the chirps in the gap
    are detected but left unassigned (no live track within 20 Hz). A
    self-contained panel: track interpolation across interior gaps is under
    discussion, and this panel may be dropped or redrawn.
C   Two assignments between fish < 7 Hz apart: (1) one chirp between two
    fish 2.7 Hz apart, assigned with confidence 0.53; (2) two chirps 77 ms
    apart between fish 6.4 Hz apart, shown in the 51-ms channel where they
    are separate, assigned to different fish with confidence 0.72 and 0.56.
    Real data have no emitter ground truth: these are uncertain, not
    verified errors.

Also prints how often the failure-prone situations occur among the chirps
of the 33 analysed trials (unassigned chirps; chirps between fish < 10 Hz
apart; low assignment confidence).

Inputs: snippet cache of ``v2_fig_extract.py``, the run's parquet tables,
the hand labels in ``eval/real_labels_365/labels``.

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_failure_modes.py

Runtime: ~20 s.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v2_fig_common as C
from style import letter_subplots, set_light_style

NAME = "plot_v2_failure_modes"
MISSED_A = 4080  # 2019-12-09-10_00, score 0.33, on a hand-labelled chirp
CLOSE_1 = 14783  # 2019-12-09-10_00
CLOSE_2 = (2974, 2975)  # 2020-03-13-10_00
H = 13.9  # figure height [cm]


def letters(fig, axes_and_offsets):
    """letter_subplots with offsets given in cm from each axes' upper left corner."""
    W, Hh = fig.get_size_inches() / C.CM
    axs, xo, yo, lets = [], [], [], []
    for ax, let, dx, dy in axes_and_offsets:
        bb = ax.get_position()
        axs.append(ax)
        lets.append(let)
        xo.append(-dx / (bb.width * W))
        yo.append(1 + dy / (bb.height * Hh))
    letter_subplots(axs, letters=lets, xoffset=xo, yoffset=yo)


def hand_labels(recording: str) -> pd.DataFrame:
    path = (C.REAL_LABELS / "2019_tube_competition" / recording
            / "traces-grid1-editable-labels.csv")
    lab = pd.read_csv(path)
    return lab[lab["category"] == "chirp"]


def match_labels(labels: pd.DataFrame, kept: pd.DataFrame):
    """Greedy one-to-one matching of centres within 0.1 s (closest pairs first),
    the evaluation's rule without the frequency-overlap condition."""
    lab_c = 0.5 * (labels.t_start_s + labels.t_end_s).to_numpy()
    det_c = kept.t_center_s.to_numpy()
    pairs = sorted((abs(a - b), i, j) for i, a in enumerate(lab_c)
                   for j, b in enumerate(det_c) if abs(a - b) <= 0.1)
    used_l, used_d = set(), set()
    for _, i, j in pairs:
        if i not in used_l and j not in used_d:
            used_l.add(i)
            used_d.add(j)
    unmatched_lab = [float(lab_c[i]) for i in range(len(lab_c)) if i not in used_l]
    unmatched_det = kept.iloc[[j for j in range(len(det_c)) if j not in used_d]]
    return lab_c, unmatched_lab, unmatched_det


def fish_labels(ax, snip, colors, order, t0, t1, x, dy=(13, -13)):
    for k, tid in enumerate(order):
        y = C.track_median(snip, tid, t0, t1) + dy[min(k, 1)]
        C.eod_label(ax, x, y, k + 1, color=colors[tid])


# ---------------------------------------------------------------------------
# A: hand-labelled region and the missed chirp
# ---------------------------------------------------------------------------
def panel_labelled(ax, ax_zoom, fig, report):
    snip = C.load_snippet("f3_labelled")
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 650, 960
    colors, order = C.fish_colors(snip, t0, t1)
    rows = C.window(snip, t0, t1)
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=0.3, vmax=0.95)
    C.draw_tracks(ax, snip, colors, t0, t1)
    drawn = C.draw_detections(ax, rows, colors, t0)
    labels = hand_labels(m["recording"])
    labels = labels[(labels.t_end_s > t0) & (labels.t_start_s < t1)]
    kept = drawn[drawn.drawn_as.isin(["assigned", "unassigned"])]
    lab_c, unmatched_lab, unmatched_det = match_labels(labels, kept)
    C.label_marks(ax, lab_c - t0, f1 - 14)
    miss = rows.set_index("chirp_id").loc[MISSED_A]
    C.draw_missed(ax, miss, t0)
    for r in unmatched_det.itertuples():
        ax.text(r.t_center_s - t0, r.f_high_hz + 4, "?", ha="center", va="bottom",
                fontweight="bold", **C.ANNOT)
    fish_labels(ax, snip, colors, order, t0, t1, 0.1)
    C.scalebar(ax, t1 - t0 - 0.15, 668, 1.0, "1 s")
    C.spec_yaxis(ax, ticks=[700, 800, 900])
    n_lab = len(labels)
    n_found = n_lab - len(unmatched_lab)
    ax.set_title(f"Hand-labelled test region: {n_found} of {n_lab} labelled chirps detected",
                 loc="left")

    # zoom on the missed chirp, the crop's own display contrast
    zt0, zt1 = miss.t_center_s - 0.4, miss.t_center_s + 0.4
    zf0, zf1 = 685, 845
    C.show_spec(ax_zoom, snip, zt0, zt1, zf0, zf1, vmin=None, vmax=None)
    C.draw_tracks(ax_zoom, snip, colors, zt0, zt1)
    C.draw_detections(ax_zoom, rows, colors, zt0)
    C.draw_missed(ax_zoom, miss, zt0)
    zl = lab_c[(lab_c > zt0) & (lab_c < zt1)]
    C.label_marks(ax_zoom, zl - zt0, zf1 - 8)
    ax_zoom.text(0.5 * (miss.t_start_s + miss.t_end_s) - zt0, miss.f_high_hz + 6,
                 f"score {miss.score:.2f}", ha="center", va="bottom", **C.ANNOT)
    C.scalebar(ax_zoom, zt1 - zt0 - 0.04, 692, 0.2, "200 ms")
    C.spec_yaxis(ax_zoom, ylabel=False)
    ax_zoom.set_yticks([])
    ax_zoom.spines["left"].set_visible(False)
    ax_zoom.set_title("Missed: faint chirp", loc="left")
    C.zoom_connect(fig, ax, (zt0 - t0, zf0, zt1 - t0, zf1), ax_zoom)

    report["A"] = {
        "recording": m["recording"], "window_s": [t0, t1],
        "n_hand_labels": int(n_lab), "n_detections_kept": int(len(kept)),
        "labels_without_detection": [round(t, 3) for t in unmatched_lab],
        "detections_without_label": unmatched_det[
            ["chirp_id", "t_center_s", "score", "track_id", "emitter_hz"]
        ].round(3).to_dict(orient="records"),
        "missed": {"chirp_id": MISSED_A, "score": float(miss.score),
                   "t_center_s": float(miss.t_center_s)},
        "tracks_hz": {int(t): round(C.track_median(snip, t, t0, t1), 1) for t in order},
    }


# ---------------------------------------------------------------------------
# B: track gap (self-contained; may be dropped or redrawn)
# ---------------------------------------------------------------------------
def panel_trackgap(ax, report):
    snip = C.load_snippet("f3_trackgap")
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 640, 880
    # both fish, ordered by frequency over a longer stretch (one track ends)
    colors, order = C.fish_colors(snip, t0 - 15, t1)
    rows = C.window(snip, t0, t1)
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=0.3, vmax=0.95)
    C.draw_tracks(ax, snip, colors, t0, t1, lw=0.8)
    drawn = C.draw_detections(ax, rows, colors, t0)
    lower = order[-1]
    tt, ff = snip["tracks"][lower]
    inside = tt[(tt >= t0 - 15) & (tt <= t1 + 15)]
    end = float(inside[inside <= t1].max())
    # the gap itself (the cache holds the tracks only +- 15 s around the
    # snippet): the run's track-gap table (track_gaps/gaps.csv)
    gaps = pd.read_csv(C.RUN / "track_gaps" / "gaps.csv")
    g = gaps[(gaps.recording == m["recording"]) & (gaps.fish_id == lower)
             & ((gaps.start_s - end).abs() < 0.01)].iloc[0]
    resume = float(g.end_s)
    rec_rows = C.detections(m["recording"])
    f_end = float(ff[tt == end][0])
    ax.plot(end - t0, f_end, "|", ms=6, mew=1.0, color=colors[lower], zorder=6)
    ax.annotate("track ends", xy=(end - t0 + 0.05, f_end - 3), xytext=(end - t0 + 0.6, 662),
                arrowprops=dict(**C.ARROW), ha="left", va="center", **C.ANNOT)
    un = drawn[drawn.drawn_as == "unassigned"].sort_values("t_center_s")
    tx = un.t_center_s.mean() - t0
    for u in un.itertuples():
        ax.annotate("", xy=(u.t_center_s - t0, u.f_high_hz + 3), xytext=(tx, 852),
                    arrowprops=dict(**C.ARROW))
    ax.text(tx, 854, "fish still chirps: chirps unassigned", ha="center", va="bottom",
            **C.ANNOT)
    fish_labels(ax, snip, colors, order, t0 - 15, t1, 1.15, dy=(13, -14))
    C.scalebar(ax, t1 - t0 - 0.15, 662, 1.0, "1 s")
    C.spec_yaxis(ax, ticks=[700, 800])
    report["B"] = {
        "recording": m["recording"], "window_s": [t0, t1],
        "track_id": int(lower), "track_last_detection_s": end,
        "track_next_detection_s": resume,
        "gap_s": resume - end,
        "gap_table": g[["fish", "kind", "start_s", "end_s", "duration_s", "chirps_in_gap",
                        "chirps_unassigned", "rescuable_by_interp"]].to_dict(),
        "n_assigned_in_window": int((drawn.drawn_as == "assigned").sum()),
        "n_unassigned_in_window": int(len(un)),
        "unassigned": un[["chirp_id", "t_center_s", "score", "emitter_hz",
                          "nearest_track_hz", "assign_status"]].round(2)
        .to_dict(orient="records"),
        "recording_chirps_beyond_tolerance": int(
            (rec_rows.chirp & (rec_rows.assign_status == "beyond_tolerance")).sum()),
        "recording_chirps": int(rec_rows.chirp.sum()),
    }
    return report["B"]


# ---------------------------------------------------------------------------
# C: assignments between close fish
# ---------------------------------------------------------------------------
def panel_close(ax, name, ids, half, channel, report, key):
    snip = C.load_snippet(name)
    rec = snip["meta"]["recording"]
    rows = C.window(snip, snip["meta"]["t_start_s"] - 1, snip["meta"]["t_end_s"] + 1)
    sel = rows.set_index("chirp_id").loc[list(ids)]
    tc = float(sel.t_center_s.mean())
    t0, t1 = tc - half, tc + half
    f0, f1 = 640, 880
    # fish identity over the snippet's surroundings (tracks are sparse in 1 s)
    colors, order = C.fish_colors(snip, t0 - 5, t1 + 5)
    C.show_spec(ax, snip, t0, t1, f0, f1, channel=channel,
                vmin=0.3 if channel == 0 else None, vmax=0.95 if channel == 0 else None)
    C.draw_tracks(ax, snip, colors, t0, t1, lw=0.7)
    win = rows[(rows.t_end_s > t0) & (rows.t_start_s < t1)]
    C.draw_detections(ax, win, colors, t0)
    for r in sel.itertuples():
        ax.text(r.t_center_s - t0, r.f_high_hz + 4, f"{r.assign_confidence:.2f}",
                ha="center", va="bottom", **C.ANNOT)
    hz = [C.track_median(snip, t, t0 - 5, t1 + 5) for t in order]
    ytop = max(hz) + (15 if channel == 0 else 52)
    C.eod_label(ax, 0.012, ytop + 19, None, color=colors[order[0]], text=r"EOD$f_1$")
    C.eod_label(ax, 0.012, ytop, None, color=colors[order[1]], text=r"EOD$f_2$")
    C.scalebar(ax, t1 - t0 - 0.02, 852, 0.1, "100 ms")
    report[key] = {
        "recording": rec,
        "rows": sel.reset_index()[["chirp_id", "t_center_s", "score", "emitter_hz",
                                   "track_id", "track_hz", "second_track_hz",
                                   "track_df_other_hz", "assign_confidence"]]
        .round(3).to_dict(orient="records"),
        "tracks_hz": {int(t): round(h, 2) for t, h in zip(order, hz, strict=True)},
        "df_hz_track_medians": round(hz[0] - hz[1], 2),
        # the pipeline's own dF at chirp onset: the emitter track's distance
        # to its closest live neighbour (track_df_other_hz)
        "df_hz": float(sel.track_df_other_hz.iloc[0]),
        "df_hz_all": sel.track_df_other_hz.round(3).tolist(),
        "chirp_gap_ms": 1e3 * float(sel.t_center_s.max() - sel.t_center_s.min()),
        "channel_window_s": snip["meta"]["windows_s"][channel],
        "other_detections_in_window": win.loc[~win.chirp_id.isin(ids), [
            "chirp_id", "t_center_s", "score", "operating", "track_id",
            "assign_confidence"]].round(3).to_dict(orient="records"),
    }
    return report[key]


def prevalence(report):
    """How often the failure-prone situations occur in the 33 analysed trials."""
    recs = sorted(C.analysed_recordings())
    d = pd.concat([C.detections(r) for r in recs], ignore_index=True)
    ch = d[d.chirp]
    assigned = ch[ch.assign_status == "assigned"]
    report["prevalence_33_trials"] = {
        "chirps": int(len(ch)),
        "assign_status": ch.assign_status.value_counts().to_dict(),
        "unassigned_fraction": float((ch.assign_status.isin(
            ["beyond_tolerance", "no_live_track"])).mean()),
        "assigned_df_other_lt_10_hz": int((assigned.track_df_other_hz < 10).sum()),
        "assigned_df_other_lt_10_fraction": float((assigned.track_df_other_hz < 10).mean()),
        "assigned_confidence_lt_0.9": int((assigned.assign_confidence < 0.9).sum()),
        "assigned_confidence_lt_0.9_fraction": float(
            (assigned.assign_confidence < 0.9).mean()),
        "n_assigned": int(len(assigned)),
    }


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {}
    fig = plt.figure(figsize=(C.FIG_WIDTH, H * C.CM))

    # key (the symbols of the pipeline figure)
    items = [
        ("boxdot", {"color": C.FISH_COLORS[0]}, r"chirp assigned to EOD$f_1$"),
        ("boxdot", {"color": C.FISH_COLORS[1]}, r"to EOD$f_2$"),
        ("boxdot", {"color": "white"}, "unassigned"),
        ("box", {"color": "white", "ls": C.MISSED_LS}, "below threshold (missed)"),
        ("label", {}, "hand label"),
        ("track", {"color": C.FISH_COLORS[0]}, "track"),
    ]
    xend = C.key_strip(fig, 0.0, H - 0.3, items)
    # centre the key on the figure: redraw with the measured width
    for a in fig.axes[:]:
        a.remove()
    for t in fig.texts[:]:
        t.remove()
    C.key_strip(fig, (17.4 - xend) / 2 + 0.35, H - 0.3, items)
    fig.text(0.5, (H - 0.62) / H, r"circles: predicted emitter EOD$f$; numbers above "
             r"boxes: assignment confidence; ?: detection without hand label",
             ha="center", va="top", fontsize="small", color=C.MUTED)

    ax_a = C.axes_cm(fig, 1.45, 9.05, 11.55, 3.25)
    ax_az = C.axes_cm(fig, 13.75, 9.05, 3.45, 3.25)
    ax_b = C.axes_cm(fig, 1.45, 4.85, 15.75, 3.05)
    ax_c1 = C.axes_cm(fig, 1.45, 0.25, 7.5, 3.45)
    ax_c2 = C.axes_cm(fig, 9.7, 0.25, 7.5, 3.45)

    panel_labelled(ax_a, ax_az, fig, report)
    b = panel_trackgap(ax_b, report)
    ax_b.set_title(f"Track gap: the lower fish's track resumes {b['gap_s']:.0f} s later",
                   loc="left")
    c1 = panel_close(ax_c1, "g_close_15388", [CLOSE_1], 0.45, 0, report, "C1")
    c2 = panel_close(ax_c2, "g_close_9394", CLOSE_2, 0.45, 1, report, "C2")
    C.spec_yaxis(ax_c1, ticks=[700, 800])
    C.spec_yaxis(ax_c2, ticks=[700, 800], ylabel=False)
    ax_c1.set_title(rf"Fish {c1['df_hz']:.1f} Hz apart: emitter uncertain", loc="left")
    ax_c2.set_title(rf"Fish {c2['df_hz']:.1f} Hz apart, two chirps "
                    rf"({1e3 * c2['channel_window_s']:.0f}-ms window)", loc="left")

    prevalence(report)
    letters(fig, [(ax_a, "A", 1.2, 0.3), (ax_b, "B", 1.2, 0.3), (ax_c1, "C", 1.2, 0.3)])
    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
