"""Figure: where chirp detection pipeline v2 fails on real recordings.

Replaces ``plot_pipeline_performances_manuscript_supplement.pdf`` (two v1
snippets with annotated failures). Real spectrograms (detector's 205-ms
channel unless stated, click-cleaned, power summed over electrodes) of the
production run ``competition2019_v2_20261113`` with its boxes at the
operating threshold, coloured by the fish they were assigned to, and the
wavetracker tracks; one key for all panels (the symbols of the pipeline
figure); display levels from ``chirp_levels`` (85th percentile of the crop
-> 99th percentile of the chirp pixels away from the baselines). Only
failures that can be checked by eye, or against hand labels, are marked:

A   2019-12-09-10_00, 4312-4328 s, inside a hand-labelled test region:
    hand labels (triangles above the image) against detections. One faint
    labelled chirp scored 0.33 and is missed at the 0.5 threshold (zoom at
    right, with the crop's own display contrast); two detections have no
    label ("?").
B   2019-12-04-10_00, 7468-7488 s: the wavetracker track of the lower fish
    has a 140-s gap while the fish keeps chirping; the chirps in the gap
    are detected but left unassigned (no live track within 20 Hz). A
    self-contained panel: the gap-filled run (competition2019_v2_20261113_
    gapfill) will replace its content.
C   Two assignments between fish < 7 Hz apart: (left) one chirp between two
    fish 2.7 Hz apart, assigned with confidence 0.53; (right) two chirps
    77 ms apart between fish 6.4 Hz apart, shown in the 51-ms channel where
    they are separate, assigned to different fish with confidence 0.72 and
    0.56. Real data have no emitter ground truth: these are uncertain, not
    verified errors.

Also prints how often the failure-prone situations occur among the chirps
of the 33 analysed trials (unassigned chirps; chirps between fish < 10 Hz
apart; low assignment confidence).

Inputs: snippet cache of ``v2_fig_extract.py``, the run's parquet tables and
track-gap table, the hand labels in ``eval/real_labels_365/labels``.

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_failure_modes.py

Runtime: ~20 s cold, ~4 s warm.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v2_fig_common as C
from style import set_light_style

NAME = "plot_v2_failure_modes"
MISSED_A = 4080  # 2019-12-09-10_00, score 0.33, on a hand-labelled chirp
CLOSE_1 = 14783  # 2019-12-09-10_00
CLOSE_2 = (2974, 2975)  # 2020-03-13-10_00
W, H = 17.4, 12.05  # figure size [cm]


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


def fish_labels(ax, snip, colors, order, t0, t1, x, dy):
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
    # floor from the background only: in this 310-Hz crop the glow of the two
    # baselines 20 Hz apart fills > 15 % of the pixels (chirp_levels' floor
    # landed above the median chirp pixel)
    lv = C.background_levels(snip, t0, t1, f0, f1, rows)
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=lv[0], vmax=lv[1])
    C.draw_tracks(ax, snip, colors, t0, t1)
    drawn = C.draw_detections(ax, rows, colors, t0)
    labels = hand_labels(m["recording"])
    labels = labels[(labels.t_end_s > t0) & (labels.t_start_s < t1)]
    kept = drawn[drawn.drawn_as.isin(["assigned", "unassigned"])]
    lab_c, unmatched_lab, unmatched_det = match_labels(labels, kept)
    C.label_marks(ax, lab_c - t0)
    miss = rows.set_index("chirp_id").loc[MISSED_A]
    C.draw_missed(ax, miss, t0)
    # "?" above the highest box at that time (clear of every box edge)
    for r in unmatched_det.itertuples():
        near = kept[(kept.t_start_s < r.t_center_s + 0.06) & (kept.t_end_s > r.t_center_s - 0.06)]
        ax.text(r.t_center_s - t0, float(near.f_high_hz.max()) + 6, "?", ha="center",
                va="bottom", fontweight="bold", **C.ANNOT)
    fish_labels(ax, snip, colors, order, t0, t1, 0.1, dy=(29, -29))
    C.scalebar(ax, t1 - t0 - 0.15, 668, 1.0, "1 s")
    C.spec_yaxis(ax, ticks=[700, 800, 900])
    n_lab = len(labels)
    n_found = n_lab - len(unmatched_lab)

    # zoom on the missed chirp with the levels of the main panel: the missed
    # chirp keeps its brightness relative to the detected chirps (its pixels
    # do not exceed the background of the same rows; not stretched)
    zt0, zt1 = miss.t_center_s - 0.4, miss.t_center_s + 0.4
    zf0, zf1 = 685, 845
    C.show_spec(ax_zoom, snip, zt0, zt1, zf0, zf1, vmin=lv[0], vmax=lv[1])
    C.draw_tracks(ax_zoom, snip, colors, zt0, zt1)
    C.draw_detections(ax_zoom, rows, colors, zt0)
    C.draw_missed(ax_zoom, miss, zt0)
    zl = lab_c[(lab_c > zt0) & (lab_c < zt1)]
    C.label_marks(ax_zoom, zl - zt0)
    ax_zoom.text(0.5 * (miss.t_start_s + miss.t_end_s) - zt0, miss.f_high_hz + 5,
                 f"score {miss.score:.2f}", ha="center", va="bottom", **C.ANNOT)
    C.scalebar(ax_zoom, zt1 - zt0 - 0.04, 692, 0.2, "200 ms")
    C.no_yaxis(ax_zoom)
    C.zoom_connect(fig, ax, (zt0 - t0, zf0, zt1 - t0, zf1), ax_zoom)

    report["A"] = {
        "recording": m["recording"], "window_s": [t0, t1], "levels": lv,
        "n_hand_labels": int(n_lab), "n_found": int(n_found),
        "n_detections_kept": int(len(kept)),
        "labels_without_detection": [round(t, 3) for t in unmatched_lab],
        "detections_without_label": unmatched_det[
            ["chirp_id", "t_center_s", "score", "track_id", "emitter_hz"]
        ].round(3).to_dict(orient="records"),
        "missed": {"chirp_id": MISSED_A, "score": float(miss.score),
                   "t_center_s": float(miss.t_center_s)},
        "tracks_hz": {int(t): round(C.track_median(snip, t, t0, t1), 1) for t in order},
    }


# ---------------------------------------------------------------------------
# B: track gap (self-contained; content follows the gap-filled run)
# ---------------------------------------------------------------------------
def panel_trackgap(ax, report):
    snip = C.load_snippet("f3_trackgap")
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 640, 880
    # both fish, ordered by frequency over a longer stretch (one track ends)
    colors, order = C.fish_colors(snip, t0 - 15, t1)
    rows = C.window(snip, t0, t1)
    lv = C.chirp_levels(snip, t0, t1, f0, f1, rows)
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=lv[0], vmax=lv[1])
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
    # label below the band, left of the track's end; dark backing over the
    # faint noise line there
    C.annotate(ax, "track ends", xy=(end - t0, f_end - 4), xytext=(end - t0 - 0.9, 655),
               rad=0.2, ha="right", shrink_b=3, **C.DARK_BACK)
    un = drawn[drawn.drawn_as == "unassigned"].sort_values("t_center_s")
    fish_labels(ax, snip, colors, order, t0 - 15, t1, 1.15, dy=(22, -22))
    C.scalebar(ax, t1 - t0 - 0.15, 662, 1.0, "1 s")
    C.spec_yaxis(ax, ticks=[700, 800])
    report["B"] = {
        "recording": m["recording"], "window_s": [t0, t1], "levels": lv,
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
def panel_close(ax, name, ids, half, channel, report, key, *, word: bool):
    snip = C.load_snippet(name)
    rec = snip["meta"]["recording"]
    rows = C.window(snip, snip["meta"]["t_start_s"] - 1, snip["meta"]["t_end_s"] + 1)
    sel = rows.set_index("chirp_id").loc[list(ids)]
    tc = float(sel.t_center_s.mean())
    t0, t1 = tc - half, tc + half
    f0, f1 = 640, 890
    # fish identity over the snippet's surroundings (tracks are sparse in 1 s)
    colors, order = C.fish_colors(snip, t0 - 5, t1 + 5)
    win = rows[(rows.t_end_s > t0) & (rows.t_start_s < t1)]
    if channel == 0:
        lv = C.chirp_levels(snip, t0, t1, f0, f1, win, channel=channel)
    else:  # 51-ms window: the merged bands dominate; per-crop percentiles
        lv = (None, None)
    lv = C.show_spec(ax, snip, t0, t1, f0, f1, channel=channel, vmin=lv[0], vmax=lv[1])
    C.draw_tracks(ax, snip, colors, t0, t1, lw=0.7)
    C.draw_detections(ax, win, colors, t0)
    # confidences above the highest of the boxes, in the colour of their box
    ytxt = float(sel.f_high_hz.max()) + 6
    for r in sel.itertuples():
        txt = f"{r.assign_confidence:.2f}"
        if word:
            txt = "confidence " + txt
        ax.text(r.t_center_s - t0, ytxt, txt, ha="center", va="bottom",
                color=colors.get(int(r.track_id), "white"), fontsize=C.ANNOT_FS,
                **C.DARK_BACK)
    hz = [C.track_median(snip, t, t0 - 5, t1 + 5) for t in order]
    # labels just above / below the pair of tracks; a dark backing where the
    # 51-ms window merges the two bands into one bright band
    off, back = (25, {}) if channel == 0 else (17, C.DARK_BACK)
    C.eod_label(ax, 0.012, max(hz) + off, None, color=colors[order[0]], text=r"EOD$f_1$",
                **back)
    C.eod_label(ax, 0.012, min(hz) - off, None, color=colors[order[1]], text=r"EOD$f_2$",
                **back)
    df_onset = float(sel.track_df_other_hz.iloc[0])
    lab = rf"$\Delta f$ = {df_onset:.1f} Hz"
    if channel != 0:
        lab += f", {1e3 * snip['meta']['windows_s'][channel]:.0f}-ms window"
    ax.text(0.012 * (t1 - t0), f1 - 5, lab, ha="left", va="top", **C.ANNOT)
    C.scalebar(ax, t1 - t0 - 0.02, 862, 0.1, "100 ms")
    report[key] = {
        "recording": rec, "levels": lv,
        "rows": sel.reset_index()[["chirp_id", "t_center_s", "score", "emitter_hz",
                                   "track_id", "track_hz", "second_track_hz",
                                   "track_df_other_hz", "assign_confidence"]]
        .round(3).to_dict(orient="records"),
        "tracks_hz": {int(t): round(h, 2) for t, h in zip(order, hz, strict=True)},
        "df_hz_track_medians": round(hz[0] - hz[1], 2),
        # the pipeline's own dF at chirp onset: the emitter track's distance
        # to its closest live neighbour (track_df_other_hz)
        "df_hz": df_onset,
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
    fig = plt.figure(figsize=(W * C.CM, H * C.CM))

    # key (the symbols of the pipeline figure), centred
    items = [
        ("boxdot", {"color": C.FISH_COLORS[0]}, r"EOD$f_1$"),
        ("boxdot", {"color": C.FISH_COLORS[1]}, r"EOD$f_2$"),
        ("boxdot", {"color": "white"}, "unassigned"),
        ("box", {"color": "white", "ls": C.MISSED_LS}, "below threshold"),
        ("label", {}, "hand label"),
        ("qmark", {}, "no hand label"),
    ]
    width = C.key_strip(fig, 0.0, H - 0.3, items, measure=True)
    C.key_strip(fig, (W - width) / 2, H - 0.3, items)

    ax_a = C.axes_cm(fig, 1.45, 7.9, 11.55, 3.25)
    ax_az = C.axes_cm(fig, 13.75, 7.9, 3.45, 3.25)
    ax_b = C.axes_cm(fig, 1.45, 4.25, 15.75, 3.05)
    ax_c1 = C.axes_cm(fig, 1.45, 0.25, 7.6, 3.45)
    ax_c2 = C.axes_cm(fig, 9.6, 0.25, 7.6, 3.45)

    panel_labelled(ax_a, ax_az, fig, report)
    panel_trackgap(ax_b, report)
    panel_close(ax_c1, "g_close_15388", [CLOSE_1], 0.45, 0, report, "C1", word=True)
    panel_close(ax_c2, "g_close_9394", CLOSE_2, 0.45, 1, report, "C2", word=False)
    C.spec_yaxis(ax_c1, ticks=[700, 800])
    C.no_yaxis(ax_c2)  # same frequency range as C left

    prevalence(report)
    C.letters_cm(fig, [(ax_a, "A", 1.2, 0.3), (ax_b, "B", 1.2, 0.3),
                       (ax_c1, "C", 1.2, 0.3)])
    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
