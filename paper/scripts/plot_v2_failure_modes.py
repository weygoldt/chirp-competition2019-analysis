"""Figure: where chirp detection pipeline v2 fails on real recordings.

Replaces ``plot_pipeline_performances_manuscript_supplement.pdf`` (two v1
snippets with annotated failures). Real spectrograms (detector's 205-ms
channel, click-cleaned, power summed over electrodes) of the production run
``competition2019_v2_20261113`` with its boxes at the operating threshold
and the wavetracker tracks, drawn as in the pipeline figure (no colour or
line style for fish identity: every detection a solid white box, its emitter
marker on the track of the fish it was assigned to; tracks thin black lines,
dashed where interpolated; dotted light grey, below the operating
threshold); one key for all panels (only the symbols drawn). Power in dB re
1 V^2, display levels the 2nd and 98th percentile of each panel's crop
(``C.CLIP_PCT``; the enlargement in A uses A's levels); no colour bars.
Only failures that can be checked by eye, or against hand labels, are
marked:

A   2019-12-09-10_00, 4312-4328 s, inside a hand-labelled test region:
    hand labels (triangles above the image) against detections. One faint
    labelled chirp scored 0.33 and is below the operating threshold
    (enlarged at right, with the levels and the frequency range of the main
    panel; two dashed lines mark the enlarged time span); if any,
    detections without a label get a "?". The tracks are drawn as the
    assignment uses them: linearly interpolated (dashed) across their short
    gaps.
B   a fish producing huge, long chirps (``CASES["B"]``);
C   chirp rates so high that the chirps smear together in the spectrogram
    (``CASES["C"]``).
    B and C are PLACEHOLDERS until the author picks the windows: set them in
    ``v2_fig_extract.SNIPPETS`` (``f3_long``, ``f3_dense``: recording, t0,
    t1, frequency band), rerun ``v2_fig_extract.py snippets --only f3_long
    f3_dense`` and set ``placeholder`` to False below.

Round 3 (author, 2026-11-25): the former B (a 140-s track gap: an upstream
tracking problem) and C (assignments between fish < 7 Hz apart: redundant
with Fig. 10E, F) are no longer drawn, but their numbers are still computed
(``stats["track_gap"]``, ``["C1"]``, ``["C2"]``; the Methods cite the
track-gap counts). The enlargement in A was a dashed rectangle inside the
panel that read as a second (bigger) below-threshold box around the missed
chirp; it is now a time span over the full frequency range.

Also prints how often the failure-prone situations occur among the chirps
of the 33 analysed trials (unassigned chirps; chirps between fish < 10 Hz
apart; low assignment confidence) and how many track gaps the patch of the
gap-filled run filled in them. The numbers the caption cites go to
write_stats ("v2_failure_modes"); the printed report has the details.

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
from manuscript_stats import write_stats
from style import set_light_style

NAME = "plot_v2_failure_modes"
MISSED_A = 4080  # 2019-12-09-10_00, score 0.33, on a hand-labelled chirp
CLOSE_1 = 14783  # 2019-12-09-10_00
CLOSE_2 = (2974, 2975)  # 2020-03-13-10_00
DRAWN: set[str] = set()  # symbols drawn in the panels (the key lists only these)

# the two hard cases for the detector (round 3). ``snippet``: a snippet of
# v2_fig_extract.SNIPPETS whose meta holds the window and the band shown;
# ``below_min_score``: also draw the stored detections from this score up to
# the operating threshold (dotted grey; None: none); ``scalebar_s``: length of
# the scale bar; ``placeholder``: the window is not chosen yet (marked in the
# panel)
CASES = {
    "B": {"key": "long", "snippet": "f3_long", "below_min_score": 0.5, "scalebar_s": 0.5,
          "placeholder": True},
    "C": {"key": "dense", "snippet": "f3_dense", "below_min_score": 0.5, "scalebar_s": 0.5,
          "placeholder": True},
}

# ---------------------------------------------------------------------------
# layout [cm]: every row from L to R (as in the pipeline figure); three rows
# of spectrograms of one height, groups separated by GROUP_GAP
# ---------------------------------------------------------------------------
W = 17.4
L, R = 1.5, 17.0
ROW_H = 3.2
ZOOM_W, ZOOM_GAP = 3.4, 0.7  # the enlargement in A and the gap for its connectors
GROUP_GAP = 0.85
BOTTOM = 0.25
C_Y = BOTTOM
B_Y = C_Y + ROW_H + GROUP_GAP
A_Y = B_Y + ROW_H + GROUP_GAP + 0.15  # room for B's letter under A
KEY_Y_FROM_TOP = 0.3  # the key's centre line below the top edge
H = A_Y + ROW_H + 0.55 + 0.55  # hand-label triangles, then the key
LETTER_DX = 1.2


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


def fish_labels(ax, snip, order, t0, t1, x, dy, **kw):
    """EOD$f_1$ / EOD$f_2$ in white just above / below their bands."""
    for k, tid in enumerate(order[:2]):
        y = C.track_median(snip, tid, t0, t1) + dy[min(k, 1)]
        C.eod_label(ax, x, y, k + 1, **kw)


def draw_tracks(ax, snip, fish, t0, t1, **kw):
    """Tracks as the assignment uses them (gaps up to its max_gap_s bridged)."""
    dashed = C.draw_tracks(ax, snip, fish, t0, t1, bridge_s=C.ASSIGN_MAX_GAP_S, **kw)
    if dashed:
        DRAWN.add("interpolated")
    return dashed


# ---------------------------------------------------------------------------
# A: hand-labelled region and the missed chirp
# ---------------------------------------------------------------------------
def panel_labelled(ax, ax_zoom, fig, report, stats):
    snip = C.load_snippet("f3_labelled")
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 650, 960
    fish, order = C.fish_styles(snip, t0, t1)
    rows = C.window(snip, t0, t1)
    _, lv = C.show_power(ax, snip, t0, t1, f0, f1)
    dashed = draw_tracks(ax, snip, fish, t0, t1)
    drawn = C.draw_detections(ax, rows, fish, t0)
    DRAWN.update(drawn.drawn_as.dropna())
    labels = hand_labels(m["recording"])
    labels = labels[(labels.t_end_s > t0) & (labels.t_start_s < t1)]
    kept = drawn[drawn.drawn_as.isin(["assigned", "unassigned"])]
    lab_c, unmatched_lab, unmatched_det = match_labels(labels, kept)
    C.label_marks(ax, lab_c - t0)
    miss = rows.set_index("chirp_id").loc[MISSED_A]
    if miss.operating:
        msg = f"chirp {MISSED_A} (score {miss.score:.2f}) is not below the threshold {C.OPERATING}"
        raise ValueError(msg)
    C.draw_missed(ax, miss, t0)
    DRAWN.add("missed")
    # "?" above the highest box at that time (clear of every box edge)
    for r in unmatched_det.itertuples():
        DRAWN.add("qmark")
        near = kept[(kept.t_start_s < r.t_center_s + 0.06) & (kept.t_end_s > r.t_center_s - 0.06)]
        ax.text(r.t_center_s - t0, float(near.f_high_hz.max()) + 6, "?", ha="center",
                va="bottom", fontweight="bold", **C.ANNOT)
    fish_labels(ax, snip, order, t0, t1, 0.1, dy=(29, -29))
    C.scalebar(ax, t1 - t0 - 0.15, 668, 1.0, "1 s")
    C.spec_yaxis(ax, ticks=[700, 800, 900])
    n_lab = len(labels)
    n_found = n_lab - len(unmatched_lab)

    # enlargement of the missed chirp: the main panel's levels (it is a crop of
    # it, so the missed chirp keeps its brightness relative to the detected
    # chirps) and its frequency range (same scale; only time is stretched)
    zt0, zt1 = miss.t_center_s - 0.4, miss.t_center_s + 0.4
    C.show_power(ax_zoom, snip, zt0, zt1, f0, f1, levels=lv)
    draw_tracks(ax_zoom, snip, fish, zt0, zt1)
    DRAWN.update(C.draw_detections(ax_zoom, rows, fish, zt0).drawn_as.dropna())
    C.draw_missed(ax_zoom, miss, zt0)
    zl = lab_c[(lab_c > zt0) & (lab_c < zt1)]
    C.label_marks(ax_zoom, zl - zt0)
    ax_zoom.text(0.5 * (miss.t_start_s + miss.t_end_s) - zt0, miss.f_high_hz + 5,
                 f"score {miss.score:.2f}", ha="center", va="bottom", **C.ANNOT)
    C.scalebar(ax_zoom, zt1 - zt0 - 0.04, 668, 0.2, "200 ms")
    C.no_yaxis(ax_zoom)
    C.zoom_span(ax, zt0 - t0, zt1 - t0, ax_zoom)

    report["A"] = {
        "recording": m["recording"], "window_s": [t0, t1], "levels_db": lv,
        "n_hand_labels": int(n_lab), "n_found": int(n_found),
        "n_detections_kept": int(len(kept)),
        "labels_without_detection": [round(t, 3) for t in unmatched_lab],
        "detections_without_label": unmatched_det[
            ["chirp_id", "t_center_s", "score", "track_id", "emitter_hz"]
        ].round(3).to_dict(orient="records"),
        "missed": {"chirp_id": MISSED_A, "score": float(miss.score),
                   "t_center_s": float(miss.t_center_s),
                   "box_hz": [float(miss.f_low_hz), float(miss.f_high_hz)]},
        "zoom_s": [zt0, zt1],
        "tracks_hz": {int(t): round(C.track_median(snip, t, t0, t1), 1) for t in order},
        "interpolated_track_pieces": dashed,
    }
    hz = [C.track_median(snip, t, t0, t1) for t in order]
    stats["labelled"] = {"n_hand_labels": n_lab, "n_found": n_found,
                         "n_detections_kept": len(kept), "missed_score": miss.score,
                         "fish_hz": hz, "fish_df_hz": hz[0] - hz[1]}


# ---------------------------------------------------------------------------
# B, C: two hard cases for the detector
# ---------------------------------------------------------------------------
def nice_ticks(f0: float, f1: float) -> list[int]:
    """Frequency ticks every 100 or 200 Hz, at most four, inside [f0, f1]."""
    for step in (100, 200, 400):
        ticks = list(range(int(np.ceil(f0 / step) * step), int(f1) + 1, step))
        if len(ticks) <= 4:
            return ticks
    return ticks


def panel_case(ax, case: dict, report, stats):
    """One hard case: the window and band of its snippet, every detection at the
    operating threshold, and (dotted grey) the stored ones below it."""
    snip = C.load_snippet(case["snippet"])
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    if not m.get("band_hz"):
        msg = f"snippet {case['snippet']} has no band_hz: give one in v2_fig_extract.SNIPPETS"
        raise ValueError(msg)
    f0, f1 = m["band_hz"]
    fish, order = C.fish_styles(snip, t0, t1)
    rows = C.window(snip, t0, t1)
    _, lv = C.show_power(ax, snip, t0, t1, f0, f1)
    dashed = draw_tracks(ax, snip, fish, t0, t1)
    lo = case["below_min_score"]
    below = rows.iloc[0:0]
    if lo is not None:
        below = rows[~rows.operating & (rows.score >= lo)]
        for _, r in below.iterrows():
            C.draw_missed(ax, r, t0)
        if len(below):
            DRAWN.add("missed")
    drawn = C.draw_detections(ax, rows, fish, t0)
    DRAWN.update(drawn.drawn_as.dropna())
    fish_labels(ax, snip, order, t0, t1, 0.012 * (t1 - t0), dy=(22, -22))
    sb = case["scalebar_s"]
    C.scalebar(ax, t1 - t0 - 0.012 * (t1 - t0), f0 + 0.06 * (f1 - f0), sb,
               f"{sb * 1e3:.0f} ms" if sb < 1 else f"{sb:g} s")
    C.spec_yaxis(ax, ticks=nice_ticks(f0, f1))
    if case["placeholder"]:
        ax.text(0.5, 0.97, "placeholder window (not chosen yet)", transform=ax.transAxes,
                ha="center", va="top", color="#ffb000", fontsize=C.ANNOT_FS, zorder=30)

    det = drawn[drawn.drawn_as.isin(["assigned", "unassigned"])].sort_values("t_start_s")
    onsets = det.t_start_s.to_numpy()
    ioi = np.diff(onsets) * 1e3
    dur = (det.t_end_s - det.t_start_s).to_numpy() * 1e3
    ext = (det.f_high_hz - det.f_low_hz).to_numpy()
    span = t1 - t0
    hz = [C.track_median(snip, t, t0, t1) for t in order]

    def rng(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())} \
            if len(x) else {"median": np.nan, "min": np.nan, "max": np.nan}

    out = {
        "duration_s": span,
        "band_hz": [f0, f1],
        "n_detections": len(det),
        "n_assigned": int((det.drawn_as == "assigned").sum()),
        "n_unassigned": int((det.drawn_as == "unassigned").sum()),
        "n_assigned_per_fish": [int((det.track_id == t).sum()) for t in order],
        "n_below_threshold": len(below),
        "n_removed": int(drawn.drawn_as.isin(["harmonic", "duplicate"]).sum()),
        "rate_per_s": len(det) / span,
        "box_duration_ms": rng(dur),
        "box_extent_hz": rng(ext),
        "onset_interval_ms": rng(ioi),
        "height_hz": rng(det.height_hz),
        "fwhm_ms": rng(det.fwhm_s * 1e3),
        "width10_ms": rng(det.width10_s * 1e3),
        "n_params_flagged": int((~det.params_ok.astype(bool)).sum()),
        "fish_hz": hz,
    }
    if len(hz) >= 2:
        out["fish_df_hz"] = hz[0] - hz[1]
    report[case["key"]] = {
        "recording": m["recording"], "window_s": [t0, t1], "levels_db": lv,
        "placeholder": case["placeholder"], "interpolated_track_pieces": dashed,
        "detections": det[["chirp_id", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz",
                           "score", "track_id", "assign_confidence", "height_hz", "fwhm_s",
                           "params_ok", "drawn_as"]].round(3).to_dict(orient="records"),
        "below_threshold": below[["chirp_id", "t_center_s", "score"]].round(3)
        .to_dict(orient="records"),
        **{k: v for k, v in out.items() if k not in ("fish_hz",)},
        "fish_hz": [round(h, 1) for h in hz],
    }
    # stats: numbers only, NaN-free (write_stats refuses NaN)
    stats[case["key"]] = {k: v for k, v in out.items()
                          if not (isinstance(v, dict) and any(np.isnan(x) for x in v.values()))}


# ---------------------------------------------------------------------------
# numbers of the former panels (no longer drawn; cited in the Methods)
# ---------------------------------------------------------------------------
def trackgap_numbers(report, stats):
    """The 140-s gap in the track of the lower fish (2019-12-04, former B).

    Source run (PAPER_DETECTIONS=rtdetr): the track ends, the chirps in the
    gap stay unassigned. Gap-filled run (rtdetr_gapfill): the gap is bridged
    by linear interpolation and the chirps in it are assigned.
    """
    snip = C.load_snippet("f3_trackgap")
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    # both fish, ordered by frequency over a longer stretch (one track ends)
    fish, order = C.fish_styles(snip, t0 - 15, t1)
    rows = C.window(snip, t0, t1)
    drawn = C.detection_kinds(rows, fish)
    lower = order[-1]
    tt, ff = snip["tracks"][lower]
    detected = ~snip["filled"][lower]
    inside = tt[detected & (tt >= t0 - 15) & (tt <= t1 + 15)]
    end = float(inside[inside <= t1].max())
    rec_rows = C.detections(m["recording"])
    if C.GAPFILL:
        # the gap and what the patch did with it: the run's patched tracks
        z = np.load(C.TRACKS_RUN / m["recording"] / "tracks_gapfill.npz")
        k = int(np.flatnonzero((z["gap_fish_id"] == lower)
                               & (np.abs(z["gap_start_s"] - end) < 0.01))[0])
        gap = {"start_s": float(z["gap_start_s"][k]), "end_s": float(z["gap_end_s"][k]),
               "f_before_hz": float(z["gap_f_before_hz"][k]),
               "f_after_hz": float(z["gap_f_after_hz"][k]), "action": str(z["gap_action"][k])}
    else:
        # the gap in the source run's track-gap table (track_gaps/gaps.csv;
        # the cache holds the tracks only +- 15 s around the snippet)
        gaps = pd.read_csv(C.SOURCE_RUN / "track_gaps" / "gaps.csv")
        g = gaps[(gaps.recording == m["recording"]) & (gaps.fish_id == lower)
                 & ((gaps.start_s - end).abs() < 0.01)].iloc[0]
        gap = g[["fish", "kind", "start_s", "end_s", "duration_s", "chirps_in_gap",
                 "chirps_unassigned", "rescuable_by_interp"]].to_dict()
    in_gap = drawn[(drawn.t_center_s > gap["start_s"]) & (drawn.t_center_s < gap["end_s"])
                   & drawn.drawn_as.isin(["assigned", "unassigned"])]
    report["track_gap"] = {
        "recording": m["recording"], "window_s": [t0, t1], "detections": C.DETECTIONS,
        "track_id": int(lower), "track_last_detection_s": end,
        "gap": gap, "gap_s": float(gap["end_s"] - gap["start_s"]),
        "chirps_in_gap_in_window": in_gap[["chirp_id", "t_center_s", "score", "emitter_hz",
                                           "track_id", "assign_status",
                                           "assign_confidence"]].round(3)
        .to_dict(orient="records"),
        "n_assigned_in_window": int((drawn.drawn_as == "assigned").sum()),
        "n_unassigned_in_window": int((drawn.drawn_as == "unassigned").sum()),
        "recording_chirps_beyond_tolerance": int(
            (rec_rows.chirp & (rec_rows.assign_status == "beyond_tolerance")).sum()),
        "recording_chirps": int(rec_rows.chirp.sum()),
    }
    r = report["track_gap"]
    stats["track_gap"] = {
        "gap_s": r["gap_s"],
        "n_assigned_in_window": r["n_assigned_in_window"],
        "n_unassigned_in_window": r["n_unassigned_in_window"],
        "recording_chirps": r["recording_chirps"],
        "recording_chirps_beyond_tolerance": r["recording_chirps_beyond_tolerance"],
    }


def close_numbers(name, ids, half, channel, report, stats, key):
    """An assignment between fish < 7 Hz apart (former C left / right)."""
    snip = C.load_snippet(name)
    rec = snip["meta"]["recording"]
    rows = C.window(snip, snip["meta"]["t_start_s"] - 1, snip["meta"]["t_end_s"] + 1)
    sel = rows.set_index("chirp_id").loc[list(ids)]
    tc = float(sel.t_center_s.mean())
    t0, t1 = tc - half, tc + half
    _, order = C.fish_styles(snip, t0 - 5, t1 + 5)
    win = rows[(rows.t_end_s > t0) & (rows.t_start_s < t1)]
    hz = [C.track_median(snip, t, t0 - 5, t1 + 5) for t in order]
    df_onset = float(sel.track_df_other_hz.iloc[0])
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
        "df_hz": df_onset,
        "df_hz_all": sel.track_df_other_hz.round(3).tolist(),
        "chirp_gap_ms": 1e3 * float(sel.t_center_s.max() - sel.t_center_s.min()),
        "channel_window_s": snip["meta"]["windows_s"][channel],
        "other_detections_in_window": win.loc[~win.chirp_id.isin(ids), [
            "chirp_id", "t_center_s", "score", "operating", "track_id",
            "assign_confidence"]].round(3).to_dict(orient="records"),
    }
    stats[key] = {"df_hz": df_onset, "confidence": sel.assign_confidence.tolist(),
                  "chirp_gap_ms": report[key]["chirp_gap_ms"]}


def track_gaps(stats):
    """The interior track gaps of the trial fish in the 33 analysed trials, as
    the gap-filled run's patch handled them (its tracks_gapfill.npz tables)."""
    if not C.GAPFILL:
        return
    actions, fish, duration = [], set(), []
    for rec in sorted(C.analysed_recordings()):
        z = np.load(C.TRACKS_RUN / rec / "tracks_gapfill.npz")
        actions += [str(a) for a in z["gap_action"]]
        fish |= {(rec, int(f)) for f in z["gap_fish_id"]}
        duration += list(z["gap_end_s"] - z["gap_start_s"])
    actions = pd.Series(actions)
    stats["track_gaps_33_trials"] = {
        "n_gaps": len(actions), "n_fish_with_gaps": len(fish),
        "n_fish": 2 * len(C.analysed_recordings()),
        "actions": actions.value_counts().to_dict(),
        "duration_s": {"median": np.median(duration), "range": [min(duration), max(duration)]},
    }


def prevalence(report, stats):
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
    unassigned = ch.assign_status.isin(["beyond_tolerance", "no_live_track"])
    stats["prevalence_33_trials"] = {
        "chirps": len(ch),
        "assign_status": ch.assign_status.value_counts().to_dict(),
        "unassigned": {"n": unassigned.sum(), "fraction": unassigned.mean()},
        "assigned": len(assigned),
        "assigned_df_lt_10_hz": {"n": (assigned.track_df_other_hz < 10).sum(),
                                 "fraction": (assigned.track_df_other_hz < 10).mean()},
        "assigned_confidence_lt_0_9": {"n": (assigned.assign_confidence < 0.9).sum(),
                                       "fraction": (assigned.assign_confidence < 0.9).mean()},
    }


def key(fig):
    """The symbols drawn, in one centred line at the top."""
    items = [
        ("boxdot", {}, "detection"),
        ("boxdot", {"color": C.UNASSIGNED_COLOR}, "unassigned"),
        ("box", {"color": C.DROPPED_COLOR, "ls": C.DROPPED_LS, "alpha": C.DROPPED_ALPHA},
         "removed"),
        ("box", {"color": C.MISSED_COLOR, "ls": C.MISSED_LS}, "below threshold"),
        ("track", {}, "track"),
        ("track", {"ls": C.FILLED_LS}, "interpolated"),
        ("label", {}, "hand label"),
        ("qmark", {}, "no hand label"),
    ]
    needed = {"unassigned": "unassigned", "removed": ("harmonic", "duplicate"),
              "below threshold": "missed", "interpolated": "interpolated",
              "no hand label": "qmark"}

    def drawn(text):
        want = needed.get(text)
        if want is None:
            return True
        return any(w in DRAWN for w in ((want,) if isinstance(want, str) else want))

    items = [it for it in items if drawn(it[2])]
    y = H - KEY_Y_FROM_TOP
    width = C.key_strip(fig, 0.0, y, items, measure=True)
    C.key_strip(fig, L + (R - L - width) / 2, y, items)
    return [it[2] for it in items]


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {}
    stats: dict = {"operating_threshold": C.OPERATING}
    fig = plt.figure(figsize=(W * C.CM, H * C.CM))

    ax_a = C.axes_cm(fig, L, A_Y, R - L - ZOOM_W - ZOOM_GAP, ROW_H)
    ax_az = C.axes_cm(fig, R - ZOOM_W, A_Y, ZOOM_W, ROW_H)
    ax_b = C.axes_cm(fig, L, B_Y, R - L, ROW_H)
    ax_c = C.axes_cm(fig, L, C_Y, R - L, ROW_H)

    panel_labelled(ax_a, ax_az, fig, report, stats)
    panel_case(ax_b, CASES["B"], report, stats)
    panel_case(ax_c, CASES["C"], report, stats)
    report["key"] = key(fig)

    # numbers of the former panels B and C (cited in the Methods / caption)
    trackgap_numbers(report, stats)
    close_numbers("g_close_15388", [CLOSE_1], 0.45, 0, report, stats, "C1")
    close_numbers("g_close_9394", CLOSE_2, 0.45, 1, report, stats, "C2")
    prevalence(report, stats)
    track_gaps(stats)
    C.letters_cm(fig, [(ax_a, "A", LETTER_DX, 0.45), (ax_b, "B", LETTER_DX, 0.3),
                       (ax_c, "C", LETTER_DX, 0.3)])
    extents = C.axes_extents(fig, [("A", ax_a), ("A zoom", ax_az), ("B", ax_b),
                                   ("C", ax_c)])
    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print(extents)
    print("\n".join(map(str, paths)))
    write_stats("v2_failure_modes", stats)


if __name__ == "__main__":
    main()
