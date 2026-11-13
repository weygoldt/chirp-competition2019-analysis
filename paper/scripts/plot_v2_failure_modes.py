"""Figure: remaining failure modes of chirp detection pipeline v2 on real recordings.

Replaces ``plot_pipeline_performances_manuscript_supplement.pdf`` (two v1
snippets with annotated failures). Real spectrograms (detector's 205-ms
channel, click-cleaned, power summed over electrodes) of the production run
``competition2019_v2_20261113`` with its boxes at the operating threshold,
coloured by the fish they were assigned to, and the wavetracker tracks.
Only failures that can be checked by eye, or against hand labels, are
annotated:

A   2019-12-09-10_00, 4312-4328 s, inside a hand-labelled test region:
    hand-labelled chirps (arrowheads) against detections; one faint
    labelled chirp scored 0.33 and is missed at the 0.5 threshold. Two
    fish 21 Hz apart; one fish chirps in a rapid series.
B   2019-12-04-10_00, 7468-7488 s: the wavetracker track of the lower fish
    ends at 7477.8 s (next detection > 10 s later) while the fish keeps
    chirping; its later chirps are detected but left unassigned (no live
    track within 20 Hz).
C   Crops: (1) one chirp between two fish 2.7 Hz apart, assigned with
    confidence 0.53 (emitter not identifiable by eye); (2) two chirps 75 ms
    apart between fish 6 Hz apart, shown in the 51-ms channel where they
    are separate, assigned to different fish with confidence 0.72 and 0.56;
    (3) a faint hand-labelled chirp scored 0.29 (missed).

Also prints how often the failure-prone situations occur among the chirps
of the 33 analysed trials (unassigned chirps; chirps between fish < 10 Hz
apart; low assignment confidence).

Inputs: snippet cache of ``v2_fig_extract.py``, the run's parquet tables,
the hand labels in ``eval/real_labels_365/labels``.

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_failure_modes.py

Runtime: ~15 s.
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
MISSED_C = 453  # 2019-12-09-10_00, score 0.29, on a hand-labelled chirp
CLOSE_1 = 14783  # 2019-12-09-10_00
CLOSE_2 = (2974, 2975)  # 2020-03-13-10_00


def axes_cm(fig, x, y, w, h):
    W, H = fig.get_size_inches() / C.CM
    return fig.add_axes((x / W, y / H, w / W, h / H))


def hand_labels(recording: str) -> pd.DataFrame:
    path = (C.REAL_LABELS / "2019_tube_competition" / recording
            / "traces-grid1-editable-labels.csv")
    lab = pd.read_csv(path)
    return lab[lab["category"] == "chirp"]


def colors_for(snip, t0, t1):
    order = C.fish_order(snip, t0, t1)
    return {tid: C.FISH_COLORS[i % 2] for i, tid in enumerate(order)}, order


def track_median(snip, tid, t0, t1):
    tt, ff = snip["tracks"][tid]
    m = (tt >= t0) & (tt <= t1)
    return float(np.median(ff[m]))


def label_marks(ax, labels, t0, t1, y):
    sel = labels[(labels.t_end_s > t0) & (labels.t_start_s < t1)]
    tc = 0.5 * (sel.t_start_s + sel.t_end_s) - t0
    ax.plot(tc, np.full(len(tc), y), "v", ms=3.2, mfc="white", mec="k", mew=0.3,
            zorder=20, clip_on=False)
    return sel


def panel_labelled(ax, report):
    snip = C.load_snippet("f3_labelled")
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 640, 960
    colors, order = colors_for(snip, t0, t1)
    rows = C.window(snip, t0, t1)
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=0.3, vmax=0.95)
    C.draw_tracks(ax, snip, colors, t0, t1)
    drawn = C.draw_detections(ax, rows, colors, t0)
    labels = label_marks(ax, hand_labels(m["recording"]), t0, t1, f1 - 12)
    miss = rows.set_index("chirp_id").loc[MISSED_A]
    C.draw_box(ax, miss, t0, "white", ls=(0, (1, 1.5)), lw=0.9, marker=False)
    ax.annotate(
        f"missed\n(score {miss.score:.2f})",
        xy=(miss.t_end_s - t0, miss.f_high_hz - 10),
        xytext=(miss.t_end_s - t0 + 1.3, 905),
        arrowprops=dict(**C.ARROW, connectionstyle="arc3,rad=0.3"),
        ha="center", va="center", **C.ANNOT,
    )
    for k, tid in enumerate(order):
        y = track_median(snip, tid, t0, t1)
        C.eod_label(ax, 0.1, y + (14 if k == 0 else -14), k + 1, color=colors[tid])
    C.scalebar(ax, t1 - t0 - 0.2, 660, 1.0, "", dy=6)
    ax.text(t1 - t0 - 0.7, 666, "1 s", ha="center", va="bottom", **C.ANNOT)
    C.clean_spec_axes(ax)
    # bookkeeping: detections vs hand labels in the window
    kept = drawn[drawn.drawn_as.isin(["assigned", "unassigned"])]
    # greedy one-to-one matching of centres within 0.1 s (closest pairs first),
    # as the evaluation's rule without the frequency-overlap condition
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
    report["A"] = {
        "recording": m["recording"], "window_s": [t0, t1],
        "n_hand_labels": int(len(labels)), "n_detections_kept": int(len(kept)),
        "labels_without_detection": [round(t, 3) for t in unmatched_lab],
        "detections_without_label": unmatched_det[
            ["chirp_id", "t_center_s", "score", "track_id", "emitter_hz"]
        ].round(3).to_dict(orient="records"),
        "missed": {"chirp_id": MISSED_A, "score": float(miss.score)},
        "tracks_hz": {int(t): round(track_median(snip, t, t0, t1), 1) for t in order},
    }


def panel_trackgap(ax, report):
    snip = C.load_snippet("f3_trackgap")
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    f0, f1 = 600, 900
    # both fish, ordered by frequency over a longer stretch (one track ends)
    colors, order = colors_for(snip, t0 - 15, t1)
    rows = C.window(snip, t0, t1)
    C.show_spec(ax, snip, t0, t1, f0, f1, vmin=0.3, vmax=0.95)
    C.draw_tracks(ax, snip, colors, t0, t1, lw=0.8)
    drawn = C.draw_detections(ax, rows, colors, t0)
    lower = order[-1]
    tt, ff = snip["tracks"][lower]
    inside = tt[(tt >= t0 - 15) & (tt <= t1 + 15)]
    end = float(inside[inside <= t1].max())
    nxt = inside[inside > end]
    f_end = float(ff[tt == end][0])
    ax.annotate(
        "track ends", xy=(end - t0, f_end - 4), xytext=(end - t0 - 1.5, 640),
        arrowprops=dict(**C.ARROW, connectionstyle="arc3,rad=-0.2"),
        ha="center", va="center", **C.ANNOT,
    )
    un = drawn[drawn.drawn_as == "unassigned"]
    first = un.sort_values("t_center_s").iloc[0]
    ax.annotate(
        "chirps unassigned", xy=(first.t_center_s - t0, first.f_high_hz + 4),
        xytext=(first.t_center_s - t0 + 2.6, 850),
        arrowprops=dict(**C.ARROW, connectionstyle="arc3,rad=0.25"),
        ha="center", va="center", **C.ANNOT,
    )
    for u in un.sort_values("t_center_s").iloc[1:].itertuples():
        ax.annotate(
            "", xy=(u.t_center_s - t0, u.f_high_hz + 4),
            xytext=(first.t_center_s - t0 + 2.6, 838),
            arrowprops=dict(**C.ARROW, connectionstyle="arc3,rad=0.15"),
        )
    for k, tid in enumerate(order):
        y = track_median(snip, tid, t0 - 15, t1)
        C.eod_label(ax, 0.15, y + 14, k + 1, color=colors[tid])
    C.scalebar(ax, t1 - t0 - 0.2, 615, 1.0, "", dy=6)
    ax.text(t1 - t0 - 0.7, 621, "1 s", ha="center", va="bottom", **C.ANNOT)
    C.clean_spec_axes(ax)
    rec_rows = C.detections(m["recording"])
    report["B"] = {
        "recording": m["recording"], "window_s": [t0, t1],
        "track_id": int(lower), "track_last_detection_s": end,
        "track_next_detection_s": float(nxt.min()) if len(nxt) else None,
        "n_assigned_in_window": int((drawn.drawn_as == "assigned").sum()),
        "n_unassigned_in_window": int(len(un)),
        "unassigned": un[["chirp_id", "t_center_s", "score", "emitter_hz",
                          "nearest_track_hz", "assign_status"]].round(2)
        .to_dict(orient="records"),
        "recording_chirps_beyond_tolerance": int(
            (rec_rows.chirp & (rec_rows.assign_status == "beyond_tolerance")).sum()),
        "recording_chirps": int(rec_rows.chirp.sum()),
    }


def crop(ax, name, ids, f0, f1, half, channel, report, key, title):
    snip = C.load_snippet(name)
    rec = snip["meta"]["recording"]
    rows = C.window(snip, snip["meta"]["t_start_s"], snip["meta"]["t_end_s"])
    sel = rows.set_index("chirp_id").loc[list(ids)]
    tc = float(sel.t_center_s.mean())
    t0, t1 = tc - half, tc + half
    # fish identity over the snippet's surroundings (tracks are sparse in 0.6 s)
    colors, order = colors_for(snip, t0 - 5, t1 + 5)
    C.show_spec(ax, snip, t0, t1, f0, f1, channel=channel,
                vmin=0.3 if channel == 0 else None, vmax=0.95 if channel == 0 else None)
    C.draw_tracks(ax, snip, colors, t0, t1, lw=0.7)
    for _, r in sel.iterrows():
        if r.operating:
            C.draw_box(ax, r, t0, colors.get(int(r.track_id), "white")
                       if np.isfinite(r.track_id) else "white", lw=0.9)
        else:
            C.draw_box(ax, r, t0, "white", ls=(0, (1, 1.5)), lw=0.9, marker=False)
    C.clean_spec_axes(ax, ylabel=False)
    ax.set_title(title, loc="left", fontsize="small", pad=2)
    report[key] = {
        "recording": rec,
        "rows": sel.reset_index()[["chirp_id", "t_center_s", "score", "emitter_hz",
                                   "track_id", "track_hz", "second_track_hz",
                                   "track_df_other_hz", "assign_confidence"]]
        .round(3).to_dict(orient="records"),
        "tracks_hz": {int(t): round(track_median(snip, t, t0 - 5, t1 + 5), 2)
                      for t in order},
        "channel_window_s": snip["meta"]["windows_s"][channel],
    }
    return snip, rows, t0, colors, order


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
    fig = plt.figure(figsize=(C.FIG_WIDTH, 12.6 * C.CM))
    ax_a = axes_cm(fig, 1.35, 8.55, 15.85, 3.6)
    ax_b = axes_cm(fig, 1.35, 4.35, 15.85, 3.6)
    ax_c = [axes_cm(fig, 1.35 + i * 5.35, 0.35, 4.95, 3.25) for i in range(3)]

    panel_labelled(ax_a, report)
    panel_trackgap(ax_b, report)

    # C1: one chirp between two fish 2.7 Hz apart
    snip, rows, t0, colors, order = crop(
        ax_c[0], "g_close_15388", [CLOSE_1], 640, 900, 0.3, 0, report, "C1",
        r"$\Delta f$ = 2.7 Hz")
    r = rows.set_index("chirp_id").loc[CLOSE_1]
    ax_c[0].text(0.97, 0.94, f"confidence {r.assign_confidence:.2f}",
                 transform=ax_c[0].transAxes, ha="right", va="top", **C.ANNOT)
    ax_c[0].set_ylabel("Frequency [Hz]")
    # C2: two chirps 75 ms apart, 51-ms channel
    snip, rows, t0, colors, order = crop(
        ax_c[1], "g_close_9394", CLOSE_2, 640, 900, 0.3, 1, report, "C2",
        r"$\Delta f$ = 6.4 Hz, 51-ms window")
    rr = rows.set_index("chirp_id").loc[list(CLOSE_2)]
    ax_c[1].text(0.97, 0.94, "confidence " + ", ".join(
        f"{v:.2f}" for v in rr.assign_confidence), transform=ax_c[1].transAxes,
        ha="right", va="top", **C.ANNOT)
    # C3: faint hand-labelled chirp below threshold
    snip, rows, t0, colors, order = crop(
        ax_c[2], "g_missed_1208", [MISSED_C], 640, 900, 0.3, 0, report, "C3",
        "faint chirp")
    lab = hand_labels(snip["meta"]["recording"])
    label_marks(ax_c[2], lab, t0, t0 + 0.6, 892)
    r = rows.set_index("chirp_id").loc[MISSED_C]
    ax_c[2].text(0.97, 0.94, f"missed (score {r.score:.2f})",
                 transform=ax_c[2].transAxes, ha="right", va="top", **C.ANNOT)
    for ax in ax_c:
        C.scalebar(ax, 0.58, 655, 0.1, "", dy=6)
        ax.text(0.53, 662, "100 ms", ha="center", va="bottom", **C.ANNOT)
    for ax in ax_c[1:]:
        ax.set_yticklabels([])

    prevalence(report)
    letter_subplots([ax_a, ax_b, ax_c[0]], xoffset=[-0.075, -0.075, -0.24],
                    yoffset=[1.05, 1.05, 1.09])
    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
