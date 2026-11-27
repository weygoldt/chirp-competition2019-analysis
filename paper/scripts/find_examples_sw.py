"""Candidate examples for Figs. 9B (harmonic copy, duplicate) and S4A (missed labelled chirp) in a run.

The examples of the pipeline-v2 figures were chosen among v2's detections; under
the single-window detector (production since 2026-11-27) two of them no longer
hold: the harmonic copy of Fig. 9B scores below the operating threshold, the
duplicate pair is kept as two chirps, and the labelled chirp that v2 missed in
Fig. S4A is detected. This lists candidates of each kind in the
run of PAPER_DETECTIONS, for the author to choose from (rendered with
``v2_fig_extract.py snippets --window`` and the figures):

``harmonics``
    Rows removed as harmonic copies (``pp_drop_reason == "harmonic"``) at the
    operating threshold with a kept chirp at the same time (centres within
    30 ms) whose box lies at half the frequencies (lower edge 0.4-0.6 of the
    copy's), in the analysed trials' dark phase; ranked by the copy's score.
    One per recording, a 0.6-s window (as Fig. 9B).
``duplicates``
    Rows removed as duplicates (``pp_drop_reason == "duplicate"``) at the
    operating threshold whose kept twin (``pp_duplicate_of``) is a chirp, in
    the analysed trials' dark phase; ranked by the lower of the two scores. One
    per recording, a 1.2-s window around the pair (as Fig. 9B).
``missed``
    Hand-labelled chirps (``eval/real_labels_365``) without a kept detection at
    the operating threshold within 0.1 s (the figure's ``match_labels``) but
    with a stored box below it that overlaps the label in time and frequency;
    for each, the 16-s window (inside its labelled region) centred on it as far
    as the region allows, with the window's labels, missed labels and
    detections without a label. Ranked: one missed label, no unlabelled
    detection, many labels.

    cd paper && PAPER_DETECTIONS=rtdetr_sw_gapfill_t080 uv run python scripts/find_examples_sw.py
"""

from __future__ import annotations

import pandas as pd

import v2_fig_common as C
from plot_v2_failure_modes import hand_labels, match_labels

WINDOW_S = 16.0  # Fig. S4A's window length
DUP_HALF_S = 0.6  # Fig. 9B's window: 1.2 s around the pair
DARK_PHASE_S = 10_800.0


def regions(recording: str) -> pd.DataFrame:
    path = (C.REAL_LABELS / "2019_tube_competition" / recording
            / "traces-grid1-editable-labels.csv")
    lab = pd.read_csv(path)
    return lab[lab["category"] == "labeled_region"]


def harmonics() -> pd.DataFrame:
    rows = []
    for rec in sorted(C.analysed_recordings()):
        d = C.detections(rec)
        chirps = d[d["chirp"]]
        harm = d[(d["pp_drop_reason"] == "harmonic") & d["operating"]
                 & (d["t_center_s"] < DARK_PHASE_S)]
        for r in harm.itertuples():
            f = chirps[((chirps.t_center_s - r.t_center_s).abs() < 0.03)
                       & (chirps.f_low_hz > 0.4 * r.f_low_hz) & (chirps.f_low_hz < 0.6 * r.f_low_hz)]
            if f.empty:
                continue
            fund = f.iloc[0]
            rows.append({"recording": rec, "fundamental": int(fund.chirp_id), "harmonic": int(r.chirp_id),
                         "fundamental_score": float(fund.score), "harmonic_score": float(r.score),
                         "fundamental_box_hz": (round(float(fund.f_low_hz)), round(float(fund.f_high_hz))),
                         "harmonic_box_hz": (round(r.f_low_hz), round(r.f_high_hz)),
                         "t0": round(r.t_center_s - 0.3, 2), "t1": round(r.t_center_s + 0.3, 2)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.sort_values("harmonic_score", ascending=False)
    return t.groupby("recording").head(1).reset_index(drop=True)


def duplicates() -> pd.DataFrame:
    rows = []
    for rec in sorted(C.analysed_recordings()):
        d = C.detections(rec)
        by = d.set_index("chirp_id")
        dup = d[(d["pp_drop_reason"] == "duplicate") & d["operating"]
                & (d["t_center_s"] < DARK_PHASE_S)]
        for r in dup.itertuples():
            twin_id = r.pp_duplicate_of
            if pd.isna(twin_id) or int(twin_id) not in by.index:
                continue
            twin = by.loc[int(twin_id)]
            if not bool(twin["chirp"]):
                continue
            tc = 0.5 * (r.t_center_s + twin["t_center_s"])
            rows.append({"recording": rec, "kept": int(twin_id), "dropped": int(r.chirp_id),
                         "kept_score": float(twin["score"]), "dropped_score": float(r.score),
                         "min_score": min(float(twin["score"]), float(r.score)),
                         "dt_ms": 1e3 * abs(float(twin["t_center_s"]) - r.t_center_s),
                         "kept_box_hz": (round(float(twin["f_low_hz"])), round(float(twin["f_high_hz"]))),
                         "dropped_box_hz": (round(r.f_low_hz), round(r.f_high_hz)),
                         "t0": round(tc - DUP_HALF_S, 2), "t1": round(tc + DUP_HALF_S, 2)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.sort_values("min_score", ascending=False)
    return t.groupby("recording").head(1).reset_index(drop=True)


def missed() -> pd.DataFrame:
    rows = []
    for f in sorted((C.REAL_LABELS / "2019_tube_competition").glob("*/traces-grid1-editable-labels.csv")):
        rec = f.parent.name
        labels = hand_labels(rec)
        d = C.detections(rec)
        kept = d[d["operating"] & d["kept"]]
        for reg in regions(rec).itertuples():
            lab = labels[(labels.t_start_s >= reg.t_start_s) & (labels.t_end_s <= reg.t_end_s)]
            if lab.empty:
                continue
            k = kept[(kept.t_center_s >= reg.t_start_s - 0.1) & (kept.t_center_s <= reg.t_end_s + 0.1)]
            lab_c, unmatched, _ = match_labels(lab, k)
            for tc in unmatched:
                row = lab.iloc[int(abs(lab_c - tc).argmin())]
                below = d[~d["operating"] & (abs(d.t_center_s - tc) <= 0.1)
                          & (d.f_low_hz < row.f_high_hz) & (d.f_high_hz > row.f_low_hz)]
                if below.empty:
                    continue
                t0 = min(max(tc - WINDOW_S / 2, reg.t_start_s), reg.t_end_s - WINDOW_S)
                t0 = max(t0, reg.t_start_s)
                t1 = t0 + WINDOW_S
                wl = lab[(lab.t_start_s >= t0) & (lab.t_end_s <= t1)]
                wk = k[(k.t_center_s >= t0) & (k.t_center_s <= t1)]
                _, wun, wdet = match_labels(wl, wk)
                rows.append({"recording": rec, "label_t_s": round(tc, 3),
                             "below_score": round(float(below.score.max()), 3),
                             "t0": round(t0, 1), "t1": round(t1, 1), "labels": len(wl),
                             "missed": len(wun), "unlabelled_detections": len(wdet)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["rank"] = ((t["missed"] == 1).astype(int) * 100 + (t["unlabelled_detections"] == 0) * 50
                 + t["labels"])
    return t.sort_values("rank", ascending=False).reset_index(drop=True)


def main() -> None:
    pd.set_option("display.width", 200)
    print(f"run {C.RUN} (operating threshold {C.OPERATING})")
    out = C.SNIPPETS
    out.mkdir(parents=True, exist_ok=True)
    for kind, fn, n in (("harmonics", harmonics, 12), ("duplicates", duplicates, 12),
                        ("missed", missed, 15)):
        t = fn()
        t.to_csv(out / f"candidates_{kind}.csv", index=False)
        print(f"\n{kind} (best first; all in {out / f'candidates_{kind}.csv'}):")
        print(t.head(n).to_string())


if __name__ == "__main__":
    main()
