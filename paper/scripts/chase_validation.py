"""Timing validation of the BORIS chase annotations: the author's re-labelled chases against BORIS.

New step (not in the old analysis). The chases were annotated in BORIS by another person. A random sample
was re-labelled by the author in the review tool (``paper/chase_review``) with a dominant-centred
criterion: onset = the dominant accelerates towards the subordinate, offset = the dominant decelerates
and/or changes its trajectory. Reviews either adjusted one BORIS chase (``mode="target"``) or marked
every chase in the reviewed window (``mode="window"``), which ``export_revisions.match_window`` matches
to the BORIS chases by overlap.

Inputs: the frozen copy ``paper_config.CHASE_VALIDATION`` (``session.json``, ``revisions.jsonl``). The
BORIS times in the session are checked against ``load_and_convert_boris_events`` (the analysis' times)
and the step stops on any difference.

Outputs (``$PAPER_OUT/data/processed/chase_validation/``):
  pairs.csv          one row per BORIS chase matched one-to-one: author and BORIS onset / offset
                     (recording s), shifts (author - BORIS, NaN where an end is censored)
  segmentation.csv   status of every BORIS chase in a whole-window review, and author-only chases
and the stats namespace ``chase_validation``: sample sizes, median / IQR / share earlier of the onset
and offset shifts, their dependence on chase duration and recording, and the segmentation counts.

Rules: every review counts (the latest save of a chase), also those saved with the BORIS markers or the
chirps visible (the author: the criterion did not depend on them; chirps were switched on only after
marking). In one-chase reviews an end on the first / last frame of the window is censored (the chase ran
past the window), as the window reviews flag it.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

from analyze_event_time_rates_trial import load_and_convert_boris_events
from manuscript_stats import write_stats
from paper_config import CHASE_VALIDATION, DATA_ROOT, PAPER, processed_dir

sys.path.insert(0, str(PAPER / "chase_review"))
import export_revisions as ex  # noqa: E402

T_MAX_S = 3 * 60 * 60


def load_inputs():
    ses_path, rev_path = CHASE_VALIDATION / "session.json", CHASE_VALIDATION / "revisions.jsonl"
    for p in (ses_path, rev_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: freeze the chase review first (paper/inputs/README.md)")
    return json.loads(ses_path.read_text()), ex.latest(rev_path)


def check_boris_times(session: dict) -> None:
    """The session's BORIS times must be the analysis' times (load_and_convert_boris_events)."""
    for rec in session["recordings"]:
        _, on_off, _, _ = load_and_convert_boris_events(DATA_ROOT / rec, rec, sr=20_000)
        mine = np.array([[c["onset_rec_s"], c["offset_rec_s"]] for c in session["chases"] if c["recording"] == rec])
        if mine.shape != on_off.reshape(-1, 2).shape or not np.allclose(mine, on_off.reshape(-1, 2), rtol=0, atol=1e-9):
            raise AssertionError(f"{rec}: chase times of the review session differ from load_and_convert_boris_events")


def target_pairs(session: dict, revisions: dict) -> list[dict]:
    """Pairs from one-chase reviews (the author's marks of that BORIS chase)."""
    chases = {c["id"]: c for c in session["chases"]}
    out = []
    for cid, r in revisions.items():
        if r.get("mode", "target") != "target" or r.get("not_a_chase"):
            continue
        c, w = chases[cid], r["window"]
        w0, w1 = w["frame0"], w["frame0"] + w["n"] - 1
        on_cens, off_cens = r["onset_frame"] == w0, r["offset_frame"] == w1
        out.append({"chase_id": cid, "recording": c["recording"], "source": "one-chase review", "own_review": True,
                    "boris_onset_rec_s": c["onset_rec_s"], "boris_offset_rec_s": c["offset_rec_s"],
                    "onset_rec_s": r["onset_rec_s"], "offset_rec_s": r["offset_rec_s"],
                    "boris_onset_frame": c["onset_frame"], "boris_offset_frame": c["offset_frame"],
                    "onset_frame": r["onset_frame"], "offset_frame": r["offset_frame"],
                    "onset_censored": on_cens, "offset_censored": off_cens,
                    "d_onset_s": np.nan if on_cens else r["d_onset_s"],
                    "d_offset_s": np.nan if off_cens else r["d_offset_s"],
                    "blind": r.get("blind"), "saved_at": r.get("saved_at")})
    return out


def window_pairs(wf: pd.DataFrame) -> list[dict]:
    m = wf[wf["status"] == "matched"]
    return [{"chase_id": r.chase_id, "recording": r.recording, "source": "window review",
             "own_review": r.chase_id == r.window_of,
             "boris_onset_rec_s": r.boris_onset_rec_s, "boris_offset_rec_s": r.boris_offset_rec_s,
             "onset_rec_s": r.onset_rec_s, "offset_rec_s": r.offset_rec_s,
             "boris_onset_frame": int(r.boris_onset_frame), "boris_offset_frame": int(r.boris_offset_frame),
             "onset_frame": int(r.onset_frame), "offset_frame": int(r.offset_frame),
             "onset_censored": bool(r.onset_censored), "offset_censored": bool(r.offset_censored),
             "d_onset_s": r.d_onset_s, "d_offset_s": r.d_offset_s, "blind": r.blind, "saved_at": r.saved_at}
            for r in m.itertuples()]


def shift_stats(d: pd.Series, dur: pd.Series, rec: pd.Series) -> dict:
    ok = d.notna()
    d, dur, rec = d[ok].to_numpy(float), dur[ok].to_numpy(float), rec[ok].to_numpy()
    q25, q50, q75 = np.percentile(d, [25, 50, 75])
    rho, p_rho = st.spearmanr(dur, d)
    groups = [d[rec == r] for r in np.unique(rec) if np.sum(rec == r) >= 3]
    kw = st.kruskal(*groups) if len(groups) >= 2 else None
    return {"n": int(len(d)), "n_recordings": int(len(np.unique(rec))), "median_s": float(q50),
            "iqr_s": [float(q25), float(q75)], "earlier_fraction": float(np.mean(d < 0)),
            "later_fraction": float(np.mean(d > 0)),
            "duration_spearman": {"rho": float(rho), "p": float(p_rho)},
            "recording_kruskal": {"n_recordings": len(groups), "h": float(kw.statistic) if kw else None,
                                  "p": float(kw.pvalue) if kw else None}}


def main() -> None:
    session, revisions = load_inputs()
    check_boris_times(session)
    wf = ex.window_table(session, revisions)
    pairs = pd.DataFrame(target_pairs(session, revisions) + window_pairs(wf))
    # a BORIS chase reviewed more than once: its own review first, then the latest save
    pairs = (pairs.sort_values(["own_review", "saved_at"], ascending=False)
             .drop_duplicates("chase_id").sort_values(["recording", "boris_onset_rec_s"]).reset_index(drop=True))
    pairs["boris_duration_s"] = pairs["boris_offset_rec_s"] - pairs["boris_onset_rec_s"]
    pairs["in_analysis_window"] = (pairs["boris_onset_rec_s"] < T_MAX_S) & (pairs["boris_offset_rec_s"] < T_MAX_S)
    out = processed_dir("chase_validation")
    pairs.to_csv(out / "pairs.csv", index=False)
    wf.to_csv(out / "segmentation.csv", index=False)

    n_window = sum(1 for r in revisions.values() if r.get("mode") == "window")
    n_target = sum(1 for r in revisions.values() if r.get("mode", "target") == "target")
    seg = wf[wf["chase_id"].notna() & wf["boris_in_window"].astype(bool)]
    counts = seg["status"].value_counts()
    values = {
        "n_reviews": len(revisions), "n_window_reviews": n_window, "n_one_chase_reviews": n_target,
        "n_not_a_chase": int(sum(1 for r in revisions.values() if r.get("not_a_chase"))),
        "n_blind_reviews": int(sum(1 for r in revisions.values() if r.get("blind"))),
        "n_pairs": len(pairs), "n_recordings": int(pairs["recording"].nunique()),
        "onset": shift_stats(pairs["d_onset_s"], pairs["boris_duration_s"], pairs["recording"]),
        "offset": shift_stats(pairs["d_offset_s"], pairs["boris_duration_s"], pairs["recording"]),
        "segmentation": {
            "n_boris_chases": int(len(seg)),
            **{k: int(counts.get(k, 0)) for k in ("matched", "split", "merged", "boris_only")},
            "author_only": int((wf["status"] == "yours_only").sum()),
            "matched_fraction": float(counts.get("matched", 0) / max(len(seg), 1)),
        },
    }
    write_stats("chase_validation", values)
    print(json.dumps(values, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
