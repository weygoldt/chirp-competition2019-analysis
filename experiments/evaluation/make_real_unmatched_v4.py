"""Candidates file of the final v4 models' unmatched detections on real data.

Writes $CHIRP_DATA/labels/candidates/real_unmatched_v4.parquet: every prediction of
dualhead_r18_sim_v4_r1 and yolo26s_sim_v4_r1 with outcome ``fp`` at score
>= 0.5 on the four hand-labelled recordings (eval/real/items.csv; boxes from
eval/real_predictions.parquet by ``pred_row``). A dual-head box and a YOLO
box of the same recording with IoU >= 0.3 are one candidate (matched
one-to-one, highest IoU first): it keeps the dual-head box, score and
predicted emitter, and lists both predictions. ``track_id`` is the live
track nearest to the dual-head's ``emitter_hz`` within 10 Hz (as in
chirpdetector's showcase). Format: chirpdetector.labeling.candidates.

    uv run python experiments/evaluation/make_real_unmatched_v4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA, REC_2019, REC_2022  # noqa: E402


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gridtools.io import load_tracks

from chirpdetector.evaluation.assignment import assign_by_frequency
from chirpdetector.labeling.candidates import read_candidates

RUNS = DATA / "runs"
DUALHEAD, YOLO = "dualhead_r18_sim_v4_r1", "yolo26s_sim_v4_r1"
RECORDINGS = {
    "2019-12-09-10_00": (str(REC_2019 / "2019-12-09-10_00"), None),
    "2020-06-03-10_10": (str(REC_2019 / "2020-06-03-10_10"), None),
    "2022-05-12-10_00": (
        str(REC_2022 / "raw/2022-05-12-10_00"),
        str(REC_2022 / "interrim/2022-05-12-10_00"),
    ),
    "2022-06-01-10_00": (
        str(REC_2022 / "raw/2022-06-01-10_00"),
        str(REC_2022 / "interrim/2022-06-01-10_00"),
    ),
}
MIN_SCORE = 0.5
DUP_IOU = 0.3
EMITTER_TOLERANCE_HZ = 10.0
BOX = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]
OUT = DATA / "labels/candidates/real_unmatched_v4.parquet"


def unmatched(run: str) -> pd.DataFrame:
    eval_dir = RUNS / run / "eval"
    items = pd.read_csv(eval_dir / "real" / "items.csv")
    preds = pd.read_parquet(eval_dir / "real_predictions.parquet")
    fp = items[(items["outcome"] == "fp") & (items["score"] >= MIN_SCORE)]
    rows = fp["pred_row"].astype(int).to_numpy()
    out = preds.iloc[rows].copy()
    assert np.allclose(out["score"].to_numpy(), fp["score"].to_numpy())
    assert (out["recording"].to_numpy() == fp["recording"].to_numpy()).all()
    out["pred_row"] = rows
    return out.reset_index(drop=True)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    dt = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    df = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = dt * df
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    dh, yo = unmatched(DUALHEAD), unmatched(YOLO)
    print(f"{DUALHEAD}: {len(dh)} unmatched >= {MIN_SCORE}; {YOLO}: {len(yo)}")
    rows = []
    for rec in sorted(set(dh["recording"]) | set(yo["recording"])):
        a = dh[dh["recording"] == rec]
        b = yo[yo["recording"] == rec]
        pairs = sorted(
            (
                (iou(x, y), i, j)
                for i, x in zip(a.index, a[BOX].to_numpy(), strict=True)
                for j, y in zip(b.index, b[BOX].to_numpy(), strict=True)
            ),
            reverse=True,
        )
        match: dict[int, tuple[int, float]] = {}
        used: set[int] = set()
        for value, i, j in pairs:
            if value < DUP_IOU:
                break
            if i not in match and j not in used:
                match[i] = (j, value)
                used.add(j)
        for i, r in a.iterrows():
            j, value = match.get(int(i), (None, np.nan))  # pyright: ignore[reportArgumentType]
            y = b.loc[j] if j is not None else None
            rows.append(
                {
                    "recording": rec,
                    **{c: float(r[c]) for c in BOX},
                    "candidate_id": f"v4fp-dualhead-{int(r['pred_row'])}",
                    "score": float(r["score"]),
                    "emitter_hz": float(r["emitter_hz"]),
                    "source": DUALHEAD if y is None else f"{DUALHEAD}+{YOLO}",
                    "found_by": "dualhead" if y is None else "both",
                    "dualhead_pred_row": int(r["pred_row"]),
                    "dualhead_score": float(r["score"]),
                    "yolo_pred_row": None if y is None else int(y["pred_row"]),
                    "yolo_score": None if y is None else float(y["score"]),
                    "iou": value,
                    "model_sha256": str(r["model_sha256"]),
                }
            )
        for j, y in b.iterrows():
            if int(j) in used:  # pyright: ignore[reportArgumentType]
                continue
            rows.append(
                {
                    "recording": rec,
                    **{c: float(y[c]) for c in BOX},
                    "candidate_id": f"v4fp-yolo-{int(y['pred_row'])}",
                    "score": float(y["score"]),
                    "emitter_hz": np.nan,
                    "source": YOLO,
                    "found_by": "yolo",
                    "dualhead_pred_row": None,
                    "dualhead_score": None,
                    "yolo_pred_row": int(y["pred_row"]),
                    "yolo_score": float(y["score"]),
                    "iou": np.nan,
                    "model_sha256": str(y["model_sha256"]),
                }
            )
    df = pd.DataFrame(rows)
    tracks = {}
    for name, (rec_dir, track_dir) in RECORDINGS.items():
        tracks[name] = load_tracks(track_dir or rec_dir)
    assigned = assign_by_frequency(df, tracks, tolerance_hz=EMITTER_TOLERANCE_HZ)
    df["track_id"] = assigned["emitter_id"].astype("Int64")
    df["tracks"] = [RECORDINGS[r][1] for r in df["recording"]]
    df["recording"] = [RECORDINGS[r][0] for r in df["recording"]]
    for c in ("dualhead_pred_row", "yolo_pred_row"):
        df[c] = df[c].astype("Int64")
    for c in ("dualhead_score", "yolo_score"):
        df[c] = df[c].astype("float64")
    df = df.sort_values(["recording", "t_start_s"]).reset_index(drop=True)
    df = df[
        [
            "recording", "tracks", *BOX, "candidate_id", "score", "emitter_hz",
            "track_id", "source", "found_by", "dualhead_pred_row", "dualhead_score",
            "yolo_pred_row", "yolo_score", "iou", "model_sha256",
        ]
    ]
    df.to_parquet(OUT, index=False)
    checked = read_candidates(OUT)  # the labeller's validation
    print(f"wrote {OUT}: {len(checked)} candidates")
    print(df.groupby(["found_by"]).size().to_string())
    print(df.groupby(df["recording"].str.rsplit("/", n=1).str[-1]).size().to_string())
    print("track_id assigned:", int(df["track_id"].notna().sum()), "of", int(df["emitter_hz"].notna().sum()), "with emitter_hz")


if __name__ == "__main__":
    sys.exit(main())
