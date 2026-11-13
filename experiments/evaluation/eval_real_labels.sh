#!/usr/bin/env bash
# Detection on the real hand labels after the labeller's confirmations.
#
# The four hand-labelled recordings hold 335 chirps in 6 exhaustively
# labelled regions (audian files). On 2026-11-13 the author labelled the
# final detectors' boxes that matched no label (label-emitters, blind to the
# model; $CHIRP_DATA/labels/emitters). Every box called a chirp whose centre
# lies in a labelled region, and that matches no hand label there, is added
# to that region as a chirp label (augmented audian files in OUT/labels/);
# boxes called not_chirp stay false positives. Then the v1 benchmark rule: centres within 100 ms, overlapping
# frequency ranges, one-to-one; boxes outside a region's band dropped.
# CPU only, seconds.
#
#   experiments/evaluation/eval_real_labels.sh [OUT]   (default $CHIRP_DATA/eval/real_labels_365)
#
# Predictions: dual-head and YOLO26s from eval_postprocess.py (predict_span
# over every labelled region, score >= 0.05, with tracks and post-processing;
# kept rows, as in production), v1 from the frozen benchmark's boxes with
# v1's NMS (IoU 0.2 per batch).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
out=${1:-$CHIRP_DATA/eval/real_labels_365}
cd "$CHIRP_REPO"
mkdir -p "$out/labels"
uv run python - "$out/labels" "$CHIRP_DATA/labels/emitters/labels" \
    "$CHIRP_REC_2019/2019-12-09-10_00" "$CHIRP_REC_2019/2020-06-03-10_10" \
    "$CHIRP_REC_2022/raw/2022-05-12-10_00" "$CHIRP_REC_2022/raw/2022-06-01-10_00" <<'PY'
import sys
from pathlib import Path
import pandas as pd

out, confirmed = Path(sys.argv[1]), Path(sys.argv[2])
name = "traces-grid1-editable-labels.csv"
for rec in map(Path, sys.argv[3:]):
    hand = pd.read_csv(rec / name)
    regions = hand[hand["category"] == "labeled_region"]
    extra = pd.DataFrame(columns=hand.columns)
    path = confirmed / f"{rec.name}.csv"
    if path.exists():
        lab = pd.read_csv(path)
        lab = lab[lab["label"].isin(["chirp", "chirp_emitter_unclear"])]
        centre = (lab["t_start_s"] + lab["t_end_s"]) / 2
        inside = pd.Series(False, index=lab.index)
        for _, r in regions.iterrows():
            inside |= centre.between(r["t_start_s"], r["t_end_s"])
        # a box on a chirp that is already labelled (matching rule: centres
        # within 100 ms, overlapping frequency ranges) adds no chirp
        chirps = hand[hand["category"] == "chirp"]
        known = (chirps["t_start_s"] + chirps["t_end_s"]) / 2
        for k, row in lab.iterrows():
            if ((known - centre[k]).abs().le(0.1)
                    & (chirps["f_low_hz"] < row["f_high_hz"])
                    & (row["f_low_hz"] < chirps["f_high_hz"])).any():
                inside[k] = False
        extra = lab.loc[inside, ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].assign(
            category="chirp", kind="span", note="confirmed " + lab.loc[inside, "item_id"])
    target = out / rec.parent.name / rec.name
    target.mkdir(parents=True, exist_ok=True)
    pd.concat([hand, extra[hand.columns.intersection(extra.columns)]], ignore_index=True)[
        hand.columns].to_csv(target / name, index=False)
    n_hand = int((hand["category"] == "chirp").sum())
    print(f"{rec.name}: {n_hand} hand-labelled + {len(extra)} confirmed chirps")
PY
labels=()
for f in "$out"/labels/*/*/traces-grid1-editable-labels.csv; do labels+=(--labels "$f"); done
tracks=(--tracks "$CHIRP_REC_2019/2019-12-09-10_00" --tracks "$CHIRP_REC_2019/2020-06-03-10_10"
        --tracks "$CHIRP_REC_2022/interrim/2022-05-12-10_00" --tracks "$CHIRP_REC_2022/interrim/2022-06-01-10_00")
for run in dualhead_r18_sim_v4_r1 yolo26s_sim_v4_r1; do
  mkdir -p "$out/$run"
  uv run python - "$CHIRP_DATA/runs/$run/eval_postprocess/real_predictions_flagged.parquet" \
      "$out/$run/predictions_kept.parquet" <<'PY'
import sys
import pandas as pd
from chirpdetector.postprocess import COLUMNS
flagged = pd.read_parquet(sys.argv[1])
kept = flagged[flagged["drop_reason"].astype(str) == ""]
kept = kept.drop(columns=[c for c in COLUMNS if c in kept])
kept.reset_index(drop=True).to_parquet(sys.argv[2])
print(f"{sys.argv[1]}: {len(kept)} of {len(flagged)} rows kept")
PY
  uv run chirpdetector evaluate "$out/$run/predictions_kept.parquet" "${labels[@]}" "${tracks[@]}" \
      --out "$out/$run"
done
uv run chirpdetector evaluate "$CHIRP_V1_BENCHMARK"/results/boxes/*.csv --nms-iou 0.2 \
    "${labels[@]}" "${tracks[@]}" --out "$out/v1"
