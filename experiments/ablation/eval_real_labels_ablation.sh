#!/usr/bin/env bash
# experiments/evaluation/eval_real_labels.sh for the front-end ablation: the
# same label augmentation (365 = 335 hand labels + 30 confirmed) and the same
# `chirpdetector evaluate` call, on each arm's eval_postprocess output instead
# of the two released runs. CPU only.
#
#   eval_real_labels_ablation.sh OUT NAME=EVAL_POSTPROCESS_DIR [NAME=DIR ...]
#
# -> OUT/labels (augmented audian files), OUT/NAME/ (predictions_kept.parquet + evaluation)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
out=$1; shift
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
for arg in "$@"; do
  name=${arg%%=*} dir=${arg#*=}
  mkdir -p "$out/$name"
  uv run python - "$dir/real_predictions_flagged.parquet" "$out/$name/predictions_kept.parquet" <<'PY'
import sys
import pandas as pd
from chirpdetector.postprocess import COLUMNS
flagged = pd.read_parquet(sys.argv[1])
kept = flagged[flagged["drop_reason"].astype(str) == ""]
kept = kept.drop(columns=[c for c in COLUMNS if c in kept])
kept.reset_index(drop=True).to_parquet(sys.argv[2])
print(f"{sys.argv[1]}: {len(kept)} of {len(flagged)} rows kept")
PY
  uv run chirpdetector evaluate "$out/$name/predictions_kept.parquet" "${labels[@]}" "${tracks[@]}" \
      --out "$out/$name"
done
