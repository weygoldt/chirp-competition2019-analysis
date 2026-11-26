"""Per-label table of the breakdown benchmark for the manuscript's ablation figure.

    uv run python experiments/ablation_scenarios/export_for_paper.py

The paper module ``paper/scripts/plot_v2_frontend_ablation.py`` runs in the
paper environment (no chirpdetector), so the benchmark is handed over as one
table, written once to
``$CHIRP_DATA/runs/ablation_frontend/scenarios/paper/bench_labels.parquet``
(its sha256 printed and stored next to it). One row per chirp label of the
benchmark (both sweeps); per model, after post-processing, from the stored
evaluations (``bench/<arm>/after_tol{025,100}/items.csv``):

- ``found25_<m>``, ``score25_<m>``: matched within 25 ms (and its score),
- ``found100_<m>``, ``score100_<m>``, ``correct100_<m>``: matched within
  100 ms (the evaluation's default), and the emitter assigned correctly.

Models: ``A`` = the released detector (three windows), ``B16`` / ``B22`` = the
single-window detector (0.2048 s in all three channels) at epochs 16 / 22,
i.e. at the released detector's training budget (arm Bc; ``make_bc_bundles``).
Matches are at score >= 0.5; the paper module applies the operating
threshold as Fig. 10 does (true positives with score >= the threshold).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

ROOT = DATA / "runs/ablation_frontend/scenarios"
ARMS = {"A": "A", "B16": "Bc16", "B22": "Bc22"}
OUT = ROOT / "paper/bench_labels.parquet"


def outcomes(arm_dir: str, tol: str, rows: np.ndarray) -> pd.DataFrame:
    it = pd.read_csv(ROOT / f"bench/{arm_dir}/after_tol{tol}/items.csv")
    it = it[(it["label"] == "chirp") & it["outcome"].isin(["tp", "fn"])].set_index("label_row")
    it = it.loc[rows]
    found = (it["outcome"] == "tp").to_numpy()
    correct = found & (
        it["pred_emitter_id"].astype("Int64").fillna(-1).to_numpy()
        == it["true_emitter_id"].astype("Int64").fillna(-2).to_numpy()
    )
    return pd.DataFrame({"found": found, "score": np.where(found, it["score"].to_numpy(float), np.nan),
                         "correct": correct}, index=rows)


def main() -> None:
    lab = pd.read_parquet(ROOT / "bench/analysis_A_Bc16/labels.parquet")
    rows = lab.index.to_numpy()
    out = pd.DataFrame(
        {
            "label_row": rows,
            "tile": lab["recording"].to_numpy(),
            "scene": lab["scene"].to_numpy(),
            "sweep": lab["sweep"].to_numpy(),
            "level": lab["level"].to_numpy(),
            "level_value": lab["level_value"].to_numpy(float),
            "emitter_id": lab["emitter_id"].to_numpy(int),
            "emitter_df_hz": lab["emitter_df_hz"].to_numpy(float),
            "dt_same_s": lab["dt_same_s"].to_numpy(float),
            "family": lab["family"].astype(str).to_numpy(),
            "snr_db": lab["snr_db"].to_numpy(float),
        }
    )
    for m, arm_dir in ARMS.items():
        o25, o100 = outcomes(arm_dir, "025", rows), outcomes(arm_dir, "100", rows)
        out[f"found25_{m}"] = o25["found"].to_numpy()
        out[f"score25_{m}"] = o25["score"].to_numpy()
        out[f"found100_{m}"] = o100["found"].to_numpy()
        out[f"score100_{m}"] = o100["score"].to_numpy()
        out[f"correct100_{m}"] = o100["correct"].to_numpy()
    # consistency with the scored tables (bench_analyse.py)
    for m, arm_dir in (("A", "A"), ("B16", "Bc16")):
        assert (out[f"found25_{m}"].to_numpy() == lab[f"found_{arm_dir}_after_025"].to_numpy()).all()
        assert (out[f"correct100_{m}"].to_numpy() == lab[f"correct_{arm_dir}_after_100"].to_numpy()).all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    meta = {
        "file": OUT.name,
        "sha256": sha,
        "n_labels": len(out),
        "n_scenes": int(out["scene"].nunique()),
        "models": {
            "A": str(DATA / "runs/dualhead_r18_sim_v4_r1/final/bundle"),
            "B16": str(DATA / "runs/dualhead_r18_sim_v4_longonly_r1c/bundles/epoch_16"),
            "B22": str(DATA / "runs/dualhead_r18_sim_v4_longonly_r1c/bundles/epoch_22"),
        },
        "source": "experiments/ablation_scenarios/export_for_paper.py",
    }
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
