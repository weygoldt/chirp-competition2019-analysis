"""Check the t001 rerun against the production run (2019/2020 recordings).

Same model and config apart from the store threshold (0.01 instead of
0.05), so the rows with score >= 0.05 must be the production rows: same
boxes, scores, emitter, assignment and post-processing flags (post-
processing references by chirp_id are compared through the referenced
box). The decoder's columns are compared separately (it runs batched on
the GPU over all stored boxes, so a different set of boxes may change its
outputs by float rounding).

    uv run python experiments/real_labels/compare_production.py

Writes ``$CHIRP_DATA/detections/competition2019_v2_t001/compare_production.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRODUCTION_2019, RUN_2019  # noqa: E402

EXACT = ["t_start_s", "t_end_s", "t_center_s", "f_low_hz", "f_high_hz", "score", "operating",
         "kept", "chirp", "emitter_hz", "track_id", "track_hz", "assign_status",
         "assign_confidence", "nearest_track_id", "second_track_id", "n_live_tracks",
         "pp_drop_reason", "pp_harmonic_of_track", "pp_reattributed"]
REFS = ["pp_duplicate_of", "pp_harmonic_pair_of"]


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
        return np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=0, equal_nan=True)
    return a.astype(str).to_numpy() == b.astype(str).to_numpy()


def main() -> None:
    rows = []
    for rec_dir in sorted(p for p in PRODUCTION_2019.iterdir() if (p / "chirps_rtdetr.parquet").exists()):
        rec = rec_dir.name
        a = pd.read_parquet(rec_dir / "chirps_rtdetr.parquet")
        full = pd.read_parquet(RUN_2019 / rec / "chirps_rtdetr.parquet")
        b = full[full["score"] >= 0.05].reset_index(drop=True)
        row = {"recording": rec, "production_rows": len(a), "t001_rows": len(full),
               "t001_rows_ge_0.05": len(b)}
        if len(a) == len(b):
            row["exact_columns_differing_rows"] = int(sum((~_same(a[c], b[c])).sum() for c in EXACT))
            for c in REFS:
                ta = pd.to_numeric(a[c].map(a.set_index("chirp_id")["t_center_s"]), errors="coerce")
                tb = pd.to_numeric(b[c].map(full.set_index("chirp_id")["t_center_s"]), errors="coerce")
                ta, tb = ta.astype(float), tb.astype(float)
                row[f"{c}_differing"] = int((~_same(ta, tb)).sum())
            row["height_max_abs_diff_hz"] = float(np.nanmax(np.abs(a["height_hz"] - b["height_hz"])))
            row["fwhm_max_abs_diff_ms"] = 1e3 * float(np.nanmax(np.abs(a["fwhm_s"] - b["fwhm_s"])))
            row["decoder_median_rel_diff_height"] = float(np.nanmedian(np.abs(a["height_hz"] - b["height_hz"])
                                                                       / a["height_hz"].abs()))
            row["params_flag_differing"] = int((~_same(a["params_flag"], b["params_flag"])).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    out = RUN_2019 / "compare_production.csv"
    df.to_csv(out, index=False)
    same_n = (df["production_rows"] == df["t001_rows_ge_0.05"]).sum()
    exact = (df.get("exact_columns_differing_rows", pd.Series(dtype=float)) == 0).sum()
    refs_ok = int((df[[f"{c}_differing" for c in REFS]].fillna(1).sum(axis=1) == 0).sum())
    print(f"{len(df)} recordings: row count equal in {same_n}; exact columns identical in {exact}; "
          f"chirp_id references identical in {refs_ok}")
    big = df.sort_values("height_max_abs_diff_hz", ascending=False).head(3)
    print("largest decoder height differences:", big[["recording", "height_max_abs_diff_hz"]].to_dict("records"))
    print(f"decoder: max |height diff| {df['height_max_abs_diff_hz'].max():.3g} Hz, max |FWHM diff| "
          f"{df['fwhm_max_abs_diff_ms'].max():.3g} ms, median relative "
          f"height difference (median over recordings) {df['decoder_median_rel_diff_height'].median():.2g}; "
          f"params_flag differs in {int(df['params_flag_differing'].sum())} of {int(df['production_rows'].sum())} rows")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
