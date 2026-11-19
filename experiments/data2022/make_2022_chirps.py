"""The 2022 chirp dataset for analysis: threshold 0.8 with a per-fish correction (author's decision, 2026-11-19).

Selection, as in the 2019/2020 runs at 0.8: kept detections (not flagged by
post-processing) with score >= 0.8, assigned (or re-attributed) to a track, from
the gap-filled 2022 run ``competition2022_v2_t001_gapfill`` (``gapfill_2022.py``;
the trial fish's track gaps filled as for 2019/2020, author's request). The
decoder's chirp parameters (``height_hz``, ``fwhm_s``, ``width10_s``,
``undershoot_hz``, calibrated, with ``params_ok`` / ``params_flag``) are already in
the table: the production pipeline decodes every stored box.

Correction: every selected detection gets ``p_chirp``, the share of detections
the expert judged chirps in its stratum of the 2022 labelling session
real_v2_2022 (``labels/real_v2_2022/report/agreement.csv``, rows ``by =
group+status``): group (pair 3 = 2022-06-20/22/24, rest) x the assigned fish's
trial status (winner / loser / other, from ``raw/meta.csv``) x score bin
(0.8-0.9, 0.9-1), with its 95 % bootstrap interval (``p_chirp_low``,
``p_chirp_high``). The labels cover the dark phase (first 3 h); detections after
it get the same values (``p_extrapolated = True``).

"Per fish" is per fish role and group: the session has 15 labels per cell,
too few for individual fish. The expected number of chirps of a fish is the
sum of ``p_chirp`` over its detections; its bounds are the sums of the bounds
(conservative).

    .venv/bin/python experiments/data2022/make_2022_chirps.py

Output ``$CHIRP_DATA/detections/competition2022_v2_gapfill_t080_corrected/``:

- per recording: ``chirps_rtdetr_t080.parquet`` (the selected rows with
  ``status, group, p_chirp, p_chirp_low, p_chirp_high, p_extrapolated``) and the
  legacy arrays ``chirp_times_rtdetr_t080.npy`` (box centre, s),
  ``chirp_ids_rtdetr_t080.npy`` (track id), ``chirp_weights_rtdetr_t080.npy``
  (``p_chirp``, aligned with them);
- ``trial_counts.csv``: per trial and fish, detections and expected chirps in the dark
  phase and the whole recording;
- ``meta.json`` (inputs with sha256, threshold, strata table), ``README.md``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2022  # noqa: E402

SRC = DATA / "detections" / "competition2022_v2_t001_gapfill"
TABLE = "chirps_rtdetr_gapfill.parquet"
AGREEMENT = DATA / "labels" / "real_v2_2022" / "report" / "agreement.csv"
OUT = DATA / "detections" / "competition2022_v2_gapfill_t080_corrected"
THRESHOLD = 0.8
PAIR3 = {"2022-06-20-10_00", "2022-06-22-10_00", "2022-06-24-10_00"}
DARK_S = 10_800.0
BINS = {"0.8-0.9": (0.8, 0.9), "0.9-1": (0.9, 1.0001)}


def markdown(df: pd.DataFrame) -> str:
    """A pipe table (pandas' to_markdown needs tabulate, which is not installed)."""
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strata() -> pd.DataFrame:
    a = pd.read_csv(AGREEMENT)
    a = a[(a["by"] == "group+status") & a["scores"].isin(BINS)]
    return a[["group", "status", "scores", "n", "agreement", "ci_low", "ci_high"]].rename(
        columns={"scores": "score_bin", "agreement": "p_chirp", "ci_low": "p_chirp_low", "ci_high": "p_chirp_high"})


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it to rebuild")
    st = strata()
    if len(st) != 2 * 3 * len(BINS):
        raise SystemExit(f"expected 12 strata in {AGREEMENT}, found {len(st)}")
    meta = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"}).set_index("recording")
    counts = []
    for rd in sorted(p for p in SRC.glob("2022-*") if p.is_dir()):
        name = rd.name
        d = pd.read_parquet(rd / TABLE)
        d = d[d["kept"] & (d["score"] >= THRESHOLD) & d["assign_status"].isin(["assigned", "reattributed"])].copy()
        fish = {}
        if name in meta.index:
            fish = {float(meta.loc[name, "Win_ID"]): "winner", float(meta.loc[name, "Lose_ID"]): "loser"}
        d["status"] = [fish.get(float(t), "other") for t in d["track_id"]]
        d["group"] = "pair3" if name in PAIR3 else "rest"
        d["score_bin"] = np.where(d["score"] >= 0.9, "0.9-1", "0.8-0.9")
        d = d.merge(st.drop(columns="n"), on=["group", "status", "score_bin"], how="left", validate="many_to_one")
        if d["p_chirp"].isna().any():
            raise SystemExit(f"{name}: detections without a stratum")
        d["p_extrapolated"] = d["t_center_s"] >= DARK_S
        d = d.sort_values("t_center_s").reset_index(drop=True)
        (OUT / name).mkdir(parents=True)
        d.to_parquet(OUT / name / "chirps_rtdetr_t080.parquet", index=False)
        np.save(OUT / name / "chirp_times_rtdetr_t080.npy", d["t_center_s"].to_numpy(np.float64))
        np.save(OUT / name / "chirp_ids_rtdetr_t080.npy", d["track_id"].to_numpy(np.float64))
        np.save(OUT / name / "chirp_weights_rtdetr_t080.npy", d["p_chirp"].to_numpy(np.float64))
        for status in ("winner", "loser"):
            for span, m in (("dark", d["t_center_s"] < DARK_S), ("whole", np.ones(len(d), bool))):
                g = d[m & (d["status"] == status)]
                counts.append({"recording": name, "pair": meta.loc[name, "Pairing"] if name in meta.index else "",
                               "group": d["group"].iloc[0] if len(d) else ("pair3" if name in PAIR3 else "rest"),
                               "status": status, "span": span, "detections": len(g),
                               "expected_chirps": float(g["p_chirp"].sum()),
                               "expected_low": float(g["p_chirp_low"].sum()),
                               "expected_high": float(g["p_chirp_high"].sum())})
        print(f"{name}: {len(d)} detections >= {THRESHOLD}", flush=True)
    c = pd.DataFrame(counts)
    c.to_csv(OUT / "trial_counts.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent).stdout.strip()
    (OUT / "meta.json").write_text(json.dumps({
        "decision": "author, 2026-11-19: 2022 uses threshold 0.8 with a per-fish correction",
        "threshold": THRESHOLD, "selection": "kept, score >= 0.8, assign_status in (assigned, reattributed)",
        "source_run": str(SRC), "agreement_table": {"path": str(AGREEMENT), "sha256": sha256(AGREEMENT)},
        "meta_csv": {"path": str(REC_2022 / "raw" / "meta.csv"), "sha256": sha256(REC_2022 / "raw" / "meta.csv")},
        "strata": st.to_dict(orient="records"), "script": str(Path(__file__).resolve()), "git_commit": commit},
        indent=1))
    s = c[c["span"] == "dark"].groupby(["group", "status"])[["detections", "expected_chirps", "expected_low",
                                                              "expected_high"]].sum()
    (OUT / "README.md").write_text(
        "# 2022 chirps at threshold 0.8 with the per-fish correction\n\n"
        f"Made by `experiments/data2022/make_2022_chirps.py` (commit {commit[:8]}); see its docstring.\n"
        "Use `p_chirp` (or `chirp_weights_rtdetr_t080.npy`) as each detection's probability of being a chirp,\n"
        "from the expert's 2022 labels (real_v2_2022); expected chirps of a fish = the sum of its weights.\n\n"
        "## Strata (agreement, 95 % interval, labelled items)\n\n" + markdown(st.round(3)) +
        "\n\n## Dark phase, summed over trials\n\n" + markdown(s.round(1)) + "\n")
    print(st.round(3).to_string(index=False))
    print(s.round(1).to_string())


if __name__ == "__main__":
    main()
