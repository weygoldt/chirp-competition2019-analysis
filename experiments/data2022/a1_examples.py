"""Example cut-outs of the expert's 2022 (and 2019/2020) decisions, by score bin and fish.

Contact sheets (see ``cutouts.py`` for the panels) of labelled real_v1
candidates and of unlabelled detections, each sheet one group:

- ``2022_winner_not``: 2022, scores 0.5-0.9, assigned to the trial's winner, judged not a chirp;
- ``2022_loser_not``: 2022, 0.5-0.9, assigned to the loser, judged not a chirp;
- ``2022_other_not``: 2022, 0.5-0.9, unassigned or other track, judged not a chirp;
- ``2022_chirps``: 2022, >= 0.5, judged a chirp (for comparison);
- ``2022_low_not``: 2022, 0.05-0.5, judged not a chirp;
- ``2019_not`` / ``2019_chirps``: the same for 2019/2020, 0.5-0.9;
- ``2022-06-20_loser_unlabelled``: random kept detections of the 2022-06-20 loser
  (track 12600) at 0.7-0.9 and >= 0.9, the recording that dominates the 2022
  0.7-1.0 bins.

Items are drawn at random (seeded) within the group and sorted by score.

    .venv/bin/python experiments/data2022/a1_examples.py

Output: ``$CHIRP_DATA/data2022/a1/examples/<group>.png`` (+ ``<group>.csv``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutouts as co
from paths import DATA

SCORED = DATA / "labels" / "real_v1" / "report" / "candidates_scored.csv"
RUN_2022 = DATA / "detections" / "competition2022_v2_t001"
OUT = DATA / "data2022" / "a1" / "examples"
N = 12
SEED = 20261119


def labelled() -> pd.DataFrame:
    t = pd.read_csv(SCORED)
    t = t[t["labelled"] & t["stratum"].str.startswith("main/")]
    return t.drop_duplicates("candidate_id")


def pick(g: pd.DataFrame, n: int = N) -> pd.DataFrame:
    if len(g) > n:
        g = g.sample(n, random_state=SEED)
    return g.sort_values("score")


def groups() -> dict[str, tuple[pd.DataFrame, str]]:
    t = labelled()
    d22 = t[t["dataset"] == "competition2022"]
    d19 = t[t["dataset"] == "competition2019"]
    mid = (t["score"] >= 0.5) & (t["score"] < 0.9)
    not_chirp = t["label"] == "not_chirp"
    out = {}
    for status in ("winner", "loser", "other"):
        g = d22[mid.loc[d22.index] & not_chirp.loc[d22.index] & (d22["status"] == status)]
        out[f"2022_{status}_not"] = (pick(g), f"2022, scores 0.5-0.9, assigned to the {status}: "
                                     f"judged NOT a chirp ({len(g)} labelled; {min(N, len(g))} shown)")
    g = d22[(d22["score"] >= 0.5) & ~not_chirp.loc[d22.index]]
    out["2022_chirps"] = (pick(g), f"2022, scores >= 0.5: judged a chirp ({len(g)} labelled; {min(N, len(g))} shown)")
    g = d22[(d22["score"] >= 0.05) & (d22["score"] < 0.5) & not_chirp.loc[d22.index]]
    out["2022_low_not"] = (pick(g), f"2022, scores 0.05-0.5: judged NOT a chirp ({len(g)} labelled; "
                           f"{min(N, len(g))} shown)")
    g = d19[mid.loc[d19.index] & not_chirp.loc[d19.index]]
    out["2019_not"] = (pick(g), f"2019/2020, scores 0.5-0.9: judged NOT a chirp ({len(g)} labelled; "
                       f"{min(N, len(g))} shown)")
    g = d19[mid.loc[d19.index] & ~not_chirp.loc[d19.index]]
    out["2019_chirps"] = (pick(g), f"2019/2020, scores 0.5-0.9: judged a chirp ({len(g)} labelled; "
                          f"{min(N, len(g))} shown)")
    d = pd.read_parquet(RUN_2022 / "2022-06-20-10_00" / "chirps_rtdetr.parquet")
    d = d[d["kept"] & (d["t_center_s"] < 10_800) & (d["track_id"] == 12600)]
    a = d[(d["score"] >= 0.7) & (d["score"] < 0.9)].sample(N // 2, random_state=SEED)
    b = d[d["score"] >= 0.9].sample(N // 2, random_state=SEED)
    out["2022-06-20_loser_unlabelled"] = (
        pd.concat([a, b]).sort_values("score"),
        f"2022-06-20, loser (track 12600), unlabelled: {N // 2} random detections at 0.7-0.9 "
        f"({len(d[(d['score'] >= 0.7) & (d['score'] < 0.9)])} in the dark phase) and {N // 2} at >= 0.9 "
        f"({len(d[d['score'] >= 0.9])})")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (rows, title) in groups().items():
        rows.to_csv(OUT / f"{name}.csv", index=False)
        co.contact_sheet(rows.reset_index(drop=True), OUT / f"{name}.png",
                         title + "\nupper panel: 5 s, 204.8 ms window (the labeller's view); lower: 0.8 s, "
                         "51.2 ms window; white dashed: box; cyan tick: predicted emitter EODf; blue dashed: tracks")
        print(f"wrote {OUT / name}.png", flush=True)


if __name__ == "__main__":
    main()
