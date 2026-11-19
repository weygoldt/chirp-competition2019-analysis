"""Gap-fill the 2022 trial fish's tracks and redo assignment, as for 2019/2020 (author's request, 2026-11-19).

Runs ``experiments/production/gapfill_reassign.py`` unchanged on the 2022
detection run ``competition2022_v2_t001`` (all 19 recordings; store threshold
0.01): the same rule (interior gaps > 10 s and <= 300 s with an EODf jump <= 15 Hz
of the two trial fish, filled by linear interpolation), the same equivalence
test first (the unpatched tracks must reproduce the stored tables exactly), the
same outputs. Only the trial table differs: 2022 has ``raw/meta.csv``
(``Win_ID``, ``Lose_ID``; no draws) instead of 2019's ``trial_summary.csv``;
2022-07-07-10_00 is not a trial (not in meta.csv) and is left as it is.

    .venv/bin/python experiments/data2022/gapfill_2022.py [--check-only]

Output: ``$CHIRP_DATA/detections/competition2022_v2_t001_gapfill/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
import gapfill_reassign as gr  # noqa: E402
from paths import DATA, REC_2022  # noqa: E402

SOURCE = DATA / "detections" / "competition2022_v2_t001"
OUT = DATA / "detections" / "competition2022_v2_t001_gapfill"


def load_trials_2022(_data: Path) -> pd.DataFrame:
    """2022 trial table in the layout gapfill_reassign expects (win_ID, lose_ID, draw, analysed)."""
    m = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"})
    return pd.DataFrame({"recording": m["recording"], "win_ID": m["Win_ID"].astype(float),
                         "lose_ID": m["Lose_ID"].astype(float), "draw": 0, "analysed": True}).set_index("recording")


if __name__ == "__main__":
    gr.load_trials = load_trials_2022
    extra = [a for a in sys.argv[1:] if a in ("--check-only", "--overwrite")]
    sys.argv = [sys.argv[0], "--source", str(SOURCE), "--out", str(OUT), "--data", str(REC_2022 / "raw"), *extra]
    sys.exit(gr.main())
