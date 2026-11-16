"""Wavetracker copies of the gap-filled trial-fish tracks, for the labeller.

The gap-filled dataset patch (competition2019_v2_20261113_gapfill, author's
decision of 2026-11-14) stores the two trial fish's patched tracks of each
trial in ``<rec>/tracks_gapfill.npz``; the labeller reads tracks only from
wavetracker directories. This writes, per trial recording,
``$CHIRP_DATA/labels/real_v1/tracks_gapfill/<rec>/{times,fund_v,ident_v,idx_v}.npy``:
the original arrays with the trial fish's rows replaced by the patched rows
(other identities and unassigned detections unchanged). Checks: ``times``
equal to the original; every unfilled patched row equals an original row.
The raw data (/mnt/data2) are only read.

    uv run python experiments/real_labels/make_gapfill_tracks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GAPFILL_2019, REC_2019, TRACKS_GAPFILL  # noqa: E402


def main() -> None:
    TRACKS_GAPFILL.mkdir(parents=True, exist_ok=True)
    report = {}
    for npz in sorted(GAPFILL_2019.glob("*/tracks_gapfill.npz")):
        rec = npz.parent.name
        z = np.load(npz)
        src = REC_2019 / rec
        orig = {k: np.load(src / f"{k}.npy") for k in ("times", "fund_v", "ident_v", "idx_v")}
        assert np.array_equal(z["times"], orig["times"]), rec
        fish = set(z["fish_id"].tolist())
        other = ~np.isin(orig["ident_v"], list(fish))  # NaN (unassigned) is kept too
        # unfilled patched rows must be original rows of the same fish
        unf = ~z["filled"]
        key_o = set(zip(orig["idx_v"][~other].tolist(), orig["ident_v"][~other].tolist(),
                        orig["fund_v"][~other].tolist(), strict=True))
        key_p = set(zip(z["idx_v"][unf].tolist(), z["ident_v"][unf].tolist(),
                        z["fund_v"][unf].tolist(), strict=True))
        assert key_p <= key_o, f"{rec}: unfilled patched rows not in the original"
        fund = np.concatenate([orig["fund_v"][other], z["fund_v"]])
        ident = np.concatenate([orig["ident_v"][other], z["ident_v"]])
        idx = np.concatenate([orig["idx_v"][other], z["idx_v"]]).astype(orig["idx_v"].dtype)
        order = np.lexsort((np.nan_to_num(ident, nan=-1.0), idx))
        out = TRACKS_GAPFILL / rec
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "times.npy", orig["times"])
        np.save(out / "fund_v.npy", fund[order])
        np.save(out / "ident_v.npy", ident[order])
        np.save(out / "idx_v.npy", idx[order])
        report[rec] = {
            "fish": sorted(fish),
            "rows_original": int(len(orig["fund_v"])),
            "rows_written": int(len(fund)),
            "filled_rows": int(z["filled"].sum()),
            "dropped_original_fish_rows": int(len(key_o) - len(key_p)),
        }
    (TRACKS_GAPFILL / "manifest.json").write_text(json.dumps(report, indent=1))
    (TRACKS_GAPFILL / "README.md").write_text(
        "# Gap-filled trial tracks for the real_v1 labelling session\n\n"
        "Written by `experiments/real_labels/make_gapfill_tracks.py` (chirp_competition_upgrade,\n"
        "branch v2-real-labels) from `competition2019_v2_20261113_gapfill/<rec>/tracks_gapfill.npz`\n"
        "and the original arrays in `/mnt/data2/2019_tube_competition/<rec>/`: the original\n"
        "wavetracker arrays with the two trial fish's rows replaced by the patched rows (linear\n"
        "interpolation across gaps <= 300 s with an EODf jump <= 15 Hz). No `sign_v`.\n"
        "`manifest.json`: rows per recording. Used only for the labeller's track display and\n"
        "click snapping (the model's assignments are read from the runs).\n"
    )
    print(f"{len(report)} recordings -> {TRACKS_GAPFILL}")
    for rec, r in list(report.items())[:3]:
        print(rec, r)


if __name__ == "__main__":
    main()
