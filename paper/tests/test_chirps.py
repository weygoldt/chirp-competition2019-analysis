"""The loader's two views of the v2 detections agree, on every recording."""

from __future__ import annotations

import numpy as np
import pytest

from chirps import load_chirp_table, load_chirps
from paper_config import DATA_ROOT

RECORDINGS = sorted(p.parent for p in DATA_ROOT.glob("*/chirps_rtdetr.parquet"))


def test_recordings_found():
    assert len(RECORDINGS) == 50


@pytest.mark.parametrize("rec", RECORDINGS, ids=lambda p: p.name)
def test_parquet_keep_equals_legacy_arrays(rec):
    times, ids = load_chirps(rec, "rtdetr")
    df = load_chirp_table(rec)
    kept = df[df["keep"]].sort_values("t_center_s", kind="stable")
    np.testing.assert_array_equal(kept["t_center_s"].to_numpy(), times)
    np.testing.assert_array_equal(kept["track_id"].to_numpy(), ids)


def test_legacy_ids_are_trial_fish():
    """Every legacy id of a trial is its winner or loser (handoff check, 43 trials)."""
    import pandas as pd

    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv")
    n = 0
    for _, row in ts.iterrows():
        _, ids = load_chirps(DATA_ROOT / row["recording"], "rtdetr")
        assert set(np.unique(ids)) <= {row["win_ID"], row["lose_ID"]}, row["recording"]
        n += len(ids)
    assert n == 91_931
