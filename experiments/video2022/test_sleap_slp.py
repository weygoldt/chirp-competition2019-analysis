"""Tests of sleap_slp.py on a tiny synthetic SLEAP labels file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleap_slp  # noqa: E402


def write_slp(path: Path) -> None:
    nodes = ["spine1", "snout", "tail"]  # metadata order differs from skeleton order on purpose
    meta = {"nodes": [{"name": n} for n in nodes],
            "skeletons": [{"nodes": [{"id": 1}, {"id": 0}, {"id": 2}], "links": []}]}
    frames = np.array([(0, 1, 500, 0, 3)], dtype=[("frame_id", "u8"), ("video", "u4"), ("frame_idx", "u8"),
                                                 ("instance_id_start", "u8"), ("instance_id_end", "u8")])
    it = [("instance_id", "i8"), ("instance_type", "u1"), ("frame_id", "u8"), ("skeleton", "u4"), ("track", "i4"),
          ("from_predicted", "i8"), ("score", "f4"), ("point_id_start", "u8"), ("point_id_end", "u8"),
          ("tracking_score", "f4")]
    inst = np.array([(0, 0, 0, 0, -1, -1, np.nan, 0, 3, np.nan),   # user
                     (1, 1, 0, 0, -1, -1, 0.9, 0, 3, np.nan),      # predicted: ignored
                     (2, 0, 0, 0, -1, -1, np.nan, 3, 6, np.nan)], dtype=it)
    pts = np.array([(10, 20, True, True), (12, 21, True, True), (90, 25, True, True),
                    (300, 300, True, True), (0, 0, False, False), (400, 310, True, True)],
                   dtype=[("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")])
    with h5py.File(path, "w") as f:
        f.create_dataset("metadata", data=np.zeros(1))
        f["metadata"].attrs["json"] = json.dumps(meta)
        vids = [json.dumps({"backend": {"filename": n}}) for n in ("a.mp4", "2022.05.10_0.mp4")]
        f.create_dataset("videos_json", data=np.array(vids, dtype=object), dtype=h5py.string_dtype())
        f.create_dataset("frames", data=frames)
        f.create_dataset("instances", data=inst)
        f.create_dataset("points", data=pts)


def test_reads_user_instances_in_skeleton_order(tmp_path: Path) -> None:
    p = tmp_path / "l.slp"
    write_slp(p)
    df, order = sleap_slp.read_user_instances(p)
    assert order == ["snout", "spine1", "tail"]
    assert list(df["instance_id"]) == [0, 2]
    assert df.loc[0, "video_file"] == "2022.05.10_0.mp4" and df.loc[0, "frame_idx"] == 500
    assert (df.loc[0, "snout_x"], df.loc[0, "spine1_x"], df.loc[0, "tail_x"]) == (10.0, 12.0, 90.0)
    assert np.isnan(df.loc[1, "spine1_x"]) and df.loc[1, "tail_y"] == 310.0


def test_duplicates_keeps_the_more_complete_instance() -> None:
    a = np.array([[0, 0], [10, 0], [20, 0.0]])
    b = np.array([[1, 1], [11, 1], [np.nan, np.nan]])  # same fish, fewer nodes
    c = np.array([[200, 0], [210, 0], [220, 0.0]])     # another fish
    d = np.array([[np.nan, np.nan], [np.nan, np.nan], [21, 0.0]])  # one shared node only: not a duplicate
    assert sleap_slp.duplicates([a, b, c, d], max_px=5) == [1]
