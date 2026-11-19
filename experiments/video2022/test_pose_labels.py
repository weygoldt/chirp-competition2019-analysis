"""Tests of pose_labels.py: round trip of the label store and the YOLO conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_labels as pl  # noqa: E402


def test_store_round_trip_and_latest_wins(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    pl.append_label(path, pl.FrameLabel("a", "draft", [pl.Fish(head=(1, 2))]))
    pl.append_label(path, pl.FrameLabel("b", "unusable"))
    pl.append_label(path, pl.FrameLabel("a", "done", [pl.Fish(head=(10, 20), tail=(30, 40)), pl.Fish(center=(5, 5))],
                                        elapsed_s=3.5))
    got = pl.load_labels(path)
    assert set(got) == {"a", "b"}
    assert got["a"].status == "done"
    assert got["a"].fish[0] == pl.Fish(head=(10.0, 20.0), tail=(30.0, 40.0))
    assert got["a"].fish[1].n_points() == 1
    assert got["a"].elapsed_s == 3.5
    assert got["b"].fish == []


def test_unknown_status_and_keypoint_are_refused() -> None:
    with pytest.raises(ValueError):
        pl.FrameLabel("a", "maybe")
    with pytest.raises(ValueError):
        pl.Fish.from_dict({"snout": [1, 2]})


def test_bbox_pads_and_clips() -> None:
    f = pl.Fish(head=(5, 10), center=(50, 30), tail=(100, 12))
    assert pl.bbox_from_keypoints(f, 10, 104, 200) == (0.0, 0.0, 104.0, 40.0)


def test_yolo_lines_normalise_and_mark_missing_keypoints() -> None:
    f = pl.Fish(head=(100, 50), tail=(300, 150))
    (line,) = pl.yolo_lines([f, pl.Fish()], width=400, height=200, pad_px=0)
    v = line.split()
    assert v[0] == "0"
    assert [float(x) for x in v[1:5]] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert [float(x) for x in v[5:8]] == pytest.approx([0.25, 0.25, 2])
    assert v[8:11] == ["0.000000", "0.000000", "0"]
    assert [float(x) for x in v[11:14]] == pytest.approx([0.75, 0.75, 2])
