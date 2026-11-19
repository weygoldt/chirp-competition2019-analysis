"""Tests of pose_labels.py: the 7-point skeleton, legacy 3-point labels, the store and the YOLO conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_labels as pl  # noqa: E402


def test_store_round_trip_and_latest_wins(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    pl.append_label(path, pl.FrameLabel("a", "draft", [pl.Fish({"snout": (1, 2)})]))
    pl.append_label(path, pl.FrameLabel("b", "unusable"))
    pl.append_label(path, pl.FrameLabel("a", "done", [pl.Fish({"snout": (10, 20), "tail": (30, 40)}, "new"),
                                                      pl.Fish({"left_fin": (5, 5)})], elapsed_s=3.5))
    got = pl.load_labels(path)
    assert set(got) == {"a", "b"}
    assert got["a"].status == "done"
    assert got["a"].fish[0].points == {"snout": (10.0, 20.0), "tail": (30.0, 40.0)}
    assert got["a"].fish[0].origin == "new"
    assert got["a"].fish[1].n_points() == 1
    assert got["b"].fish == []


def test_legacy_three_point_labels_map_to_the_skeleton() -> None:
    f = pl.Fish.from_dict({"head": [1, 2], "center": None, "tail": [3, 4], "origin": "prefill"})
    assert f.points == {"snout": (1.0, 2.0), "tail": (3.0, 4.0)}
    assert f.origin == "prefill"


def test_unknown_status_and_keypoint_are_refused() -> None:
    with pytest.raises(ValueError):
        pl.FrameLabel("a", "maybe")
    with pytest.raises(ValueError):
        pl.Fish.from_dict({"nose": [1, 2]})
    with pytest.raises(ValueError):
        pl.Fish({"nose": (1, 2)})


def test_flip_idx_swaps_the_fins_only() -> None:
    sk = list(pl.SKELETON)
    assert [sk[i] for i in pl.FLIP_IDX] == ["snout", "spine1", "spine2", "spine3", "tail", "right_fin", "left_fin"]


def test_bbox_pads_and_clips() -> None:
    f = pl.Fish({"snout": (5, 10), "spine1": (50, 30), "tail": (100, 12)})
    assert pl.bbox_from_points(f, 10, 104, 200) == (0.0, 0.0, 104.0, 40.0)


def test_yolo_lines_normalise_and_mark_missing_keypoints() -> None:
    f = pl.Fish({"snout": (100, 50), "tail": (300, 150)})
    (line,) = pl.yolo_lines([f, pl.Fish()], width=400, height=200, pad_px=0)
    v = line.split()
    assert v[0] == "0"
    assert [float(x) for x in v[1:5]] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert len(v) == 5 + 3 * len(pl.SKELETON)
    assert [float(x) for x in v[5:8]] == pytest.approx([0.25, 0.25, 2])  # snout
    assert v[8:11] == ["0.000000", "0.000000", "0"]  # spine1 not labelled
    assert [float(x) for x in v[17:20]] == pytest.approx([0.75, 0.75, 2])  # tail (5th point)


def test_dataset_yaml_names_the_skeleton() -> None:
    y = pl.dataset_yaml(Path("/d"), {"train": "images/train", "val": ["a", "b"]})
    assert "kpt_shape: [7, 3]" in y and "flip_idx: [0, 1, 2, 3, 4, 6, 5]" in y
    assert 'val: ["a", "b"]' in y
