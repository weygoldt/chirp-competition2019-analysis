"""Tests of the SLEAP NIX reader on a tiny synthetic file with the 2022 layout.

    .venv/bin/python -m pytest experiments/video2022/test_sleap_nix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sleap_nix import instances_per_frame, node_names, read_instances, read_video_info  # noqa: E402

NODES = ["head", "center", "tail"]
FRAME = np.array([5, 3, 3, 7])  # deliberately unsorted
TRACK = np.array([1, 0, 1, 0])
POS = np.array([  # (n, 2, 3): x/y of head, center, tail
    [[10, 11, 12], [20, 21, 22]],
    [[np.nan, 31, 32], [np.nan, 41, 42]],
    [[50, 51, 52], [60, 61, 62]],
    [[70, np.nan, 72], [80, np.nan, 82]],
], dtype=np.float32)
NODE_SCORE = np.array([[0.9, 0.8, 0.7], [0, 0.6, 0.5], [1.0, 1.0, 1.0], [0.4, 0, 0.3]], dtype=np.float32)
INSTANCE_SCORE = np.array([0.5, 0.25, 0.9, 0.3], dtype=np.float32)


def _array(g: h5py.Group, name: str, data, labels: dict[int, list[str]] | None = None) -> h5py.Group:
    a = g.create_group(name)
    a.attrs["name"] = name
    a.create_dataset("data", data=data)
    dims = a.create_group("dimensions")
    for k in range(1, np.ndim(data) + 1):
        dims.create_group(str(k))
        if labels and k in labels:
            dims[str(k)].create_dataset("labels", data=np.array(labels[k], dtype=object),
                                        dtype=h5py.string_dtype())
    return a


def make_file(path: Path, video: str = "2022.05.10_0.mp4") -> Path:
    with h5py.File(path, "w") as f:
        g = f.create_group(f"data/{video}")
        g.attrs["type"] = "nix.tracking_results"
        g.attrs["name"] = video
        a = g.create_group("data_arrays")
        frame = _array(a, "frame", FRAME.astype(np.int64))
        arrays = [
            _array(a, "position", POS, {2: ["x", "y"], 3: NODES}),
            _array(a, "node score", NODE_SCORE, {2: NODES}),
            _array(a, "instance score", INSTANCE_SCORE),
            _array(a, "tracking score", np.array([0, 0.1, 0.2, 0.3], dtype=np.float32)),
            _array(a, "centroid", np.nanmean(POS, axis=2), {2: ["x", "y"]}),
            _array(a, "track", TRACK.astype(np.int64)),
        ]
        for arr in arrays:  # every array's first dimension links to the frame array
            arr["dimensions/1"].create_group("link")
            arr["dimensions/1/link/2fffa813-ea63-4286-9b7b-a2af57e518d8"] = frame
        props = g.create_group(f"sources/{video}/metadata/properties")
        for k, v in dict(fps=np.array([25.0]), frames=np.array([10]), width=np.array([2048]),
                         height=np.array([1536])).items():
            props.create_dataset(k, data=v)
        props.create_dataset("filename", data=np.array([b"/data/x.mp4"], dtype=object),
                             dtype=h5py.string_dtype())
    return path


@pytest.fixture
def nix(tmp_path: Path) -> Path:
    return make_file(tmp_path / "test.analysis.nix")


def test_video_info(nix: Path) -> None:
    info = read_video_info(nix)
    assert info.name == "2022.05.10_0.mp4"
    assert (info.fps, info.n_frames, info.width, info.height) == (25.0, 10, 2048, 1536)
    assert info.filename == "/data/x.mp4"
    assert node_names(nix) == NODES


def test_instances_sorted_and_complete(nix: Path) -> None:
    df = read_instances(nix)
    assert list(df["frame"]) == [3, 3, 5, 7]
    assert list(df["track"]) == [0, 1, 1, 0]
    assert len(df) == len(FRAME)
    expected = {"frame", "track", "instance_score", "tracking_score", "centroid_x", "centroid_y"}
    expected |= {f"{n}_{c}" for n in NODES for c in ("x", "y", "score")}
    assert set(df.columns) == expected


def test_node_values_follow_their_instance(nix: Path) -> None:
    df = read_instances(nix).set_index(["frame", "track"])
    r = df.loc[(5, 1)]  # input row 0
    assert (r["head_x"], r["head_y"], r["tail_x"], r["tail_y"]) == (10, 20, 12, 22)
    assert r["center_score"] == pytest.approx(0.8)
    assert r["instance_score"] == pytest.approx(0.5)
    assert r["centroid_x"] == pytest.approx(11.0)


def test_missing_nodes_are_nan(nix: Path) -> None:
    df = read_instances(nix).set_index(["frame", "track"])
    r = df.loc[(3, 0)]  # input row 1: head missing
    assert np.isnan(r["head_x"])
    assert np.isnan(r["head_score"])
    assert r["center_score"] == pytest.approx(0.6)
    r = df.loc[(7, 0)]  # input row 3: centre missing
    assert np.isnan(r["center_x"])
    assert np.isnan(r["center_score"])
    assert r["tail_score"] == pytest.approx(0.3)


def test_instances_per_frame(nix: Path) -> None:
    df = read_instances(nix)
    c = instances_per_frame(df["frame"].to_numpy(), read_video_info(nix).n_frames)
    assert c.tolist() == [0, 0, 0, 2, 0, 1, 0, 1, 0, 0]
    with pytest.raises(ValueError, match="outside"):
        instances_per_frame(np.array([10]), 10)


def test_wrong_layout_is_refused(tmp_path: Path) -> None:
    path = make_file(tmp_path / "bad.analysis.nix")
    with h5py.File(path, "a") as f:
        del f["data/2022.05.10_0.mp4/data_arrays/track/data"]
        f["data/2022.05.10_0.mp4/data_arrays/track"].create_dataset("data", data=np.zeros(3, np.int64))
    with pytest.raises(ValueError, match="track"):
        read_instances(path)
