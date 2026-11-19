"""Read the SLEAP video tracks of the 2022 competitions (NIX/HDF5 export).

The 2022 videos were tracked with SLEAP (top-down: a centroid model finds each
fish, a centred-instance model places the keypoints) and exported with SLEAP's
``NixAdaptor`` to ``interrim/<rec>/<project>_<video>.analysis.nix``. The file is
plain HDF5 in the ``nix.tracking`` layout; this module reads it with ``h5py``
(no ``nixio`` needed).

Layout (one block per video under ``data/``; ``n`` = number of instances, one
instance = one detected fish in one frame)::

    data/<video>/data_arrays/
        frame/data            (n,)       video frame of each instance
        position/data         (n, 2, K)  x/y (px) of K nodes; NaN = node not found
        node score/data       (n, K)     confidence of each node; 0 = not found
        instance score/data   (n,)       centroid-model confidence (>= 0.2)
        tracking score/data   (n,)       SLEAP tracker's matching score
        centroid/data         (n, 2)     mean of the found nodes (px)
        track/data            (n,)       SLEAP track id (index of the track map)
    data/<video>/data_frames/track map/data   (name, index) of every track
    data/<video>/sources/<video>/metadata/properties/{fps,frames,width,height,filename}

The frame index of each instance is also the target of every data array's
first dimension (``<array>/dimensions/1/link/<uuid>/data``); the reader uses
that link and checks that all arrays have the same length. Node names come
from the ``position`` array's third dimension (``head``, ``center``, ``tail``
in all 2022 files). The ``index`` column of the track map is int8 in the file
and overflows, so it is not used; ``track`` values are ids, not names.

Instance score: for SLEAP top-down models it is the centroid peak value (the
lower bound 0.2 is the centroid model's peak threshold); the node scores are
the centred-instance model's peak values (can exceed 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VideoInfo:
    """Metadata of the tracked video, as stored in the export."""

    name: str  # block name, the video file name SLEAP saw
    filename: str  # path of the video on the tracking machine
    fps: float
    n_frames: int
    width: int
    height: int


def _video_group(f: h5py.File) -> h5py.Group:
    """The one tracking block under ``data/`` (raises if there is not exactly one)."""
    groups = [g for g in f["data"].values() if g.attrs.get("type") == "nix.tracking_results"]
    if len(groups) != 1:
        raise ValueError(f"{f.filename}: expected one nix.tracking_results block, found {len(groups)}")
    return groups[0]


def _labels(dim: h5py.Group) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in dim["labels"][:]]


def _frame_index(arrays: h5py.Group) -> np.ndarray:
    """Frame of each instance, read through the dimension link of ``position``."""
    link = arrays["position/dimensions/1/link"]
    keys = list(link.keys())
    if len(keys) != 1:
        raise ValueError(f"position/dimensions/1/link holds {len(keys)} targets, expected 1")
    return link[keys[0]]["data"][:].astype(np.int64)


def read_video_info(path: str | Path) -> VideoInfo:
    with h5py.File(path, "r") as f:
        g = _video_group(f)
        props = next(iter(g["sources"].values()))["metadata/properties"]

        def get(key):
            v = props[key][0]
            return v.decode() if isinstance(v, bytes) else v

        return VideoInfo(
            name=g.attrs["name"] if isinstance(g.attrs["name"], str) else g.attrs["name"].decode(),
            filename=str(get("filename")),
            fps=float(get("fps")),
            n_frames=int(get("frames")),
            width=int(get("width")),
            height=int(get("height")),
        )


def read_instances(path: str | Path) -> pd.DataFrame:
    """One row per instance (detected fish in one frame), sorted by frame.

    Columns: ``frame``, ``track``, ``instance_score``, ``tracking_score``,
    ``centroid_x``, ``centroid_y`` and, for each node (``head``, ``center``,
    ``tail``): ``<node>_x``, ``<node>_y`` (px, NaN if the node was not found)
    and ``<node>_score`` (NaN if not found). Pixel coordinates are in the
    full video frame (2048 x 1536 in 2022), origin top left.
    """
    with h5py.File(path, "r") as f:
        a = _video_group(f)["data_arrays"]
        frame = _frame_index(a)
        nodes = _labels(a["position/dimensions/3"])
        axes = _labels(a["position/dimensions/2"])
        if axes != ["x", "y"]:
            raise ValueError(f"{path}: position axes {axes}, expected ['x', 'y']")
        if _labels(a["node score/dimensions/2"]) != nodes:
            raise ValueError(f"{path}: node score and position name different nodes")
        pos = a["position/data"][:]
        node_score = a["node score/data"][:].astype(float)
        cols = {
            "frame": frame,
            "track": a["track/data"][:].astype(np.int64),
            "instance_score": a["instance score/data"][:].astype(float),
            "tracking_score": a["tracking score/data"][:].astype(float),
            "centroid_x": a["centroid/data"][:, 0].astype(float),
            "centroid_y": a["centroid/data"][:, 1].astype(float),
        }
    n = len(frame)
    for name, v in cols.items():
        if len(v) != n:
            raise ValueError(f"{path}: array {name} has {len(v)} rows, frame has {n}")
    if pos.shape != (n, 2, len(nodes)) or node_score.shape != (n, len(nodes)):
        raise ValueError(f"{path}: position {pos.shape} / node score {node_score.shape} do not match {n} instances")
    for k, node in enumerate(nodes):
        found = np.isfinite(pos[:, 0, k])
        cols[f"{node}_x"] = pos[:, 0, k].astype(float)
        cols[f"{node}_y"] = pos[:, 1, k].astype(float)
        cols[f"{node}_score"] = np.where(found, node_score[:, k], np.nan)
    df = pd.DataFrame(cols)
    return df.sort_values(["frame", "track"], kind="stable").reset_index(drop=True)


def node_names(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return _labels(_video_group(f)["data_arrays/position/dimensions/3"])


def instances_per_frame(frames: np.ndarray, n_frames: int) -> np.ndarray:
    """Number of instances in each video frame 0 .. n_frames-1."""
    frames = np.asarray(frames, dtype=np.int64)
    if len(frames) and (frames.min() < 0 or frames.max() >= n_frames):
        raise ValueError(f"instance frames {frames.min()}..{frames.max()} outside 0..{n_frames - 1}")
    return np.bincount(frames, minlength=n_frames)
