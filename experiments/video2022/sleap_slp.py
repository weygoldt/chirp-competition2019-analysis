"""Read hand-made (user) instances from a SLEAP labels file (``.slp``, HDF5), with h5py only.

Layout (SLEAP labels format, as in ``sleap_tracking/labels.v000.slp``):

- ``metadata`` attribute ``json``: ``nodes`` (names) and ``skeletons`` (each
  with its ``nodes`` as ids into ``nodes``: the order of an instance's points);
- ``videos_json``: one JSON per video (``backend.filename``);
- ``frames``: ``frame_id, video, frame_idx, instance_id_start, instance_id_end``;
- ``instances``: ``instance_id, instance_type`` (0 = user, 1 = predicted),
  ``frame_id, skeleton, ..., point_id_start, point_id_end``;
- ``points``: ``x, y, visible, complete`` of user instances (predicted ones
  point into ``pred_points``).

Only user instances are returned; predicted instances stored alongside them
are model output, not labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_user_instances(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """One row per user instance: ``frame_id, video, video_file, frame_idx, instance_id``
    and per node ``<node>_x, <node>_y`` (NaN if not visible). Returns the table and the node names."""
    with h5py.File(path, "r") as f:
        meta = json.loads(f["metadata"].attrs["json"])
        names = [n["name"] for n in meta["nodes"]]
        if len(meta["skeletons"]) != 1:
            raise ValueError(f"{path}: {len(meta['skeletons'])} skeletons, expected 1")
        order = [names[n["id"]] for n in meta["skeletons"][0]["nodes"]]
        videos = [json.loads(v)["backend"]["filename"] for v in f["videos_json"][:]]
        frames = f["frames"][:]
        inst = f["instances"][:]
        pts = f["points"][:]
    frame_of = {int(r["frame_id"]): r for r in frames}
    rows = []
    for i in inst[inst["instance_type"] == 0]:
        p = pts[i["point_id_start"]:i["point_id_end"]]
        if len(p) != len(order):
            raise ValueError(f"{path}: instance {i['instance_id']} has {len(p)} points, skeleton {len(order)}")
        fr = frame_of[int(i["frame_id"])]
        row = {"frame_id": int(i["frame_id"]), "video": int(fr["video"]), "video_file": videos[int(fr["video"])],
               "frame_idx": int(fr["frame_idx"]), "instance_id": int(i["instance_id"])}
        for k, node in enumerate(order):
            ok = bool(p["visible"][k]) and np.isfinite(p["x"][k]) and np.isfinite(p["y"][k])
            row[f"{node}_x"] = float(p["x"][k]) if ok else np.nan
            row[f"{node}_y"] = float(p["y"][k]) if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows), order


def duplicates(xy: list[np.ndarray], max_px: float) -> list[int]:
    """Indices of instances to drop as duplicates of another in the same frame.

    ``xy``: per instance an (n_nodes, 2) array, NaN where not visible. Two
    instances are duplicates if they share at least two visible nodes and the
    median distance over the shared nodes is below ``max_px``; the one with
    fewer visible nodes (the later one on ties) is dropped.
    """
    drop: set[int] = set()
    n_vis = [int(np.isfinite(a[:, 0]).sum()) for a in xy]
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if i in drop or j in drop:
                continue
            shared = np.isfinite(xy[i][:, 0]) & np.isfinite(xy[j][:, 0])
            if shared.sum() >= 2 and np.median(np.linalg.norm(xy[i][shared] - xy[j][shared], axis=1)) < max_px:
                drop.add(j if n_vis[i] >= n_vis[j] else i)
    return sorted(drop)
