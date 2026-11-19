"""Fish keypoint labels of video frames: storage and conversion to the YOLO pose format.

The skeleton is the 2022 SLEAP hand-label skeleton (author's choice, 2026-11-19):
``snout, spine1, spine2, spine3, tail, left_fin, right_fin`` (``SKELETON``).
``spine1`` lies 18 % of the way from snout to tail, where the pectoral fins
attach; it is also the "center" node of the 3-node SLEAP exports. Older
3-point labels (``head``, ``center``, ``tail``: the mined frames and the first
labelling session) are read as ``snout``, ``spine1``, ``tail``
(``LEGACY_3``); their other points are simply not labelled.

A frame label lists every fish visible in the frame with its visible points
(pixel coordinates of the full video frame, origin top left). Labelling rule
(author, 2026-11-19): a partly hidden fish is labelled by its visible part; a
fish that cannot be seen at all is not labelled.

Storage: one JSON object per saved frame in a ``labels.jsonl``; the latest
line of an item wins (:func:`load_labels`).

YOLO pose format (one ``.txt`` per image, one line per fish)::

    0 cx cy w h  x1 y1 v1 ... x7 y7 v7

normalised to the image size, ``v`` = 2 (labelled) or 0 (not labelled; x = y = 0).
The box is the extent of the fish's labelled points padded by ``pad_px`` and
clipped to the image (:func:`bbox_from_points`), the same rule for every source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SKELETON = ("snout", "spine1", "spine2", "spine3", "tail", "left_fin", "right_fin")
# a mirror image (horizontal or vertical flip) swaps the fish's left and right fin
FLIP_IDX = [0, 1, 2, 3, 4, 6, 5]
EDGES = [("snout", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"), ("spine3", "tail"),
         ("spine1", "left_fin"), ("spine1", "right_fin")]
LEGACY_3 = {"head": "snout", "center": "spine1", "tail": "tail"}
STATUSES = ("done", "unusable", "draft")


@dataclass
class Fish:
    """One fish: its labelled points (name -> (x, y) px); missing names are not labelled."""

    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    origin: str = ""  # e.g. "sleap", "sleap_hand", "model", "model_edited", "new", "prefill", "prefill_edited"

    def __post_init__(self) -> None:
        unknown = set(self.points) - set(SKELETON)
        if unknown:
            raise ValueError(f"unknown keypoints {sorted(unknown)}")

    def n_points(self) -> int:
        return len(self.points)

    @classmethod
    def from_dict(cls, d: dict) -> Fish:
        """From ``{"points": {...}, "origin": ...}`` or the legacy ``{"head": [x, y], "center": ..., "tail": ...}``."""
        if "points" in d:
            raw = d["points"]
        else:
            unknown = set(d) - set(LEGACY_3) - {"origin"}
            if unknown:
                raise ValueError(f"unknown keypoints {sorted(unknown)}")
            raw = {LEGACY_3[k]: v for k, v in d.items() if k in LEGACY_3}
        pts = {k: (float(v[0]), float(v[1])) for k, v in raw.items() if v is not None}
        return cls(points=pts, origin=str(d.get("origin", "")))

    def to_json(self) -> dict:
        return {"points": {k: list(v) for k, v in self.points.items()}, "origin": self.origin}


@dataclass
class FrameLabel:
    item_id: str
    status: str
    fish: list[Fish] = field(default_factory=list)
    labeller: str = ""
    labelled_at: str = ""
    elapsed_s: float = float("nan")

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")

    @classmethod
    def from_json(cls, d: dict) -> FrameLabel:
        return cls(item_id=str(d["item_id"]), status=str(d["status"]),
                   fish=[Fish.from_dict(f) for f in d.get("fish", [])], labeller=str(d.get("labeller", "")),
                   labelled_at=str(d.get("labelled_at", "")), elapsed_s=float(d.get("elapsed_s", float("nan"))))

    def to_json(self) -> dict:
        return {"item_id": self.item_id, "status": self.status, "fish": [f.to_json() for f in self.fish],
                "labeller": self.labeller, "labelled_at": self.labelled_at, "elapsed_s": self.elapsed_s}


def append_label(path: Path, label: FrameLabel) -> None:
    with Path(path).open("a") as f:
        f.write(json.dumps(label.to_json()) + "\n")


def load_labels(path: Path) -> dict[str, FrameLabel]:
    """Latest label per item (later lines replace earlier ones); empty if the file does not exist."""
    out: dict[str, FrameLabel] = {}
    path = Path(path)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            lab = FrameLabel.from_json(json.loads(line))
            out[lab.item_id] = lab
    return out


def bbox_from_points(fish: Fish, pad_px: float, width: int, height: int) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the fish's labelled points padded by ``pad_px`` and clipped to the image."""
    if not fish.points:
        raise ValueError("a fish without points has no box")
    pts = np.array(list(fish.points.values()), dtype=float)
    x0, y0 = np.maximum(pts.min(axis=0) - pad_px, 0.0)
    x1, y1 = np.minimum(pts.max(axis=0) + pad_px, [width, height])
    return float(x0), float(y0), float(x1), float(y1)


def yolo_lines(fish: list[Fish], width: int, height: int, pad_px: float,
               skeleton: tuple[str, ...] = SKELETON) -> list[str]:
    """YOLO pose label lines (class 0) for the fish of one image."""
    lines = []
    for f in fish:
        if f.n_points() == 0:
            continue
        x0, y0, x1, y1 = bbox_from_points(f, pad_px, width, height)
        vals: list[float | int] = [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width,
                                   (y1 - y0) / height]
        for n in skeleton:
            p = f.points.get(n)
            vals += [0.0, 0.0, 0] if p is None else [p[0] / width, p[1] / height, 2]
        lines.append("0 " + " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals))
    return lines


def dataset_yaml(root: Path, splits: dict[str, str | list[str]], skeleton: tuple[str, ...] = SKELETON,
                 flip_idx: list[int] = FLIP_IDX) -> str:
    """The Ultralytics dataset file: one class ``fish``, the skeleton's keypoints."""
    lines = [f"path: {root}"]
    for k, v in splits.items():
        lines.append(f"{k}: {json.dumps(v) if isinstance(v, list) else v}")
    lines += [f"kpt_shape: [{len(skeleton)}, 3]", f"flip_idx: {flip_idx}", "names:", "  0: fish",
              f"kpt_names: {{0: [{', '.join(skeleton)}]}}"]
    return "\n".join(lines) + "\n"
