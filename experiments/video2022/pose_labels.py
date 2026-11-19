"""Fish keypoint labels of video frames: storage and conversion to the YOLO pose format.

A frame label lists every fish visible in the frame, each with up to three
keypoints (``head``, ``center``, ``tail``; pixel coordinates of the full video
frame, origin top left). A keypoint that is not visible is missing (``None``).
``center`` is SLEAP's node of that name: just behind the head, at the pectoral
fins, 18 % of the way from snout to tail along head-centre-tail (median of
the mined frames, 10th-90th percentile 15-21 %), not the middle of the body.
Labelling rule (author, 2026-11-19): a partly hidden fish is labelled by its
visible part (e.g. only the head peeking out of the tube); a fish that cannot
be seen at all is not labelled.

Storage: the labeller appends one JSON object per saved frame to
``<session>/labels.jsonl``; the latest line of an item wins (:func:`load_labels`).

YOLO pose format (one ``.txt`` per image, one line per fish)::

    0 cx cy w h  x_head y_head v_head  x_center y_center v_center  x_tail y_tail v_tail

all coordinates normalised to the image size, ``v`` = 2 (visible) or 0 (not
labelled; then x = y = 0). The box is not drawn by hand: it is the extent of
the fish's keypoints padded by ``pad_px`` on every side and clipped to the
image (:func:`bbox_from_keypoints`), the same rule for mined and hand-labelled
frames.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

NODES = ("head", "center", "tail")
FLIP_IDX = [0, 1, 2]  # a horizontal flip keeps head, centre and tail (no left/right pairs)
STATUSES = ("done", "unusable", "draft")


@dataclass
class Fish:
    """One fish: keypoint name -> (x, y) in px, or None if not visible."""

    head: tuple[float, float] | None = None
    center: tuple[float, float] | None = None
    tail: tuple[float, float] | None = None
    origin: str = ""  # labeller: "prefill" (kept as shown), "prefill_edited" or "new"

    def points(self) -> dict[str, tuple[float, float] | None]:
        return {n: getattr(self, n) for n in NODES}

    def n_points(self) -> int:
        return sum(p is not None for p in self.points().values())

    @classmethod
    def from_dict(cls, d: dict) -> Fish:
        unknown = set(d) - set(NODES) - {"origin"}
        if unknown:
            raise ValueError(f"unknown keypoints {sorted(unknown)}")
        pts = {n: (None if d.get(n) is None else (float(d[n][0]), float(d[n][1]))) for n in NODES}
        return cls(**pts, origin=str(d.get("origin", "")))


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
        return asdict(self)


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


def bbox_from_keypoints(fish: Fish, pad_px: float, width: int, height: int) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the fish's keypoints padded by ``pad_px`` and clipped to the image."""
    pts = np.array([p for p in fish.points().values() if p is not None], dtype=float)
    if len(pts) == 0:
        raise ValueError("a fish without keypoints has no box")
    x0, y0 = np.maximum(pts.min(axis=0) - pad_px, 0.0)
    x1, y1 = np.minimum(pts.max(axis=0) + pad_px, [width, height])
    return float(x0), float(y0), float(x1), float(y1)


def yolo_lines(fish: list[Fish], width: int, height: int, pad_px: float) -> list[str]:
    """YOLO pose label lines (class 0) for the fish of one image."""
    lines = []
    for f in fish:
        if f.n_points() == 0:
            continue
        x0, y0, x1, y1 = bbox_from_keypoints(f, pad_px, width, height)
        vals = [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height]
        for p in f.points().values():
            vals += [0.0, 0.0, 0] if p is None else [p[0] / width, p[1] / height, 2]
        lines.append("0 " + " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals))
    return lines


def dataset_yaml(root: Path, splits: dict[str, str]) -> str:
    """The Ultralytics dataset file: one class ``fish``, three keypoints."""
    lines = [f"path: {root}"] + [f"{k}: {v}" for k, v in splits.items()]
    lines += ["kpt_shape: [3, 3]", f"flip_idx: {FLIP_IDX}", "names:", "  0: fish",
              f"kpt_names: {{0: [{', '.join(NODES)}]}}"]
    return "\n".join(lines) + "\n"
