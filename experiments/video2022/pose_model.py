"""Run a trained YOLO pose model (``b3_train.py``) on video frames and read out the fish.

:func:`predict` returns, per image, the detected fish with the box confidence
and each keypoint's position and confidence (``pose_labels.SKELETON`` order).
:func:`to_fish` turns a prediction into a ``pose_labels.Fish`` with only the
confident points (pseudo labels / pre-fill for the labeller), and
:func:`frame_uncertainty` scores how much a frame is worth labelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import pose_labels as pl

BODY = ("snout", "spine1", "spine2", "spine3", "tail")


@dataclass
class PredFish:
    box_conf: float
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    kpt_conf: dict[str, float] = field(default_factory=dict)


def load(weights: str):  # noqa: ANN201
    from ultralytics import YOLO

    return YOLO(str(weights))


def predict(model, images: list[np.ndarray], imgsz: int = 1280, conf: float = 0.2, batch: int = 1,  # noqa: ANN001
            half: bool = True) -> list[list[PredFish]]:
    """Fish in each image (grey uint8 arrays), boxes with confidence >= ``conf``."""
    out: list[list[PredFish]] = []
    for i in range(0, len(images), batch):
        chunk = [np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img for img in images[i:i + batch]]
        for r in model.predict(chunk, imgsz=imgsz, conf=conf, half=half, verbose=False, device=0):
            fish = []
            if r.keypoints is not None and len(r.boxes):
                xy = r.keypoints.xy.cpu().numpy()
                kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
                for j, bc in enumerate(r.boxes.conf.cpu().numpy()):
                    fish.append(PredFish(float(bc), {n: (float(xy[j, k, 0]), float(xy[j, k, 1]))
                                                     for k, n in enumerate(pl.SKELETON)},
                                         {n: float(kc[j, k]) for k, n in enumerate(pl.SKELETON)}))
            out.append(fish)
    return out


def to_fish(p: PredFish, min_kpt_conf: float = 0.5) -> pl.Fish:
    """The prediction as a label: only points with confidence >= ``min_kpt_conf``."""
    return pl.Fish({n: xy for n, xy in p.points.items() if p.kpt_conf[n] >= min_kpt_conf}, origin="model")


def frame_uncertainty(fish: list[PredFish], confident: float = 0.5) -> dict[str, float]:
    """How doubtful a frame's prediction is.

    - ``count_off``: |number of fish with box confidence >= ``confident`` - 2| (the tank holds two);
    - ``ambiguous``: fish with box confidence between 0.2 and ``confident``;
    - ``weak_points``: 1 - the lowest mean body-point confidence among the confident fish
      (1 if there is none).

    ``score`` is their sum; higher = more worth labelling.
    """
    sure = [p for p in fish if p.box_conf >= confident]
    ambiguous = sum(1 for p in fish if p.box_conf < confident)
    weak = 1.0 - min((np.mean([p.kpt_conf[n] for n in BODY]) for p in sure), default=0.0)
    count_off = abs(len(sure) - 2)
    return {"n_confident": len(sure), "count_off": count_off, "ambiguous": ambiguous, "weak_points": float(weak),
            "score": count_off + ambiguous + float(weak)}
