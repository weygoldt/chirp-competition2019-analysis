"""Build a 7-point keypoint-correction session with a trained pose model (milestone B3, one per round).

Items, in one fixed random order (``kind`` in ``truth.csv``):

- ``test`` (``--with-test``): the blind test frames of ``labels/b2_v1``
  (recordings never trained on). Never pre-filled by a model, so the labels are
  not anchored on it; where the author already labelled a frame in b2_v1 (3
  points), those labels are shown so that only the other points need adding.
- ``hard``: b2_v1's hard frames (dropouts, close pairs, light phase) of train
  and val recordings. Frames the author already labelled in b2_v1 come first
  (his points shown, the missing ones to add), the rest pre-filled with the
  model; ``--n-hard`` in total, frames used in an earlier session are skipped.
- ``new``: ``--n-new`` frames chosen by the model's doubt
  (``pose_model.frame_uncertainty``: not exactly two confident fish, ambiguous
  detections, weak body points) among ``--candidates`` frames per train / val
  recording, spread evenly over the electric recording. At most
  ``--per-recording`` per recording, >= 30 s apart, highest score first;
  pre-filled with the model.

Model pre-fill: fish with box confidence >= 0.5, points with confidence >= 0.5.

    .venv/bin/python experiments/video2022/b3_session.py r1 --model runs/<run>/weights/best.pt --with-test

Output: ``$CHIRP_DATA/video2022/labels/<name>/`` with ``items.json``,
``session.json`` (the 7-point skeleton), ``images/``, ``truth.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1  # noqa: E402
import b2_mine  # noqa: E402
import frames as fr  # noqa: E402
import pose_labels as pl  # noqa: E402
import pose_model as pm  # noqa: E402
from paths import DATA  # noqa: E402
from recordings2022 import list_recordings  # noqa: E402

LABELS = DATA / "video2022" / "labels"
B2 = LABELS / "b2_v1"
JPEG_QUALITY = 92
MIN_GAP_S = 30.0
PREFILL_BOX_CONF = 0.5
SEED = 20261119
COLORS = {"snout": "#ff4d4d", "spine1": "#ffd24d", "spine2": "#c7e9b4", "spine3": "#7fcdbb", "tail": "#4dc3ff",
          "left_fin": "#f781bf", "right_fin": "#a65628"}
NOTE = ("spine1 = just behind the head, where the pectoral fins attach (~1/5 of the way to the tail); "
        "spine2, spine3 split the rest of the body to the tail; left / right fin = the fish's own left and right "
        "pectoral fin (skip with s if not visible).")


def used_frames() -> set[tuple[str, int]]:
    """(recording, frame) of every item of every earlier session except b2_v1 (whose items are reused here)."""
    out = set()
    for t in LABELS.glob("*/truth.csv"):
        if t.parent.name == "b2_v1":
            continue
        d = pd.read_csv(t)
        out |= set(zip(d["recording"], d["frame"].astype(int), strict=True))
    return out


def model_prefill(preds: list[pm.PredFish]) -> list[dict]:
    return [pm.to_fish(p).to_json() for p in preds if p.box_conf >= PREFILL_BOX_CONF and pm.to_fish(p).n_points()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--model", required=True, help="weights (.pt), absolute or relative to $CHIRP_DATA/video2022")
    ap.add_argument("--with-test", action="store_true", help="include b2_v1's blind test frames")
    ap.add_argument("--n-hard", type=int, default=60)
    ap.add_argument("--n-new", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=250, help="frames scored per train / val recording")
    ap.add_argument("--per-recording", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()
    out = LABELS / args.name
    if out.exists():
        raise SystemExit(f"{out} exists; sessions are never rebuilt")
    weights = Path(args.model) if Path(args.model).is_absolute() else DATA / "video2022" / args.model
    rng = np.random.default_rng(SEED)
    model = pm.load(weights)
    (out / "images").mkdir(parents=True)
    b2_truth = pd.read_csv(B2 / "truth.csv")
    b2_labels = pl.load_labels(B2 / "labels.jsonl")
    done = {k for k, v in b2_labels.items() if v.status == "done"}
    used = used_frames()
    items, truth = [], []

    def add(kind, rec, frame, t_s, stratum, prefill, image_from=None, img=None, score=np.nan, source=""):  # noqa: ANN001, ANN202
        iid = secrets.token_hex(5)
        dst = out / "images" / f"{iid}.jpg"
        if image_from is not None:
            os.symlink(image_from.resolve(), dst)
        else:
            import cv2

            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        items.append({"item_id": iid, "image": f"images/{iid}.jpg", "width": 2048, "height": 1536, "prefill": prefill})
        truth.append({"item_id": iid, "kind": kind, "split": "test" if kind == "test" else b2_mine.split_of(rec),
                      "recording": rec, "frame": int(frame), "t_s": float(t_s),
                      "phase": "dark" if t_s < b1.LIGHT_ON_S else "light", "stratum": stratum, "uncertainty": score,
                      "source_item": source})

    # test: b2_v1's blind items, the author's own earlier points as the only pre-fill
    if args.with_test:
        for r in b2_truth[b2_truth["kind"] == "test"].itertuples():
            pre = [f.to_json() for f in b2_labels[r.item_id].fish] if r.item_id in done else []
            add("test", r.recording, r.frame, r.t_s, r.stratum, pre, image_from=B2 / "images" / f"{r.item_id}.jpg",
                source=f"b2_v1/{r.item_id}")
    # hard: b2_v1's hard items, the author's labelled ones first
    hard = b2_truth[(b2_truth["kind"] == "hard")
                    & ~pd.Series(list(zip(b2_truth["recording"], b2_truth["frame"], strict=True))).isin(used).to_numpy()]
    hard = hard.assign(done=hard["item_id"].isin(done)).sample(frac=1.0, random_state=SEED)
    hard = hard.sort_values("done", ascending=False, kind="stable").head(args.n_hard)
    need = hard[~hard["done"]]
    imgs = {}
    for name, g in need.groupby("recording"):
        rec = next(x for x in list_recordings() if x.name == name)
        imgs.update({(name, f): im for f, im in fr.read_seek(rec.video, g["frame"])})
    keys = list(imgs)
    preds = dict(zip(keys, pm.predict(model, [imgs[k] for k in keys], imgsz=args.imgsz), strict=True))
    for r in hard.itertuples():
        pre = ([f.to_json() for f in b2_labels[r.item_id].fish] if r.done else model_prefill(preds[(r.recording, r.frame)]))
        add("hard", r.recording, r.frame, r.t_s, r.stratum, pre, image_from=B2 / "images" / f"{r.item_id}.jpg",
            source=f"b2_v1/{r.item_id}")
    print(f"test {sum(t['kind'] == 'test' for t in truth)}, hard {len(hard)} ({int(hard['done'].sum())} with b2_v1 labels)",
          flush=True)
    # new: the model's most doubtful frames
    taken = used | set(zip(b2_truth["recording"], b2_truth["frame"].astype(int), strict=True))
    scored = []
    for rec in list_recordings():
        if b2_mine.split_of(rec.name) == "test":
            continue
        t_frame = np.load(b1.TABLES / "frame_time" / f"{rec.name}.npy")
        inside = np.flatnonzero((t_frame >= 0) & (t_frame < 14_400))
        cand = inside[np.linspace(0, len(inside) - 1, args.candidates).astype(int)]
        cand = cand + rng.integers(0, 25, len(cand))  # avoid a fixed phase relative to the 1-s grid
        cand = [int(f) for f in cand if (rec.name, int(f)) not in taken and f < len(t_frame)]
        got = dict(fr.read_seek(rec.video, cand))
        fs = list(got)
        for f, p in zip(fs, pm.predict(model, [got[f] for f in fs], imgsz=args.imgsz), strict=True):
            u = pm.frame_uncertainty(p)
            scored.append({"recording": rec.name, "frame": f, "t_s": float(t_frame[f]), **u,
                           "prefill": model_prefill(p)})
        print(f"{rec.name}: scored {len(fs)}", flush=True)
    sc = pd.DataFrame(scored).sort_values("score", ascending=False)
    sc.drop(columns=["prefill"]).to_csv(out / "candidates_scored.csv", index=False)
    chosen = []
    for r in sc.itertuples():
        mine = [c for c in chosen if c.recording == r.recording]
        if len(mine) >= args.per_recording or any(abs(c.t_s - r.t_s) < MIN_GAP_S for c in mine):
            continue
        chosen.append(r)
        if len(chosen) >= args.n_new:
            break
    for r in chosen:
        rec = next(x for x in list_recordings() if x.name == r.recording)
        (_, img), = fr.read_seek(rec.video, [r.frame])
        stratum = "count_off" if r.count_off else ("ambiguous" if r.ambiguous else "weak_points")
        add("new", r.recording, r.frame, r.t_s, stratum, r.prefill, img=img, score=r.score)
    order = rng.permutation(len(items))
    (out / "items.json").write_text(json.dumps([items[i] for i in order]))
    pd.DataFrame(truth).iloc[order].to_csv(out / "truth.csv", index=False)
    (out / "session.json").write_text(json.dumps({"nodes": list(pl.SKELETON), "edges": [list(e) for e in pl.EDGES],
                                                  "colors": COLORS, "note": NOTE, "model": str(weights)}, indent=1))
    t = pd.DataFrame(truth)
    print(t.groupby(["kind", "stratum"]).size().to_string())
    print(f"{len(t)} items -> {out}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
