"""Build the hand-labelling session of milestone B2 (fish keypoints in 2022 video frames).

Two kinds of items, in one fixed random order:

- **test** (evaluation of the pose model against SLEAP; recordings never
  mined, ``b2_mine.SPLITS``): per test recording 20 frames drawn uniformly over
  the electric recording (an unbiased sample of what the model will see, ~1/4
  in the light phase) and 20 hard frames (strata below). Shown **without** SLEAP's
  points, so the labels are not anchored on SLEAP.
- **hard** (training and validation data for what mining cannot provide): per
  train / val recording ``--hard-per-recording`` frames from the strata below,
  shown **with** SLEAP's points pre-filled to correct.

Hard strata (B1, ``tables/``): a dropout (fewer than two instances for >= 1 s)
0.5 s after its start, by where the lost fish was last seen (tube, block, edge,
open water); two instances less than 1 body length apart (mostly one fish
detected twice, the other missed); a random light-phase frame (fish under the
blocks and in the tube). Within a recording items are at least 5 s apart.

Labelling rule (author, 2026-11-19): every visible fish, partly hidden ones by
their visible part; a fish that cannot be seen is not labelled.

    .venv/bin/python experiments/video2022/b2_session.py [--out DIR]

Output (``$CHIRP_DATA/video2022/labels/b2_v1/``): ``items.json`` (what the
labeller reads), ``images/<item>.jpg``, ``truth.csv`` (recording, frame, time,
stratum, SLEAP instances of each item; never served), ``README.md``.
"""

from __future__ import annotations

import argparse
import json
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
from paths import DATA  # noqa: E402
from recordings2022 import list_recordings  # noqa: E402

OUT = DATA / "video2022" / "labels" / "b2_v1"
NODES = ("head", "center", "tail")
FPS = 25.0
LOOK_S = 0.5
MIN_SPACING_S = 5.0
TEST_RANDOM, TEST_HARD = 20, 20
HARD_STRATA = ("dropout_tube", "dropout_block", "dropout_edge", "dropout_open", "close_pair", "light_random")
JPEG_QUALITY = 92
SEED = 20261119


def strata_frames(name: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Candidate frames of one recording per stratum (inside the electric recording)."""
    recs = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")
    bl = float(recs.loc[name, "body_length_px"])
    inst = pd.read_parquet(b1.TABLES / "instances" / f"{name}.parquet",
                           columns=["frame", "phase", "n_in_frame", "pair_dist_px"])
    drop = pd.read_parquet(b1.TABLES / "dropouts.parquet")
    drop = drop[(drop["recording"] == name) & drop["phase"].isin([1, 2]) & (drop["dur_s"] >= 1.0) & ~drop["censored"]]
    look = (drop["start_frame"] + round(LOOK_S * FPS)).to_numpy()
    out = {f"dropout_{loc}": look[(drop["lost1_loc"] == loc).to_numpy()] for loc in ("tube", "block", "edge", "open")}
    close = inst[(inst["n_in_frame"] == 2) & (inst["pair_dist_px"] < bl) & inst["phase"].isin([1, 2])]
    out["close_pair"] = np.unique(close["frame"].to_numpy())
    t = np.load(b1.TABLES / "frame_time" / f"{name}.npy")
    phase = b1.phase_codes(t, float(recs.loc[name, "rec_dur_s"]))
    out["light_random"] = np.flatnonzero(phase == 2)
    out["random"] = np.flatnonzero(np.isin(phase, [1, 2]))
    return {k: rng.permutation(v) for k, v in out.items()}


def pick(pools: dict[str, np.ndarray], wants: list[tuple[str, int]], t: np.ndarray) -> list[tuple[str, int]]:
    """Frames per stratum, round-robin, each at least MIN_SPACING_S from every frame already taken."""
    taken: list[tuple[str, int]] = []
    left = dict(wants)
    pos = {k: 0 for k in left}
    while any(left.values()):
        progressed = False
        for k in list(left):
            if left[k] == 0:
                continue
            pool = pools.get(k, np.array([], int))
            while pos[k] < len(pool):
                f = int(pool[pos[k]])
                pos[k] += 1
                if all(abs(t[f] - t[g]) >= MIN_SPACING_S for _, g in taken):
                    taken.append((k, f))
                    left[k] -= 1
                    progressed = True
                    break
            else:
                left[k] = 0  # stratum exhausted
        if not progressed:
            break
    return taken


def sleap_fish(inst: pd.DataFrame, frame: int) -> list[dict]:
    rows = inst[inst["frame"] == frame]
    out = []
    for _, r in rows.iterrows():
        out.append({n: (None if not np.isfinite(r[f"{n}_x"]) else [float(r[f"{n}_x"]), float(r[f"{n}_y"])])
                    for n in NODES})
    return out


def build(out: Path, hard_per_recording: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    per_stratum = max(1, round(hard_per_recording / len(HARD_STRATA)))
    for rec in list_recordings():
        split = b2_mine.split_of(rec.name)
        pools = strata_frames(rec.name, rng)
        t = np.load(b1.TABLES / "frame_time" / f"{rec.name}.npy")
        if split == "test":
            n_hard = round(TEST_HARD / len(HARD_STRATA))
            wants = [("random", TEST_RANDOM)] + [(s, n_hard) for s in HARD_STRATA]
        else:
            wants = [(s, per_stratum) for s in HARD_STRATA]
        inst = pd.read_parquet(b1.TABLES / "instances" / f"{rec.name}.parquet")
        for stratum, frame in pick(pools, wants, t):
            rows.append({"item_id": secrets.token_hex(5), "kind": "test" if split == "test" else "hard",
                         "split": split, "recording": rec.name, "frame": frame, "t_s": float(t[frame]),
                         "phase": "dark" if t[frame] < b1.LIGHT_ON_S else "light", "stratum": stratum,
                         "sleap": json.dumps(sleap_fish(inst, frame))})
    items = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    size = {}
    for name, g in items.groupby("recording"):
        rec = next(r for r in list_recordings() if r.name == name)
        ids = dict(zip(g["frame"], g["item_id"], strict=True))
        for frame, img in fr.read_seek(rec.video, g["frame"]):
            import cv2

            cv2.imwrite(str(out / "images" / f"{ids[frame]}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            size[ids[frame]] = (img.shape[1], img.shape[0])
        print(f"{name}: {len(g)} items", flush=True)
    queue = [{"item_id": r.item_id, "image": f"images/{r.item_id}.jpg", "width": size[r.item_id][0],
              "height": size[r.item_id][1], "prefill": json.loads(r.sleap) if r.kind == "hard" else []}
             for r in items.itertuples()]
    (out / "items.json").write_text(json.dumps(queue))
    items.to_csv(out / "truth.csv", index=False)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--hard-per-recording", type=int, default=14)
    args = ap.parse_args()
    if (args.out / "labels.jsonl").exists():
        raise SystemExit(f"{args.out} already has labels; refusing to rebuild it")
    items = build(args.out, args.hard_per_recording)
    print(items.groupby(["kind", "stratum"]).size().to_string())
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
