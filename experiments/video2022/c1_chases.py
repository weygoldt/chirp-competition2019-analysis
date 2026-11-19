"""Chase events in every recording of a pose run (milestone C; author's go 2026-11-19).

For each recording of ``$CHIRP_DATA/video2022/poses/<run>/`` that is complete:
``chases.detect`` (tracks, kinematics, chase frames, events; parameters
``chases.Params``), and the time both fish were visible (frames with two
confident, full-length detections), so that chase rates can be given per hour
with both fish in view: in some trials one fish spends most of the dark phase
hidden in the tube.

    .venv/bin/python experiments/video2022/c1_chases.py poses_r3

Output: ``$CHIRP_DATA/video2022/chases/<run>/``: ``<rec>/tracks.parquet`` (the clean
detections with ``track``), ``<rec>/kinematics.parquet``, ``<rec>/chase_frames.parquet``,
``events.csv`` (all recordings), ``recordings.csv`` (per recording: visible times,
events, rates), ``params.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1
import b3_predict as bp
import chases as ch
from paths import DATA, REC_2022

OUT = DATA / "video2022" / "chases"
FPS = 25.0


def pairing() -> pd.Series:
    meta = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"})
    return meta.set_index("recording")["Pairing"]


def run_recording(run: str, rec: str, bl: float, stride: int, p: ch.Params, out: Path) -> tuple[pd.DataFrame, dict]:
    d = bp.load(run, rec)
    tr, kin, cf, ev = ch.detect(d, bl, stride, FPS, p)
    rd = out / rec
    rd.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(rd / "tracks.parquet", index=False)
    kin.to_parquet(rd / "kinematics.parquet", index=False)
    cf.to_parquet(rd / "chase_frames.parquet", index=False)
    frames_total = len(bp.dark_frames(rec, stride)[0])
    per_frame = tr.groupby("frame").size()
    dt = stride / FPS
    info = {"recording": rec, "body_length_px": bl, "frames": frames_total,
            "hours_dark": frames_total * dt / 3600,
            "hours_one_or_more": float((per_frame >= 1).sum() * dt / 3600),
            "hours_both_visible": float((per_frame >= 2).sum() * dt / 3600),
            "hours_three_or_more": float((per_frame >= 3).sum() * dt / 3600),
            "tracks": int(tr["track"].nunique()),
            "track_median_s": float(tr.groupby("track").size().median() * dt),
            "events": len(ev), "chase_s": float(ev["duration_s"].sum())}
    info["events_per_hour_both_visible"] = (len(ev) / info["hours_both_visible"]
                                            if info["hours_both_visible"] > 0 else float("nan"))
    return ev.assign(recording=rec), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--recordings", nargs="*")
    args = ap.parse_args()
    src = bp.ROOT / args.run
    stride = json.loads((src / "meta.json").read_text())["stride"]
    p = ch.Params()
    out = OUT / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.json").write_text(json.dumps({"stride": stride, **dataclasses.asdict(p)}, indent=1))
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    pairs = pairing()
    recs = sorted(r.parent.name for r in src.glob("*/done.json"))
    if args.recordings:
        recs = [r for r in recs if r in args.recordings]
    evs, infos = [], []
    for rec in recs:
        ev, info = run_recording(args.run, rec, float(bl[rec]), stride, p, out)
        info["pairing"] = pairs.get(rec, "control")
        evs.append(ev)
        infos.append(info)
        print(f"{rec}: {info['events']} events, both fish visible {info['hours_both_visible']:.2f} h of "
              f"{info['hours_dark']:.2f} h, tracks {info['tracks']} (median {info['track_median_s']:.1f} s)", flush=True)
    ev_all = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
    if not ev_all.empty:
        ev_all.insert(0, "event", [f"{r}_{k:04d}" for r, k in zip(ev_all["recording"], ev_all.groupby("recording").cumcount(),
                                                                    strict=True)])
    ev_all.to_csv(out / "events.csv", index=False)
    pd.DataFrame(infos).to_csv(out / "recordings.csv", index=False)
    print(f"{len(ev_all)} events in {len(recs)} recordings -> {out}")


if __name__ == "__main__":
    main()
