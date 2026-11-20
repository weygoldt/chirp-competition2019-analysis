"""Train and test the chaser-centric chase model (``chase_model.py``) on the author's labelled clips.

Leave one clip out: the model is fitted on all other labelled clips and predicts
the held-out one (both directions, "A chases B" and "B chases A"); chases are
decoded from the probabilities (``chase_model.decode``). Compared with the labels
and with the current rule-based detector (``c1_chases.py`` events in the clip's
time window):

- frames: recall / precision of chase frames (either direction) and their overlap
  (intersection over union);
- chases: a labelled chase is *found* if a predicted chase overlaps it; for found
  chases the onset and offset error (predicted - labelled; offsets of chases still
  going at the clip's end are left out) and whether the chaser is right; predicted
  chases overlapping no labelled one are *extra*.

Frame labels: inside a labelled chase, the chaser (the fish that swims towards the
other more, ``c7_chase_labels_kinematics.roles``) chases the other; everything else
is "not chasing".

The two fish of a clip (``--series``):

- ``identity`` (default): the winner and the loser (``identity_tracks.py``: pose tracks
  cut at the swaps the electrodes show and named by them); fish A = winner, B = loser;
- ``chain``: the clip's two event tracks, each chained to its continuations
  (``c5_chase_timeseries.chain``), as in the first prototype.

    .venv/bin/python experiments/video2022/c8_chase_model.py chase_v1 [chase_v2 ...] [--name proto]

Output: ``$CHIRP_DATA/video2022/chases/model/<name>/`` (``clips.png``, ``summary.json``,
``chases.csv``, ``coefficients.csv``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1
import c1_chases as c1
import c5_chase_timeseries as c5
import c6_chase_label_session as c6
import c7_chase_labels_kinematics as c7
import chase_model as cm
import chases as ch
import identity_tracks as it

OUT = c1.OUT / "model"


def load_clips(sessions: list[str], run: str, series: str = "identity") -> list[dict]:
    """Per labelled clip: its series, features of both directions, frame labels, labelled chases, detector chases."""
    root = c1.OUT / run
    params = json.loads((root / "params.json").read_text())
    stride = int(params["stride"])
    p = ch.Params(**{k: v for k, v in params.items() if k in ch.Params.__dataclass_fields__})
    blpx = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    events = pd.read_csv(root / "events.csv")
    cache: dict = {}
    clips = []
    for sess in sessions:
        sd = c6.LABELS / sess
        truth = pd.read_csv(sd / "truth.csv").set_index("clip_id")
        for cid, lab in c7.labels_of(sd).items():
            row = truth.loc[cid]
            rec = row["recording"]
            if rec not in cache:
                if series == "identity":
                    cache = {rec: it.load(run, rec)}
                else:
                    kin = pd.read_parquet(root / rec / "kinematics.parquet")
                    tr = pd.read_parquet(root / rec / "tracks.parquet", columns=["track", "frame", "tail_x", "tail_y"])
                    cache = {rec: (kin, tr, c5.spans_of(kin))}
            if series == "identity":
                f0 = int(row["frame0"])
                i_lo, i_hi = f0 // stride, (f0 + int(row["n_frames"])) // stride
                kin_id = cache[rec]
                s = c7.pair_frame(it.on_grid(kin_id, "winner", i_lo, i_hi), it.on_grid(kin_id, "loser", i_lo, i_hi),
                                  f0, stride, float(blpx[rec]))
                s = s[s["t"] >= 0].reset_index(drop=True)
            else:
                kin, tr, spans = cache[rec]
                s = c7.clip_series(row, kin, tr, spans, float(blpx[rec]), stride, p).reset_index(drop=True)
            feats = cm.directed_features(s)
            t = s["t"].to_numpy()
            y = {"A": np.zeros(len(s), int), "B": np.zeros(len(s), int)}
            chases = []
            for c in lab["chases"]:
                chaser, _, _ = c7.roles(s, c["onset_s"], c["offset_s"])
                inside = (t >= c["onset_s"]) & (t <= c["offset_s"])
                y[chaser][inside] = 1
                chases.append({**c, "chaser": chaser})
            dur = row["n_frames"] / c1.FPS
            ev = events[(events["recording"] == rec) & (events["t_end"] >= row["t0"])
                        & (events["t_start"] <= row["t0"] + dur)]
            det = [(a - row["t0"], b - row["t0"]) for a, b in zip(ev["t_start"], ev["t_end"], strict=True)]
            clips.append({"clip_id": cid, "session": sess, "event": row["event"], "kind": row.get("kind", "detected"),
                          "s": s, "t": t, "feats": feats, "y": y, "chases": chases, "no_chase": lab["no_chase"],
                          "detector": det, "duration": dur})
    return clips


def matrix(clips: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = list(clips[0]["feats"]["A"].columns)
    X = np.vstack([c["feats"][d][cols].to_numpy(float) for c in clips for d in ("A", "B")])
    y = np.concatenate([c["y"][d] for c in clips for d in ("A", "B")])
    return X, y, cols


def intervals(t: np.ndarray, pieces: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [(float(t[a]), float(t[b])) for a, b in pieces]


def score_clip(c: dict, pred: list[tuple[float, float, str]]) -> tuple[list[dict], dict]:
    """Per labelled chase: found, onset / offset error, chaser right; and frame counts."""
    t = c["t"]
    lab_f = np.zeros(len(t), bool)
    for x in c["chases"]:
        lab_f |= (t >= x["onset_s"]) & (t <= x["offset_s"])
    pred_f = np.zeros(len(t), bool)
    for a, b, _ in pred:
        pred_f |= (t >= a) & (t <= b)
    rows = []
    for x in c["chases"]:
        ov = [(min(b, x["offset_s"]) - max(a, x["onset_s"]), a, b, d) for a, b, d in pred
              if min(b, x["offset_s"]) > max(a, x["onset_s"])]
        r = {"clip_id": c["clip_id"], "onset_s": x["onset_s"], "offset_s": x["offset_s"], "continues": x["continues"],
             "duration_s": x["offset_s"] - x["onset_s"], "found": bool(ov)}
        if ov:
            _, a, b, d = max(ov)
            r |= {"onset_err_s": a - x["onset_s"], "offset_err_s": np.nan if x["continues"] else b - x["offset_s"],
                  "chaser_right": d == x["chaser"]}
        rows.append(r)
    extra = sum(1 for a, b, _ in pred
                if not any(min(b, x["offset_s"]) > max(a, x["onset_s"]) for x in c["chases"]))
    return rows, {"lab": int(lab_f.sum()), "pred": int(pred_f.sum()), "both": int((lab_f & pred_f).sum()),
                  "union": int((lab_f | pred_f).sum()), "extra": extra, "n_pred": len(pred)}


def summarize(rows: list[dict], frames: list[dict]) -> dict:
    d = pd.DataFrame(rows)
    f = pd.DataFrame(frames).sum()
    found = d[d["found"]] if len(d) else d
    return {"chases": len(d), "found": int(d["found"].sum()) if len(d) else 0,
            "extra_chases": int(f["extra"]), "predicted_chases": int(f["n_pred"]),
            "frame_recall": round(f["both"] / max(f["lab"], 1), 3),
            "frame_precision": round(f["both"] / max(f["pred"], 1), 3),
            "frame_iou": round(f["both"] / max(f["union"], 1), 3),
            "onset_err_median_s": round(float(found["onset_err_s"].median()), 2) if len(found) else None,
            "onset_abs_err_median_s": round(float(found["onset_err_s"].abs().median()), 2) if len(found) else None,
            "offset_err_median_s": round(float(found["offset_err_s"].median()), 2) if len(found) else None,
            "offset_abs_err_median_s": round(float(found["offset_err_s"].abs().median()), 2) if len(found) else None,
            "chaser_right": int(found["chaser_right"].sum()) if "chaser_right" in found else None}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--run", default="poses_r3")
    ap.add_argument("--name", default="proto")
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--series", choices=("identity", "chain"), default="identity")
    args = ap.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    clips = load_clips(args.sessions, args.run, args.series)
    dec = cm.Decoder()
    rows_m, frames_m, rows_d, frames_d, preds = [], [], [], [], {}
    for k, c in enumerate(clips):
        X, y, cols = matrix([x for j, x in enumerate(clips) if j != k])
        model = cm.Logistic(args.l2).fit(X, y)
        pa = model.predict(c["feats"]["A"][cols].to_numpy(float))
        pb = model.predict(c["feats"]["B"][cols].to_numpy(float))
        pred = [(a, b, "A") for a, b in intervals(c["t"], cm.decode(pa, dec))] + \
               [(a, b, "B") for a, b in intervals(c["t"], cm.decode(pb, dec))]
        preds[c["clip_id"]] = (pa, pb, pred)
        r, f = score_clip(c, pred)
        rows_m += [x | {"method": "model"} for x in r]
        frames_m.append(f)
        r, f = score_clip(c, [(a, b, "?") for a, b in c["detector"]])
        rows_d += [x | {"method": "detector"} for x in r]
        frames_d.append(f)
    summary = {"clips": len(clips), "sessions": args.sessions, "series": args.series, "decoder": dec.__dict__,
               "l2": args.l2, "model": summarize(rows_m, frames_m), "detector": summarize(rows_d, frames_d)}
    summary["detector"].pop("chaser_right")
    kind_of = {c["clip_id"]: c["kind"] for c in clips}
    for kind in sorted(set(kind_of.values())):
        ids = [c["clip_id"] for c in clips if c["kind"] == kind]
        idx = [k for k, c in enumerate(clips) if c["kind"] == kind]
        for name, rows, frames in (("model", rows_m, frames_m), ("detector", rows_d, frames_d)):
            summary[f"{name}_{kind}"] = summarize([r for r in rows if r["clip_id"] in ids], [frames[k] for k in idx])
            summary[f"{name}_{kind}"].pop("chaser_right", None)
            summary[f"{name}_{kind}"]["clips"] = len(ids)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(rows_m + rows_d).to_csv(out / "chases.csv", index=False)
    X, y, cols = matrix(clips)
    full = cm.Logistic(args.l2).fit(X, y)
    coef = pd.Series(full.coef[1:], index=cols).sort_values(key=np.abs, ascending=False)
    coef.to_csv(out / "coefficients.csv", header=["coef_standardised"])

    n = len(clips)
    fig, axes = plt.subplots(n, 1, figsize=(10, 0.95 * n + 0.8), sharex=True, constrained_layout=True)
    for ax, c in zip(np.atleast_1d(axes), clips, strict=True):
        pa, pb, pred = preds[c["clip_id"]]
        ax.plot(c["t"], pa, color="#e6820e", lw=0.8)
        ax.plot(c["t"], pb, color="#1fb4d2", lw=0.8)
        for x in c["chases"]:
            ax.axvspan(x["onset_s"], x["offset_s"], ymin=0.82, ymax=1.0, color="#c0392b")
        for a, b, d in pred:
            ax.axvspan(a, b, ymin=0.62, ymax=0.8, color="#e6820e" if d == "A" else "#1fb4d2")
        for a, b in c["detector"]:
            ax.axvspan(max(a, 0), min(b, c["duration"]), ymin=0.42, ymax=0.6, color="0.55")
        ax.set_ylim(0, 1.25)
        ax.set_yticks([])
        ax.text(0.003, 0.05, f"{c['kind']}  {c['event']}", transform=ax.transAxes, fontsize=6)
    axes[-1].set_xlabel("clip time (s).  Bars, top to bottom: labelled chases (red), model chases (colour = chaser), "
                        "current detector (grey).  Lines: model probability, fish A (orange) / B (cyan) chasing",
                        fontsize=7)
    fig.suptitle("Chase model, leave one clip out", fontsize=10)
    fig.savefig(out / "clips.png", dpi=130)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("clips", "model", "detector"))}, indent=1))
    print("largest standardised coefficients:\n", coef.head(12).round(2).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
