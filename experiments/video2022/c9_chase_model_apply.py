"""Chases in every 2022 trial from the chaser-centric model, on the winner / loser trajectories.

The model (``chase_model.py``) is fitted on all labelled clips of the given sessions
(``c8_chase_model.load_clips``, identity series) and applied to the whole dark phase
of every trial (``identity_tracks.py``: winner = fish A, loser = fish B). Both
directions are decoded (``chase_model.decode``); a chase's chaser is the fish of the
direction it was found in. Where both directions are called at once (a conflict),
both chases are kept and flagged ``overlap``.

    .venv/bin/python experiments/video2022/c9_chase_model_apply.py chase_v1 chase_v2 [--name model_v1]

Output: ``$CHIRP_DATA/video2022/chases/model/<name>/``: ``events.csv`` (recording,
chaser, chasee, t_start, t_end, duration_s, p_mean, p_max, overlap), ``trials.csv`` (per
trial: hours with both fish in view, chases by chaser, rate, old detector's count),
``trials.png``, ``model.json`` (coefficients, standardisation, decoder).
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
import c7_chase_labels_kinematics as c7
import c8_chase_model as c8
import chase_model as cm
import identity_tracks as it

FISH = {"A": "winner", "B": "loser"}


def recording_series(kin: pd.DataFrame, bl: float, stride: int) -> pd.DataFrame:
    """The winner (A) / loser (B) pair table over the whole recording (model frame grid)."""
    i_lo, i_hi = int(kin["i"].min()), int(kin["i"].max())
    s = c7.pair_frame(it.on_grid(kin, "winner", i_lo, i_hi), it.on_grid(kin, "loser", i_lo, i_hi), 0, stride, bl)
    t = it.on_grid(kin, "winner", i_lo, i_hi)["t_s"].fillna(it.on_grid(kin, "loser", i_lo, i_hi)["t_s"])
    s["t_s"] = t.interpolate(limit_direction="both").to_numpy()
    return s.reset_index(drop=True)


def chases_in(s: pd.DataFrame, model: cm.Logistic, cols: list[str], dec: cm.Decoder) -> pd.DataFrame:
    feats = cm.directed_features(s)
    rows = []
    probs = {}
    for d in ("A", "B"):
        p = model.predict(feats[d][cols].to_numpy(float))
        probs[d] = p
        for a, b in cm.decode(p, dec):
            seg = p[a:b + 1]
            rows.append({"chaser": FISH[d], "chasee": FISH["B" if d == "A" else "A"], "i_start": a, "i_end": b,
                         "t_start": float(s["t_s"].iloc[a]), "t_end": float(s["t_s"].iloc[b]),
                         "duration_s": (b - a + 1) * cm.DT, "p_mean": float(np.nanmean(seg)),
                         "p_max": float(np.nanmax(seg)), "both_seen": float(np.isfinite(s["dist"].iloc[a:b + 1]).mean())})
    ev = pd.DataFrame(rows)
    if len(ev):
        ev["overlap"] = False
        for k, r in ev.iterrows():
            other = ev[(ev["chaser"] != r["chaser"]) & (ev["i_start"] <= r["i_end"]) & (ev["i_end"] >= r["i_start"])]
            ev.loc[k, "overlap"] = len(other) > 0
        ev = ev.sort_values("t_start", ignore_index=True)
    return ev


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--run", default="poses_r3")
    ap.add_argument("--name", default="model_v1")
    ap.add_argument("--l2", type=float, default=3.0)
    args = ap.parse_args()
    out = c8.OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    clips = c8.load_clips(args.sessions, args.run, "identity")
    X, y, cols = c8.matrix(clips)
    model = cm.Logistic(args.l2).fit(X, y)
    dec = cm.Decoder()
    (out / "model.json").write_text(json.dumps({
        "sessions": args.sessions, "clips": len(clips), "l2": args.l2, "decoder": dec.__dict__, "features": cols,
        "coef": model.coef.tolist(), "mu": model.mu.tolist(), "sd": model.sd.tolist()}, indent=1))
    stride = int(json.loads((c1.OUT / args.run / "params.json").read_text())["stride"])
    blpx = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    cov = pd.read_csv(it.OUT / args.run / "coverage.csv").set_index("recording")
    old = pd.read_csv(c1.OUT / args.run / "events.csv")
    pairs = c1.pairing()
    evs, rows = [], []
    for rec in cov.index:
        s = recording_series(it.load(args.run, rec), float(blpx[rec]), stride)
        ev = chases_in(s, model, cols, dec)
        evs.append(ev.assign(recording=rec))
        h_both = cov.loc[rec, "share_both_seen"] * cov.loc[rec, "frames"] * stride / c1.FPS / 3600
        n_w = int((ev["chaser"] == "winner").sum()) if len(ev) else 0
        n_l = int((ev["chaser"] == "loser").sum()) if len(ev) else 0
        rows.append({"recording": rec, "pairing": pairs.get(rec, ""), "hours_both_seen": h_both,
                     "chases": len(ev), "winner_chases": n_w, "loser_chases": n_l,
                     "overlaps": int(ev["overlap"].sum()) if len(ev) else 0,
                     "chase_s": float(ev["duration_s"].sum()) if len(ev) else 0.0,
                     "median_duration_s": float(ev["duration_s"].median()) if len(ev) else np.nan,
                     "chases_per_hour_both_seen": len(ev) / h_both if h_both > 0 else np.nan,
                     "old_detector_events": int((old["recording"] == rec).sum())})
        print(f"{rec}: {len(ev)} chases ({n_w} by the winner, {n_l} by the loser), old detector "
              f"{rows[-1]['old_detector_events']}", flush=True)
    ev_all = pd.concat(evs, ignore_index=True)
    ev_all.to_csv(out / "events.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(out / "trials.csv", index=False)

    t = t.sort_values("recording").reset_index(drop=True)
    x = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.bar(x - 0.2, t["winner_chases"] / t["hours_both_seen"], 0.4, color="#2b8cbe", label="winner chases")
    ax.bar(x - 0.2, t["loser_chases"] / t["hours_both_seen"], 0.4,
           bottom=t["winner_chases"] / t["hours_both_seen"], color="#d95f0e", label="loser chases")
    ax.bar(x + 0.2, t["old_detector_events"] / t["hours_both_seen"], 0.4, color="0.75", label="old detector")
    ax.set_xticks(x, [f"{r[5:10]} {p}" for r, p in zip(t["recording"], t["pairing"], strict=True)], rotation=60,
                  ha="right", fontsize=8)
    ax.set_ylabel("chases per hour with both fish in view")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{len(ev_all)} chases in {len(t)} trials; the winner chases in "
                 f"{int(t['winner_chases'].sum())}, the loser in {int(t['loser_chases'].sum())}", fontsize=9)
    ax = axes[1]
    if len(ev_all):
        ax.hist(ev_all["duration_s"], bins=np.arange(0, max(10, ev_all["duration_s"].quantile(0.99)) + 0.5, 0.5),
                color="0.4")
    ax.set_xlabel("chase duration (s)")
    ax.set_ylabel("chases")
    ax.set_title(f"median {ev_all['duration_s'].median():.1f} s (old detector: "
                 f"{old['duration_s'].median():.1f} s)", fontsize=9)
    fig.suptitle(f"Chaser-centric model (trained on {len(clips)} labelled clips) on the winner / loser "
                 f"trajectories, dark phase", fontsize=10)
    fig.savefig(out / "trials.png", dpi=150)
    plt.close(fig)
    print(t[["recording", "pairing", "hours_both_seen", "chases", "winner_chases", "loser_chases", "overlaps",
             "old_detector_events"]].round(2).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
