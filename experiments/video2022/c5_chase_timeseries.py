"""Time series of chase candidates: speed, acceleration, alignment, distance, and which chase criteria hold.

For each event (``c1_chases.py``), from ``BEFORE_S`` before its onset to ``AFTER_S``
after its offset. Tracks are often broken after a few seconds, so each fish's track
is chained to its continuation (:func:`chain`: the next track that starts within
``CHAIN_GAP_S`` near where it ended), to show what happens after the event; the
joins are marked. Panels:

1. speed of leader (orange) and follower (cyan), BL/s; the speed threshold dashed;
2. acceleration (change of speed), BL/s^2;
3. alignment: cosine of the angle between the two fish's swimming directions (black;
   1 = same direction) and whether the follower swims towards the leader (grey;
   cosine between its direction and the line to the leader);
4. distance: spine1 to spine1 (black), follower snout to leader tail (grey), follower
   to the leader's recent path (purple; the "on the path" criterion); thresholds dashed;
5. which chase criteria hold (both fast, close, aligned, heads apart, on the path)
   and all of them together (= chase frame).

The detected event is shaded. Also ``aligned.png``: medians (and quartiles) over all
events, aligned to the onsets and to the offsets.

    .venv/bin/python experiments/video2022/c5_chase_timeseries.py poses_r3 [--events ... | --videos]

``--videos``: the events rendered in ``<run>/videos/``. Output: ``$CHIRP_DATA/video2022/chases/<run>/timeseries/``.
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
import chases as ch

BEFORE_S = 3.0
AFTER_S = 6.0
CHAIN_GAP_S = 1.0
CHAIN_BL = 1.0
CHAIN_SPEED_BL = 3.0
LEADER, FOLLOWER = "#e6820e", "#1fb4d2"
CRITERIA = ("fast", "close", "aligned", "apart", "on_path", "chase")


def spans_of(kin: pd.DataFrame) -> pd.DataFrame:
    g = kin.sort_values("t_s").groupby("track")
    first, last = g.first(), g.last()
    return pd.DataFrame({"t_first": first["t_s"], "t_last": last["t_s"], "x_first": first["x"], "y_first": first["y"],
                         "x_last": last["x"], "y_last": last["y"]})


def chain(spans: pd.DataFrame, track: int, t_lo: float, t_hi: float, bl: float, exclude: set) -> list[int]:
    """The track and its continuations (forward to t_hi, backward to t_lo), in time order."""
    seq = [track]
    for forward in (True, False):
        cur = track
        while (spans.loc[cur, "t_last"] < t_hi) if forward else (spans.loc[cur, "t_first"] > t_lo):
            e = spans.loc[cur]
            if forward:
                cand = spans[(spans["t_first"] > e["t_last"]) & (spans["t_first"] <= e["t_last"] + CHAIN_GAP_S)]
                gap = cand["t_first"] - e["t_last"]
                d = np.hypot(cand["x_first"] - e["x_last"], cand["y_first"] - e["y_last"])
            else:
                cand = spans[(spans["t_last"] < e["t_first"]) & (spans["t_last"] >= e["t_first"] - CHAIN_GAP_S)]
                gap = e["t_first"] - cand["t_last"]
                d = np.hypot(cand["x_last"] - e["x_first"], cand["y_last"] - e["y_first"])
            ok = (d <= (CHAIN_BL + CHAIN_SPEED_BL * gap) * bl) & ~cand.index.isin(list(exclude | set(seq)))
            if not ok.any():
                break
            cur = d[ok].idxmin()
            seq = [*seq, cur] if forward else [cur, *seq]
    return seq


def series(kin: pd.DataFrame, tr: pd.DataFrame, tracks: list[int], i_lo: int, i_hi: int) -> pd.DataFrame:
    """One fish's chained kinematics on the grid i_lo..i_hi (NaN where unseen), with its tail and track id."""
    k = kin[kin["track"].isin(tracks) & kin["i"].between(i_lo, i_hi)]
    k = k.merge(tr[["track", "frame", "tail_x", "tail_y"]], on=["track", "frame"], how="left")
    k = k.sort_values("i").drop_duplicates("i").set_index("i")
    return k.reindex(np.arange(i_lo, i_hi + 1))


def features(L: pd.DataFrame, F: pd.DataFrame, bl: float, lag_max: int, dt: float, p: ch.Params) -> pd.DataFrame:
    t = L["t_s"].fillna(F["t_s"]).to_numpy(float)
    out = pd.DataFrame(index=L.index)
    out["t"] = t
    for name, s in (("L", L), ("F", F)):
        out[f"speed_{name}"] = s["speed"].to_numpy(float)
        out[f"accel_{name}"] = np.gradient(out[f"speed_{name}"].to_numpy(), dt)
    vl, vf = L[["vx", "vy"]].to_numpy(float), F[["vx", "vy"]].to_numpy(float)
    pl_, pf = L[["x", "y"]].to_numpy(float), F[["x", "y"]].to_numpy(float)
    out["cos"] = (vl * vf).sum(1) / (np.linalg.norm(vl, axis=1) * np.linalg.norm(vf, axis=1) + 1e-12)
    to_l = pl_ - pf
    out["aim"] = (vf * to_l).sum(1) / (np.linalg.norm(vf, axis=1) * np.linalg.norm(to_l, axis=1) + 1e-12)
    out["dist"] = np.linalg.norm(to_l, axis=1) / bl
    out["gap"] = np.hypot(F["sx"] - L["tail_x"], F["sy"] - L["tail_y"]).to_numpy(float) / bl
    sep = np.hypot(F["sx"] - L["sx"], F["sy"] - L["sy"]).to_numpy(float) / bl
    out["path"] = ch.path_distance(pf[:, 0], pf[:, 1], pl_[:, 0], pl_[:, 1], lag_max)[0] / bl
    out.loc[~np.isfinite(out["dist"]), "path"] = np.nan
    both = np.isfinite(out["dist"]).to_numpy()
    out["fast"] = both & (out["speed_L"] >= p.v_min) & (out["speed_F"] >= p.v_min)
    out["close"] = both & (out["dist"] <= p.d_max)
    out["aligned"] = both & (out["cos"] >= p.cos_min)
    out["apart"] = both & (sep >= p.min_sep)
    out["on_path"] = both & (out["path"] <= p.r_path)
    out["chase"] = out[["fast", "close", "aligned", "apart", "on_path"]].all(axis=1)
    out["L_track"], out["F_track"] = L["track"].to_numpy(), F["track"].to_numpy()
    return out


def event_features(e: pd.Series, kin: pd.DataFrame, tr: pd.DataFrame, spans: pd.DataFrame, bl: float, stride: int,
                   p: ch.Params, before: float = BEFORE_S, after: float = AFTER_S) -> pd.DataFrame:
    lag_max = round(p.lag_max_s * c1.FPS / stride)
    n_b, n_a = round(before * c1.FPS / stride), round(after * c1.FPS / stride)
    i_lo, i_hi = int(e["i_start"]) - n_b - lag_max, int(e["i_end"]) + n_a
    t_lo, t_hi = e["t_start"] - before - p.lag_max_s, e["t_end"] + after
    lt = chain(spans, int(e["leader"]), t_lo, t_hi, bl, {int(e["follower"])})
    ft = chain(spans, int(e["follower"]), t_lo, t_hi, bl, set(lt))
    f = features(series(kin, tr, lt, i_lo, i_hi), series(kin, tr, ft, i_lo, i_hi), bl, lag_max, stride / c1.FPS, p)
    f = f.loc[int(e["i_start"]) - n_b:]
    f["rel"] = (f.index - int(e["i_start"])) * stride / c1.FPS
    f["rel_end"] = (f.index - int(e["i_end"])) * stride / c1.FPS
    return f


def plot_event(e: pd.Series, f: pd.DataFrame, p: ch.Params, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(7.5, 8.2), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.2, 0.8]})
    x = f["rel"].to_numpy()
    end = (e["i_end"] - e["i_start"]) * (f["rel"].iloc[1] - f["rel"].iloc[0]) if len(f) > 1 else 0
    for ax in axes:
        ax.axvspan(0, end, color="0.88", zorder=0)
        for col, k in (("L_track", LEADER), ("F_track", FOLLOWER)):
            tracks = f[col].to_numpy()
            ok = ~pd.isna(tracks)
            joins = np.flatnonzero(ok[1:] & ok[:-1] & (tracks[1:] != tracks[:-1]))
            for j in joins:
                ax.axvline(x[j + 1], color=k, lw=0.8, ls=":")
    a = axes[0]
    a.plot(x, f["speed_L"], color=LEADER, label="leader")
    a.plot(x, f["speed_F"], color=FOLLOWER, label="follower")
    a.axhline(p.v_min, color="0.4", ls="--", lw=0.8)
    a.set_ylabel("speed\n(BL/s)")
    a.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    a = axes[1]
    a.plot(x, f["accel_L"], color=LEADER)
    a.plot(x, f["accel_F"], color=FOLLOWER)
    a.axhline(0, color="0.4", lw=0.6)
    a.set_ylabel("acceleration\n(BL/s²)")
    a = axes[2]
    a.plot(x, f["cos"], color="k", label="same direction")
    a.plot(x, f["aim"], color="0.55", label="follower swims towards leader")
    a.axhline(p.cos_min, color="0.4", ls="--", lw=0.8)
    a.set_ylim(-1.05, 1.05)
    a.set_ylabel("alignment\n(cos)")
    a.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    a = axes[3]
    a.plot(x, f["dist"], color="k", label="spine1 to spine1")
    a.plot(x, f["gap"], color="0.55", label="follower snout to leader tail")
    a.plot(x, f["path"], color="#7b3294", label="follower to leader's path")
    a.axhline(p.r_path, color="#7b3294", ls="--", lw=0.8)
    a.axhline(p.d_max, color="0.4", ls="--", lw=0.8)
    a.set_ylim(0, max(3.2, float(np.nanmax(f[["dist"]].to_numpy(), initial=0)) * 1.05))
    a.set_ylabel("distance\n(BL)")
    a.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    a = axes[4]
    for r, c in enumerate(CRITERIA):
        on = f[c].to_numpy(bool)
        a.scatter(x[on], np.full(on.sum(), r), marker="|", s=40, color="k" if c == "chase" else "0.45")
    a.set_yticks(range(len(CRITERIA)), ["both fast", "close", "aligned", "heads apart", "on path", "CHASE"],
                 fontsize=7)
    a.set_ylim(len(CRITERIA) - 0.5, -0.5)
    a.set_xlabel("time from the detected onset (s); shaded: detected event; dotted: track joins")
    fig.suptitle(f"{e['event']}  (recording time {e['t_start'] / 60:.1f} min, detected {e['duration_s']:.1f} s)",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aligned(feats: list[pd.DataFrame], path: Path, p: ch.Params) -> None:
    import matplotlib.pyplot as plt

    rows = [("speed_L", "speed_F", "speed (BL/s)"), ("cos", None, "same direction (cos)"),
            ("dist", "gap", "distance (BL)"), ("path", None, "follower to leader's path (BL)")]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 8), sharex="col", sharey="row", constrained_layout=True)
    for col, (key, lo, hi) in enumerate((("rel", -BEFORE_S, 3.0), ("rel_end", -3.0, AFTER_S))):
        grid = np.round(np.arange(round(lo / 0.08), round(hi / 0.08) + 1) * 0.08, 2)
        for r, (c1_, c2_, label) in enumerate(rows):
            ax = axes[r, col]
            for c, color in ((c1_, LEADER if c1_ == "speed_L" else "k"), (c2_, FOLLOWER if c2_ == "speed_F" else "0.55")):
                if c is None:
                    continue
                m = np.array([f.set_index(f[key].round(2))[c].reindex(grid).to_numpy() for f in feats])
                q1, med, q3 = np.nanpercentile(m, [25, 50, 75], axis=0)
                ax.fill_between(grid, q1, q3, color=color, alpha=0.15, lw=0)
                ax.plot(grid, med, color=color, lw=1.6)
            ax.axvline(0, color="k", lw=0.8, ls=":")
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
        axes[-1, col].set_xlabel("time from detected onset (s)" if key == "rel" else "time from detected offset (s)")
    axes[0, 0].plot([], [], color=LEADER, label="leader")
    axes[0, 0].plot([], [], color=FOLLOWER, label="follower")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[2, 0].plot([], [], color="k", label="spine1 to spine1")
    axes[2, 0].plot([], [], color="0.55", label="follower snout to leader tail")
    axes[2, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"All {len(feats)} detected chases: medians and quartiles, aligned to onset (left) and offset (right)",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--events", nargs="*")
    ap.add_argument("--videos", action="store_true", help="the events rendered in <run>/videos/")
    args = ap.parse_args()
    root = c1.OUT / args.run
    params = json.loads((root / "params.json").read_text())
    stride = int(params["stride"])
    p = ch.Params(**{k: v for k, v in params.items() if k in ch.Params.__dataclass_fields__})
    ev = pd.read_csv(root / "events.csv")
    if args.videos:
        show = set(pd.read_csv(root / "videos" / "rendered.csv")["event"])
    else:
        show = set(args.events or [])
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    out = root / "timeseries"
    out.mkdir(exist_ok=True)
    feats = []
    for rec, g in ev.groupby("recording"):
        kin = pd.read_parquet(root / rec / "kinematics.parquet")
        tr = pd.read_parquet(root / rec / "tracks.parquet", columns=["track", "frame", "tail_x", "tail_y"])
        spans = spans_of(kin)
        for _, e in g.iterrows():
            f = event_features(e, kin, tr, spans, float(bl[rec]), stride, p)
            feats.append(f)
            if e["event"] in show:
                plot_event(e, f, p, out / f"{e['event']}.png")
        print(f"{rec}: {len(g)} events", flush=True)
    plot_aligned(feats, out / "aligned.png", p)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
