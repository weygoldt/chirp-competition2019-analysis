"""Kinematics around the author's labelled chase onsets and offsets (``c6`` session), from the chaser's side.

For every labelled chase: both fish's tracks over the clip (chained,
``c5_chase_timeseries.chain``), and the *chaser*: the fish that swims towards the
other more of the chase (mean cosine between its swimming direction and the line to
the other fish). Aligned at the labelled onsets and at the labelled offsets
(chases still going at the clip's end have no offset): speed and acceleration of
chaser and chasee, their distance, how fast it changes, and alignment. Also: how
the detector's event in the clip compares with the labels.

Author's observation (2026-11-20, 15 clips): chases start spontaneously, get
interrupted, or end in contact; an important cue is a subtle deceleration of the
chaser while the chasee keeps evading. The figure is meant to check that.

    .venv/bin/python experiments/video2022/c7_chase_labels_kinematics.py chase_v1 [--run poses_r3]

Output: ``$CHIRP_DATA/video2022/chases/labels/<session>/analysis/`` (``aligned.png``,
``clips.png``, ``chases.csv``).
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
import chases as ch

WIN_S = 3.0
CHASER, CHASEE = "#c0392b", "#7f8c8d"


def labels_of(session: Path) -> dict[str, dict]:
    out = {}
    for line in (session / "labels.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["clip_id"]] = r
    return out


def clip_series(row: pd.Series, kin: pd.DataFrame, tr: pd.DataFrame, spans: pd.DataFrame, bl: float, stride: int,
                p: ch.Params) -> pd.DataFrame:
    """Both fish (the clip's event leader = A, follower = B) over the clip, on the model's frame grid."""
    t0 = row["t0"]
    t1 = t0 + row["n_frames"] / c1.FPS
    a = c5.chain(spans, int(row["leader"]), t0 - 1, t1 + 1, bl, {int(row["follower"])})
    b = c5.chain(spans, int(row["follower"]), t0 - 1, t1 + 1, bl, set(a))
    lag_max = round(p.lag_max_s * c1.FPS / stride)
    i_lo, i_hi = int(row["frame0"]) // stride - lag_max, (int(row["frame0"]) + int(row["n_frames"])) // stride
    A, B = c5.series(kin, tr, a, i_lo, i_hi), c5.series(kin, tr, b, i_lo, i_hi)
    out = pd.DataFrame(index=A.index)
    out["t"] = (A.index * stride - int(row["frame0"])) / c1.FPS  # clip time (s)
    dt = stride / c1.FPS
    for name, s in (("A", A), ("B", B)):
        out[f"x_{name}"], out[f"y_{name}"] = s["x"].to_numpy(float), s["y"].to_numpy(float)
        out[f"vx_{name}"], out[f"vy_{name}"] = s["vx"].to_numpy(float), s["vy"].to_numpy(float)
        out[f"speed_{name}"] = s["speed"].to_numpy(float)
        out[f"accel_{name}"] = np.gradient(out[f"speed_{name}"].to_numpy(), dt)
    dx, dy = out["x_B"] - out["x_A"], out["y_B"] - out["y_A"]  # from A to B
    out["dist"] = np.hypot(dx, dy) / bl
    out["ddist"] = np.gradient(out["dist"].to_numpy(), dt)
    va, vb = out[["vx_A", "vy_A"]].to_numpy(), out[["vx_B", "vy_B"]].to_numpy()
    na, nb, nd = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1), np.hypot(dx, dy).to_numpy()
    out["cos"] = (va * vb).sum(1) / (na * nb + 1e-12)
    out["aim_A"] = (va[:, 0] * dx + va[:, 1] * dy) / (na * nd + 1e-12)  # A swims towards B
    out["aim_B"] = -(vb[:, 0] * dx + vb[:, 1] * dy) / (nb * nd + 1e-12)  # B swims towards A
    return out[out["t"] >= 0]


def roles(s: pd.DataFrame, on: float, off: float) -> tuple[str, str, float]:
    """(chaser, chasee, chaser's mean aim) within [on, off]: the fish that swims towards the other more."""
    w = s[(s["t"] >= on) & (s["t"] <= off)]
    aim_a, aim_b = np.nanmean(w["aim_A"]), np.nanmean(w["aim_B"])
    return ("A", "B", aim_a) if aim_a >= aim_b else ("B", "A", aim_b)


def aligned(s: pd.DataFrame, t_event: float, chaser: str, chasee: str, grid: np.ndarray) -> dict[str, np.ndarray]:
    rel = np.round((s["t"] - t_event) / 0.08).astype(int)
    idx = pd.Series(range(len(s)), index=rel)
    idx = idx[~idx.index.duplicated()]
    take = idx.reindex(np.round(grid / 0.08).astype(int)).to_numpy()
    ok = ~np.isnan(take)

    def col(c: str) -> np.ndarray:
        out = np.full(len(grid), np.nan)
        out[ok] = s[c].to_numpy()[take[ok].astype(int)]
        return out

    return {"speed_chaser": col(f"speed_{chaser}"), "speed_chasee": col(f"speed_{chasee}"),
            "accel_chaser": col(f"accel_{chaser}"), "accel_chasee": col(f"accel_{chasee}"), "dist": col("dist"),
            "ddist": col("ddist"), "cos": col("cos"), "aim_chaser": col(f"aim_{chaser}")}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--run", default="poses_r3")
    args = ap.parse_args()
    sess = c6.LABELS / args.session
    out = sess / "analysis"
    out.mkdir(exist_ok=True)
    root = c1.OUT / args.run
    params = json.loads((root / "params.json").read_text())
    stride = int(params["stride"])
    p = ch.Params(**{k: v for k, v in params.items() if k in ch.Params.__dataclass_fields__})
    truth = pd.read_csv(sess / "truth.csv").set_index("clip_id")
    labs = labels_of(sess)
    blpx = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    grid = np.round(np.arange(-round(WIN_S / 0.08), round(WIN_S / 0.08) + 1) * 0.08, 2)
    cache, series, rows = {}, {}, []
    on_al, off_al = [], []
    for cid, lab in labs.items():
        row = truth.loc[cid]
        rec = row["recording"]
        if rec not in cache:
            kin = pd.read_parquet(root / rec / "kinematics.parquet")
            cache = {rec: (kin, pd.read_parquet(root / rec / "tracks.parquet",
                                                columns=["track", "frame", "tail_x", "tail_y"]), c5.spans_of(kin))}
        kin, tr, spans = cache[rec]
        s = clip_series(row, kin, tr, spans, float(blpx[rec]), stride, p)
        series[cid] = (s, lab, row)
        for k, c in enumerate(lab["chases"]):
            chaser, chasee, aim = roles(s, c["onset_s"], c["offset_s"])
            on_al.append(aligned(s, c["onset_s"], chaser, chasee, grid))
            if not c["continues"]:
                off_al.append(aligned(s, c["offset_s"], chaser, chasee, grid))
            w = s[(s["t"] >= c["onset_s"]) & (s["t"] <= c["offset_s"])]
            rows.append({"clip_id": cid, "event": row["event"], "chase": k, "onset_s": c["onset_s"],
                         "offset_s": c["offset_s"], "continues": c["continues"], "duration_s": c["offset_s"] - c["onset_s"],
                         "chaser_is_detector_follower": chaser == "B", "chaser_aim": aim,
                         "seen_both_share": float(np.isfinite(w["dist"]).mean()) if len(w) else np.nan,
                         "detected_onset_s": row["detected_onset_s"], "detected_offset_s": row["detected_offset_s"]})
    d = pd.DataFrame(rows)
    d.to_csv(out / "chases.csv", index=False)

    panels = [("speed", "speed (BL/s)"), ("accel", "acceleration (BL/s²)"), ("dist", "distance (BL)"),
              ("ddist", "distance change (BL/s)"), ("cos", "same direction (cos)")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(9.5, 10), sharex=True, sharey="row", constrained_layout=True)
    for col, (al, name) in enumerate(((on_al, "labelled onset"), (off_al, "labelled offset"))):
        for r, (key, label) in enumerate(panels):
            ax = axes[r, col]
            pairs = ([("speed_chaser", CHASER), ("speed_chasee", CHASEE)] if key == "speed" else
                     [("accel_chaser", CHASER), ("accel_chasee", CHASEE)] if key == "accel" else
                     [("cos", "k"), ("aim_chaser", CHASER)] if key == "cos" else [(key, "k")])
            for c, color in pairs:
                m = np.array([a[c] for a in al])
                for tr_ in m:
                    ax.plot(grid, tr_, color=color, lw=0.5, alpha=0.25)
                ax.plot(grid, np.nanmedian(m, axis=0), color=color, lw=2.2)
            ax.axvline(0, color="k", ls=":", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
        axes[0, col].set_title(f"aligned to the {name} (n = {len(al)})", fontsize=9)
        axes[-1, col].set_xlabel(f"time from the {name} (s)")
    axes[0, 0].plot([], [], color=CHASER, lw=2, label="chaser")
    axes[0, 0].plot([], [], color=CHASEE, lw=2, label="chasee")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[-1, 0].plot([], [], color="k", lw=2, label="same direction")
    axes[-1, 0].plot([], [], color=CHASER, lw=2, label="chaser swims towards chasee")
    axes[-1, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"Author-labelled chases ({len(d)} chases in {len(labs)} clips): thin = each chase, thick = median",
                 fontsize=10)
    fig.savefig(out / "aligned.png", dpi=150)
    plt.close(fig)

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.55 * n), sharex=True, constrained_layout=True)
    for ax, (s, lab, row) in zip(np.atleast_1d(axes), series.values(), strict=True):
        ax.axvspan(row["detected_onset_s"], row["detected_offset_s"], color="0.85", zorder=0)
        for c in lab["chases"]:
            ax.axvspan(c["onset_s"], c["offset_s"], ymin=0.9, ymax=1.0, color="#c0392b", alpha=0.8)
        ax.plot(s["t"], s["speed_A"], color="#e6820e", lw=1, label="detector leader")
        ax.plot(s["t"], s["speed_B"], color="#1fb4d2", lw=1, label="detector follower")
        ax2 = ax.twinx()
        ax2.plot(s["t"], s["dist"], color="k", lw=0.8, ls="--")
        ax2.set_ylim(0, 4)
        ax2.tick_params(labelsize=6)
        ax.set_ylim(0, 6)
        ax.tick_params(labelsize=6)
        ax.text(0.005, 0.8, f"{row['event']}", transform=ax.transAxes, fontsize=6)
    axes[0].legend(fontsize=6, frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("clip time (s). Red bar: labelled chases; grey: detected event; lines: speed (BL/s, left); "
                        "dashed: distance (BL, right)", fontsize=8)
    fig.savefig(out / "clips.png", dpi=130)
    plt.close(fig)

    def window(al: list[dict], key: str, lo: float, hi: float) -> np.ndarray:
        sel = (grid >= lo) & (grid < hi)
        return np.array([np.nanmean(a[key][sel]) for a in al])

    for al, name in ((on_al, "onset"), (off_al, "offset")):
        for who in ("chaser", "chasee"):
            before = window(al, f"speed_{who}", -1.0, 0.0)
            after = window(al, f"speed_{who}", 0.0, 1.0)
            acc = window(al, f"accel_{who}", -0.5, 0.5)
            print(f"{name}: {who} speed 1 s before {np.nanmedian(before):.2f}, 1 s after {np.nanmedian(after):.2f} "
                  f"BL/s (slower after in {int(np.sum(after < before))} of {int(np.sum(np.isfinite(after - before)))}); "
                  f"accel +-0.5 s {np.nanmedian(acc):.2f} BL/s^2")
        dd = window(al, "ddist", 0.0, 1.0)
        print(f"{name}: distance change in the 1 s after: median {np.nanmedian(dd):.2f} BL/s, "
              f"growing in {int(np.sum(dd > 0))} of {int(np.sum(np.isfinite(dd)))}")
    print(d[["duration_s", "chaser_aim", "seen_both_share"]].describe().round(2).loc[["count", "50%", "min", "max"]])
    print("chaser = the detector's follower in", int(d.chaser_is_detector_follower.sum()), "of", len(d))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
