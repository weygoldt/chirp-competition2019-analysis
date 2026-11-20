"""Figures and numbers of the electrode identity of the 2022 video tracks (``d1_identity.py``).

    .venv/bin/python experiments/video2022/d2_identity_figures.py [electrodes evidence size frames chases localize]

Outputs in ``$CHIRP_DATA/video2022/identity/poses_r3/figures/`` (and ``validation/`` tables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import d1_identity as d1  # noqa: E402
import identity as idn  # noqa: E402

FIG = d1.RES / "figures"
VAL = d1.RES / "validation"
WIN, LOS, UNK = "#d95f02", "#1b6ca8", "#9e3ea8"
COLOR = {"winner": WIN, "loser": LOS, "unknown": UNK}
ARENA = (136, 236, 1885, 1330)


def plt_():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def recs_done() -> list[str]:
    return sorted(p.parent.name for p in d1.RES.glob("*/track_identity.csv"))


def pair_label(rec: str) -> str:
    m = d1.meta()
    return f"{rec[5:10]} {m['Pairing'].get(rec, 'control')}"


# ----------------------------------------------------------------------------- electrodes
def fig_electrodes() -> None:
    plt = plt_()
    from PIL import Image

    recs = recs_done()
    bg = np.asarray(Image.open(d1.b1.OUT / "background" / f"{recs[0]}.png"))
    models = {r: d1.load_models(r) for r in recs}
    rows = [row for r in recs for row in d1.heldout_fit(r)]
    held = pd.DataFrame(rows)
    held.to_csv(VAL / "heldout_fit.csv", index=False)
    fig = plt.figure(figsize=(13, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255)
    E0 = idn.ELECTRODES_IMAGE
    for r in recs:
        m = models[r]["all"]
        if m is None:
            continue
        E = np.asarray(m["electrodes"])
        g = np.asarray(m["gain"])
        ok = 20 * np.log10(g) > -40  # a dead channel has no position worth showing
        ax.scatter(E[ok, 0], E[ok, 1], s=6, c="tab:red", alpha=0.6, lw=0)
        ax.scatter(*m["ref"], s=14, marker="x", c="tab:purple")
    ax.scatter(E0[:, 0], E0[:, 1], s=160, facecolors="none", edgecolors="yellow", lw=1.5)
    for i, (x, y) in enumerate(E0):
        ax.annotate(f"{i}", (x, y), xytext=(10, -10), textcoords="offset points", color="yellow", fontsize=11,
                    weight="bold")
    refs = np.array([models[r]["all"]["ref"] for r in recs if models[r]["all"]])
    ax.annotate("reference (fitted),\noutside the view", refs.mean(0), xytext=(250, 1480), color="tab:purple",
                arrowprops={"arrowstyle": "->", "color": "tab:purple"}, fontsize=9)
    ax.set_xlim(-50, 2098)
    ax.set_ylim(max(1720, refs[:, 1].max() + 40), -10)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Electrodes 0-10 (yellow: holders seen in the video; red dots: positions fitted to the signals,"
                 " one per recording)", fontsize=9)
    ax = fig.add_subplot(gs[0, 2])
    held = held.groupby("recording")[["explained_better", "explained_swapped", "decisive"]].mean()
    y = np.arange(len(held))
    ax.scatter(held["explained_better"], y, c="k", s=18, label="right assignment")
    ax.scatter(held["explained_swapped"], y, facecolors="none", edgecolors="k", s=18, label="swapped")
    for yi, (_, r) in zip(y, held.iterrows(), strict=True):
        ax.text(1.02, yi, f"{100 * r['decisive']:.0f} %", va="center", fontsize=7)
    ax.set_yticks(y, [pair_label(r) for r in held.index], fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.set_xlabel("share of the measured pattern explained\n(median, fish >= 3 BL apart, held-out half)")
    ax.legend(loc="center left", fontsize=7, frameon=False)
    ax.set_title("Fitted field explains the patterns\n(right: share of windows decided by > 0.3)", fontsize=9)
    fig.suptitle("The 11 electrodes sit around the tank, not on a line: 8 on the walls, 3 at the shelter", fontsize=11)
    fig.savefig(FIG / "electrodes.png", dpi=110)
    plt.close(fig)
    pos = []
    for r in recs:
        m = models[r]["all"]
        if m is None:
            continue
        for i in range(idn.N_CH):
            pos.append({"recording": r, "channel": i, "name": idn.ELECTRODE_NAMES[i], "x_image": E0[i, 0],
                        "y_image": E0[i, 1], "x_fit": m["electrodes"][i][0], "y_fit": m["electrodes"][i][1],
                        "gain_db": 20 * np.log10(m["gain"][i])})
        pos.append({"recording": r, "channel": "ref", "name": "reference", "x_fit": m["ref"][0], "y_fit": m["ref"][1]})
    pd.DataFrame(pos).to_csv(d1.RES / "electrodes_fitted.csv", index=False)
    pd.DataFrame({"channel": range(idn.N_CH), "name": idn.ELECTRODE_NAMES, "x_px": E0[:, 0],
                  "y_px": E0[:, 1]}).to_csv(d1.RES / "electrodes.csv", index=False)


# ----------------------------------------------------------------------------- evidence vs distance
def pairing_table() -> pd.DataFrame:
    p = pd.read_csv(VAL / "pairing_test.csv")
    lab = pd.cut(p["dist_bl"], d1.PAIR_BINS, right=False)
    rows = []
    for (iv, tandem), g in p.groupby([lab, "tandem"], observed=True):
        rows.append({"dist_from": iv.left, "dist_to": iv.right, "tandem": bool(tandem), "windows": len(g),
                     "accuracy": g["correct"].mean()})
    for iv, g in p.groupby(lab, observed=True):
        rows.append({"dist_from": iv.left, "dist_to": iv.right, "tandem": "all", "windows": len(g),
                     "accuracy": g["correct"].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(VAL / "pairing_by_distance.csv", index=False)
    return t


def fig_evidence() -> None:
    plt = plt_()
    pt = pairing_table()
    tb = pd.read_csv(VAL / "test_blocks.csv")
    ci = pd.read_csv(VAL / "clashes_by_distance_independent.csv")
    cj = pd.read_csv(VAL / "clashes_by_distance.csv")
    mk = pd.read_parquet(VAL / "masked.parquet")
    mk = mk[~mk["masked"]]
    pc = pd.read_csv(VAL / "pieces.csv")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    ax = axs[0]
    for key, lab, sty in (("all", "all configurations", "o-"), ("True", "one behind the other (as in a chase)", "s--")):
        g = pt[pt["tandem"].astype(str) == key]
        mid = (g["dist_from"] + g["dist_to"]) / 2
        ax.plot(mid, g["accuracy"], sty, label=lab, ms=4, color="k" if key == "all" else "tab:red")
    order = ["[0.0, 0.5)", "[0.5, 1.0)", "[1.0, 1.5)", "[1.5, 2.0)", "[2.0, 3.0)"]
    mids = [0.25, 0.75, 1.25, 1.75, 2.5]
    a = tb.groupby("dist")["correct"].agg(["mean", "size"]).reindex(order)
    ok = a["size"] >= 50
    ax.plot(np.array(mids)[ok.to_numpy()], a.loc[ok, "mean"], "^:", c="0.5", ms=5,
            label="real 1-s blocks, fish not touching")
    ax.axhline(0.5, c="0.7", ls=":", lw=0.8)
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("distance between the two fish (spine1 to spine1, body lengths)")
    ax.set_ylabel("share pointing to the right fish")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title(f"Evidence of a single window (0.16 s), {len(pd.read_csv(VAL / 'pairing_test.csv'))} pairs\n"
                 "of real moments of each fish combined at that distance", fontsize=9)
    ax = axs[1]
    names = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">=3"]
    x = np.arange(len(names))
    for c, lab, sty in ((ci, "each track judged on its own", "o-"), (cj, "joint (method used)", "s--")):
        ax.plot(x[:len(c)], 100 * c["clash_rate_of_known"], sty, ms=4, label=lab,
                c="k" if "own" in lab else "tab:blue")
    ax.set_ylim(bottom=-0.6)
    for xi, n in zip(x, ci["both_known"], strict=False):
        ax.text(xi, -0.5, f"n={int(n)}", fontsize=6, ha="center", va="bottom")
    ax.set_xticks(x, names)
    ax.set_xlabel("distance between the two fish (body lengths)")
    ax.set_ylabel("% of frames: both fish get the same identity")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title("Two fish in view: how often both are called the same\n(numbers: frames with both labelled)",
                 fontsize=9)
    ax = axs[2]
    bins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001])
    for col, lab, mk_ in (("p_heldout", "pieces of tracks, beta fitted on the other recordings", "o"),):
        pcb = pc.assign(b=pd.cut(pc[col], bins))
        g = pcb.groupby("b", observed=True).agg(p=(col, "mean"), acc=("correct", "mean"), n=("correct", "size"))
        ax.plot(g["p"], g["acc"], mk_ + "-", c="k", label=lab)
        for _, r in g.iterrows():
            ax.text(r["p"], r["acc"] - 0.025, f"{int(r['n'])}", fontsize=6, ha="center")
    ax.plot([0.5, 1], [0.5, 1], c="0.6", lw=0.8)
    ax.set_xlabel("stated probability of being right")
    ax.set_ylabel("observed share right")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Calibration of the stated confidence", fontsize=9)
    fig.suptitle("The electrodes tell the two fish apart in most single windows even when they are close;"
                 " errors grow below one body length", fontsize=10)
    fig.savefig(FIG / "evidence_by_distance.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- body size
def size_table() -> pd.DataFrame:
    """Per recording: skeleton length (px) of frames labelled winner vs loser (straight, confident fish),
    and the measured body sizes of the two fish (meta2.csv, cm)."""
    m2 = pd.read_csv(d1.REC_2022 / "raw" / "meta2.csv", encoding="utf-7").rename(columns={"Unnamed: 0": "recording"})
    m2 = m2.set_index("recording")
    rows = []
    for rec in recs_done():
        o = pd.read_parquet(d1.RES / rec / "observations_labelled.parquet")
        tr = d1.load_tracks(rec)
        import chases as ch

        P = idn.poses(tr)
        chord = np.hypot(*(P[:, -1] - P[:, 0]).T)
        length = ch.skeleton_length(tr)
        tr = tr.assign(length=length, straight=chord / length)
        x = o.merge(tr[["frame", "track", "length", "straight"]], on=["frame", "track"])
        x = x[(x["straight"] >= 0.95) & (x["n_vis"] == 2) & (x["dist_bl"] >= 1.5)]
        r = {"recording": rec, "pairing": d1.meta()["Pairing"].get(rec, "control")}
        for lab in ("winner", "loser"):
            s = x[x["label"] == lab]
            r[f"{lab}_frames"] = len(s)
            r[f"{lab}_px"] = float(s["length"].median()) if len(s) else np.nan
        if rec in m2.index:
            mm = m2.loc[rec]
            win_first = mm["Winner_pitch"] == mm["Fish1_pitch"]
            r["winner_cm"] = float(mm["Fish1_size"] if win_first else mm["Fish2_size"])
            r["loser_cm"] = float(mm["Fish2_size"] if win_first else mm["Fish1_size"])
        rows.append(r)
    t = pd.DataFrame(rows)
    t["ratio_video"] = t["winner_px"] / t["loser_px"]
    t["ratio_measured"] = t["winner_cm"] / t["loser_cm"]
    return t


def fig_size() -> None:
    plt = plt_()
    t = size_table()
    t.to_csv(VAL / "size_check.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    tt = t.dropna(subset=["ratio_measured"])
    ax.scatter(tt["ratio_measured"], tt["ratio_video"], c="k", s=20)
    for _, r in tt.iterrows():
        ax.annotate(r["pairing"], (r["ratio_measured"], r["ratio_video"]), xytext=(3, 3), textcoords="offset points",
                    fontsize=7)
    lim = [0.93, 1.2]
    ax.plot(lim, lim, c="0.6", lw=0.8)
    ax.axhline(1, c="0.8", lw=0.8)
    ax.axvline(1, c="0.8", lw=0.8)
    ax.set_xlim(lim)
    ax.set_xlabel("winner / loser body length, measured by hand (meta2.csv)")
    ax.set_ylabel("winner / loser skeleton length in the video\n(frames labelled by the electrodes)")
    ax.set_title("Where the winner is the larger fish, the frames labelled\n'winner' show the longer fish"
                 " (independent check)", fontsize=10)
    fig.savefig(FIG / "size_check.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- example frames
def fig_frames(n_frames: int = 8, seed: int = 1) -> None:
    plt = plt_()
    import frames as frm
    from recordings2022 import list_recordings

    rng = np.random.default_rng(seed)
    recs = recs_done()
    picks = []
    kinds = ["far", "near", "touch", "far", "near", "touch", "one", "near"]
    for i, kind in enumerate(kinds[:n_frames]):
        rec = recs[(i * 5 + 2) % len(recs)]
        o = pd.read_parquet(d1.RES / rec / "observations_labelled.parquet")
        two = o[o["other"] >= 0]
        sel = {"far": two[two["dist_bl"] >= 3], "near": two[(two["dist_bl"] >= 1) & (two["dist_bl"] < 2)],
               "touch": two[two["contact_bl"] <= 0.3], "one": o[o["n_vis"] == 1]}[kind]
        if not len(sel):
            continue
        picks.append((rec, int(sel["frame"].iloc[rng.integers(len(sel))]), kind))
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.2), constrained_layout=True)
    videos = {r.name: r.video for r in list_recordings()}
    for ax, (rec, f, kind) in zip(axs.flat, picks, strict=False):
        img = dict(frm.read_seek(videos[rec], [f]))[f]
        o = pd.read_parquet(d1.RES / rec / "observations_labelled.parquet")
        tr = d1.load_tracks(rec)
        d = tr[tr["frame"] == f]
        lab = o[o["frame"] == f].set_index("track")
        ax.imshow(img, cmap="gray")
        pat = d1.load_patterns(rec)
        t = float(d["t_s"].iloc[0])
        k = idn.match_windows(np.array([t]), pat["t"])[0]
        models = d1.load_models(rec)
        mdl = d1.model_for(rec, int(d1.parity(np.array([t]))[0]), models)[0]
        pts = [idn.poses(d).reshape(-1, 2)]
        for p, (_, row) in zip(idn.poses(d), d.iterrows(), strict=True):
            lb = lab["label"].get(row["track"], "unknown")
            ax.plot(p[:, 0], p[:, 1], "-", c=COLOR[lb], lw=2.5)
            ax.plot(*p[0], "o", c=COLOR[lb], ms=4)
            ax.plot(*idn.body_point(p[None], mdl.center)[0], "o", mfc="none", mec=COLOR[lb], ms=9, mew=1.5)
        # electric-only estimate of each fish's dipole centre
        for role in ("winner", "loser"):
            v = pat[role][k:k + 1]
            if np.isfinite(v).all():
                c, _s = idn.localize(mdl, v, d1.body_length(rec), ARENA)
                ax.plot(*c[0], "+", c=COLOR[role], ms=20, mew=2.5)
                pts.append(c)
        pts = np.concatenate(pts)
        lo, hi = pts.min(0) - 150, pts.max(0) + 150
        span = max(hi[0] - lo[0], (hi[1] - lo[1]) * 4 / 3)
        mid = (lo + hi) / 2
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] + span * 3 / 8, mid[1] - span * 3 / 8)
        ax.set_title(f"{pair_label(rec)}, t = {t:.0f} s ({kind})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Video fish coloured by the identity assigned (orange winner, blue loser, purple unknown; circle ="
                 " model's EOD centre on the body). + = where the electrodes alone place each fish's EOD centre"
                 " (typical error 0.5 body lengths)", fontsize=10)
    fig.savefig(FIG / "example_frames.png", dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- localisation error
def localization_error(max_windows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Electric-only position of each fish against the video fish of the same label (far-apart frames and
    close frames), in body lengths and cm (the recording's median body size, meta2)."""
    rng = np.random.default_rng(seed)
    rows = []
    for rec in recs_done():
        o = pd.read_parquet(d1.RES / rec / "observations_labelled.parquet")
        o = o[(o["label"] != "unknown")]
        pat = d1.load_patterns(rec)
        models = d1.load_models(rec)
        tr = d1.load_tracks(rec).set_index(["frame", "track"])
        bl = d1.body_length(rec)
        for kind, sel in (("apart (>= 3 BL)", o[o["dist_bl"] >= 3]), ("close (< 1 BL)", o[o["dist_bl"] < 1]),
                          ("one fish in view", o[o["n_vis"] == 1])):
            if not len(sel):
                continue
            s = sel.iloc[rng.choice(len(sel), min(max_windows, len(sel)), replace=False)]
            for h in (0, 1):
                ss = s[d1.parity(s["t_s"].to_numpy()) == h]
                if not len(ss):
                    continue
                mdl = d1.model_for(rec, h, models)[0]
                for lab in ("winner", "loser"):
                    x = ss[ss["label"] == lab]
                    if not len(x):
                        continue
                    v = pat[lab][x["k"].to_numpy()]
                    c, _sc = idn.localize(mdl, v, bl, ARENA)
                    pose = idn.poses(tr.loc[list(zip(x["frame"], x["track"], strict=True))].reset_index())
                    true_c = idn.body_point(pose, mdl.center)
                    err = np.hypot(*(c - true_c).T) / bl
                    rows += [{"recording": rec, "kind": kind, "role": lab, "err_bl": e} for e in err if np.isfinite(e)]
    return pd.DataFrame(rows)


def stage_localize() -> None:
    t = localization_error()
    t.to_csv(VAL / "localization_error.csv", index=False)
    print(t.groupby("kind")["err_bl"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2).to_string())


# ----------------------------------------------------------------------------- chases
def chase_size_table() -> pd.DataFrame:
    """Per trial (video only): share of chase events in which the follower has the longer skeleton,
    next to the measured winner / loser size ratio."""
    ev = pd.read_csv(d1.RES / "events_identity.csv")
    sz = pd.read_csv(VAL / "size_check.csv").set_index("recording")
    rows = []
    for rec, g in ev.groupby("recording"):
        g = g.dropna(subset=["skeleton_px_leader", "skeleton_px_follower"])
        k = int((g["skeleton_px_follower"] > g["skeleton_px_leader"]).sum())
        lo, hi = d1.wilson(k, len(g))
        rows.append({"recording": rec, "pairing": g["pairing"].iloc[0] if len(g) else "", "events": len(g),
                     "follower_longer": k, "share_follower_longer": k / len(g) if len(g) else np.nan,
                     "ci_low": lo, "ci_high": hi,
                     "ratio_measured": sz["ratio_measured"].get(rec, np.nan)})
    t = pd.DataFrame(rows)
    t.to_csv(VAL / "chase_size_check.csv", index=False)
    return t


def fig_chases() -> None:
    plt = plt_()
    per = pd.read_csv(d1.RES / "chases_by_trial.csv").sort_values("recording")
    ev = pd.read_csv(d1.RES / "events_identity.csv")
    cs = chase_size_table()
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axs[0]
    y = np.arange(len(per))
    ok = per["identified"] > 0
    s = per.loc[ok, "share_winner_chases"]
    ax.errorbar(s, y[ok], xerr=[np.clip(s - per.loc[ok, "ci_low"], 0, None),
                                np.clip(per.loc[ok, "ci_high"] - s, 0, None)], fmt="o", c="k", capsize=2)
    for yi, (_, r) in zip(y, per.iterrows(), strict=True):
        ax.text(1.03, yi, f"{int(r['winner_chases'])}/{int(r['identified'])} of {int(r['events'])}", va="center",
                fontsize=7)
    ax.set_yticks(y, [f"{r[5:10]} {p}" for r, p in zip(per["recording"], per["pairing"], strict=True)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.3)
    ax.axvline(0.5, c="0.8", lw=0.8)
    ax.set_xlabel("share of identified chases in which the winner chases (95 % CI)")
    ax.set_title("Electrodes: who chases whom\n(right: winner chases / identified / all events)", fontsize=9)
    ax = axs[1]
    conf = (ev["p_outside_leader"] >= 0.9) | (ev["p_outside_leader"] <= 0.1)
    e = ev[conf]
    agree = (e["S_inside_leader"] > 0) == (e["p_outside_leader"] > 0.5)
    ax.scatter(e["S_inside_leader"], e["p_outside_leader"], s=8, c=np.where(agree, "k", "tab:red"))
    ax.axvline(0, c="0.8", lw=0.8)
    ax.axhline(0.5, c="0.8", lw=0.8)
    ax.set_xlabel("evidence inside the chase alone\n(s of evidence that the leader is the winner)")
    ax.set_ylabel("P(leader = winner) from before / after the chase")
    ax.set_title(f"Check 1: inside and outside the chase agree in\n{int(agree.sum())} of {len(e)} events"
                 f" ({agree.mean() * 100:.1f} %; red: disagree)", fontsize=9)
    ax = axs[2]
    c = cs[cs["events"] >= 10]
    ax.errorbar(c["ratio_measured"], c["share_follower_longer"],
                yerr=[np.clip(c["share_follower_longer"] - c["ci_low"], 0, None),
                      np.clip(c["ci_high"] - c["share_follower_longer"], 0, None)], fmt="o", c="k", capsize=2)
    for i, (_, r) in enumerate(c.sort_values(["ratio_measured", "share_follower_longer"]).iterrows()):
        ax.annotate(f"{r['pairing']} (n={int(r['events'])})", (r["ratio_measured"], r["share_follower_longer"]),
                    xytext=(5, -12 + 9 * (i % 3)), textcoords="offset points", fontsize=7)
    ax.axhline(0.5, c="0.8", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("winner / loser body length, measured by hand")
    ax.set_ylabel("share of chases: the follower is the longer fish")
    ax.set_title("Check 2, video only: the chaser is the longer fish\nwhere the winner is clearly larger"
                 " (trials with >= 10 events)", fontsize=9)
    k, n = int(per["winner_chases"].sum()), int(per["identified"].sum())
    fig.suptitle(f"The winner is the chaser in {k} of {n} identified chases ({100 * k / n:.1f} %;"
                 f" {int(per['events'].sum())} chase events in {len(per)} trials)", fontsize=11)
    fig.savefig(FIG / "chases_by_trial.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f} %"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, float) and np.isfinite(v) else
                         ("" if isinstance(v, float) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def recording_table() -> pd.DataFrame:
    seg = pd.read_csv(VAL / "segments_all.csv")
    held = pd.read_csv(VAL / "heldout_fit.csv").groupby("recording")[["explained_better", "explained_swapped",
                                                                          "decisive"]].mean()
    per = pd.read_csv(d1.RES / "chases_by_trial.csv").set_index("recording")
    rows = []
    for rec, g in seg.groupby("recording"):
        tid = g.groupby("track")["track_identity"].first()
        m = d1.load_models(rec)["all"]
        dead = [str(i) for i, gn in enumerate(m["gain"]) if 20 * np.log10(gn) < -40] if m else []
        rows.append({"recording": rec, "pairing": d1.meta()["Pairing"].get(rec, ""),
                     "frames": int(g["n_frames"].sum()),
                     "labelled": float(g.loc[g["identity"] != "unknown", "n_frames"].sum() / g["n_frames"].sum()),
                     "winner": float(g.loc[g["identity"] == "winner", "n_frames"].sum() / g["n_frames"].sum()),
                     "tracks": len(tid), "tracks_with_switch": int((tid == "switch").sum()),
                     "explained_right": held["explained_better"].get(rec, np.nan),
                     "explained_swapped": held["explained_swapped"].get(rec, np.nan),
                     "dead_channels": ",".join(dead) or "-",
                     "chases": int(per["events"].get(rec, 0)), "identified": int(per["identified"].get(rec, 0)),
                     "winner_chases": int(per["winner_chases"].get(rec, 0))})
    return pd.DataFrame(rows)


def disc_gain(elec: pd.DataFrame) -> float:
    g = elec[elec["channel"].isin(["4", "5"]) & (elec["gain_db"] > -40)]
    return float(g["gain_db"].median())


def wall_gain(elec: pd.DataFrame) -> float:
    g = elec[elec["channel"].isin([str(c) for c in (0, 1, 2, 3, 6, 7, 8, 9)]) & (elec["gain_db"] > -40)]
    return float(g["gain_db"].median())


def n_big_longer_txt(big: pd.DataFrame) -> str:
    return f"{int((big['ratio_video'] > 1).sum())} of the {len(big)}"


def stage_report() -> None:
    s = json.loads((VAL / "summary.json").read_text())
    rt = recording_table()
    rt.to_csv(VAL / "recordings.csv", index=False)
    pt = pd.read_csv(VAL / "pairing_by_distance.csv")
    ci = pd.read_csv(VAL / "clashes_by_distance_independent.csv")
    cj = pd.read_csv(VAL / "clashes_by_distance.csv")
    mkd = pd.read_csv(VAL / "masked_by_distance.csv")
    sz = pd.read_csv(VAL / "size_check.csv")
    cs = pd.read_csv(VAL / "chase_size_check.csv")
    loc = pd.read_csv(VAL / "localization_error.csv")
    per = pd.read_csv(d1.RES / "chases_by_trial.csv")
    ev = pd.read_csv(d1.RES / "events_identity.csv")
    elec = pd.read_csv(d1.RES / "electrodes_fitted.csv")
    conf = (ev["p_outside_leader"] >= 0.9) | (ev["p_outside_leader"] <= 0.1)
    e = ev[conf]
    agree = (e["S_inside_leader"] > 0) == (e["p_outside_leader"] > 0.5)
    k, n = int(per["winner_chases"].sum()), int(per["identified"].sum())
    shift = elec[elec["channel"] != "ref"].assign(
        shift=lambda d: np.hypot(d["x_fit"] - d["x_image"], d["y_fit"] - d["y_image"]))
    shift = shift[shift["gain_db"] > -40]
    ref = elec[elec["channel"] == "ref"]
    pt_all = pt[pt["tandem"] == "all"].copy()
    pt_tan = pt[pt["tandem"] == "True"].set_index("dist_from")["accuracy"]
    pt_all["tandem_accuracy"] = pt_all["dist_from"].map(pt_tan)
    pt_all = pt_all.rename(columns={"dist_from": "from (BL)", "dist_to": "to (BL)", "accuracy": "all configurations",
                                    "tandem_accuracy": "one behind the other"})
    clash = ci[["dist", "both_known", "clash_rate_of_known"]].rename(
        columns={"both_known": "frames", "clash_rate_of_known": "same identity, judged alone"})
    clash["same identity, method"] = cj["clash_rate_of_known"].to_numpy()
    lq = loc.groupby("kind")["err_bl"].quantile([0.5, 0.9]).unstack().reset_index()
    lq.columns = ["frames", "median error (BL)", "90th percentile (BL)"]
    switch_tracks = int(rt["tracks_with_switch"].sum())
    big = sz[sz["ratio_measured"] >= 1.04]
    p1 = sz[sz["pairing"].str.startswith("1")]
    p2 = sz[sz["pairing"].str.startswith("2")]
    cs2 = cs[cs["pairing"].str.startswith("2")]
    csb = cs[(cs["ratio_measured"] >= 1.04) & (cs["events"] >= 10)]
    pc = pd.read_csv(VAL / "pieces.csv")
    hi_p, lo_p = pc[pc["p_heldout"] >= 0.9], pc[pc["p_heldout"] < 0.9]
    ch_shift = shift.assign(channel=shift["channel"].astype(int)).groupby("channel")["shift"].median()
    worst = int(ch_shift.idxmax())
    praw = pd.read_csv(VAL / "pairing_test.csv")
    touch = praw[praw["contact_bl"] < 0.1]
    lines = f"""# Winner / loser identity of the 2022 video tracks, from the electrodes

Made by `experiments/video2022/d1_identity.py` (stages `amplitudes`, `calibrate`, `evidence`, `assign`,
`pairing`, `events`) and `d2_identity_figures.py` (figures, this README), branch `link2022`. Input: the
round-3 tracks and chase events of `video2022/chases/poses_r3/` (dark phase, 12.5 frames/s), the raw
electrode traces and the gap-filled wavetracker tracks of the two trial fish
(`detections/competition2022_v2_t001_gapfill`). The control 2022-07-07 has no winner / loser and is left out.

## In short

- **The electrodes are not a line.** `fishgrid.cfg` says 1 x 11 only so that 11 channels are recorded.
  The 11 electrodes are visible in the video: 3 on the top wall, 3 on the bottom wall, 1 on each side wall,
  1 at the left end of the tube and 2 on the suction-cup discs left and right of the tube. The data confirm
  which channel is which (below), and the reference electrode lies outside the camera view, bottom left.
  Because the electrodes surround the tank, the two fish can be told apart in 2-D, not only along a line.
- **{pct(s['frames_known_share'])} of the video detections get an identity** (winner or loser with a
  probability of at least {s['p_min']}); the rest are "unknown".
- **Reliability.** A single 0.16-s window points to the right fish in {pct(pt_all.iloc[-1]['all configurations'])}
  of cases with the fish 2-3 body lengths (BL) apart, {pct(pt_all.iloc[3]['all configurations'])} at
  0.75-1 BL and {pct(pt_all.iloc[0]['all configurations'])} below 0.25 BL (one behind the other, as in a chase:
  {pct(pt_all.iloc[0]['one behind the other'])}). Tracks pool many windows, so labels are more reliable than
  single windows; when each track is judged on its own, two fish in view get the same identity in
  {pct(s['two_in_view_clash_rate_independent'], 2)} of frames (below 0.5 BL: {pct(ci.iloc[0]['clash_rate_of_known'])}).
- **Independent check, body size.** In {n_big_longer_txt(big)} trials in which the winner was measured
  >= 4 % longer by hand (pairs 3-6), the frames labelled "winner" show the longer fish, with a winner / loser
  length ratio close to the measured one; in pair 2 (equal size) the ratio is
  {p2['ratio_video'].min():.2f}-{p2['ratio_video'].max():.2f} (`figures/size_check.png`).
- **The tracker swaps the fish when they touch.** The electrodes show a change of identity inside
  {switch_tracks} of {int(rt['tracks'].sum())} tracks; these tracks are cut into segments (one identity each).
- **Who chases whom.** In {k} of {n} identified chase events ({pct(k / n)}) the winner is the follower (the
  chaser); {int(per['events'].sum()) - n} of {int(per['events'].sum())} events stay unknown. Two checks that do
  not use the chase's own label: the evidence inside each chase agrees with the identity carried in from before
  and after it in {int(agree.sum())} of {len(e)} events; and, from the video alone, the follower is the longer
  fish in {int(csb['follower_longer'].sum())} of {int(csb['events'].sum())} chases of the trials in which the
  winner was measured >= 4 % longer, but only in {int(cs2['follower_longer'].sum())} of {int(cs2['events'].sum())}
  in pair 2 (equal size; there the labelled winner is not the longer fish), so this is not an effect of the
  chasing posture.

## Where the electrodes are, and how we know

`electrodes.csv` (positions of the electrode holders in the video, px, 2048 x 1536) and
`figures/electrodes.png`:

{md_table(pd.read_csv(d1.RES / 'electrodes.csv'), '.0f')}

- **Seen in the video**: the small black electrode holders on the walls with their cables, one at the left end
  of the tube (cable from the top), and one on top of each disc next to the tube. They sit in the same place in
  all 19 background frames (`b1_sleap/background/`).
- **Channel order, from the data**: for each channel, the video positions of the fish at the moments that
  channel had the strongest EOD (wavetracker `sign_v`) cluster at one of these electrodes. The field model
  below was then fitted with the electrode positions free (pulled towards the video positions with an sd of
  60 px); per recording, `electrodes_fitted.csv`. The fitted positions stay close to the holders (median
  shift {shift['shift'].median():.0f} px, 90th percentile {shift['shift'].quantile(0.9):.0f} px, i.e. about
  {shift['shift'].median() / 20:.0f} cm; furthest: electrode {worst}, median {ch_shift.max():.0f} px). The discs
  (4, 5) have a median gain of {disc_gain(elec):.0f} dB relative to the strongest channel, the wall
  electrodes {wall_gain(elec):.0f} dB.
- **Reference**: fitted at x = {ref['x_fit'].median():.0f}, y = {ref['y_fit'].median():.0f} px (median over
  recordings), i.e. outside the view below the bottom-left corner.
- **Dead channels** (gain < -40 dB): see the table at the end.
- **How well the fitted field explains the patterns**: on frames with the fish >= 3 BL apart, scored with the
  model fitted on the other half of the recording, the right assignment explains a median
  {rt['explained_right'].median():.2f} of each fish's pattern, the swapped one {rt['explained_swapped'].median():.2f}.

The 50 cm spacing of `fishgrid.cfg` is a default; the electrodes span the tank (~100 x 60 cm in view).

## Method

1. **Amplitudes** (`eod_amplitudes.py`). The wavetracker's power per electrode (`sign_v`) averages over
   1.64-s windows, in which a chasing fish swims half a metre. Instead, the raw traces are mixed down by each
   fish's own EOD frequency (its gap-filled wavetracker track) and the complex amplitude on every electrode is
   taken on 0.16-s windows every 0.08 s. The EOD field is quasi-static, so each fish gives a real, signed
   pattern over the 11 electrodes (up to one sign); the two fish are separated by their frequencies, also when
   they touch.
2. **Field model** (`identity.FieldModel`). A fish is a point dipole on its body axis at a fitted fraction of
   the way from snout to tail (about 0.65 of the keypoint index, between spine2 and spine3), in a shallow
   layer of water (2-D field, softened); each channel records gain x (potential at its electrode - potential at
   the reference). Fitted per recording (electrode positions, gains, reference, softening, dipole position) on
   frames with the two video fish >= 3 BL apart; which video fish is which is found during the fit (two
   hypotheses per frame, hard EM). **Cross-fitted**: frames in 10-min blocks of one parity are scored with the
   model fitted on the blocks of the other parity.
3. **Evidence per detection**: how much of the winner's and the loser's pattern the detection's pose explains
   (`cos2`). With two fish in view the two assignments are compared jointly; a number between -1 and 1,
   > 0 = this track is the winner.
4. **Tracks to identities** (`identity.track_posterior`). Per track a two-state hidden Markov chain over its
   frames: the identity may change where the tracker could have swapped the fish (the two skeletons within
   0.5 BL, or a gap > 0.3 s; prior {s['eps_episode']} per such episode) and almost never elsewhere
   ({s['eps_calm']:g} per frame). Evidence is weighted by beta = {s['beta_per_s']:.1f} log-odds per second of
   evidence, calibrated on pieces of tracks with a known identity (below). Each frame gets P(winner); frames
   with P >= {s['p_min']} are "winner", <= {1 - s['p_min']:.1f} "loser", else "unknown"; runs of one label form
   the segments of `track_identity.csv`.
5. **Chases** (`events_identity.csv`): the leader's and the follower's mean P(winner) over the event;
   P(winner leads) = mean of P(leader is winner) and 1 - P(follower is winner); the chaser (follower) is the
   loser if this is >= {s['p_min']}, the winner if <= {1 - s['p_min']:.1f}, else unknown.

## Reliability

**Reference identities** come from stretches of a track without contact or gap (the tracker cannot have
swapped the fish there) in which the fish were >= 3 BL apart in at least 3 seconds, 80 % of which agree
({s['reference_runs']} stretches).

**By distance between the fish, one window at a time (pairing test, `figures/evidence_by_distance.png`
left).** The two fish's patterns are measured separately, so the winner's pattern at one moment and the
loser's at another, both from reference stretches, together are what the electrodes would record with the
fish at these two poses at once. Pairs were drawn at every distance ({int(pt_all['windows'].sum())} pairs,
held-out model). Share of single windows pointing to the right fish:

{md_table(pt_all[['from (BL)', 'to (BL)', 'windows', 'all configurations', 'one behind the other']])}

This test uses poses of fish that were far apart (well detected); real close fish also have worse poses.

**Two fish in view: how often both get the same identity** (an error of at least one of them). Judged alone =
each track with the evidence of its own pose only; method = joint evidence (what is used):

{md_table(clash)}

**Whole method with the far evidence removed**: rerun with the evidence of every frame with the fish
>= 3 BL apart in the reference stretches set to zero, the frames of these stretches are labelled in
{pct(s['masked_known_share'])} and right in {pct(s['masked_accuracy_known'], 2)} of the labelled. These
stretches contain no contacts by construction, so this is the reliability away from contacts. By distance:

{md_table(mkd[['dist', 'frames', 'known', 'accuracy_known']], '.4f')}

**Calibration** (`figures/evidence_by_distance.png` right): pieces of reference stretches with the length of
real tracks; beta fitted on half of the recordings and applied to the other half. Pieces with a stated
probability >= 0.9 were right in {pct(hi_p['correct'].mean())} (mean stated {pct(hi_p['p_heldout'].mean())},
n = {len(hi_p)}); below 0.9 in {pct(lo_p['correct'].mean())} (mean stated {pct(lo_p['p_heldout'].mean())},
n = {len(lo_p)}): the stated probability is conservative where it is low.

**Body size (independent of the electrodes)** (`figures/size_check.png`): median skeleton length of straight
fish (frames with both fish in view >= 1.5 BL apart), by label:

{md_table(sz[['recording', 'pairing', 'winner_px', 'loser_px', 'ratio_video', 'winner_cm', 'loser_cm', 'ratio_measured']], '.2f')}

In pair 1 both fish were measured at 10 cm, but the winner weighed 3.1 g and the loser 1.0 g
(`weight_size.ods`); the video shows the winner {pct(p1['ratio_video'].mean() - 1, 0)} longer.

**Electric-only position** (`figures/example_frames.png`; `validation/localization_error.csv`): placing a fish
by its pattern alone (grid search, no video) puts its EOD centre a median 0.5 BL from where the video fish
of the same identity has it:

{md_table(lq, '.2f')}

The identity does not rely on this free localisation: it compares the two assignments at the video poses,
which is much more selective.

## Who chases whom

{md_table(per[['recording', 'pairing', 'events', 'identified', 'winner_chases', 'loser_chases', 'share_winner_chases', 'ci_low', 'ci_high']])}

- Pooled: the winner chases in {k} of {n} identified events ({pct(k / n)}).
- **Check 1, inside vs outside**: with the evidence inside all chases removed, the identity the method
  carries into the chase from before and after is confident for {len(e)} events; the evidence inside the
  chase alone points the same way in {int(agree.sum())} ({pct(agree.mean())}).
- **Check 2, video only** (`validation/chase_size_check.csv`): share of chases in which the follower has the
  longer skeleton:

{md_table(cs[cs['events'] >= 10][['recording', 'pairing', 'events', 'share_follower_longer', 'ci_low', 'ci_high', 'ratio_measured']])}

## Limits

- **Contacts.** With the skeletons within 0.1 BL of each other, single windows are right in
  {pct(touch['correct'].mean())} of the pairing test, and the tracker swaps the fish at contacts; identities
  through a contact rest mostly on the evidence before and after it. Chases are mostly contact (the follower's
  snout at the leader's tail).
- **Reference identities** come from the same electrodes (far-apart frames); the body-size checks are the only
  fully independent ones, and they only show the direction, not an error rate.
- **The pairing test** uses well-detected poses; in real close encounters the pose model errs more often.
- **Calibration** is measured away from contacts; the stated probability inside a contact may be optimistic.
- Hidden fish (tube, blocks) are not in the video tracks, so their identity is not needed; the electrodes do
  locate them (e.g. a fish in the tube), which is not used here.
- Chase events and tracks are those of `chases/poses_r3` unchanged; events between a fish and a mis-detection
  are not removed.

## Decisions for the author

- The probability threshold for "unknown" ({s['p_min']}) and the switch prior per contact ({s['eps_episode']})
  are choices (`--p-min`, `--eps-episode` of the `assign` stage).
- Whether tracks with a switch should be cut in the chase detector as well (now only in the identity table).
- The control 2022-07-07 (pair "2l4w": the loser of pair 2 with the winner of pair 4) could get fish
  identities the same way (by EOD frequency) if useful.

## Per recording

{md_table(rt, '.3f')}

## Files

- `<recording>/track_identity.csv`: one row per segment of a track (a run of frames with one label):
  `track`, `segment`, `n_segments` (of the track), `identity` (winner / loser / unknown), `p` (mean
  probability of that identity over the segment), `p_min`, `track_identity` (winner / loser / switch /
  unknown for the whole track), `t_start`, `t_end`, `duration_s`, `n_frames`, `n_evidence`, `evidence_s` (net
  seconds of evidence for "winner"), `share_two_in_view`, `share_far`, `dist_bl_median`, `contact_frames`,
  `skeleton_px`, `skeleton_bl`.
- `<recording>/observations_labelled.parquet`: every detection with its evidence and P(winner).
- `events_identity.csv`: `events.csv` plus `p_leader_winner`, `p_follower_winner`, `p_winner_leads`,
  `leader_identity`, `follower_identity`, `chaser` (winner / loser / unknown), `chaser_confidence`, and the
  check columns (`S_inside_*`, `p_outside_*`, `skeleton_px_*`). `chases_by_trial.csv`: per trial.
- `electrodes.csv`, `electrodes_fitted.csv`, `models/<recording>.json` (fitted field models: `half0`,
  `half1`, `all`).
- `validation/`: `summary.json`, `pairing_test.csv`, `pairing_by_distance.csv`, `clashes_by_distance*.csv`,
  `masked.parquet`, `masked_by_distance.csv`, `pieces.csv`, `test_blocks.csv`, `heldout_fit.csv`,
  `size_check.csv`, `chase_size_check.csv`, `localization_error.csv`, `segments_all.csv`, `recordings.csv`.
- `figures/`: `electrodes.png`, `evidence_by_distance.png`, `size_check.png`, `example_frames.png`,
  `chases_by_trial.png`.
- Amplitudes (per window and fish, complex, 11 channels): `video2022/identity/amplitudes/<recording>.npz`.
"""
    (d1.RES / "README.md").write_text(lines)
    print("README written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", nargs="*",
                    default=["electrodes", "evidence", "size", "frames", "chases", "localize", "report"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for w in args.what:
        {"electrodes": fig_electrodes, "evidence": fig_evidence, "size": fig_size, "frames": fig_frames,
         "chases": fig_chases, "localize": stage_localize, "report": stage_report}[w]()
        print(w, "done", flush=True)


if __name__ == "__main__":
    main()
