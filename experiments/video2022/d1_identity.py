"""Winner / loser identity of the 2022 video tracks from the electrode recordings.

Stages, in this order (outputs under ``$CHIRP_DATA/video2022/identity/``)::

    PY=/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python
    $PY experiments/video2022/d1_identity.py amplitudes   # amplitudes/<rec>.npz from the raw traces (~2.5 min each)
    $PY experiments/video2022/d1_identity.py calibrate    # poses_r3/models/<rec>.json: field model per recording
    $PY experiments/video2022/d1_identity.py evidence     # poses_r3/<rec>/observations.parquet (cross-fitted)
    $PY experiments/video2022/d1_identity.py assign       # validation/, <rec>/track_identity.csv (HMM labels)
    $PY experiments/video2022/d1_identity.py pairing      # validation/pairing_test.csv
    $PY experiments/video2022/d1_identity.py events       # events_identity.csv, chases_by_trial.csv
    $PY experiments/video2022/d2_identity_figures.py      # figures/, README.md

``amplitudes`` and ``calibrate`` are long: run them detached (``systemd-run --user`` + ``scripts/guarded``).
See ``identity/poses_r3/README.md`` (written by ``d2_identity_figures.py report``) for method and results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import b1_assess_sleap as b1  # noqa: E402
import eod_amplitudes as ea  # noqa: E402
import identity as idn  # noqa: E402
import pandas as pd  # noqa: E402
from paths import DATA, REC_2022  # noqa: E402
from recordings2022 import LIGHT_ON_S, list_recordings  # noqa: E402

OUT = DATA / "video2022" / "identity"
GAPFILL = DATA / "detections" / "competition2022_v2_t001_gapfill"
RUN = "poses_r3"
CHASES = DATA / "video2022" / "chases" / RUN
RES = OUT / RUN
HALF_S = 600.0  # cross-fitting: the model of 10-min blocks of one parity is fitted on the other parity
MIN_DIST_CAL = 3.0  # BL: calibration windows have the two video fish at least this far apart
MAX_IMAG = 0.2  # windows whose pattern is not real enough (imag_frac) are dropped


def recordings(names: list[str] | None) -> list:
    recs = list_recordings()
    return [r for r in recs if not names or r.name in names]


def meta() -> pd.DataFrame:
    m = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"})
    return m.set_index("recording")


def body_length(rec: str) -> float:
    return float(pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording").loc[rec, "body_length_px"])


def load_patterns(rec: str) -> dict[str, np.ndarray]:
    """Window times and the signed patterns of winner and loser (NaN where missing or not clean)."""
    z = np.load(OUT / "amplitudes" / f"{rec}.npz")
    out = {"t": z["t"]}
    for role in ("winner", "loser"):
        v, imag = ea.signed_pattern(z[f"A_{role}"].astype(complex))
        v[~(imag <= MAX_IMAG)] = np.nan
        out[role] = v
        out[f"imag_{role}"] = imag
        out[f"amp_{role}"] = np.abs(z[f"A_{role}"]).max(axis=1)
    return out


def load_tracks(rec: str) -> pd.DataFrame:
    return pd.read_parquet(CHASES / rec / "tracks.parquet")


def parity(t: np.ndarray) -> np.ndarray:
    return (np.floor(np.asarray(t) / HALF_S) % 2).astype(int)


def calibration_windows(tracks: pd.DataFrame, pat: dict, bl: float, min_dist_bl: float = MIN_DIST_CAL
                        ) -> dict[str, np.ndarray]:
    """Windows with exactly two video fish at least ``min_dist_bl`` apart and both patterns clean."""
    d = tracks.sort_values(["frame", "track"])
    n = d.groupby("frame")["track"].transform("size")
    d = d[n == 2]
    P = idn.poses(d).reshape(-1, 2, len(idn.BODY), 2)
    t = d["t_s"].to_numpy(float)[::2]
    k = idn.match_windows(t, pat["t"])
    dist = np.hypot(*(P[:, 0, 1] - P[:, 1, 1]).T) / bl
    ok = (k >= 0) & (dist >= min_dist_bl)
    ok &= np.isfinite(pat["winner"][np.maximum(k, 0)]).all(1) & np.isfinite(pat["loser"][np.maximum(k, 0)]).all(1)
    k = k[ok]
    return {"k": k, "t": pat["t"][k], "pa": P[ok, 0], "pb": P[ok, 1], "vw": pat["winner"][k],
            "vl": pat["loser"][k], "dist": dist[ok]}


def stage_calibrate(names: list[str] | None) -> None:
    """Fit the field model per recording: on each half (cross-fitting) and on all windows."""
    out = RES / "models"
    out.mkdir(parents=True, exist_ok=True)
    for rec in recordings(names):
        if not (OUT / "amplitudes" / f"{rec.name}.npz").exists():
            print(f"{rec.name}: no amplitudes", flush=True)
            continue
        bl = body_length(rec.name)
        pat = load_patterns(rec.name)
        cal = calibration_windows(load_tracks(rec.name), pat, bl)
        half = parity(cal["t"])
        res = {"recording": rec.name, "n_windows": len(cal["k"])}
        tic = time.time()
        for name, sel in [("half0", half == 0), ("half1", half == 1), ("all", np.ones(len(half), bool))]:
            if sel.sum() < 200:
                res[name] = None
                continue
            m, _ = idn.fit_field(cal["vw"][sel], cal["vl"][sel], cal["pa"][sel], cal["pb"][sel])
            res[name] = m.to_dict()
        (out / f"{rec.name}.json").write_text(json.dumps(res, indent=1))
        print(f"{rec.name}: {res['n_windows']} calibration windows, {time.time() - tic:.0f} s", flush=True)


def heldout_fit(rec: str) -> list[dict]:
    """Far-apart windows of each half scored with the model fitted on the other half."""
    models = load_models(rec)
    cal = calibration_windows(load_tracks(rec), load_patterns(rec), body_length(rec))
    half = parity(cal["t"])
    rows = []
    for h in (0, 1):
        sel = half == h
        other = models[f"half{1 - h}"]
        if other is None or sel.sum() == 0:
            continue
        ca, cb = idn.joint_cos2(idn.FieldModel.from_dict(other), cal["vw"][sel], cal["vl"][sel],
                                cal["pa"][sel], cal["pb"][sel])
        best, worse = np.maximum(ca, cb), np.minimum(ca, cb)
        rows.append({"recording": rec, "half": h, "windows": int(sel.sum()),
                     "explained_better": float(np.median(best)), "explained_swapped": float(np.median(worse)),
                     "decisive": float(np.mean(best - worse > 0.3)), "weak": float(np.mean(best - worse < 0.1))})
    return rows


def stage_amplitudes(names: list[str] | None, force: bool) -> None:
    out = OUT / "amplitudes"
    out.mkdir(parents=True, exist_ok=True)
    p = ea.DEFAULT
    for rec in recordings(names):
        dst = out / f"{rec.name}.npz"
        if dst.exists() and not force:
            print(f"{rec.name}: exists", flush=True)
            continue
        src = GAPFILL / rec.name / "tracks_gapfill.npz"
        if not src.exists():  # the control 2022-07-07 has no trial fish (not in meta.csv)
            print(f"{rec.name}: no gap-filled trial tracks, skipped", flush=True)
            continue
        refs = ea.fish_references(src)
        t_end = min(LIGHT_ON_S, rec.recording_duration_s())
        tic = time.time()
        res = ea.recording_amplitudes(rec.raw / "traces-grid1.raw", refs, t_end, p=p)
        ids = {role: fid for role, (fid, _, _) in refs.items()}
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, **res, roles=np.array(list(refs)), fish_id=np.array([ids[r] for r in refs]),
                 params=json.dumps(p.__dict__))
        tmp.rename(dst)
        print(f"{rec.name}: {len(res['t'])} windows, {time.time() - tic:.0f} s", flush=True)


def load_models(rec: str) -> dict:
    return json.loads((RES / "models" / f"{rec}.json").read_text())


def fallback_model(rec: str) -> tuple[dict, str]:
    """For a recording with too few calibration windows: the model of the other bout of the same pair with
    the most calibration windows (same tank, same electrodes; gains may differ)."""
    pair = meta()["Pairing"].get(rec, "")[:1]
    best = None
    for f in sorted((RES / "models").glob("*.json")):
        r = json.loads(f.read_text())
        if r["recording"] == rec or r["all"] is None:
            continue
        key = (meta()["Pairing"].get(r["recording"], "")[:1] == pair, r["n_windows"])
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        raise RuntimeError(f"{rec}: no field model to fall back on")
    return best[1]["all"], f"fallback:{best[1]['recording']}"


def model_for(rec: str, h: int, models: dict | None = None) -> tuple[idn.FieldModel, str]:
    """The field model that scores windows of parity ``h``: fitted on the other parity, else on all
    windows, else borrowed from another bout of the pair (:func:`fallback_model`)."""
    models = models or load_models(rec)
    if models.get(f"half{1 - h}") is not None:
        return idn.FieldModel.from_dict(models[f"half{1 - h}"]), f"half{1 - h}"
    if models.get("all") is not None:
        return idn.FieldModel.from_dict(models["all"]), "all"
    m, src = fallback_model(rec)
    return idn.FieldModel.from_dict(m), src


def stage_evidence(names: list[str] | None) -> None:
    """Per recording: every video detection with its electrode evidence (cross-fitted models)."""
    for rec in recordings(names):
        f = RES / "models" / f"{rec.name}.json"
        if not f.exists():
            print(f"{rec.name}: no model", flush=True)
            continue
        models = load_models(rec.name)
        bl = body_length(rec.name)
        pat = load_patterns(rec.name)
        tr = load_tracks(rec.name)
        tr = tr[tr["t_s"] <= pat["t"][-1] + 0.04]
        par = parity(tr["t_s"].to_numpy())
        parts = []
        for h in (0, 1):
            m, src = model_for(rec.name, h, models)
            o = idn.observations(tr[par == h], pat["t"], pat["winner"], pat["loser"], m, bl)
            parts.append(o.assign(model=src))
        obs = pd.concat(parts, ignore_index=True).sort_values(["frame", "track"], ignore_index=True)
        obs["delta_solo"] = obs["cw"] - obs["cl"]
        (RES / rec.name).mkdir(parents=True, exist_ok=True)
        obs.to_parquet(RES / rec.name / "observations.parquet", index=False)
        print(f"{rec.name}: {len(obs)} detections, {np.isfinite(obs['delta']).mean():.2f} with evidence", flush=True)


DIST_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
P_MIN = 0.9  # stretches whose posterior probability is below this are "unknown"
FAR_BL = 3.0
EPS_EPISODE = 0.1  # prior probability that the tracker swapped the fish in one contact episode
EPS_CALM = 1e-4  # ... between two consecutive windows without contact


def load_obs(rec: str) -> pd.DataFrame:
    obs = pd.read_parquet(RES / rec / "observations.parquet")
    return obs.sort_values(["track", "frame"], ignore_index=True)


def with_runs(obs: pd.DataFrame, eps_episode: float) -> pd.DataFrame:
    """Risky transitions and runs (calm stretches / contact episodes) of every track."""
    o = idn.track_posterior(obs, 0.0, eps_episode, EPS_CALM)  # beta 0: only the risky flags are used
    return o.assign(run=idn.runs(o)).drop(columns=["logodds"])


def dist_label(d: np.ndarray) -> np.ndarray:
    lab = pd.cut(d, DIST_BINS, right=False).astype(str)
    return np.where(np.isfinite(d), lab, "one fish in view")


def window_validation(obs: dict[str, pd.DataFrame], seed: int = 0) -> tuple[float, dict[str, pd.DataFrame]]:
    """How informative the electrode evidence is at each distance between the fish (1-s blocks of
    reference runs that are not far), and ``beta`` (log-odds per second of evidence) calibrated on
    track-like pieces of these blocks."""
    tests, pieces, truths = [], [], []
    lengths = np.concatenate([o.groupby("track").size().to_numpy() / 12.5 for o in obs.values()])
    lengths = np.maximum(1, np.round(lengths)).astype(int)
    rng = np.random.default_rng(seed)
    for rec, o in obs.items():
        truth = idn.run_truth(o, FAR_BL)
        b = idn.block_evidence(o, by="run")
        b = b.merge(o.groupby("run")["risky"].first().rename("risky_run"), left_on="run", right_index=True)
        test = b[b["run"].isin(truth.index) & ~idn.is_far(b, FAR_BL)].copy()
        test["correct"] = np.sign(test["delta"]).to_numpy() == truth.reindex(test["run"]).to_numpy()
        tests.append(test.assign(recording=rec))
        pieces.append(idn.chop(test, truth, lengths, rng, by="run").assign(recording=rec))
        truths.append(pd.DataFrame({"run": truth.index, "truth": truth.to_numpy(), "recording": rec}))
    test = pd.concat(tests, ignore_index=True)
    test["dist"] = dist_label(test["dist_bl"].to_numpy())
    pc = pd.concat(pieces, ignore_index=True)
    beta = idn.fit_beta(pc["S"], pc["correct"])
    recs = sorted(pc["recording"].unique())
    fold = pc["recording"].map({r: i % 2 for i, r in enumerate(recs)})
    pc["p_heldout"] = np.nan
    for f in (0, 1):
        if (fold != f).any() and (fold == f).any():
            bf = idn.fit_beta(pc.loc[fold != f, "S"], pc.loc[fold != f, "correct"])
            pc.loc[fold == f, "p_heldout"] = idn.probability(pc.loc[fold == f, "S"], bf)
    pc["p"] = idn.probability(pc["S"], beta)
    pc["dist"] = dist_label(pc["dist_bl"].to_numpy())
    by_dist = test.groupby("dist").agg(blocks=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    by_rec = test.groupby(["recording", "dist"]).agg(blocks=("correct", "size"),
                                                     accuracy=("correct", "mean")).reset_index()
    return beta, {"blocks_by_distance": by_dist, "blocks_by_recording": by_rec, "pieces": pc,
                  "test_blocks": test, "truth": pd.concat(truths, ignore_index=True)}


def masked_validation(obs: dict[str, pd.DataFrame], truth: pd.DataFrame, beta: float, eps_episode: float,
                      p_min: float) -> pd.DataFrame:
    """The whole method rerun with the far evidence (fish >= FAR_BL apart) of every reference run
    removed. Per window of a reference run: its label and probability against the reference."""
    rows = []
    for rec, o in obs.items():
        tr = truth[truth["recording"] == rec].set_index("run")["truth"]
        far_rows = (o["run"].isin(tr.index) & (o["dist_bl"] >= FAR_BL)).to_numpy()
        m = idn.track_posterior(o.assign(delta=np.where(far_rows, 0.0, o["delta"])), beta, eps_episode, EPS_CALM)
        m = idn.label_segments(m, p_min)
        m = m[m["run"].isin(tr.index)]
        ref = tr.reindex(m["run"]).to_numpy()
        rows.append(pd.DataFrame({"recording": rec, "track": m["track"].to_numpy(), "run": m["run"].to_numpy(),
                                  "t_s": m["t_s"].to_numpy(), "dist_bl": m["dist_bl"].to_numpy(),
                                  "masked": (m["dist_bl"] >= FAR_BL).to_numpy(), "risky_run": m["risky"].to_numpy(),
                                  "p_winner": m["p_winner"].to_numpy(), "label": m["label"].to_numpy(),
                                  "truth": np.where(ref > 0, "winner", "loser")}))
    out = pd.concat(rows, ignore_index=True)
    out["p_truth"] = np.where(out["truth"] == "winner", out["p_winner"], 1 - out["p_winner"])
    out["dist"] = dist_label(out["dist_bl"].to_numpy())
    return out


def segment_table(rec: str, o: pd.DataFrame) -> pd.DataFrame:
    """Identity and evidence summary of every segment (run of one label) of every track of one recording."""
    import chases as ch

    tr = load_tracks(rec)
    bl = body_length(rec)
    tr = tr.assign(skeleton_px=ch.skeleton_length(tr))
    o = o.merge(tr[["frame", "track", "skeleton_px"]], on=["frame", "track"], how="left")
    o["p_label"] = np.where(o["label"] == "loser", 1 - o["p_winner"],
                            np.where(o["label"] == "winner", o["p_winner"],
                                     np.maximum(o["p_winner"], 1 - o["p_winner"])))
    o["far"] = o["dist_bl"] >= FAR_BL
    o["two"] = o["n_vis"] == 2
    g = o.groupby("seg")
    out = pd.DataFrame({"track": g["track"].first(), "segment": g["segment"].first(), "identity": g["label"].first(),
                        "p": g["p_label"].mean(), "p_min": g["p_label"].min(), "t_start": g["t_s"].min(),
                        "t_end": g["t_s"].max(), "n_frames": g.size(), "n_evidence": g["delta"].count(),
                        "evidence_s": g["delta"].sum() * 0.08, "share_two_in_view": g["two"].mean(),
                        "share_far": g["far"].mean(), "dist_bl_median": g["dist_bl"].median(),
                        "contact_frames": g["risky"].sum(), "skeleton_px": g["skeleton_px"].median()})
    out["duration_s"] = out["t_end"] - out["t_start"]
    out["skeleton_bl"] = out["skeleton_px"] / bl
    missing = sorted(set(tr["track"]) - set(out["track"]))
    if missing:  # tracks without any electric window
        gt = tr[tr["track"].isin(missing)].groupby("track")
        extra = pd.DataFrame({"track": missing, "segment": 0, "identity": "unknown", "p": 0.5,
                              "t_start": gt["t_s"].min().to_numpy(), "t_end": gt["t_s"].max().to_numpy(),
                              "n_frames": gt.size().to_numpy(), "n_evidence": 0,
                              "skeleton_px": gt["skeleton_px"].median().to_numpy()})
        extra["duration_s"] = extra["t_end"] - extra["t_start"]
        extra["skeleton_bl"] = extra["skeleton_px"] / bl
        out = pd.concat([out.reset_index(drop=True), extra], ignore_index=True)
    out = out.reset_index(drop=True)
    out["n_segments"] = out.groupby("track")["segment"].transform("size")
    known = out[out["identity"] != "unknown"].groupby("track")["identity"].agg(lambda s: "/".join(sorted(set(s))))
    tid = out["track"].map(known).fillna("unknown")
    out["track_identity"] = np.where(tid.str.contains("/"), "switch", tid)
    cols = ["track", "segment", "n_segments", "identity", "p", "p_min", "track_identity", "t_start", "t_end",
            "duration_s", "n_frames", "n_evidence", "evidence_s", "share_two_in_view", "share_far", "dist_bl_median",
            "contact_frames", "skeleton_px", "skeleton_bl"]
    return out[cols].sort_values(["track", "segment"], ignore_index=True)


def window_clashes(o: pd.DataFrame) -> pd.DataFrame:
    """Frames with two fish in view: both labels and whether they clash (the same known identity)."""
    two = o[o["other"] >= 0][["frame", "track", "other", "label", "dist_bl", "contact_bl"]]
    m = two.merge(two[["frame", "track", "label"]].rename(columns={"track": "other", "label": "label_other"}),
                  on=["frame", "other"])
    m = m[m["track"] < m["other"]].copy()
    m["both_known"] = (m["label"] != "unknown") & (m["label_other"] != "unknown")
    m["clash"] = m["both_known"] & (m["label"] == m["label_other"])
    return m


def stage_assign(names: list[str] | None, p_min: float, eps_episode: float) -> None:
    recs = [r.name for r in recordings(names) if (RES / r.name / "observations.parquet").exists()]
    vdir = RES / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    obs = {r: with_runs(load_obs(r), eps_episode) for r in recs}
    beta, val = window_validation(obs)
    for k, v in val.items():
        v.to_csv(vdir / f"{k}.csv", index=False)
    masked = masked_validation(obs, val["truth"], beta, eps_episode, p_min)
    masked.to_parquet(vdir / "masked.parquet", index=False)
    summary = {"beta_per_s": beta, "p_min": p_min, "eps_episode": eps_episode, "eps_calm": EPS_CALM,
               "far_bl": FAR_BL, "reference_runs": len(val["truth"]), "pieces": len(val["pieces"])}
    segs, clashes = [], []
    for r in recs:
        o = idn.label_segments(idn.track_posterior(obs[r], beta, eps_episode, EPS_CALM), p_min)
        o.to_parquet(RES / r / "observations_labelled.parquet", index=False)
        t = segment_table(r, o)
        t.to_csv(RES / r / "track_identity.csv", index=False)
        segs.append(t.assign(recording=r))
        clashes.append(window_clashes(o).assign(recording=r))
    seg = pd.concat(segs, ignore_index=True)
    seg.to_csv(vdir / "segments_all.csv", index=False)
    cl = pd.concat(clashes, ignore_index=True)
    cl["dist"] = dist_label(cl["dist_bl"].to_numpy())
    cl_tab = cl.groupby("dist").agg(frames=("clash", "size"), both_known=("both_known", "sum"),
                                    clash=("clash", "sum")).reset_index()
    cl_tab["clash_rate_of_known"] = cl_tab["clash"] / cl_tab["both_known"]
    cl_tab.to_csv(vdir / "clashes_by_distance.csv", index=False)
    mk = masked[~masked["masked"]].assign(known=lambda d: d["label"] != "unknown")
    mk["correct"] = mk["label"] == mk["truth"]
    mk_tab = mk.groupby("dist").agg(frames=("known", "size"), known=("known", "mean"),
                                    correct_of_known=("correct", "sum"), n_known=("known", "sum")).reset_index()
    mk_tab["accuracy_known"] = mk_tab["correct_of_known"] / mk_tab["n_known"]
    mk_tab.to_csv(vdir / "masked_by_distance.csv", index=False)
    # independent assignment: every track judged on its own pose only (no joint evidence); how often do
    # two fish in view end up with the same identity?
    solo = {r: o.assign(delta=o["delta_solo"]) for r, o in obs.items()}
    beta_solo, _ = window_validation(solo)
    cs = []
    for r in recs:
        os_ = idn.label_segments(idn.track_posterior(solo[r], beta_solo, eps_episode, EPS_CALM), p_min)
        cs.append(window_clashes(os_).assign(recording=r))
    cs = pd.concat(cs, ignore_index=True)
    cs["dist"] = dist_label(cs["dist_bl"].to_numpy())
    cs_tab = cs.groupby("dist").agg(frames=("clash", "size"), both_known=("both_known", "sum"),
                                    clash=("clash", "sum")).reset_index()
    cs_tab["clash_rate_of_known"] = cs_tab["clash"] / cs_tab["both_known"]
    cs_tab.to_csv(vdir / "clashes_by_distance_independent.csv", index=False)
    summary["beta_per_s_independent"] = beta_solo
    summary["two_in_view_clash_rate_independent"] = float(cs.loc[cs["both_known"], "clash"].mean())
    print(cs_tab.to_string())
    track_ids = seg.groupby(["recording", "track"])["track_identity"].first()
    summary.update({
        "tracks": len(track_ids), "tracks_with_switch": int((track_ids == "switch").sum()),
        "segments": len(seg), "segments_known": int((seg["identity"] != "unknown").sum()),
        "frames_known_share": float(seg.loc[seg["identity"] != "unknown", "n_frames"].sum() / seg["n_frames"].sum()),
        "two_in_view_clash_rate": float(cl.loc[cl["both_known"], "clash"].mean()),
        "masked_known_share": float(mk["known"].mean()),
        "masked_accuracy_known": float(mk.loc[mk["known"], "correct"].mean()),
    })
    (vdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(val["blocks_by_distance"].to_string())
    print(cl_tab.to_string())
    print(mk_tab.to_string())


PAIR_BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])


def stage_pairing(names: list[str] | None, per_bin: int = 300, seed: int = 0) -> None:
    """Pairing test (``identity.pairing_test``): per-window evidence at every distance between the fish,
    from windows in which each fish's identity is known (reference runs, fish >= FAR_BL apart)."""
    rng = np.random.default_rng(seed)
    parts = []
    for rec in [r.name for r in recordings(names) if (RES / r.name / "observations.parquet").exists()]:
        o = with_runs(load_obs(rec), EPS_EPISODE)
        truth = idn.run_truth(o, FAR_BL)
        far = o[o["run"].isin(truth.index) & (o["dist_bl"] >= FAR_BL)].copy()
        far["who"] = np.where(truth.reindex(far["run"]).to_numpy() > 0, "winner", "loser")
        pat = load_patterns(rec)
        tr = load_tracks(rec).set_index(["frame", "track"])
        models = load_models(rec)
        bl = body_length(rec)
        for h in (0, 1):
            m = models[f"half{1 - h}"]
            if m is None:
                continue
            f = far[parity(far["t_s"].to_numpy()) == h]
            w, lo = f[f["who"] == "winner"], f[f["who"] == "loser"]
            if len(w) < 50 or len(lo) < 50:
                continue
            pw = idn.poses(tr.loc[list(zip(w["frame"], w["track"], strict=True))].reset_index())
            pl = idn.poses(tr.loc[list(zip(lo["frame"], lo["track"], strict=True))].reset_index())
            res = idn.pairing_test(idn.FieldModel.from_dict(m), pat["winner"][w["k"].to_numpy()], pw,
                                   pat["loser"][lo["k"].to_numpy()], pl, bl, PAIR_BINS, per_bin, rng)
            parts.append(res.assign(recording=rec, half=h))
        print(rec, "done", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out = out[np.isfinite(out["delta"])]
    out.to_csv(RES / "validation" / "pairing_test.csv", index=False)
    lab = pd.cut(out["dist_bl"], PAIR_BINS, right=False)
    tab = out.groupby([lab, "tandem"], observed=True)["correct"].agg(["size", "mean"]).unstack()
    print(tab.to_string())
    cl = pd.cut(out["contact_bl"], [0, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0], right=False)
    print(out.groupby(cl, observed=True)["correct"].agg(["size", "mean"]).to_string())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(mid - half), float(mid + half)


def event_identities(ev: pd.DataFrame, o: pd.DataFrame, p_min: float) -> pd.DataFrame:
    """Leader / follower identity of each chase event from the labelled windows of their tracks during it.

    ``p_leader_winner``: mean posterior P(winner) of the leader's windows in the event; likewise for
    the follower. The two fish are different, so the event's evidence is
    ``p_event = (p_leader_winner + 1 - p_follower_winner) / 2`` = P(the winner leads, i.e. the loser chases).
    """
    rows = []
    for _, e in ev.iterrows():
        r = {}
        for role in ("leader", "follower"):
            w = o[(o["track"] == e[role]) & (o["t_s"] >= e["t_start"] - 0.04) & (o["t_s"] <= e["t_end"] + 0.04)]
            r[f"p_{role}_winner"] = float(w["p_winner"].mean()) if len(w) else np.nan
            r[f"n_{role}_windows"] = len(w)
        p_lead = r["p_leader_winner"]
        p_fol = r["p_follower_winner"]
        r["p_winner_leads"] = np.nanmean([p_lead, 1 - p_fol]) if np.isfinite([p_lead, p_fol]).any() else np.nan
        for role, p in (("leader", p_lead), ("follower", p_fol)):
            r[f"{role}_identity"] = ("winner" if p >= p_min else "loser" if p <= 1 - p_min else "unknown") \
                if np.isfinite(p) else "unknown"
        pe = r["p_winner_leads"]
        r["chaser"] = ("loser" if pe >= p_min else "winner" if pe <= 1 - p_min else "unknown") \
            if np.isfinite(pe) else "unknown"
        r["chaser_confidence"] = max(pe, 1 - pe) if np.isfinite(pe) else np.nan
        r["leader_follower_clash"] = (r["leader_identity"] == r["follower_identity"]) & (r["leader_identity"] != "unknown")
        rows.append(r)
    return pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def event_checks(ev: pd.DataFrame, rec: str, beta: float, eps_episode: float) -> pd.DataFrame:
    """Two checks per event that do not use the event's own label.

    - inside vs outside: the evidence inside the event alone (``S_inside``, seconds of evidence that
      the leader is the winner: the leader's joint ``delta`` summed over the event) against the identity
      the method gives the leader when the evidence inside all events is removed (``p_outside``: the
      HMM posterior carried in from before and after the event).
    - body size: median skeleton length (px) of leader and follower during the event, over frames in
      which the fish is straight (snout-tail chord >= 0.95 of the skeleton; a bent fish looks shorter).
    """
    import chases as ch

    o = with_runs(load_obs(rec), eps_episode)
    inside = np.zeros(len(o), bool)
    tk, t = o["track"].to_numpy(), o["t_s"].to_numpy()
    for _, e in ev.iterrows():
        inside |= np.isin(tk, [e["leader"], e["follower"]]) & (t >= e["t_start"] - 0.04) & (t <= e["t_end"] + 0.04)
    o = o.assign(delta_all=o["delta"], delta=np.where(inside, 0.0, o["delta"]))
    m = idn.track_posterior(o, beta, eps_episode, EPS_CALM)
    m["p_winner"] = 1 / (1 + np.exp(-m["logodds"]))
    tr = load_tracks(rec)
    P = idn.poses(tr)
    tr = tr.assign(skeleton_px=ch.skeleton_length(tr))
    tr = tr[np.hypot(*(P[:, -1] - P[:, 0]).T) >= 0.95 * tr["skeleton_px"]]
    rows = []
    for _, e in ev.iterrows():
        r = {}
        for role in ("leader", "follower"):
            sel = (m["track"] == e[role]) & (m["t_s"] >= e["t_start"] - 0.04) & (m["t_s"] <= e["t_end"] + 0.04)
            r[f"S_inside_{role}"] = float(np.nansum(m.loc[sel, "delta_all"]) * 0.08)
            r[f"p_outside_{role}"] = float(m.loc[sel, "p_winner"].mean()) if sel.any() else np.nan
            s = tr[(tr["track"] == e[role]) & (tr["t_s"] >= e["t_start"]) & (tr["t_s"] <= e["t_end"])]
            r[f"skeleton_px_{role}"] = float(s["skeleton_px"].median()) if len(s) else np.nan
        rows.append(r)
    return pd.DataFrame(rows, index=ev.index)


def stage_events(p_min: float) -> None:
    ev = pd.read_csv(CHASES / "events.csv")
    m = meta()
    summary = json.loads((RES / "validation" / "summary.json").read_text())
    parts = []
    for rec, g in ev.groupby("recording"):
        f = RES / rec / "observations_labelled.parquet"
        if not f.exists():
            print(f"{rec}: not labelled", flush=True)
            continue
        e = event_identities(g, pd.read_parquet(f), p_min)
        checks = event_checks(g, rec, summary["beta_per_s"], summary["eps_episode"])
        parts.append(pd.concat([e, checks.reset_index(drop=True)], axis=1))
    out = pd.concat(parts, ignore_index=True)
    out["pairing"] = out["recording"].map(m["Pairing"]).fillna("control")
    out.to_csv(RES / "events_identity.csv", index=False)
    rows = []
    for rec, g in out.groupby("recording"):
        k = int((g["chaser"] == "winner").sum())
        n = int((g["chaser"] != "unknown").sum())
        lo, hi = wilson(k, n)
        rows.append({"recording": rec, "pairing": g["pairing"].iloc[0], "events": len(g), "identified": n,
                     "winner_chases": k, "loser_chases": n - k, "share_winner_chases": k / n if n else np.nan,
                     "ci_low": lo, "ci_high": hi, "leader_follower_clash": int(g["leader_follower_clash"].sum())})
    per = pd.DataFrame(rows)
    per.to_csv(RES / "chases_by_trial.csv", index=False)
    print(per.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["amplitudes", "calibrate", "evidence", "assign", "events", "pairing"])
    ap.add_argument("--recordings", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--p-min", type=float, default=P_MIN)
    ap.add_argument("--eps-episode", type=float, default=EPS_EPISODE)
    args = ap.parse_args()
    if args.stage == "amplitudes":
        stage_amplitudes(args.recordings, args.force)
    elif args.stage == "calibrate":
        stage_calibrate(args.recordings)
    elif args.stage == "evidence":
        stage_evidence(args.recordings)
    elif args.stage == "assign":
        stage_assign(args.recordings, args.p_min, args.eps_episode)
    elif args.stage == "events":
        stage_events(args.p_min)
    elif args.stage == "pairing":
        stage_pairing(args.recordings)


if __name__ == "__main__":
    main()
