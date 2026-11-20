"""Tests of the electrode-identity tools (eod_amplitudes.py, identity.py) on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eod_amplitudes as ea
import identity as idn

FS = ea.FS


def two_fish_traces(dur=4.0, f1=650.3, f2=702.9, seed=0):
    """11 channels: fish 1 with a fixed signed pattern, fish 2 another, plus noise and a harmonic."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    p1 = np.linspace(-1, 1, 11) * 1e-3
    p2 = np.r_[np.ones(5), -np.ones(6)] * 3e-4
    x = (np.outer(np.sin(2 * np.pi * f1 * t + 0.3), p1) + np.outer(np.sin(2 * np.pi * f2 * t + 1.1), p2)
         + 0.3 * np.outer(np.sin(2 * np.pi * 2 * f1 * t), p1) + rng.normal(0, 1e-5, (len(t), 11)))
    return t, x, p1, p2


def test_demodulation_recovers_signed_pattern_and_amplitude():
    t, x, p1, p2 = two_fish_traces()
    centers = np.arange(0.5, 3.5, 0.08)
    for f, p in [(650.3, p1), (702.9, p2)]:
        # reference frequency off by 1.5 Hz: the +-4 Hz search must find the true peak
        z, tb = ea.baseband(x, 0.0, np.array([0.0, 4.0]), np.array([f - 1.5, f - 1.5]))
        A, df = ea.window_amplitudes(z, tb, centers)
        assert np.all(np.abs(df - 1.5) <= 0.25)
        v, imag = ea.signed_pattern(A)
        assert np.all(imag < 1e-3)
        cos = np.abs(v @ p) / (np.linalg.norm(v, axis=1) * np.linalg.norm(p))
        assert np.all(cos > 0.999)
        amp = np.abs(A).max(axis=1)
        assert np.allclose(amp, np.abs(p).max(), rtol=0.05)


def test_windows_outside_the_data_are_nan():
    t, x, p1, _ = two_fish_traces(dur=1.0)
    z, tb = ea.baseband(x, 10.0, np.array([0.0, 20.0]), np.array([650.3, 650.3]))
    A, _ = ea.window_amplitudes(z, tb, np.array([10.02, 10.5, 10.99]))
    assert np.isnan(A[0]).all() and np.isfinite(A[1]).all() and np.isnan(A[2]).all()


def test_near_track():
    tt = np.array([0.0, 1.0, 5.0])
    assert ea.near_track(np.array([0.5, 2.5, 4.2, 9.0]), tt, 1.0).tolist() == [True, False, True, False]


# ----------------------------------------------------------------------------- identity.py
TRUE = idn.FieldModel(electrodes=idn.ELECTRODES_IMAGE + np.array([15.0, -10.0]),
                      gain=np.linspace(0.3, 1.0, 11), ref=np.array([200.0, 1600.0]), height=40.0, center=0.65)


def random_poses(n, rng, length=200.0):
    xy = np.stack([rng.uniform(250, 1750, n), rng.uniform(300, 1250, n)], axis=1)
    return idn.straight_poses(xy, rng.uniform(0, 2 * np.pi, n), length)


def test_dipole_sign_flips_with_heading():
    rng = np.random.default_rng(1)
    p = random_poses(5, rng)
    m = idn.FieldModel(center=0.5)
    flipped = p[:, ::-1]  # same body, head and tail exchanged
    assert np.allclose(m.pattern(p), -m.pattern(flipped))


def test_cos2_is_scale_and_sign_free():
    v = np.array([[1.0, -2.0, 3.0]])
    assert np.isclose(idn.cos2(v, -5 * v)[0], 1.0)
    assert np.isclose(idn.cos2(v, np.array([[2.0, 1.0, 0.0]]))[0], 0.0)


def test_fit_field_recovers_assignment_and_geometry():
    rng = np.random.default_rng(2)
    n = 1500
    pw, pl = random_poses(n, rng), random_poses(n, rng)
    noise = 0.02
    vw = TRUE.pattern(pw) * rng.uniform(0.5, 2, (n, 1))
    vl = -TRUE.pattern(pl) * rng.uniform(0.5, 2, (n, 1))
    vw += rng.normal(0, noise, vw.shape) * np.abs(vw).max(1, keepdims=True)
    vl += rng.normal(0, noise, vl.shape) * np.abs(vl).max(1, keepdims=True)
    swap = rng.random(n) < 0.5  # video fish a / b in random order
    pa = np.where(swap[:, None, None], pl, pw)
    pb = np.where(swap[:, None, None], pw, pl)
    model, a_is_winner = idn.fit_field(vw, vl, pa, pb, p=idn.FitParams(max_samples=800))
    assert np.mean(a_is_winner == ~swap) > 0.99
    assert np.median(np.hypot(*(model.electrodes - TRUE.electrodes).T)) < 20
    assert np.allclose(model.gain / model.gain.max(), TRUE.gain / TRUE.gain.max(), atol=0.1)


def two_fish_tracks(pw, pl):
    rows = []
    for i in range(len(pw)):
        for tk, p in ((0, pw[i]), (1, pl[i])):
            r = {"frame": 2 * i, "t_s": 0.08 * i, "track": tk}
            for j, name in enumerate(idn.BODY):
                r[f"{name}_x"], r[f"{name}_y"] = p[j]
            rows.append(r)
    return pd.DataFrame(rows)


def test_observations_joint_evidence_points_to_the_right_fish():
    rng = np.random.default_rng(3)
    n = 50
    pw, pl = random_poses(n, rng), random_poses(n, rng)
    t_win = 0.08 * np.arange(n)
    obs = idn.observations(two_fish_tracks(pw, pl), t_win, TRUE.pattern(pw), 3 * TRUE.pattern(pl), TRUE, bl=200.0)
    assert len(obs) == 2 * n
    w = obs[obs["track"] == 0].sort_values("frame")
    lo = obs[obs["track"] == 1].sort_values("frame")
    assert np.all(w["delta"].to_numpy() > 0) and np.allclose(w["delta"].to_numpy(), -lo["delta"].to_numpy())
    assert np.all(w["other"] == 1) and np.all(np.isfinite(w["dist_bl"])) and np.all(np.isfinite(w["contact_bl"]))


def test_localize_finds_a_fish():
    rng = np.random.default_rng(4)
    p = random_poses(3, rng)
    found, score = idn.localize(TRUE, TRUE.pattern(p), 200.0, (200, 250, 1800, 1300), grid_px=10, n_heading=36)
    assert np.all(np.hypot(*(found - idn.body_point(p, TRUE.center)).T) < 20)
    assert np.all(score > 0.99)


def test_fit_beta_recovers_the_calibration():
    rng = np.random.default_rng(5)
    S = rng.uniform(-2, 2, 20000)
    correct = rng.random(len(S)) < idn.probability(S, 3.0)
    assert abs(idn.fit_beta(S, correct) - 3.0) < 0.3


def test_hmm_without_switches_adds_evidence():
    e = np.array([0.5, 0.5, -0.2, 0.5])
    assert np.allclose(idn.hmm_logodds(e, np.full(4, 1e-12)), e.sum())


def test_hmm_with_free_switches_keeps_each_row_alone():
    e = np.array([0.5, -1.0, 2.0])
    assert np.allclose(idn.hmm_logodds(e, np.full(3, 0.5)), e)


def test_hmm_follows_a_switch_at_a_risky_moment():
    e = np.r_[np.full(20, 1.0), np.full(20, -1.0)]
    eps = np.full(40, 1e-6)
    eps[20] = 0.1
    lo = idn.hmm_logodds(e, eps)
    assert np.all(lo[:20] > 3) and np.all(lo[20:] < -3)


def test_risky_and_switch_probability():
    t = np.array([0.0, 0.08, 0.16, 0.8, 0.88, 0.96, 1.04])
    contact = np.array([np.nan, np.nan, np.nan, np.nan, 0.2, 0.3, np.nan])
    r = idn.risky(t, contact)
    assert r.tolist() == [False, False, False, True, True, True, True]
    eps = idn.switch_probability(r, 0.1, 1e-4)
    assert np.isclose(1 - np.prod(1 - eps[3:]), 0.1)
    assert np.allclose(eps[1:3], 1e-4)


def test_label_segments_runs():
    obs = pd.DataFrame({"track": [1, 1, 1, 1, 2, 2], "logodds": [5.0, 5.0, 0.0, -5.0, -5.0, -5.0]})
    lab = idn.label_segments(obs, 0.9)
    assert lab["label"].tolist() == ["winner", "winner", "unknown", "loser", "loser", "loser"]
    assert lab["segment"].tolist() == [0, 0, 1, 2, 0, 0]


def test_run_truth_bridges_a_contact_only_if_both_sides_agree():
    # track 1: far winner - contact - far winner; track 2: far winner - contact - far loser (a swap)
    rows = []
    for tk, after in ((1, 1.0), (2, -1.0)):
        t = 0.0
        for part, d, n in (("a", 1.0, 60), ("c", 0.0, 10), ("b", after, 60)):
            for _ in range(n):
                rows.append({"track": tk, "frame": int(round(t * 12.5)), "t_s": t,
                             "delta": 0.0 if part == "c" else 0.8 * d, "dist_bl": 0.5 if part == "c" else 5.0,
                             "n_vis": 2, "contact_bl": 0.1 if part == "c" else np.nan})
                t += 0.08
    obs = pd.DataFrame(rows).sort_values(["track", "frame"], ignore_index=True)
    obs["risky"] = np.concatenate([idn.risky(g["t_s"].to_numpy(), g["contact_bl"].to_numpy())
                                   for _, g in obs.groupby("track")])
    obs["run"] = idn.runs(obs)
    truth = idn.run_truth(obs)
    contact_runs = obs[obs["risky"]].groupby("track")["run"].first()
    assert truth[contact_runs[1]] == 1
    assert contact_runs[2] not in truth.index
