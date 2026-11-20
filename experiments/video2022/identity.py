"""Which video track is the winner and which the loser: evidence from the electrodes.

The 11 electrodes of the 2022 tank are not a line (``fishgrid.cfg`` says 1 x 11 only so that
the software records 11 channels). They are visible in the video: 8 hang on the walls
(3 top, 3 bottom, 1 left, 1 right), 1 sits at the left end of the tube and 2 on the
suction-cup discs left and right of the tube (:data:`ELECTRODES_IMAGE`, channel order
established from the data, see ``d1_identity.py``). The recordings are differential against
a reference outside the camera view (fitted, bottom left).

Model (:class:`FieldModel`). A fish is a point dipole on its body axis, at fraction ``center``
of the way from snout to tail, pointing from snout to tail. Its potential at electrode ``e``
in a shallow layer of water (2-D field, softened by ``height`` px):

    phi(e) = (e - c) . u / (|e - c|^2 + height^2)

and channel ``i`` records ``gain_i * (phi(e_i) - phi(ref))``. The EOD is quasi-static, so a
fish's amplitudes on the 11 electrodes form a real, signed pattern (``eod_amplitudes``); the
model predicts it up to one free factor (EOD strength, sign of the oscillation).

Fit of a pattern to a pose: ``cos2(v, m)``, the share of the observed pattern ``v`` explained
by the model pattern ``m`` (1 = perfect, 0 = orthogonal).

Evidence per window (``delta``, between -1 and 1; > 0 means "this track is the winner"):

- one track visible: ``cos2(v_winner, m_track) - cos2(v_loser, m_track)``;
- two tracks visible (a, b): the joint hypotheses (a = winner, b = loser) against the swap,
  ``(cos2(v_w, m_a) + cos2(v_l, m_b) - cos2(v_w, m_b) - cos2(v_l, m_a)) / 2``;
  ``delta`` of b is minus that of a.

Per track, a two-state hidden Markov chain over its windows (:func:`track_posterior`) adds the
evidence up (``beta`` log-odds per second of evidence, calibrated as ``logit p = beta * |S|`` on pieces
of tracks with a known identity, :func:`fit_beta`) and lets the identity change only where the
tracker may have swapped the fish (contacts, gaps). Validation helpers: :func:`run_truth`,
:func:`chop`, :func:`pairing_test`; electric-only localisation: :func:`localize`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BODY = ("snout", "spine1", "spine2", "spine3", "tail")
N_CH = 11
# Electrode positions (px) in the background image (the electrode holders), channels 0-10.
ELECTRODES_IMAGE = np.array([
    [388, 1331],   # 0 bottom wall, left
    [969, 1321],   # 1 bottom wall, middle
    [1601, 1316],  # 2 bottom wall, right
    [128, 770],    # 3 left wall
    [714, 780],    # 4 disc left of the tube
    [1306, 775],   # 5 disc right of the tube
    [1867, 750],   # 6 right wall
    [388, 214],    # 7 top wall, left
    [969, 214],    # 8 top wall, middle
    [1581, 204],   # 9 top wall, right
    [867, 755],    # 10 left end of the tube
], float)
ELECTRODE_NAMES = ("bottom left", "bottom middle", "bottom right", "left wall", "disc left",
                   "disc right", "right wall", "top left", "top middle", "top right", "tube")
REF_INIT = np.array([190.0, 1650.0])


# ----------------------------------------------------------------------------- field model
def poses(d: pd.DataFrame) -> np.ndarray:
    """Body points snout, spine1-3, tail of each detection: (n, 5, 2) px."""
    return np.stack([d[[f"{n}_x", f"{n}_y"]].to_numpy(float) for n in BODY], axis=1)


def body_point(pose: np.ndarray, f: float) -> np.ndarray:
    """Point at fraction ``f`` of the snout -> tail polyline index (0 = snout, 1 = tail): (n, 2)."""
    seg = np.linspace(0, 1, pose.shape[1])
    i = int(np.clip(np.searchsorted(seg, f) - 1, 0, len(seg) - 2))
    w = (f - seg[i]) / (seg[i + 1] - seg[i])
    return pose[:, i] * (1 - w) + pose[:, i + 1] * w


@dataclass(frozen=True)
class FieldModel:
    electrodes: np.ndarray = field(default_factory=lambda: ELECTRODES_IMAGE.copy())
    gain: np.ndarray = field(default_factory=lambda: np.ones(N_CH))
    ref: np.ndarray = field(default_factory=lambda: REF_INIT.copy())
    height: float = 50.0
    center: float = 0.65

    def potential(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Potential of the dipole of each pose at ``points`` (m, 2): (n, m)."""
        c = body_point(pose, self.center)
        u = pose[:, -1] - pose[:, 0]
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        dv = points[None] - c[:, None]
        return (dv * u[:, None]).sum(-1) / ((dv ** 2).sum(-1) + self.height ** 2)

    def pattern(self, pose: np.ndarray) -> np.ndarray:
        """Predicted signed amplitudes on the 11 channels (arbitrary scale): (n, 11)."""
        phi = self.potential(self.electrodes, pose)
        return (phi - self.potential(self.ref[None], pose)) * self.gain

    def to_dict(self) -> dict:
        return {"electrodes": self.electrodes.tolist(), "gain": self.gain.tolist(), "ref": self.ref.tolist(),
                "height": self.height, "center": self.center}

    @classmethod
    def from_dict(cls, d: dict) -> FieldModel:
        return cls(np.asarray(d["electrodes"], float), np.asarray(d["gain"], float), np.asarray(d["ref"], float),
                   float(d["height"]), float(d["center"]))


def cos2(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Share of each row of ``v`` explained by the matching row of ``m`` (sign-free), NaN-safe."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.sum(v * m, -1) ** 2 / (np.sum(v * v, -1) * np.sum(m * m, -1))


# ----------------------------------------------------------------------------- calibration of the model
@dataclass(frozen=True)
class FitParams:
    prior_px: float = 60.0  # electrode positions are pulled towards the image positions with this sd
    prior_weight: float = 0.1
    max_samples: int = 3000
    iterations: int = 3
    max_nfev: int = 300  # cap per least-squares fit (a flat optimum can otherwise crawl for an hour)


def _pack(m: FieldModel) -> np.ndarray:
    return np.r_[m.electrodes.ravel(), np.log(m.gain), m.ref, m.height, m.center]


def _unpack(x: np.ndarray) -> FieldModel:
    return FieldModel(x[:22].reshape(N_CH, 2), np.exp(x[22:33]), x[33:35], float(x[35]), float(x[36]))


def _residual_explained(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    mn = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-30)
    return vn - np.sum(vn * mn, 1, keepdims=True) * mn


def joint_cos2(model: FieldModel, vw: np.ndarray, vl: np.ndarray, pa: np.ndarray, pb: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Mean explained share of the two fish under (a = winner, b = loser) and under the swap."""
    ma, mb = model.pattern(pa), model.pattern(pb)
    return (cos2(vw, ma) + cos2(vl, mb)) / 2, (cos2(vw, mb) + cos2(vl, ma)) / 2


def fit_field(vw: np.ndarray, vl: np.ndarray, pa: np.ndarray, pb: np.ndarray, init: FieldModel | None = None,
              p: FitParams | None = None) -> tuple[FieldModel, np.ndarray]:
    """Fit the field model on windows with two visible, well separated fish (hard EM).

    ``vw``, ``vl``: (n, 11) signed patterns of winner and loser; ``pa``, ``pb``: (n, 5, 2) poses of
    the two video fish. Which video fish is which is unknown: every iteration assigns each window to
    the better of the two hypotheses, then refits. Returns the model and the final assignment
    (True = a is the winner).
    """
    p = p or FitParams()
    model = init or FieldModel()
    x = _pack(model)
    x_img = ELECTRODES_IMAGE.ravel()
    sub = np.arange(len(vw))[:: max(1, len(vw) // p.max_samples)]
    for _ in range(p.iterations):
        ca, cb = joint_cos2(_unpack(x), vw[sub], vl[sub], pa[sub], pb[sub])
        h = ca >= cb
        pw = np.where(h[:, None, None], pa[sub], pb[sub])
        pl = np.where(h[:, None, None], pb[sub], pa[sub])

        def resid(xx, pw=pw, pl=pl):
            m = _unpack(xx)
            return np.concatenate([_residual_explained(vw[sub], m.pattern(pw)).ravel(),
                                   _residual_explained(vl[sub], m.pattern(pl)).ravel(),
                                   p.prior_weight * (xx[:22] - x_img) / p.prior_px])

        scale = np.r_[np.full(22, 50.0), np.ones(N_CH), 200, 200, 20, 0.1]
        lb = np.r_[np.full(22, -np.inf), np.full(N_CH, -15), -np.inf, -np.inf, 1.0, 0.05]
        ub = np.r_[np.full(22, np.inf), np.full(N_CH, 15), np.inf, np.inf, 1000.0, 0.95]
        x = least_squares(resid, np.clip(x, lb + 1e-9, ub - 1e-9), x_scale=scale, bounds=(lb, ub),
                          max_nfev=p.max_nfev).x
    model = _unpack(x)
    model = replace(model, gain=model.gain / model.gain.max())
    ca, cb = joint_cos2(model, vw, vl, pa, pb)
    return model, ca >= cb


# ----------------------------------------------------------------------------- per-window evidence
def match_windows(t_frames: np.ndarray, t_windows: np.ndarray, tol: float = 0.041) -> np.ndarray:
    """Index of the window nearest to each frame time, -1 if none within ``tol`` s (windows evenly spaced)."""
    step = t_windows[1] - t_windows[0]
    k = np.round((t_frames - t_windows[0]) / step).astype(int)
    k = np.clip(k, 0, len(t_windows) - 1)
    return np.where(np.abs(t_windows[k] - t_frames) <= tol, k, -1)


def observations(tracks: pd.DataFrame, t_windows: np.ndarray, vw: np.ndarray, vl: np.ndarray,
                 model: FieldModel, bl: float) -> pd.DataFrame:
    """One row per video detection matched to an electrode window, with the evidence ``delta``.

    Columns: ``frame``, ``t_s``, ``k`` (window), ``track``, ``x``, ``y`` (spine1, px), ``n_vis``
    (detections in this frame), ``other`` (the other track if exactly two), ``dist_bl`` (spine1 to
    spine1, body lengths; NaN unless two), ``cw``, ``cl`` (explained share of the winner's / loser's
    pattern by this detection's pose), ``delta`` (see module doc; NaN where a pattern is missing).
    """
    d = tracks.sort_values(["frame", "track"]).reset_index(drop=True)
    k = match_windows(d["t_s"].to_numpy(float), t_windows)
    d = d[k >= 0].reset_index(drop=True)
    k = k[k >= 0]
    pose = poses(d)
    m = model.pattern(pose)
    cw, cl = cos2(vw[k], m), cos2(vl[k], m)
    n_vis = d.groupby("frame")["track"].transform("size").to_numpy()
    out = pd.DataFrame({"frame": d["frame"].to_numpy(), "t_s": d["t_s"].to_numpy(float), "k": k,
                        "track": d["track"].to_numpy(), "x": d["spine1_x"].to_numpy(float),
                        "y": d["spine1_y"].to_numpy(float), "n_vis": n_vis, "cw": cw, "cl": cl})
    out["other"] = -1
    out["dist_bl"] = np.nan
    out["contact_bl"] = np.nan
    delta = cw - cl
    two = np.flatnonzero(n_vis == 2)
    a, b = two[0::2], two[1::2]  # rows are sorted by frame: pairs are adjacent
    if len(a):
        assert np.all(out["frame"].to_numpy()[a] == out["frame"].to_numpy()[b])
        joint = (cw[a] + cl[b] - cw[b] - cl[a]) / 2
        delta[a], delta[b] = joint, -joint
        out.loc[a, "other"] = out["track"].to_numpy()[b]
        out.loc[b, "other"] = out["track"].to_numpy()[a]
        dist = np.hypot(out["x"].to_numpy()[a] - out["x"].to_numpy()[b],
                        out["y"].to_numpy()[a] - out["y"].to_numpy()[b]) / bl
        out.loc[a, "dist_bl"] = dist
        out.loc[b, "dist_bl"] = dist
        contact = np.linalg.norm(pose[a][:, :, None] - pose[b][:, None, :], axis=-1).min(axis=(1, 2)) / bl
        out.loc[a, "contact_bl"] = contact
        out.loc[b, "contact_bl"] = contact
    out["delta"] = delta
    return out


# ----------------------------------------------------------------------------- switches inside tracks
# The tracker sometimes swaps the two fish where they touch (88 % of the sign changes of the evidence
# inside tracks of two test recordings lay within 1 s of a contact, while 4 % of the windows are
# contacts). A track's identity is therefore a two-state hidden Markov chain over its windows that may
# switch with probability ``eps_episode`` per contact episode (or gap), and almost never elsewhere.
def risky(t: np.ndarray, contact_bl: np.ndarray, gap_s: float = 0.3, contact_max: float = 0.5) -> np.ndarray:
    """Per row of one track (time order): True if the track may have jumped to the other fish between
    the previous row and this one (a gap > ``gap_s`` or a contact <= ``contact_max`` BL in either row)."""
    t = np.asarray(t, float)
    c = np.asarray(contact_bl, float) <= contact_max
    out = np.zeros(len(t), bool)
    out[1:] = (np.diff(t) > gap_s) | c[1:] | c[:-1]
    return out


def switch_probability(risk: np.ndarray, eps_episode: float, eps_calm: float) -> np.ndarray:
    """Switch probability of each transition: a run of ``n`` risky transitions (one episode) gets
    ``eps_episode`` in total, every other transition ``eps_calm``."""
    eps = np.full(len(risk), eps_calm)
    starts = np.flatnonzero(risk & ~np.r_[False, risk[:-1]])
    ends = np.flatnonzero(risk & ~np.r_[risk[1:], False])
    for a, b in zip(starts, ends, strict=True):
        n = b - a + 1
        eps[a:b + 1] = 1.0 - (1.0 - eps_episode) ** (1.0 / n)
    eps[0] = 0.5  # no transition into the first row
    return eps


def _carry(x: float, eps: float) -> float:
    """Log-odds after one step of a symmetric two-state chain with switch probability ``eps``."""
    if eps >= 0.5:
        return 0.0
    le, l1 = math.log(eps), math.log1p(-eps)
    return float(np.logaddexp(l1 + x, le) - np.logaddexp(le + x, l1))


def hmm_logodds(e: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Posterior log-odds (state 1 vs 0) of each row of a two-state chain (forward-backward).

    ``e``: per-row evidence as log-likelihood ratio; ``eps[i]``: switch probability between rows i-1, i.
    """
    e = np.nan_to_num(np.asarray(e, float))
    n = len(e)
    fwd = np.empty(n)
    bwd = np.zeros(n)
    x = 0.0
    for i in range(n):
        x = (_carry(x, eps[i]) if i else 0.0) + e[i]
        fwd[i] = x
    for i in range(n - 2, -1, -1):
        bwd[i] = _carry(e[i + 1] + bwd[i + 1], eps[i + 1])
    return fwd + bwd


def track_posterior(obs: pd.DataFrame, beta: float, eps_episode: float, eps_calm: float,
                    dt: float = 0.08, col: str = "delta") -> pd.DataFrame:
    """``obs`` sorted by track and frame, with ``risky``, ``eps`` and ``logodds`` (winner vs loser)."""
    obs = obs.sort_values(["track", "frame"]).reset_index(drop=True)
    lo = np.zeros(len(obs))
    rk = np.zeros(len(obs), bool)
    ep = np.zeros(len(obs))
    for _, idx in obs.groupby("track").indices.items():
        g = obs.iloc[idx]
        r = risky(g["t_s"].to_numpy(), g["contact_bl"].to_numpy())
        eps = switch_probability(r, eps_episode, eps_calm)
        lo[idx] = hmm_logodds(beta * dt * g[col].to_numpy(), eps)
        rk[idx], ep[idx] = r, eps
    return obs.assign(risky=rk, eps=ep, logodds=lo)


def label_segments(obs: pd.DataFrame, p_min: float) -> pd.DataFrame:
    """Per row a label (winner if P(winner) >= ``p_min``, loser if <= 1 - ``p_min``, else unknown) and
    ``segment``: runs of one label within a track, numbered from 0; ``seg`` = track * 1000 + segment."""
    p = 1.0 / (1.0 + np.exp(-obs["logodds"].to_numpy()))
    lab = np.where(p >= p_min, "winner", np.where(p <= 1 - p_min, "loser", "unknown"))
    new = np.r_[True, (lab[1:] != lab[:-1]) | (obs["track"].to_numpy()[1:] != obs["track"].to_numpy()[:-1])]
    seg = np.cumsum(new) - 1
    first = pd.Series(seg).groupby(obs["track"].to_numpy()).transform("min").to_numpy()
    segment = seg - first
    return obs.assign(p_winner=p, label=lab, segment=segment, seg=obs["track"].to_numpy() * 1000 + segment)


def runs(obs: pd.DataFrame) -> np.ndarray:
    """Run index per row (obs sorted by track, frame, with ``risky``): calm stretches and risky episodes
    alternate; a new run starts at every change of the risky flag and at every new track."""
    r = obs["risky"].to_numpy()
    tk = obs["track"].to_numpy()
    new = np.r_[True, (r[1:] != r[:-1]) | (tk[1:] != tk[:-1])]
    return np.cumsum(new) - 1


# ----------------------------------------------------------------------------- per-track evidence
def block_evidence(obs: pd.DataFrame, block_s: float = 1.0, by: str = "track") -> pd.DataFrame:
    """Mean ``delta`` per (``by``, 1-s block), with the block's median fish distance and window count."""
    o = obs[np.isfinite(obs["delta"])]
    o = o.assign(block=np.floor(o["t_s"] / block_s).astype(int))
    g = o.groupby([by, "block"])
    return g.agg(delta=("delta", "mean"), n=("delta", "size"), dist_bl=("dist_bl", "median"),
                 n_two=("n_vis", lambda s: int((s == 2).sum()))).reset_index()


def evidence_sum(blocks: pd.DataFrame, by: str = "track") -> pd.DataFrame:
    """Per ``by``: number of blocks, sum ``S`` and mean of the block deltas, share of blocks agreeing
    with the sign of ``S``."""
    g = blocks.groupby(by)["delta"]
    out = pd.DataFrame({"n_blocks": g.size(), "S": g.sum(), "mean_delta": g.mean()})
    sign = np.sign(out["S"])
    agree = blocks.assign(s=blocks[by].map(sign)).eval("delta * s > 0").groupby(blocks[by]).mean()
    out["agree"] = agree
    return out


def probability(S: np.ndarray, beta: float) -> np.ndarray:
    """P(the sign of S is right) under the calibration ``logit p = beta * |S|``."""
    return 1.0 / (1.0 + np.exp(-beta * np.abs(np.asarray(S, float))))


def fit_beta(S: np.ndarray, correct: np.ndarray) -> float:
    """Maximum-likelihood ``beta`` of ``P(correct) = 1 / (1 + exp(-beta |S|))``."""
    from scipy.optimize import minimize_scalar

    a = np.abs(np.asarray(S, float))
    c = np.asarray(correct, bool)

    def nll(beta):
        z = beta * a
        return np.sum(np.logaddexp(0, -z)[c]) + np.sum(np.logaddexp(0, z)[~c])

    return float(minimize_scalar(nll, bounds=(1e-3, 100.0), method="bounded").x)



# ----------------------------------------------------------------------------- validation
def is_far(blocks: pd.DataFrame, far_bl: float) -> np.ndarray:
    """Blocks in which both fish were in view in every window and at least ``far_bl`` apart (median)."""
    return ((blocks["n_two"] == blocks["n"]) & (blocks["dist_bl"] >= far_bl)).to_numpy()


def far_truth(blocks: pd.DataFrame, far_bl: float = 3.0, min_blocks: int = 3, min_agree: float = 0.8,
              by: str = "track") -> pd.Series:
    """Reference identity (+1 winner, -1 loser) of the tracks with clear evidence while far from the other
    fish: >= ``min_blocks`` far blocks of which >= ``min_agree`` agree in sign."""
    far = blocks[is_far(blocks, far_bl)]
    s = evidence_sum(far, by)
    s = s[(s["n_blocks"] >= min_blocks) & (s["agree"] >= min_agree)]
    return np.sign(s["S"]).astype(int)


def run_truth(obs: pd.DataFrame, far_bl: float = 3.0, min_blocks: int = 3, min_agree: float = 0.8
              ) -> pd.Series:
    """Reference identity (+1 / -1) per run (``obs`` with ``run``, ``risky``, sorted by track, frame).

    A calm run (no contact, no gap) gets the identity of its far blocks (:func:`far_truth`). A risky run
    (a contact episode) gets it if the calm runs just before and after it in the same track both have
    one and agree (the tracker did not swap the fish there, or swapped them twice).
    """
    b = block_evidence(obs, by="run")
    calm = ~obs.groupby("run")["risky"].first().astype(bool)
    truth = far_truth(b[b["run"].map(calm)], far_bl, min_blocks, min_agree, by="run")
    info = obs.groupby("run").agg(track=("track", "first"), risky=("risky", "first"))
    extra = {}
    for _, g in info.groupby("track"):
        ids = g.index.to_numpy()
        for j in range(1, len(ids) - 1):
            before, after = ids[j - 1], ids[j + 1]
            if (g["risky"].iloc[j] and before in truth.index and after in truth.index
                    and truth[before] == truth[after]):
                extra[ids[j]] = truth[before]
    return pd.concat([truth, pd.Series(extra, dtype=int)]).sort_index()


def chop(blocks: pd.DataFrame, truth: pd.Series, lengths: np.ndarray, rng: np.random.Generator,
         by: str = "track") -> pd.DataFrame:
    """Cut the blocks of each reference track into consecutive pieces with lengths drawn from ``lengths``.

    Returns one row per piece: ``by``, number of blocks, evidence sum ``S``, median fish distance,
    share of blocks with both fish in view, and whether the sign of ``S`` matches the reference.
    """
    rows = []
    for key, g in blocks[blocks[by].isin(truth.index)].groupby(by):
        g = g.sort_values("block")
        i = 0
        while i < len(g):
            n = int(rng.choice(lengths))
            piece = g.iloc[i:i + n]
            i += n
            S = float(piece["delta"].sum())
            rows.append({by: key, "n_blocks": len(piece), "S": S, "dist_bl": float(piece["dist_bl"].median()),
                         "share_two": float((piece["n_two"] > 0).mean()),
                         "correct": bool(np.sign(S) == truth[key]) if S != 0 else False})
    return pd.DataFrame(rows)


def pairing_test(model: FieldModel, vw: np.ndarray, pw: np.ndarray, vl: np.ndarray, pl: np.ndarray, bl: float,
                 bins_bl: np.ndarray, per_bin: int, rng: np.random.Generator) -> pd.DataFrame:
    """Joint evidence for two fish at a chosen distance, built from moments when they were far apart.

    The two fish's patterns are measured separately (each at its own EOD frequency), so the winner's
    pattern at one moment (``vw``, pose ``pw``) and the loser's at another (``vl``, ``pl``) together are
    what the electrodes would record with the fish at these two poses at once. For winner/loser
    windows whose spine1 positions lie at a distance within each bin of ``bins_bl``, the joint
    ``delta`` is computed as in :func:`observations`; > 0 is right. Returns one row per synthetic pair
    with the spine1 distance, the closest distance between the two skeletons (``contact_bl``) and
    whether the follower-like fish is behind the other (``tandem``: heads within 45 deg and one
    behind the other).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(pl[:, 1])
    rows = []
    order = rng.permutation(len(pw))
    want = {i: per_bin for i in range(len(bins_bl) - 1)}
    for i in order:
        if not any(want.values()):
            break
        cand = tree.query_ball_point(pw[i, 1], bins_bl[-1] * bl)
        if not cand:
            continue
        cand = np.asarray(cand)
        d = np.hypot(*(pl[cand, 1] - pw[i, 1]).T) / bl
        b = np.digitize(d, bins_bl) - 1
        for k in range(len(bins_bl) - 1):
            if want[k] and (b == k).any():
                j = cand[rng.choice(np.flatnonzero(b == k))]
                rows.append((i, j, k))
                want[k] -= 1
    if not rows:
        return pd.DataFrame()
    i, j, k = map(np.asarray, zip(*rows, strict=True))
    ma, mb = model.pattern(pw[i]), model.pattern(pl[j])
    delta = (cos2(vw[i], ma) + cos2(vl[j], mb) - cos2(vw[i], mb) - cos2(vl[j], ma)) / 2
    contact = np.linalg.norm(pw[i][:, :, None] - pl[j][:, None, :], axis=-1).min(axis=(1, 2)) / bl
    ua = pw[i, 0] - pw[i, -1]
    ub = pl[j, 0] - pl[j, -1]
    cosh = np.sum(ua * ub, 1) / (np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1))
    ab = pl[j, 1] - pw[i, 1]
    along = np.sum(ab * ua, 1) / np.linalg.norm(ua, axis=1) / np.linalg.norm(ab, axis=1).clip(1e-9)
    tandem = (cosh > np.cos(np.pi / 4)) & (np.abs(along) > np.cos(np.pi / 4))
    return pd.DataFrame({"bin": k, "dist_bl": np.hypot(*ab.T) / bl, "contact_bl": contact, "tandem": tandem,
                         "delta": delta, "correct": delta > 0})


ARC_FRACTIONS = np.array([0.0, 0.183, 0.412, 0.656, 1.0])  # median position of the body points (2022-06-20)


def straight_poses(spine1: np.ndarray, heading: np.ndarray, length: float) -> np.ndarray:
    """Straight fish with spine1 at ``spine1`` (n, 2) px, head pointing to ``heading`` (n, rad): (n, 5, 2)."""
    u = np.stack([np.cos(heading), np.sin(heading)], axis=1)  # snout direction
    s = spine1 + ARC_FRACTIONS[1] * length * u
    return s[:, None, :] - ARC_FRACTIONS[None, :, None] * length * u[:, None, :]


def localize(model: FieldModel, v: np.ndarray, length: float, arena: tuple[float, float, float, float],
             grid_px: float = 20.0, n_heading: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Electric-only estimate of where a fish is: the straight pose whose pattern explains ``v`` best.

    ``v`` (n, 11) patterns; ``arena`` (x0, y0, x1, y1) px. Grid search over spine1 positions and
    headings. Returns the dipole centre of the best pose (n, 2; the point at ``model.center`` of the
    body) and its explained share (n,). The centre, not the head: a pattern does not tell head from
    tail (a fish turned around at the same centre gives the same pattern with the opposite sign).
    """
    xs = np.arange(arena[0], arena[2] + 1, grid_px)
    ys = np.arange(arena[1], arena[3] + 1, grid_px)
    hs = np.arange(n_heading) * 2 * np.pi / n_heading
    X, Y, H = np.meshgrid(xs, ys, hs, indexing="ij")
    cand_poses = straight_poses(np.stack([X.ravel(), Y.ravel()], axis=1), H.ravel(), length)
    cand = body_point(cand_poses, model.center)
    m = model.pattern(cand_poses)  # (c, 11)
    mn = m / np.linalg.norm(m, axis=1, keepdims=True)
    best = np.full((len(v), 2), np.nan)
    score = np.full(len(v), np.nan)
    for i0 in range(0, len(v), 256):
        vv = v[i0:i0 + 256]
        ok = np.isfinite(vv).all(1)
        if not ok.any():
            continue
        vn = vv[ok] / np.linalg.norm(vv[ok], axis=1, keepdims=True)
        c2 = (vn @ mn.T) ** 2
        j = np.argmax(c2, axis=1)
        idx = np.flatnonzero(ok) + i0
        best[idx] = cand[j]
        score[idx] = c2[np.arange(len(j)), j]
    return best, score
