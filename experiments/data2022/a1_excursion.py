"""Does the box contain an upward frequency excursion of its fish? A check of the "swelling" false positives.

Many detections the expert judged "not a chirp" at scores 0.5-0.9 look, in
the spectrogram, like a brief symmetric brightening of a fish's baseline (an
amplitude change), not like a chirp (a brief upward excursion of the fish's
EOD frequency). This script measures, independently of the detector, the
instantaneous frequency of the assigned fish around each labelled candidate:

1. electrode: the one where the fish's power is highest relative to every
   other live fish (one FFT of the 0.5 s before the box, power within +-3 Hz
   of each fish's track frequency);
2. band-pass that electrode (4th-order Butterworth, zero phase) from 20 Hz
   below the fish's baseline to ``min(baseline + 250 Hz, other fish - 15 Hz)``
   for another fish above it (and at least 15 Hz above another fish below);
3. instantaneous frequency from the analytic signal's phase, smoothed over
   3 ms; amplitude envelope likewise;
4. ``rise_hz``: maximum of the smoothed instantaneous frequency within the box
   (+-20 ms) minus its median over 0.5 s before the box (excluding 50 ms next
   to it), only where the envelope is at least 30 % of its pre-box median (the
   phase of a collapsed envelope is noise); ``drop_db``: the envelope's minimum
   in the box relative to the same pre-box median; ``fall_hz``: the most
   negative excursion.

On real data this is an independent estimate (not ground truth). The script
reports how well ``rise_hz`` separates the expert's chirps from his
not-chirps (ROC AUC), per dataset, for the 0.5-0.9 bins.

    .venv/bin/python experiments/data2022/a1_excursion.py

Output: ``$CHIRP_DATA/data2022/a1/excursion/`` (``items.csv``, ``auc.csv``, ``excursion.png``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutouts as co
from paths import DATA

SCORED = DATA / "labels" / "real_v1" / "report" / "candidates_scored.csv"
OUT = DATA / "data2022" / "a1" / "excursion"
PRE_S, GAP_S, MARGIN_S, SMOOTH_S = 0.5, 0.05, 0.02, 0.003
MIN_ENV_REL = 0.3  # instantaneous frequency only where the envelope is >= 30 % of its pre-box median
BELOW_HZ, ABOVE_HZ, GUARD_HZ = 20.0, 250.0, 15.0
POSITIVE = ("chirp", "chirp_emitter_unclear")


def best_electrode(x: np.ndarray, fs: float, f_hz: float, others_hz: list[float]) -> int:
    """Electrode where the fish (at ``f_hz``) is strongest relative to every other live fish.

    ``x``: samples (n, electrodes) before the box; power per electrode at each
    fish's frequency from one Hann-windowed FFT (+-3 Hz).
    """
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)

    def power(fr: float) -> np.ndarray:
        return 10 * np.log10(spec[np.abs(f - fr) <= 3.0].max(axis=0) + 1e-30)

    own = power(f_hz)
    rel = own.copy()
    for o in others_hz:
        rel = np.minimum(rel, own - power(o))
    return int(np.argmax(rel))


def measure(row: pd.Series) -> dict:
    rec, tracks = co.open_rec(row["recording"]), co.open_tracks(row["recording"])
    ident = row["track_id"]
    out = {"candidate_id": row["candidate_id"], "electrode": -1, "rise_hz": np.nan, "fall_hz": np.nan,
           "drop_db": np.nan, "band_low_hz": np.nan, "band_high_hz": np.nan}
    if not np.isfinite(ident):
        return out
    live = tracks.frequencies_at(row["t_start_s"], 10.0)
    if int(ident) not in live:
        return out
    fb = live[int(ident)]
    others = {i: f for i, f in live.items() if i != int(ident)}
    lo, hi = fb - BELOW_HZ, fb + ABOVE_HZ
    for f in others.values():
        if f > fb:
            hi = min(hi, f - GUARD_HZ)
        else:
            lo = max(lo, f + GUARD_HZ)
    if hi - lo < 25.0:
        return out
    fs = rec.samplerate_hz
    t0 = max(0.0, row["t_start_s"] - PRE_S - GAP_S - 0.1)
    t1 = min(rec.duration_s, row["t_end_s"] + 0.1)
    xa = rec.read(int(t0 * fs), int(t1 * fs)).astype(np.float64)
    n_pre = int((row["t_start_s"] - GAP_S - t0) * fs)
    if n_pre < int(0.2 * fs):
        return out
    el = best_electrode(xa[:n_pre], fs, fb, list(others.values()))
    x = xa[:, el]
    sos = butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
    z = hilbert(sosfiltfilt(sos, x))
    n = max(1, round(SMOOTH_S * fs))
    kern = np.ones(n) / n
    inst = np.convolve(np.diff(np.unwrap(np.angle(z))) * fs / (2 * np.pi), kern, "same")
    env = np.convolve(np.abs(z[1:]), kern, "same")
    t = t0 + (np.arange(len(inst)) + 0.5) / fs
    pre = (t >= row["t_start_s"] - GAP_S - PRE_S) & (t < row["t_start_s"] - GAP_S)
    box = (t >= row["t_start_s"] - MARGIN_S) & (t <= row["t_end_s"] + MARGIN_S)
    if pre.sum() < 100 or box.sum() < 100:
        return out
    f_pre, a_pre = np.median(inst[pre]), np.median(env[pre])
    # the phase of a band-passed signal is unreliable where its envelope collapses
    ok = box & (env >= MIN_ENV_REL * a_pre)
    out.update(electrode=el, band_low_hz=lo, band_high_hz=hi,
               drop_db=float(20 * np.log10(env[box].min() / a_pre)))
    if ok.sum() >= 20:
        out.update(rise_hz=float(inst[ok].max() - f_pre), fall_hz=float(inst[ok].min() - f_pre))
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > a random negative), ties half."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def plot(items: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, constrained_layout=True)
    for ax, ds in zip(axes, ("competition2019", "competition2022"), strict=True):
        g = items[(items["dataset"] == ds) & items["mid"]]
        for k, (_label, colour) in enumerate((("not a chirp", "#d95f0e"), ("chirp", "#2b8cbe"))):
            v = g.loc[g["y"] == k, "rise_hz"].dropna()
            ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.2, 0.2, len(v)), v.clip(-20, 300),
                       s=14, alpha=0.6, color=colour, edgecolors="none")
            ax.text(k, 305, f"n = {len(v)}", ha="center", fontsize=8)
        a = auc(g.loc[g["y"] == 1, "rise_hz"].dropna().to_numpy(), g.loc[g["y"] == 0, "rise_hz"].dropna().to_numpy())
        ax.set_xticks([0, 1], ["judged not a chirp", "judged a chirp"])
        ax.set_title(f"{'2019/2020' if ds == 'competition2019' else '2022'}, scores 0.5-0.9 (AUC {a:.2f})", fontsize=10)
        ax.set_xlim(-0.6, 1.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("largest upward frequency excursion in the box (Hz)\n(instantaneous frequency of the assigned fish)")
    fig.suptitle("A simple frequency-excursion check separates the expert's decisions only weakly", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(SCORED)
    t = t[t["labelled"] & t["stratum"].str.startswith("main/") & (t["score"] >= 0.2)]
    t = t.drop_duplicates("candidate_id").sort_values(["recording", "t_center_s"])
    rows = []
    for k, (_, r) in enumerate(t.iterrows()):
        rows.append(measure(r))
        if k % 50 == 0:
            print(f"{k}/{len(t)}", flush=True)
    m = pd.DataFrame(rows)
    items = t[["candidate_id", "dataset", "recording", "score", "score_bin", "status", "label", "w",
               "height_hz", "fwhm_s", "track_hz", "emitter_hz"]].merge(m, on="candidate_id")
    items["y"] = items["label"].isin(POSITIVE).astype(int)
    items["mid"] = (items["score"] >= 0.5) & (items["score"] < 0.9)
    items.to_csv(OUT / "items.csv", index=False)
    res = []
    for ds in ("competition2019", "competition2022"):
        for name, sel in (("0.5-0.9", items["mid"]), (">=0.9", items["score"] >= 0.9),
                          ("0.2-0.5", items["score"] < 0.5)):
            g = items[(items["dataset"] == ds) & sel & items["rise_hz"].notna()]
            for feat in ("rise_hz", "drop_db", "height_hz"):
                sign = -1 if feat == "drop_db" else 1
                res.append({"dataset": ds, "scores": name, "feature": feat, "n_chirp": int(g["y"].sum()),
                            "n_not": int((1 - g["y"]).sum()),
                            "auc": auc(sign * g.loc[g["y"] == 1, feat].to_numpy(),
                                       sign * g.loc[g["y"] == 0, feat].to_numpy())})
    res = pd.DataFrame(res)
    res.to_csv(OUT / "auc.csv", index=False)
    plot(items, OUT / "excursion.png")
    print(res.round(3).to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
