"""What the detector sees in 2019/2020 and in 2022: signal measures per recording.

For every recording of the two t001 runs (``competition2019_v2_t001``,
``competition2022_v2_t001``), 24 segments of 4 s are drawn at random from the
dark phase (recording time 60 .. 10,740 s; seeded by the recording's name;
recordings shorter than 10 min are skipped),
and in each segment:

- **spectrum**: Welch power per electrode (204.8 ms Hann windows, the
  detector's long window, 50 % overlap), summed over electrodes;
- **floor**: median of the summed spectrum over 300-2000 Hz;
- **fish**: every track live at the segment's centre (10-s gaps, the tracks
  the labeller showed); its peak = maximum of the summed spectrum within
  +-3 Hz of the track's frequency; ``snr_db`` = peak over floor; the number
  of electrodes on which the fish is within 10 dB of its best electrode;
- **lines**: spectral peaks of the summed spectrum in 250-2000 Hz at least
  10 dB above their surroundings (prominence) that no fish explains, i.e.
  further than 6 Hz from every ``a f1 + b f2`` (|a|, |b| <= 3; fundamentals,
  harmonics and intermodulation of the live fish); their number and the
  strongest one's power relative to the weakest fish's peak;
- **transients**: broadband clicks: the signal high-passed at 3 kHz, its
  power in 1-ms bins per electrode; an event is a run of bins at least 20 dB
  above that electrode's median; events per second (any electrode) and the
  share of events seen on at least half of the electrodes at once (what
  chirpdetector's click removal requires before it repairs anything).

Per recording, from the run's ``meta.json``: the fraction of samples the
click removal repaired; from the tracks: per trial fish (winner, loser) the
share of dark-phase seconds with a track detection, and the median
winner-loser frequency difference; from the detections: kept detections per
hour of dark phase by score bin and the assigned fish's trial status
(2019/2020: the gap-filled production run, the paper's assignment, scores
>= 0.05; 2022: the t001 run).

    .venv/bin/python experiments/data2022/a1_signal.py [--segments 24] [--only 2022-06-20-10_00 ...]

Output: ``$CHIRP_DATA/data2022/a1/signal/`` (``segments.csv``, ``fish.csv``,
``recordings.csv``, ``rates.csv``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutouts as co
from paths import DATA, REC_2019, REC_2022

OUT = DATA / "data2022" / "a1" / "signal"
RUNS = {
    "2019/2020": DATA / "detections" / "competition2019_v2_t001",
    "2022": DATA / "detections" / "competition2022_v2_t001",
}
RATES_2019 = DATA / "detections" / "competition2019_v2_20261113_gapfill"
DARK_S = (60.0, 10_740.0)
MIN_DURATION_S = 600.0
SEG_S = 4.0
WIN_S = 0.2048
FLOOR_BAND = (300.0, 2000.0)
LINE_BAND = (250.0, 2000.0)
LINE_PROMINENCE_DB = 10.0
LINE_EXPLAINED_HZ = 6.0
FISH_HALF_WIDTH_HZ = 3.0
CLICK_HP_HZ = 3000.0
CLICK_DB = 20.0
SCORE_BINS = [0.05, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0001]
SCORE_NAMES = ["0.05-0.2", "0.2-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1"]


def trial_fish() -> dict[str, dict[str, float]]:
    """recording -> {"winner": track id, "loser": track id} (2019/2020 analysed and draw trials, 2022 meta.csv)."""
    out: dict[str, dict[str, float]] = {}
    t = pd.read_csv(REC_2019 / "trial_summary.csv")
    for r in t.itertuples():
        out[r.recording] = {"winner": float(r.win_ID), "loser": float(r.lose_ID)}
    m = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"})
    for r in m.itertuples():
        out[r.recording] = {"winner": float(r.Win_ID), "loser": float(r.Lose_ID)}
    return out


def segment_times(name: str, n: int, duration_s: float) -> np.ndarray:
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(DARK_S[0], min(DARK_S[1], duration_s) - SEG_S, n))


def explained(f: float, fish_hz: list[float]) -> bool:
    """Is ``f`` within LINE_EXPLAINED_HZ of a fundamental, harmonic or intermodulation product of the fish?"""
    if not fish_hz:
        return False
    combos = set()
    if len(fish_hz) == 1:
        combos = {a * fish_hz[0] for a in (1, 2, 3)}
    else:
        for i, f1 in enumerate(fish_hz):
            for f2 in fish_hz[i + 1:]:
                for a in range(-3, 4):
                    for b in range(-3, 4):
                        if a or b:
                            combos.add(abs(a * f1 + b * f2))
    return any(abs(f - c) <= LINE_EXPLAINED_HZ for c in combos)


def transients(x: np.ndarray, fs: float) -> tuple[float, float]:
    """(events per second on any electrode, share of events on >= half the electrodes)."""
    sos = butter(4, CLICK_HP_HZ, "highpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x, axis=0)
    nb = int(fs * 1e-3)
    n = y.shape[0] // nb
    p = (y[: n * nb] ** 2).reshape(n, nb, -1).mean(axis=1)
    hot = p > np.median(p, axis=0) * 10 ** (CLICK_DB / 10)
    any_hot = hot.any(axis=1)
    starts = np.flatnonzero(any_hot & ~np.r_[False, any_hot[:-1]])
    stops = np.flatnonzero(any_hot & ~np.r_[any_hot[1:], False]) + 1
    if len(starts) == 0:
        return 0.0, np.nan
    wide = [hot[a:b].any(axis=0).sum() >= 0.5 * hot.shape[1] for a, b in zip(starts, stops, strict=True)]
    return len(starts) / (x.shape[0] / fs), float(np.mean(wide))


def measure_segment(rec, tracks, t0: float) -> tuple[dict, list[dict]]:
    fs = rec.samplerate_hz
    x = rec.read(int(t0 * fs), int((t0 + SEG_S) * fs)).astype(np.float64)
    f, p = welch(x, fs=fs, nperseg=round(WIN_S * fs), axis=0)
    ps = p.sum(axis=1)
    fb = (f >= FLOOR_BAND[0]) & (f <= FLOOR_BAND[1])
    floor = float(np.median(ps[fb]))
    live = tracks.frequencies_at(t0 + SEG_S / 2, 10.0)
    fish = []
    for ident, fr in sorted(live.items(), key=lambda kv: kv[1]):
        m = np.abs(f - fr) <= FISH_HALF_WIDTH_HZ
        if not m.any():
            continue
        k = np.flatnonzero(m)[np.argmax(ps[m])]
        per_el = 10 * np.log10(p[k] + 1e-30)
        fish.append({"track_id": float(ident), "f_hz": fr, "peak": float(ps[k]),
                     "snr_db": 10 * np.log10(ps[k] / floor),
                     "n_electrodes_10db": int((per_el >= per_el.max() - 10).sum()),
                     "best_electrode": int(np.argmax(per_el))})
    lb = (f >= LINE_BAND[0]) & (f <= LINE_BAND[1])
    db = 10 * np.log10(ps + 1e-30)
    idx, _ = find_peaks(db[lb], prominence=LINE_PROMINENCE_DB)
    fl = f[lb][idx]
    unexplained = [i for i, fv in enumerate(fl) if not explained(fv, [d["f_hz"] for d in fish])]
    weakest = min((d["peak"] for d in fish), default=np.nan)
    strongest_line = max((ps[lb][idx[i]] for i in unexplained), default=np.nan)
    rate, wide = transients(x, fs)
    seg = {"t0_s": t0, "n_fish": len(fish), "floor": floor, "n_peaks": len(fl),
           "n_unexplained_lines": len(unexplained),
           "unexplained_line_hz": ";".join(f"{fl[i]:.0f}" for i in unexplained),
           "strongest_line_vs_weakest_fish_db": 10 * np.log10(strongest_line / weakest),
           "strongest_line_vs_floor_db": 10 * np.log10(strongest_line / floor),
           "transients_per_s": rate, "transients_wide_share": wide}
    return seg, fish


def track_coverage(tracks, ident: float) -> float:
    """Share of dark-phase seconds with at least one detection of the track."""
    try:
        tr = tracks.track(int(ident))
    except (KeyError, ValueError):
        return 0.0
    t = tr.times_s[(tr.times_s >= 0) & (tr.times_s < 10_800)]
    return len(np.unique(np.floor(t))) / 10_800


def detection_rates(dataset: str, name: str, fish: dict[str, float]) -> pd.DataFrame:
    path = (RATES_2019 / name / "chirps_rtdetr_gapfill.parquet" if dataset == "2019/2020"
            else RUNS["2022"] / name / "chirps_rtdetr.parquet")
    d = pd.read_parquet(path, columns=["t_center_s", "score", "kept", "track_id"])
    d = d[d["kept"] & (d["t_center_s"] < 10_800) & (d["score"] >= SCORE_BINS[0])]
    status = np.full(len(d), "other", dtype=object)
    for s, ident in fish.items():
        status[d["track_id"].to_numpy() == ident] = s
    d = d.assign(status=status, score_bin=pd.cut(d["score"], SCORE_BINS, right=False, labels=SCORE_NAMES))
    out = d.groupby(["status", "score_bin"], observed=False).size().rename("n").reset_index()
    out["per_hour"] = out["n"] / 3.0
    out.insert(0, "recording", name)
    out.insert(0, "dataset", dataset)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--segments", type=int, default=24)
    ap.add_argument("--only", nargs="*", default=None, help="recordings to measure (default: all)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fish_ids = trial_fish()
    segs, fishes, recs, rates = [], [], [], []
    for dataset, run in RUNS.items():
        names = sorted(p.parent.name for p in run.glob("*/meta.json"))
        for name in names:
            if args.only and name not in args.only:
                continue
            rec, tracks = co.open_rec(name), co.open_tracks(name)
            if rec.duration_s < MIN_DURATION_S:
                print(f"{dataset} {name}: skipped ({rec.duration_s:.0f} s long)", flush=True)
                continue
            ids = fish_ids.get(name, {})
            for t0 in segment_times(name, args.segments, rec.duration_s):
                seg, fish = measure_segment(rec, tracks, t0)
                segs.append({"dataset": dataset, "recording": name, **seg})
                for d in fish:
                    status = next((s for s, i in ids.items() if i == d["track_id"]), "other")
                    fishes.append({"dataset": dataset, "recording": name, "t0_s": t0, "status": status, **d})
            meta = json.loads((run / name / "meta.json").read_text())
            row = {"dataset": dataset, "recording": name, "n_electrodes": rec.n_channels,
                   "click_fraction_repaired": meta.get("click_removal", {}).get("fraction_repaired", np.nan),
                   "tracks_unassigned_fraction": meta.get("tracks", {}).get("unassigned_fraction", np.nan),
                   "trial": bool(ids)}
            for s, ident in ids.items():
                row[f"coverage_{s}"] = track_coverage(tracks, ident)
            if ids:
                fw = [d["f_hz"] for d in fishes if d["recording"] == name and d["status"] == "winner"]
                fl = [d["f_hz"] for d in fishes if d["recording"] == name and d["status"] == "loser"]
                row["median_df_hz"] = abs(np.median(fw) - np.median(fl)) if fw and fl else np.nan
                rates.append(detection_rates(dataset, name, ids))
            recs.append(row)
            print(f"{dataset} {name}: done", flush=True)
    seg_df, fish_df = pd.DataFrame(segs), pd.DataFrame(fishes)
    seg_df.to_csv(args.out / "segments.csv", index=False)
    fish_df.to_csv(args.out / "fish.csv", index=False)
    rec_df = pd.DataFrame(recs)
    agg = seg_df.groupby("recording").agg(
        floor_median=("floor", "median"), unexplained_lines_median=("n_unexplained_lines", "median"),
        strongest_line_vs_weakest_fish_db_median=("strongest_line_vs_weakest_fish_db", "median"),
        transients_per_s_median=("transients_per_s", "median"),
        transients_wide_share_median=("transients_wide_share", "median"))
    for s in ("winner", "loser"):
        g = fish_df[fish_df["status"] == s].groupby("recording")
        agg[f"snr_db_{s}"] = g["snr_db"].median()
        agg[f"n_electrodes_10db_{s}"] = g["n_electrodes_10db"].median()
    rec_df = rec_df.merge(agg, left_on="recording", right_index=True, how="left")
    rec_df.to_csv(args.out / "recordings.csv", index=False)
    if rates:
        pd.concat(rates).to_csv(args.out / "rates.csv", index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
