"""Two follow-ups to A1 (2026-11-19, from the author's labelling of the 2022 chirp session).

1. **The all-electrode view.** The labeller and the detector's front end sum power over
   all electrodes. Per tracked fish: SNR (peak over the 300-2000 Hz median) in the summed
   spectrum and on the fish's best electrode alone, 8 random 4-s segments of the dark phase
   (``a1_signal.segment_times``) in 6 recordings per dataset. ``loss_db`` = best - summed.
   Also the 99th percentile of the summed spectrum over its median (strong narrow lines).
2. **Fish without a track.** Kept detections >= 0.5 in the dark phase whose predicted
   emitter has no track within 20 Hz (``assign_status`` beyond_tolerance / no_live_track),
   and per 1-s bin of the dark phase the number of tracks live under the labeller's 10-s
   gap rule.

    .venv/bin/python experiments/data2022/a1_view_and_tracks.py

Output: ``$CHIRP_DATA/data2022/a1/view_and_tracks/`` (``view_snr.csv``, ``view_snr_summary.csv``,
``untracked.csv``, ``live_tracks.csv``, ``summary.csv``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_signal as a1  # noqa: E402
import cutouts as co  # noqa: E402
from paths import DATA  # noqa: E402

OUT = DATA / "data2022" / "a1" / "view_and_tracks"
VIEW_RECS = {"2019/2020": ["2019-11-27-10_00", "2019-12-03-10_01", "2020-03-13-10_00", "2020-05-12-10_00",
                           "2020-06-04-10_00", "2020-03-18-10_34"],
             "2022": ["2022-05-12-10_00", "2022-06-01-10_00", "2022-06-08-10_00", "2022-06-20-10_00",
                      "2022-06-27-10_00", "2022-07-04-10_00"]}


def view_snr() -> pd.DataFrame:
    rows = []
    for ds, names in VIEW_RECS.items():
        for name in names:
            rec, tracks = co.open_rec(name), co.open_tracks(name)
            fs = rec.samplerate_hz
            for t0 in a1.segment_times(name, 8, rec.duration_s):
                x = rec.read(int(t0 * fs), int((t0 + 4) * fs)).astype(np.float64)
                f, p = welch(x, fs=fs, nperseg=round(0.2048 * fs), axis=0)
                band = (f >= 300) & (f <= 2000)
                ps = p.sum(axis=1)
                for _ident, fr in tracks.frequencies_at(t0 + 2, 10.0).items():
                    m = np.abs(f - fr) <= 3
                    if not m.any():
                        continue
                    k = np.flatnonzero(m)[np.argmax(ps[m])]
                    el = int(np.argmax(p[k]))
                    rows.append({"dataset": ds, "recording": name, "t0_s": t0, "f_hz": fr, "n_electrodes": p.shape[1],
                                 "snr_summed_db": 10 * np.log10(ps[k] / np.median(ps[band])),
                                 "snr_best_db": 10 * np.log10(p[k, el] / np.median(p[band, el])),
                                 "lines_p99_over_median_db": 10 * np.log10(np.percentile(ps[band], 99) / np.median(ps[band]))})
    d = pd.DataFrame(rows)
    d["loss_db"] = d["snr_best_db"] - d["snr_summed_db"]
    return d


def tracks() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, live = [], []
    for ds, run in a1.RUNS.items():
        for f in sorted(run.glob("*/chirps_rtdetr.parquet")):
            name = f.parent.name
            d = pd.read_parquet(f, columns=["t_center_s", "score", "kept", "assign_status"])
            d = d[d["kept"] & (d["t_center_s"] < 10_800) & (d["score"] >= 0.5)]
            untracked = d["assign_status"].isin(["beyond_tolerance", "no_live_track"])
            rows.append({"dataset": ds, "recording": name, "detections": len(d), "untracked": int(untracked.sum())})
            bins = np.arange(0, 10_800) + 0.5
            n_live = np.zeros(len(bins), int)
            for t in co.open_tracks(name):
                ts = t.times_s
                if len(ts) < 2:
                    continue
                i = np.searchsorted(ts, bins)
                ok = (i > 0) & (i < len(ts))
                gap = np.full(len(bins), np.inf)
                gap[ok] = ts[i[ok]] - ts[i[ok] - 1]
                n_live += gap <= 10.0
            live.append({"dataset": ds, "recording": name,
                         **{f"share_{k}_live": float(np.mean(np.minimum(n_live, 2) == k)) for k in range(3)}})
    return pd.DataFrame(rows), pd.DataFrame(live)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v = view_snr()
    v.to_csv(OUT / "view_snr.csv", index=False)
    vs = v.groupby("dataset")[["n_electrodes", "snr_summed_db", "snr_best_db", "loss_db",
                               "lines_p99_over_median_db"]].median()
    vs.to_csv(OUT / "view_snr_summary.csv")
    u, live = tracks()
    u.to_csv(OUT / "untracked.csv", index=False)
    live.to_csv(OUT / "live_tracks.csv", index=False)
    s = u.groupby("dataset")[["detections", "untracked"]].sum()
    s["untracked_share"] = s["untracked"] / s["detections"]
    s = s.join(live.groupby("dataset")[["share_0_live", "share_1_live", "share_2_live"]].median())
    s.to_csv(OUT / "summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(vs.round(1).to_string())
        print(s.round(4).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
