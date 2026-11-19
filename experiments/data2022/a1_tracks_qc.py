"""Per 2022 recording: do the wavetracker tracks follow the fish the detector finds chirps on?

For each recording, wavetracker's own spectrogram of the dark phase
(``interrim/<rec>/fill_spec.npy``: a raw float64 memmap, stored time-major,
i.e. (times, frequencies) although ``fill_spec_shape.npy`` lists
(frequencies, times); checked against the raw recording's spectrum),
frequencies ``fill_freqs.npy``, times ``fill_times.npy``, 500-1300 Hz,
maximum over ~5-s blocks, in dB, with:

- the tracks (``fund_v`` / ``ident_v``; trial winner and loser from ``meta.csv`` coloured,
  other tracks grey);
- kept detections with score >= 0.5: the predicted emitter EODf, coloured by assignment
  (assigned / no track within 20 Hz);
- ``meta.csv``'s Win_EODf / Lose_EODf as ticks on the right.

Second panel: 5 minutes around the densest untracked detections, at full resolution.

    .venv/bin/python experiments/data2022/a1_tracks_qc.py [--only REC ...]

Output: ``$CHIRP_DATA/data2022/a1/tracks_qc/<rec>.png`` and ``untracked_by_recording.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2022  # noqa: E402

OUT = DATA / "data2022" / "a1" / "tracks_qc"
RUN = DATA / "detections" / "competition2022_v2_t001"
F_BAND = (500.0, 1300.0)
BLOCK_S = 5.0
ZOOM_S = 300.0


def spectrogram(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = REC_2022 / "interrim" / name
    shape = tuple(int(x) for x in np.load(d / "fill_spec_shape.npy"))
    freqs = np.load(d / "fill_freqs.npy")
    times = np.load(d / "fill_times.npy")[: shape[1]]
    spec = np.memmap(d / "fill_spec.npy", dtype=np.float64, mode="r", shape=shape[::-1])  # (times, freqs)
    fsel = np.flatnonzero((freqs >= F_BAND[0]) & (freqs <= F_BAND[1]))
    return spec, freqs[fsel], times, fsel


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"}).set_index("recording")
    rows = []
    for rd in sorted(RUN.glob("2022-*")):
        name = rd.name
        if args.only and name not in args.only:
            continue
        spec, f, t, fsel = spectrogram(name)
        tsel = np.flatnonzero(t < 10_800)
        step = max(1, round(BLOCK_S / np.median(np.diff(t))))
        nb = len(tsel) // step
        sub = np.asarray(spec[tsel[0]:tsel[0] + nb * step, fsel[0]:fsel[-1] + 1]).T
        over = 10 * np.log10(sub.reshape(len(fsel), nb, step).max(axis=2) + 1e-20)
        tb = t[tsel[0]:tsel[0] + nb * step].reshape(nb, step)[:, 0]
        d = pd.read_parquet(rd / "chirps_rtdetr.parquet")
        d = d[d["kept"] & (d["t_center_s"] < 10_800) & (d["score"] >= 0.5)]
        untracked = d["assign_status"].isin(["beyond_tolerance", "no_live_track"])
        fund = np.load(REC_2022 / "interrim" / name / "fund_v.npy")
        ident = np.load(REC_2022 / "interrim" / name / "ident_v.npy")
        idx = np.load(REC_2022 / "interrim" / name / "idx_v.npy")
        tt = np.load(REC_2022 / "interrim" / name / "times.npy")[idx]
        fish = {}
        if name in meta.index:
            fish = {"winner": float(meta.loc[name, "Win_ID"]), "loser": float(meta.loc[name, "Lose_ID"])}
        # densest untracked 5 minutes
        if untracked.any():
            c = np.histogram(d.loc[untracked, "t_center_s"], bins=np.arange(0, 10_800 + ZOOM_S, ZOOM_S))[0]
            z0 = float(np.argmax(c) * ZOOM_S)
        else:
            z0 = 3600.0
        zs = np.flatnonzero((t >= z0) & (t < z0 + ZOOM_S))
        zoom = 10 * np.log10(np.asarray(spec[zs[0]:zs[-1] + 1, fsel[0]:fsel[-1] + 1]).T + 1e-20)
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
        for ax, img, ext, (lo, hi) in ((axes[0], over, (tb[0], tb[-1], f[0], f[-1]), (0, 10_800)),
                                       (axes[1], zoom, (t[zs[0]], t[zs[-1]], f[0], f[-1]), (z0, z0 + ZOOM_S))):
            vmax = np.percentile(img, 99.9)
            ax.imshow(img, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=vmax - 50, vmax=vmax,
                      interpolation="nearest")
            for ident_v in np.unique(ident[np.isfinite(ident)]):
                m = (ident == ident_v) & (tt >= lo) & (tt < hi)
                if m.sum() < 2:
                    continue
                role = next((k for k, v in fish.items() if v == ident_v), None)
                ax.plot(tt[m], fund[m], ".", ms=0.6 if ax is axes[0] else 1.5,
                        color={"winner": "#4dc3ff", "loser": "#7fff7f"}.get(role, "#bbbbbb"),
                        label=f"track {int(ident_v)} ({role or 'other'})" if ax is axes[1] else None)
            dd = d[(d["t_center_s"] >= lo) & (d["t_center_s"] < hi)]
            ut = dd["assign_status"].isin(["beyond_tolerance", "no_live_track"])
            ax.plot(dd.loc[~ut, "t_center_s"], dd.loc[~ut, "emitter_hz"], "o", ms=3, mfc="none", mec="white", mew=0.6,
                    label="detection >= 0.5, assigned" if ax is axes[1] else None)
            ax.plot(dd.loc[ut, "t_center_s"], dd.loc[ut, "emitter_hz"], "x", ms=5, color="#ff4040", mew=1.2,
                    label="detection >= 0.5, no track within 20 Hz" if ax is axes[1] else None)
            for k, col in (("Win_EODf", "#4dc3ff"), ("Lose_EODf", "#7fff7f")):
                if name in meta.index:
                    ax.axhline(float(meta.loc[name, k]), xmin=0.985, xmax=1.0, color=col, lw=3)
            ax.set_ylabel("frequency (Hz)")
            ax.set_ylim(*F_BAND)
        axes[0].set_title(f"{name}: dark phase, wavetracker spectrogram (max per {BLOCK_S:.0f} s); blue = winner track, "
                          f"green = loser, grey = other tracks; red x = detections without a track; "
                          f"ticks right = meta.csv EODf", fontsize=9)
        axes[1].set_title(f"5 min with the most untracked detections ({z0 / 60:.0f}-{(z0 + ZOOM_S) / 60:.0f} min)", fontsize=9)
        axes[1].set_xlabel("recording time (s)")
        axes[1].legend(loc="upper right", fontsize=7, markerscale=3)
        fig.savefig(OUT / f"{name}.png", dpi=110)
        plt.close(fig)
        rows.append({"recording": name, "detections": len(d), "untracked": int(untracked.sum()),
                     "untracked_share": float(untracked.mean()) if len(d) else np.nan,
                     "untracked_emitter_median_hz": float(d.loc[untracked, "emitter_hz"].median()) if untracked.any() else np.nan,
                     "zoom_start_s": z0})
        print(f"{name}: {int(untracked.sum())}/{len(d)} untracked", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "untracked_by_recording.csv", index=False)


if __name__ == "__main__":
    main()
