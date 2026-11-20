"""A 2022 dataset folder in the layout the 2019 paper scripts read, for the chase-triggered chirp rates.

Author (2026-11-20): only the chase onset / offset-triggered chirp rates of the 2019
pipeline (``paper/scripts/analyze_event_time_rates_trial.py --ag-events``), run
unchanged on 2022; no contacts; nothing of ``paper/`` or the manuscript changed. The
script reads, per trial (``CHIRP_REC_2019/<rec>/``):

- ``YYYY-MM-DD.csv`` (BORIS: ``Start (s)`` in video time, ``Behavior`` 0 = chase onset,
  1 = chase offset, 2 = contact, ``FPS``): here the **locked** model chases
  (``video2022/chases/model/model_v2/events.csv``, all chases, both chasers), converted
  from recording time to video frames with the LED-synchronised frame clock
  (``video2022/b1_sleap/tables/frame_time/<rec>.npy``); the script maps them back with
  its own LED map (:func:`check_round_trip`). The 2019 loader pairs every onset with the
  next offset, so chases that overlap in time (the model called both fish chaser at once)
  are merged into one (:func:`merge_overlaps`; counted in ``contents.csv``). No contacts
  (none detected in 2022).
- ``LED_frames.npy`` (``interrim/<rec>/``), ``led_idxs.csv`` (``raw/<rec>/``): copied.
- ``chirp_times_<SUFFIX>.npy``, ``chirp_ids_<SUFFIX>.npy``: the 2022 detections at the
  operating threshold 0.8 (``competition2022_v2_gapfill_t080_corrected``), unweighted as
  in 2019 (the per-fish correction is not applied by the 2019 scripts).
- ``times.npy`` (wavetracker times of the gap-filled tracks), ``analysis/ids.npy``
  (winner, loser) and ``analysis/rise_idx.npy``: the script loads frequency rises even
  for the chase centrings; 2022 has no rise detection, so this is an empty (NaN) array.
- root ``trial_summary.csv``: the 18 trials of ``raw/meta.csv`` (2022-07-07, a control,
  is left out), ``group`` = 5 for all (the 2019 code keeps only groups >= 5, its
  BORIS-annotated trials), ``draw`` = 0, sex unknown (not recorded in meta.csv).

    .venv/bin/python experiments/paper2022/build_dataset.py [--check]

Output: ``$CHIRP_DATA/paper2022/dataset/``; ``--check`` runs the 2019 loader on every
trial and compares the chase times it returns with the model's (needs the paper's
Python environment, see ``README.md``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2022

OUT = DATA / "paper2022" / "dataset"
CHASES = DATA / "video2022" / "chases" / "model" / "model_v2" / "events.csv"
CHIRPS = DATA / "detections" / "competition2022_v2_gapfill_t080_corrected"
TRACKS = DATA / "detections" / "competition2022_v2_t001_gapfill"
FRAME_TIME = DATA / "video2022" / "b1_sleap" / "tables" / "frame_time"
FPS = 25.0
SUFFIX = "rtdetr_t080"
PAPER_SCRIPTS = Path("/home/weygoldt/wrk/analyses/chirp_competition_upgrade/paper/scripts")


def video_seconds(t_rec: np.ndarray, frame_time: np.ndarray) -> np.ndarray:
    """Recording times -> video time (s) of the nearest frame (frame / FPS), via the frame clock."""
    frames = np.round(np.interp(t_rec, frame_time, np.arange(len(frame_time)))).astype(int)
    return frames / FPS


def merge_overlaps(chases: pd.DataFrame) -> pd.DataFrame:
    """Chases of one recording with overlapping ones merged (earliest onset to latest offset)."""
    c = chases.sort_values("t_start")
    out: list[list[float]] = []
    for a, b in zip(c["t_start"], c["t_end"], strict=True):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return pd.DataFrame(out, columns=["t_start", "t_end"])


def boris_table(chases: pd.DataFrame, frame_time: np.ndarray) -> pd.DataFrame:
    """Chase onsets (Behavior 0) and offsets (1) in the BORIS columns the 2019 loader reads."""
    on = video_seconds(chases["t_start"].to_numpy(), frame_time)
    off = video_seconds(chases["t_end"].to_numpy(), frame_time)
    rows = pd.DataFrame({"Start (s)": np.r_[on, off], "Behavior": np.r_[np.zeros(len(on), int), np.ones(len(off), int)]})
    rows = rows.sort_values(["Start (s)", "Behavior"], kind="stable").reset_index(drop=True)
    rows["FPS"] = FPS
    rows["Comment start"] = "model_v2 chase (chaser-centric model, locked 2026-11-20)"
    return rows[["FPS", "Behavior", "Start (s)", "Comment start"]]


def trial_table() -> pd.DataFrame:
    meta = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"})
    return pd.DataFrame({"recording": meta["recording"], "group": 5, "pairing": meta["Pairing"],
                         "win_ID": meta["Win_ID"].astype(float), "lose_ID": meta["Lose_ID"].astype(float),
                         "sex_win": "na", "sex_lose": "na", "EODf_win": meta["Win_EODf"],
                         "EODf_lose": meta["Lose_EODf"], "draw": 0})


def build() -> pd.DataFrame:
    trials = trial_table()
    chases = pd.read_csv(CHASES)
    OUT.mkdir(parents=True, exist_ok=True)
    trials.to_csv(OUT / "trial_summary.csv")
    rows = []
    for r in trials.itertuples():
        d = OUT / r.recording
        (d / "analysis").mkdir(parents=True, exist_ok=True)
        np.save(d / "times.npy", np.load(TRACKS / r.recording / "tracks_gapfill.npz")["times"])
        np.save(d / "analysis" / "ids.npy", np.array([r.win_ID, r.lose_ID]))
        np.save(d / "analysis" / "rise_idx.npy", np.full((2, 1), np.nan))
        shutil.copyfile(REC_2022 / "interrim" / r.recording / "LED_frames.npy", d / "LED_frames.npy")
        shutil.copyfile(REC_2022 / "raw" / r.recording / "led_idxs.csv", d / "led_idxs.csv")
        for kind in ("times", "ids"):
            shutil.copyfile(CHIRPS / r.recording / f"chirp_{kind}_rtdetr_t080.npy", d / f"chirp_{kind}_{SUFFIX}.npy")
        c_all = chases[chases["recording"] == r.recording]
        c = merge_overlaps(c_all)
        frame_time = np.load(FRAME_TIME / f"{r.recording}.npy")
        boris_table(c, frame_time).to_csv(d / ("-".join(r.recording.split("-")[:3]) + ".csv"), index=False)
        rows.append({"recording": r.recording, "pairing": r.pairing, "model_chases": len(c_all), "chases": len(c),
                     "chirps": len(np.load(d / f"chirp_times_{SUFFIX}.npy"))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "contents.csv", index=False)
    return out


def check_round_trip() -> pd.DataFrame:
    """The 2019 loader on every trial: chase times it returns vs the model's (recording time)."""
    sys.path.insert(0, str(PAPER_SCRIPTS))
    import analyze_event_time_rates_trial as etr

    chases = pd.read_csv(CHASES)
    rows = []
    for rec in pd.read_csv(OUT / "trial_summary.csv")["recording"]:
        contact, on_off, _, _ = etr.load_and_convert_boris_events(OUT / rec, rec, sr=20_000)
        c = merge_overlaps(chases[chases["recording"] == rec])
        err = np.r_[np.sort(on_off[:, 0]) - np.sort(c["t_start"].to_numpy()),
                    np.sort(on_off[:, 1]) - np.sort(c["t_end"].to_numpy())] if len(c) == len(on_off) else np.nan
        rows.append({"recording": rec, "chases": len(c), "loaded": len(on_off), "contacts": len(contact),
                     "max_abs_err_ms": float(np.nanmax(np.abs(err)) * 1e3) if len(c) else np.nan,
                     "median_abs_err_ms": float(np.nanmedian(np.abs(err)) * 1e3) if len(c) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "round_trip.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only the round-trip check (paper environment)")
    args = ap.parse_args()
    if args.check:
        print(check_round_trip().round(2).to_string(index=False))
    else:
        print(build().to_string(index=False))
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
