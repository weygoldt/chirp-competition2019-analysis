"""Cache the raw-data inputs of the three pipeline-v2 figures.

The plotting scripts (``plot_v2_*.py``) run in the paper environment, which
has no chirpdetector or gridtools. This module computes, once, everything
that needs them and writes it to ``$PAPER_OUT/data/processed/v2_figures/``:

``snippets``
    Spectrogram snippets of the 2019/2020 tube competition recordings,
    computed exactly as the production run did (chirpdetector's
    `span_spectrogram` with the released detector bundle: click removal on
    grids of at least 9 electrodes, then the three-window front end: 205,
    51 and 13 ms Hann windows on one 4.88-Hz grid, 5-ms hop, 500-1500 Hz,
    power summed over electrodes), normalised to [0, 1] over the snippet
    with the front end's own rule (60 dB range ending 10 dB above the median
    per-frame peak of the reference channel), plus the wavetracker tracks
    around the snippet. One ``<name>.npz`` per snippet.
``traces``
    Instantaneous-frequency traces of the example chirps of Fig. 1E,
    extracted from the raw recording with gridtools' harvest extraction
    (emitter's field pattern, other fish projected out, pulse-pair estimator
    with a 1-ms adaptive window) and per-EOD-period frequencies from the
    upward zero crossings of the same combined signal. ``traces.npz``.
``raw``
    The raw voltage of the electrode with the largest RMS during the
    front-end example (``f1_fe_e``), for the "recording" thumbnail of
    Fig. 1. ``raw.npz``.
``assign-baseline``
    Emitter assignment on the simulated test scenes after post-processing,
    per true positive at score >= 0.5: the emitter head (as evaluated) and
    two box-only baselines (nearest live track to the box's lower edge, and
    to the lower edge plus the median offset emitter - f_low of the training
    chirps). Recomputed from the saved post-processed predictions and the
    scenes' tracks. ``assign_baseline.csv``.

Environment: the upgrade repository's root venv (chirpdetector 2.0.0,
gridtools 2.0.0), run from ``paper/``::

    /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python \
        scripts/v2_fig_extract.py snippets traces
    ~/wrk/analyses/chirp_competition_upgrade/scripts/guarded --name v2fig-assign \
        --mem 16G --vram 0.2 -- \
        /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python \
        scripts/v2_fig_extract.py assign-baseline

CPU only. ``snippets`` + ``traces`` take about a minute; ``assign-baseline``
a few minutes (loads the tracks of 300 test scenes).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config

log = logging.getLogger("v2_fig_extract")

CHIRP_DATA = Path("/mnt/data1/chirp_v2")
# the detections follow PAPER_DETECTIONS (paper_config): the production run,
# or its dataset patch with the trial fish's track gaps filled
RUNS = {
    "rtdetr": ("competition2019_v2_20261113", "chirps_rtdetr.parquet"),
    "rtdetr_gapfill": ("competition2019_v2_20261113_gapfill",
                       "chirps_rtdetr_gapfill.parquet"),
}
DETECTIONS = paper_config.DETECTIONS
RUN = CHIRP_DATA / "detections" / RUNS[DETECTIONS][0]
PARQUET = RUNS[DETECTIONS][1]
BUNDLE = CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_r1" / "final" / "bundle"
REC_2019 = Path("/mnt/data2/2019_tube_competition")
SIM_V4 = Path("~/chirp_v2_fast/datasets/sim_v4").expanduser()
PP_EVAL = CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_r1" / "eval_postprocess"

OUT = paper_config.processed("v2_figures", "x").parent

# name -> (recording, t_start_s, t_end_s); times in seconds from the
# recording's first sample (the clock of chirps_rtdetr.parquet)
SNIPPETS: dict[str, tuple[str, float, float]] = {
    # Fig. 1A: front end; two fish 29 Hz apart, a small and a big chirp
    "f1_fe_e": ("2020-06-05-10_00", 89.9, 91.1),
    # Fig. 1B-E: one detector tile (tile 4932: 3.2 s from 4932 * 2.2 s)
    "f1_tile": ("2020-05-29-10_00", 10850.4, 10853.6),
    # Fig. 1D: a duplicate box on one chirp
    "f1_dup": ("2020-03-18-10_34", 16980.9, 16982.1),
    # Fig. 3A: hand-labelled test region (2019-12-09, region 4260-4423 s)
    "f3_labelled": ("2019-12-09-10_00", 4312.0, 4328.0),
    # Fig. 3B: the track of one fish ends while it keeps chirping
    "f3_trackgap": ("2019-12-04-10_00", 7468.0, 7488.0),
    # Fig. 3C crops
    "g_close_15388": ("2019-12-09-10_00", 15387.9, 15389.1),
    "g_close_9394": ("2020-03-13-10_00", 9393.9, 9395.1),
    "g_missed_1208": ("2019-12-09-10_00", 1207.9, 1209.1),
}

# Fig. 1E: chirps of the f1_tile recording whose traces are extracted
TRACE_CHIRPS: dict[str, tuple[str, int]] = {
    "small": ("2020-05-29-10_00", 2085),
    "big": ("2020-05-29-10_00", 2086),
}


def _run_tracks(rec_name: str) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """The tracks the run assigned to: (ident, times, freqs, filled) per track.

    The wavetracker tracks of the recording; for the gap-filled run, the two
    trial fish's tracks are replaced by the patched ones of
    ``<run>/<rec>/tracks_gapfill.npz`` (``filled``: interpolated samples).
    """
    from gridtools.io import load_tracks

    out = {}
    for tr in load_tracks(REC_2019 / rec_name):
        out[int(tr.ident)] = (np.asarray(tr.times_s, float), np.asarray(tr.freqs_hz, float),
                              np.zeros(len(tr.times_s), bool))
    patched = RUN / rec_name / "tracks_gapfill.npz"
    if DETECTIONS == "rtdetr_gapfill" and patched.exists():
        z = np.load(patched)
        times = z["times"][z["idx_v"]]
        for ident in np.unique(z["ident_v"][np.isfinite(z["ident_v"])]):
            m = z["ident_v"] == ident
            order = np.argsort(times[m], kind="stable")
            out[int(ident)] = (times[m][order], z["fund_v"][m][order], z["filled"][m][order])
    return [(k, *v) for k, v in out.items()]


def _tracks_window(tracks, t0: float, t1: float) -> dict[str, np.ndarray]:
    ids, offsets, times, freqs, filled = [], [0], [], [], []
    for ident, tt, ff, fl in tracks:
        m = (tt >= t0) & (tt <= t1)
        if not m.any():
            continue
        ids.append(ident)
        times.append(tt[m])
        freqs.append(ff[m])
        filled.append(fl[m])
        offsets.append(offsets[-1] + int(m.sum()))
    return {
        "track_ids": np.asarray(ids, dtype=np.int64),
        "track_offsets": np.asarray(offsets, dtype=np.int64),
        "track_times_s": np.concatenate(times) if times else np.zeros(0),
        "track_freqs_hz": np.concatenate(freqs) if freqs else np.zeros(0),
        "track_filled": np.concatenate(filled) if filled else np.zeros(0, bool),
    }


def snippets(names: list[str] | None = None) -> None:
    from chirpdetector.bundle import load_bundle
    from chirpdetector.inference.spans import span_bounds, span_spectrogram
    from chirpdetector.spectrogram import normalize, reference_level_db
    from gridtools.io import open_recording

    bundle = load_bundle(BUNDLE)
    config = bundle.spectrogram
    track_cache: dict[str, object] = {}
    for name, (rec_name, t0, t1) in SNIPPETS.items():
        if names and name not in names:
            continue
        recording = open_recording(REC_2019 / rec_name)
        fs = recording.samplerate_hz
        start, stop = span_bounds(bundle, fs, recording.n_samples, t0, t1)
        spec = span_spectrogram(bundle, recording, start, stop, device="cpu")
        img = normalize(spec, config).cpu().numpy().astype(np.float16)
        if rec_name not in track_cache:
            track_cache[rec_name] = _run_tracks(rec_name)
        tw = _tracks_window(track_cache[rec_name], t0 - 15.0, t1 + 15.0)
        meta = {
            "recording": rec_name,
            "t_start_s": t0,
            "t_end_s": t1,
            "n_electrodes": int(recording.n_channels),
            "samplerate_hz": float(fs),
            "reference_db": reference_level_db(spec),
            "windows_s": list(config.window_durations_s),
            "click_removal": bundle.preprocessing.get("click_removal") is not None
            and int(recording.n_channels) >= 9,
            "bundle": str(BUNDLE),
            "weights_sha256": bundle.weights_sha256,
            "detections": DETECTIONS,
            "run": str(RUN),
        }
        np.savez_compressed(
            OUT / f"{name}.npz",
            img=img,
            times_s=spec.times_s,
            freqs_hz=spec.freqs_hz,
            meta=json.dumps(meta),
            **tw,
        )
        log.info("%s: %s %.1f-%.1f s, %s", name, rec_name, t0, t1, img.shape)


def traces() -> None:
    from gridtools.io import load_tracks
    from gridtools.templates.harvest import (
        HarvestConfig,
        harvest_candidate,
        open_recording_lenient,
    )

    cfg = HarvestConfig()
    out: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for key, (rec_name, chirp_id) in TRACE_CHIRPS.items():
        table = pd.read_parquet(RUN / rec_name / PARQUET)
        row = table.set_index("chirp_id").loc[chirp_id]
        # the harvest uses the wavetracker tracks; with the gap-filled run this
        # is only valid if the chirp does not lie in a filled gap of its fish
        patched = RUN / rec_name / "tracks_gapfill.npz"
        if patched.exists():
            g = np.load(patched)
            inside = ((g["gap_fish_id"] == row.track_id) & (g["gap_action"] == "filled")
                      & (g["gap_start_s"] < row.t_end_s) & (g["gap_end_s"] > row.t_start_s))
            if inside.any():
                raise RuntimeError(f"{rec_name} chirp {chirp_id} lies in a filled track gap")
        recording, _ = open_recording_lenient(REC_2019 / rec_name)
        tracks = load_tracks(REC_2019 / rec_name)
        box = pd.Series(
            {
                "box_index": int(chirp_id),
                "t_start_s": row.t_start_s,
                "t_end_s": row.t_end_s,
                "f_low_hz": row.f_low_hz,
                "f_high_hz": row.f_high_hz,
                "score": row.score,
                "track_id": int(row.track_id),
                "emitter_hz": row.emitter_hz,
            }
        )
        result = harvest_candidate(recording, tracks, box, cfg, recording_name=rec_name)
        ex = result.extraction
        if ex is None:
            raise RuntimeError(f"no extraction for {rec_name} {chirp_id}: {result.record}")
        fs = ex.samplerate_hz
        t = ex.window_start_s + np.arange(len(ex.signal)) / fs
        freq = np.asarray(ex.traces["smooth_freq"], dtype=float)
        v = np.asarray(ex.signal, dtype=float)
        up = np.flatnonzero((v[:-1] < 0) & (v[1:] >= 0))
        tz = (up + (-v[up]) / (v[up + 1] - v[up])) / fs + ex.window_start_s
        f_period = 1.0 / np.diff(tz)
        t_period = 0.5 * (tz[1:] + tz[:-1])
        # baseline: median of the smooth trace in the 200 ms before the box
        pre = (t > row.t_start_s - 0.25) & (t < row.t_start_s - 0.05)
        baseline = float(np.nanmedian(freq[pre]))
        step = max(1, int(round(fs / 4000)))  # keep 4 kHz of the smooth trace
        out[f"{key}_t_s"] = (t[::step] - row.t_center_s).astype(np.float32)
        out[f"{key}_freq_hz"] = freq[::step].astype(np.float32)
        out[f"{key}_period_t_s"] = (t_period - row.t_center_s).astype(np.float32)
        out[f"{key}_period_freq_hz"] = f_period.astype(np.float32)
        meta[key] = {
            "recording": rec_name,
            "chirp_id": int(chirp_id),
            "t_center_s": float(row.t_center_s),
            "trace_baseline_hz": baseline,
            "harvest_status": result.record.get("status"),
            "harvest_reason": result.record.get("reason"),
        }
        log.info("trace %s: %s %d, baseline %.1f Hz (%s %s)", key, rec_name, chirp_id,
                 baseline, result.record.get("status"), result.record.get("reason"))
    np.savez_compressed(OUT / "traces.npz", meta=json.dumps(meta), **out)


def raw() -> None:
    from gridtools.io import open_recording

    rec_name, t0, t1 = SNIPPETS["f1_fe_e"]
    recording = open_recording(REC_2019 / rec_name)
    block = recording.read_time(t0, t1)
    rms = np.sqrt(np.mean((block - block.mean(0)) ** 2, axis=0))
    ch = int(np.argmax(rms))
    np.savez_compressed(
        OUT / "raw.npz",
        signal=block[:, ch].astype(np.float32),
        samplerate_hz=float(recording.samplerate_hz),
        meta=json.dumps({"recording": rec_name, "t_start_s": t0, "t_end_s": t1,
                         "channel": ch, "rms": rms.round(6).tolist()}),
    )
    log.info("raw: %s %.1f-%.1f s, channel %d of %d", rec_name, t0, t1, ch, len(rms))


def assign_baseline() -> None:
    from chirpdetector.data import load_dataset
    from chirpdetector.evaluation import assign_by_frequency
    from chirpdetector.postprocess import COLUMNS
    from gridtools.io import load_tracks

    dataset = load_dataset(SIM_V4, check_files=False)
    ids = list(dataset.manifest.image_ids("test"))
    cache, tracks = {}, {}
    for image_id in ids:
        source = dataset.manifest.image(image_id).source
        if source not in cache:
            cache[source] = load_tracks(dataset.tracks_dir(source))
        tracks[image_id] = cache[source]
    log.info("%d test tiles, %d scenes", len(ids), len(cache))
    flagged = pd.read_parquet(PP_EVAL / "sim_test_predictions_flagged.parquet")
    keep = flagged["drop_reason"].astype(str).to_numpy() == ""
    # the saved table predates later post-processing columns; drop what exists
    kept = flagged.loc[keep].drop(columns=[c for c in COLUMNS if c in flagged.columns])
    kept = kept.reset_index(drop=True)
    items = pd.read_csv(PP_EVAL / "sim_test_after" / "items.csv")
    tp = items[(items["outcome"] == "tp") & (items["label"] == "chirp")].copy()
    rows = tp["pred_row"].astype(int).to_numpy()
    # consistency: the items' predicted emitter must be the kept rows' one
    pred_ids = kept["emitter_id"].astype("Int64").to_numpy()[rows]
    agree = np.mean(
        pd.array(pred_ids, dtype="Int64").fillna(-1).to_numpy()
        == tp["pred_emitter_id"].astype("Int64").fillna(-1).to_numpy()
    )
    log.info("items vs kept predictions: emitter ids agree for %.4f of TPs", agree)
    train =dataset.boxes[
        (dataset.boxes["kind"] == "chirp")
        & dataset.boxes["image_id"].isin(set(dataset.manifest.image_ids("train")))
    ]
    offset = float((train["emitter_hz"] - train["f_low_hz"]).median())
    log.info("median emitter - f_low of the training chirps: %.2f Hz", offset)
    true_ids = tp["true_emitter_id"].astype("Int64").fillna(-2).to_numpy(int)
    out = tp[["recording", "label_row", "pred_row", "score", "true_emitter_id",
              "pred_emitter_id", "delta_f_hz", "n_live_tracks", "df_bin"]].copy()
    out["head_correct"] = (
        tp["pred_emitter_id"].astype("Int64").fillna(-1).to_numpy(int) == true_ids
    )
    for name, shift in (("edge", 0.0), ("edge_offset", offset)):
        base = kept.iloc[rows].copy().reset_index(drop=True)
        base["emitter_hz"] = base["f_low_hz"].to_numpy(float) + shift
        base = assign_by_frequency(base, tracks, 1000.0, max_gap_s=10.0)
        found = base["emitter_id"].astype("Int64").fillna(-1).to_numpy(int)
        out[f"{name}_correct"] = found == true_ids
    out.attrs["offset_hz"] = offset
    path = OUT / "assign_baseline.csv"
    out.to_csv(path, index=False)
    (OUT / "assign_baseline.json").write_text(
        json.dumps({"offset_hz": offset, "n_tp": int(len(out)),
                    "items_vs_kept_agreement": float(agree)}, indent=1) + "\n"
    )
    log.info("wrote %s", path)
    print(out.groupby("df_bin")[["head_correct", "edge_correct", "edge_offset_correct"]].mean())
    print(out[["head_correct", "edge_correct", "edge_offset_correct"]].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="+", choices=["snippets", "traces", "raw", "assign-baseline"])
    parser.add_argument("--only", nargs="*", help="snippet names (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        if step == "snippets":
            snippets(args.only)
        elif step == "traces":
            traces()
        elif step == "raw":
            raw()
        else:
            assign_baseline()


if __name__ == "__main__":
    main()
