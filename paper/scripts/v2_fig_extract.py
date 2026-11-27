"""Cache the raw-data inputs of the three pipeline-v2 figures.

The plotting scripts (``plot_v2_*.py``) run in the paper environment, which
has no chirpdetector or gridtools. This module computes, once, everything
that needs them and writes it to ``$PAPER_OUT/data/processed/v2_figures/``:

``snippets``
    Spectrogram snippets of the 2019/2020 tube competition recordings,
    computed exactly as the production run did (chirpdetector's
    `span_spectrogram` with the pipeline's detector bundle,
    ``paper_config.PIPELINE["detector"]``: click removal on grids of at
    least 9 electrodes, then the front end: a 205-ms Hann window, 5-ms hop,
    4.88-Hz bins, 500-1500 Hz, power summed over electrodes, in all three
    channels for the single-window detector; pipeline v2's adds 51- and 13-ms
    windows on the same grid as the second and third channel), normalised
    to [0, 1] over the snippet
    with the front end's own rule (60 dB range ending 10 dB above the median
    per-frame peak of the reference channel; ``img``, clipped to [0, 1] as
    the detector sees it), and an analysis spectrogram unclipped in dB
    (``power_db``: always the three windows of pipeline v2's front end,
    205, 51 and 13 ms, ``ANALYSIS_BUNDLE``, whatever the detector; the
    figures draw its 205-ms channel, the same in both front ends; 10 log10
    of the calibrated linear power summed over the
    electrodes, dB re 1 V^2; a sinusoid of amplitude A on a bin reads
    A^2 / 2 per electrode; float16, steps <= 0.125 dB), plus the
    wavetracker tracks around the snippet. One ``<name>.npz`` per snippet.
``traces``
    Instantaneous-frequency traces of the example chirps of the pipeline
    figure (panel D),
    extracted from the raw recording with gridtools' harvest extraction
    (emitter's field pattern, other fish projected out, pulse-pair estimator
    with a 1-ms adaptive window) and per-EOD-period frequencies from the
    upward zero crossings of the same combined signal. ``traces.npz``.
``case-heights``
    The instantaneous-frequency height (and widths) of every chirp in the
    windows of ``CASE_HEIGHT_SNIPPETS`` (failure-mode figure B, long chirps),
    from the same harvest extraction, next to the decoder's estimates.
    ``<snippet>_trace_heights.json``.
``raw``
    The raw voltage of the electrode with the largest RMS in the window of
    snippet ``f1_fe_e``, for the "recording" thumbnail of the pipeline
    figure. ``raw.npz``.
``assign-baseline``
    Emitter assignment on the simulated test scenes after post-processing,
    per true positive at score >= 0.5: the emitter head (as evaluated) and
    two box-only baselines (nearest live track to the box's lower edge, and
    to the lower edge plus the median offset emitter - f_low of the training
    chirps). Recomputed from the saved post-processed predictions and the
    scenes' tracks. Also the emitter's ΔF: the distance of the true
    emitter's track to the nearest other live track at the label's centre
    (``emitter_df_hz``, ``emitter_df_bin``; the evaluation's ``delta_f_hz``
    is the closest pair of any two live tracks, which in scenes with three
    or more fish need not include the emitter). ``assign_baseline.csv``.

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
import os
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config
from v2_examples import example_id

log = logging.getLogger("v2_fig_extract")

CHIRP_DATA = paper_config.CHIRP_DATA
# the detections follow PAPER_DETECTIONS (paper_config: the pipeline, v2 or
# the single-window "sw", and its run: the production run, its patch with the
# trial fish's track gaps filled, or that patch at another threshold)
RUNS = paper_config.RUNS  # run directory, chirp table, run holding tracks_gapfill.npz (or None)
DETECTIONS = paper_config.DETECTIONS
RUN = CHIRP_DATA / "detections" / RUNS[DETECTIONS][0]
PARQUET = RUNS[DETECTIONS][1]
TRACKS_RUN = (CHIRP_DATA / "detections" / RUNS[DETECTIONS][2]) if RUNS[DETECTIONS][2] else None
BUNDLE = paper_config.pipeline_path("detector")  # the pipeline's detector
REC_2019 = paper_config.DATA_ROOT
# the simulated dataset the detector was trained and tested on, and its
# test-set evaluation after post-processing (eval_postprocess.py)
SIM_DATASET = (Path(os.environ.get("CHIRP_FAST", str(CHIRP_DATA))).expanduser()
               / paper_config.PIPELINE["sim_dataset"])
PP_EVAL = paper_config.pipeline_path("sim_eval")
# the analysis spectrogram stored with every snippet (``power_db``): the
# three-window front end of pipeline v2 (205, 51 and 13 ms on one 4.88-Hz grid),
# whatever the detector, so that the figures' spectrograms (the 205-ms channel,
# the same in both front ends) and the detector-free chirp count of the
# failure-mode figure C (the 51-ms channel) do not depend on the detector's input
ANALYSIS_BUNDLE = CHIRP_DATA / paper_config.PIPELINES["v2"]["detector"]

OUT = paper_config.PROCESSED / "v2_figures"  # created by main(), not at import

# name -> (recording, t_start_s, t_end_s) or (recording, t_start_s, t_end_s,
# (f_low_hz, f_high_hz)); times in seconds from the recording's first sample
# (the clock of chirps_rtdetr.parquet). The optional band is the frequency
# range the figure shows (stored in the snippet's meta as ``band_hz``; the
# spectrogram itself always covers the front end's 500-1500 Hz), so that a
# window chosen for a figure is set here, in one place
SNIPPETS: dict[str, tuple] = {
    # pipeline figure, "recording" thumbnail (raw voltage; the former front-end
    # panel, dropped 2026-11-27): two fish 29 Hz apart, a small and a big chirp
    "f1_fe_e": ("2020-06-05-10_00", 89.9, 91.1),
    # pipeline figure B: detection and assignment, 8.2 s (2245.2-2253.4 s)
    # with 13 chirps of both fish (author's choice of three candidates,
    # 2026-11-24); the extraction keeps a margin for the crop
    "f1_detect": ("2020-06-02-10_00", 2244.8, 2254.0),
    # pipeline figure C (harmonic) and D (decoder examples): one detector tile
    # (tile 4932: 3.2 s from 4932 * 2.2 s)
    "f1_tile": ("2020-05-29-10_00", 10850.4, 10853.6),
    # pipeline figure C: a duplicate box on one chirp, both boxes >= 0.8
    # (author's choice 2026-11-16)
    "f1_dup": ("2019-12-02-10_00", 1196.5, 1197.7),
    # failure-mode figure A: hand-labelled test region (2019-12-09, region
    # 4260-4423 s)
    "f3_labelled": ("2019-12-09-10_00", 4312.0, 4328.0),
    # failure-mode figure (stats only since 2026-11-25): the track of one fish
    # has a 140-s gap while it keeps chirping (the Methods cite its numbers)
    "f3_trackgap": ("2019-12-04-10_00", 7468.0, 7488.0),
    # failure-mode figure (stats only since 2026-11-25): assignments between
    # close fish
    "g_close_15388": ("2019-12-09-10_00", 15387.9, 15389.1),
    "g_close_9394": ("2020-03-13-10_00", 9393.9, 9395.1),
    # failure-mode figure B and C (round 3, the author's choice of 2026-11-25
    # among the candidates of a search; 15-20 s centred on the chosen 6-s
    # candidate, edges between boxes, the band from the boxes and tracks plus
    # a margin so that no box is cut; the plot script checks both):
    # (B) huge, long chirps of the 821-Hz fish (track 7) of an analysed trial
    # (candidate 841-847 s)
    "f3_long": ("2020-06-02-10_00", 835.0, 853.0, (620.0, 1340.0)),
    # (C) chirp rates so high that the chirps smear together: the ~702-Hz fish
    # (track 0) of a draw, not one of the 33 analysed trials (candidate
    # 11406-11412 s)
    "f3_dense": ("2020-03-18-10_34", 11400.3, 11417.62, (560.0, 1040.0)),
}

# failure-mode figure B: snippets whose chirps get an instantaneous-frequency
# height from the raw recording (``case-heights``), to check the decoder's
# heights of long chirps
CASE_HEIGHT_SNIPPETS = ("f3_long",)

# pipeline figure (decoder panel): chirps of the f1_tile recording whose
# traces are extracted, by their pipeline-v2 chirp_id (v2_examples.example_id
# finds the selected run's box)
TRACE_CHIRPS: dict[str, tuple[str, int]] = {
    "small": ("2020-05-29-10_00", 2085),
    "big": ("2020-05-29-10_00", 2086),
}


def to_db(power: np.ndarray) -> np.ndarray:
    """Linear power -> dB (10 log10), float16; zeros at the float32 floor, as
    chirpdetector's ``normalize`` does before it clips."""
    tiny = np.finfo(np.float32).tiny
    return (10.0 * np.log10(np.maximum(np.asarray(power, np.float64), tiny))).astype(np.float16)


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
    patched = TRACKS_RUN / rec_name / "tracks_gapfill.npz" if TRACKS_RUN else None
    if patched is not None and patched.exists():
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


def snippet_window(entry: tuple) -> tuple[str, float, float, tuple[float, float] | None]:
    """(recording, t0, t1, band or None) of a ``SNIPPETS`` entry; fails loudly
    on anything else."""
    if len(entry) == 3:
        rec_name, t0, t1 = entry
        band = None
    elif len(entry) == 4:
        rec_name, t0, t1, band = entry
        band = (float(band[0]), float(band[1]))
        if not 500.0 <= band[0] < band[1] <= 1500.0:
            msg = f"band {band} outside the front end's 500-1500 Hz"
            raise ValueError(msg)
    else:
        msg = f"snippet entry {entry!r}: (recording, t0, t1[, (f0, f1)])"
        raise ValueError(msg)
    if not float(t0) < float(t1):
        msg = f"snippet entry {entry!r}: t0 must be before t1"
        raise ValueError(msg)
    return str(rec_name), float(t0), float(t1), band


def parse_window(values: list[str]) -> tuple[str, tuple]:
    """``--window NAME RECORDING T0 T1 [F0 F1]`` -> (name, SNIPPETS entry)."""
    if len(values) not in (4, 6):
        msg = f"--window takes NAME RECORDING T0 T1 [F0 F1], got {values}"
        raise ValueError(msg)
    name, rec_name, t0, t1 = values[0], values[1], float(values[2]), float(values[3])
    entry = (rec_name, t0, t1) if len(values) == 4 else (
        rec_name, t0, t1, (float(values[4]), float(values[5])))
    snippet_window(entry)  # validate
    return name, entry


def snippets(names: list[str] | None = None) -> None:
    from chirpdetector.bundle import load_bundle
    from chirpdetector.inference.spans import span_bounds, span_spectrogram
    from chirpdetector.spectrogram import normalize, reference_level_db
    from gridtools.io import open_recording

    bundle = load_bundle(BUNDLE)
    config = bundle.spectrogram
    analysis = load_bundle(ANALYSIS_BUNDLE)
    same = Path(ANALYSIS_BUNDLE).resolve() == Path(BUNDLE).resolve()
    track_cache: dict[str, object] = {}
    for name, entry in SNIPPETS.items():
        if names and name not in names:
            continue
        rec_name, t0, t1, band = snippet_window(entry)
        recording = open_recording(REC_2019 / rec_name)
        fs = recording.samplerate_hz
        start, stop = span_bounds(bundle, fs, recording.n_samples, t0, t1)
        spec = span_spectrogram(bundle, recording, start, stop, device="cpu")
        img = normalize(spec, config).cpu().numpy().astype(np.float16)
        if same:
            aspec = spec
        else:
            if span_bounds(analysis, fs, recording.n_samples, t0, t1) != (start, stop):
                raise RuntimeError(f"{name}: the analysis front end spans other samples")
            aspec = span_spectrogram(analysis, recording, start, stop, device="cpu")
            if not (np.array_equal(aspec.times_s, spec.times_s)
                    and np.array_equal(aspec.freqs_hz, spec.freqs_hz)):
                raise RuntimeError(f"{name}: analysis and detector spectrograms on other grids")
        power_db = to_db(aspec.power.cpu().numpy())
        if rec_name not in track_cache:
            track_cache[rec_name] = _run_tracks(rec_name)
        tw = _tracks_window(track_cache[rec_name], t0 - 15.0, t1 + 15.0)
        meta = {
            "recording": rec_name,
            "t_start_s": t0,
            "t_end_s": t1,
            "band_hz": list(band) if band is not None else None,
            "n_electrodes": int(recording.n_channels),
            "samplerate_hz": float(fs),
            "reference_db": reference_level_db(spec),
            "dynamic_range_db": float(config.dynamic_range_db),
            "headroom_db": float(config.headroom_db),
            "power_db_unit": "dB re 1 V^2 (calibrated one-sided power, summed over electrodes)",
            "windows_s": list(analysis.spectrogram.window_durations_s),  # of power_db
            "detector_windows_s": list(config.window_durations_s),  # of img
            "analysis_bundle": str(ANALYSIS_BUNDLE),
            "click_removal": bundle.preprocessing.get("click_removal") is not None
            and int(recording.n_channels) >= 9,
            "bundle": str(BUNDLE),
            "weights_sha256": bundle.weights_sha256,
            "detections": DETECTIONS,
            "run": str(RUN),
            "tracks_run": str(TRACKS_RUN) if TRACKS_RUN else "wavetracker",
        }
        np.savez_compressed(
            OUT / f"{name}.npz",
            img=img,
            power_db=power_db,
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
    for key, (rec_name, v2_id) in TRACE_CHIRPS.items():
        table = pd.read_parquet(RUN / rec_name / PARQUET)
        chirp_id = example_id(table, rec_name, v2_id)
        row = table.set_index("chirp_id").loc[chirp_id]
        # the harvest uses the wavetracker tracks; with the gap-filled run this
        # is only valid if the chirp does not lie in a filled gap of its fish
        patched = TRACKS_RUN / rec_name / "tracks_gapfill.npz" if TRACKS_RUN else None
        if patched is not None and patched.exists():
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


def peak_excursion(t: np.ndarray, freq: np.ndarray, row) -> dict:
    """A chirp's height in its instantaneous-frequency trace: the largest
    excursion of the trace from the box start to 100 ms after the box end
    above its baseline (median of the 200 ms that end 50 ms before the box),
    and the trace's widths at 50 % and 10 % of that height (the contiguous run
    around the peak)."""
    pre = (t > row.t_start_s - 0.25) & (t < row.t_start_s - 0.05)
    base = float(np.nanmedian(freq[pre]))
    inside = (t >= row.t_start_s) & (t <= row.t_end_s + 0.1)
    ti, fi = t[inside], freq[inside] - base
    k = int(np.nanargmax(fi))
    height = float(fi[k])

    def width(frac):
        above = fi > frac * height
        a = b = k
        while a > 0 and above[a - 1]:
            a -= 1
        while b < len(fi) - 1 and above[b + 1]:
            b += 1
        return float(ti[b] - ti[a])

    return {"baseline_hz": base, "height_hz": height, "fwhm_s": width(0.5),
            "width10_s": width(0.1)}


def case_heights() -> None:
    """Instantaneous-frequency heights of the chirps of ``CASE_HEIGHT_SNIPPETS``
    (gridtools harvest extraction, as ``traces``), to compare with the
    decoder's heights: ``<snippet>_trace_heights.json``, one record per
    operating chirp in the snippet's window."""
    from gridtools.io import load_tracks
    from gridtools.templates.harvest import (
        HarvestConfig,
        harvest_candidate,
        open_recording_lenient,
    )

    cfg = HarvestConfig()
    for name in CASE_HEIGHT_SNIPPETS:
        rec_name, t0, t1, _ = snippet_window(SNIPPETS[name])
        table = pd.read_parquet(RUN / rec_name / PARQUET)
        rows = table[table.chirp & (table.t_end_s > t0) & (table.t_start_s < t1)]
        patched = TRACKS_RUN / rec_name / "tracks_gapfill.npz" if TRACKS_RUN else None
        gaps = np.load(patched) if patched is not None and patched.exists() else None
        recording, _ = open_recording_lenient(REC_2019 / rec_name)
        tracks = load_tracks(REC_2019 / rec_name)
        out = []
        for row in rows.itertuples():
            rec = {"chirp_id": int(row.chirp_id), "t_center_s": float(row.t_center_s),
                   "track_id": float(row.track_id), "decoder_height_hz": float(row.height_hz),
                   "decoder_fwhm_s": float(row.fwhm_s),
                   "decoder_width10_s": float(row.width10_s)}
            # the harvest uses the wavetracker tracks: not valid in a filled gap
            if gaps is not None and (
                    (gaps["gap_fish_id"] == row.track_id) & (gaps["gap_action"] == "filled")
                    & (gaps["gap_start_s"] < row.t_end_s)
                    & (gaps["gap_end_s"] > row.t_start_s)).any():
                out.append({**rec, "status": "in_filled_gap"})
                continue
            box = pd.Series({"box_index": int(row.chirp_id), "t_start_s": row.t_start_s,
                             "t_end_s": row.t_end_s, "f_low_hz": row.f_low_hz,
                             "f_high_hz": row.f_high_hz, "score": row.score,
                             "track_id": int(row.track_id), "emitter_hz": row.emitter_hz})
            result = harvest_candidate(recording, tracks, box, cfg, recording_name=rec_name)
            ex = result.extraction
            rec.update(status=str(result.record.get("status")),
                       reason=str(result.record.get("reason")))
            if ex is not None:
                t = ex.window_start_s + np.arange(len(ex.signal)) / ex.samplerate_hz
                freq = np.asarray(ex.traces["smooth_freq"], dtype=float)
                rec.update({f"trace_{k}": v for k, v in peak_excursion(t, freq, row).items()})
            out.append(rec)
            log.info("%s chirp %d: decoder %.0f Hz, trace %s Hz (%s)", name, row.chirp_id,
                     row.height_hz, f"{rec.get('trace_height_hz', float('nan')):.0f}",
                     rec["status"])
        (OUT / f"{name}_trace_heights.json").write_text(json.dumps(
            {"snippet": name, "recording": rec_name, "window_s": [t0, t1],
             "detections": DETECTIONS, "run": str(RUN), "chirps": out}, indent=1) + "\n")


def raw() -> None:
    from gridtools.io import open_recording

    rec_name, t0, t1, _ = snippet_window(SNIPPETS["f1_fe_e"])
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


DF_EDGES_HZ = (0.0, 10.0, 25.0, 50.0, 100.0)  # the evaluation's ΔF bins (left-closed)
DF_NAMES = ("0-10", "10-25", "25-50", "50-100", "100+")
TRACK_MAX_GAP_S = 1.0  # a track is live within 1 s of its detections (evaluation default)


def df_bin_names(df_hz: np.ndarray) -> np.ndarray:
    """ΔF bin names of the evaluation; ``unknown`` where ΔF is NaN."""
    idx = np.clip(np.digitize(np.nan_to_num(df_hz, nan=0.0), DF_EDGES_HZ) - 1, 0, len(DF_NAMES) - 1)
    return np.where(np.isfinite(df_hz), np.asarray(DF_NAMES, dtype=object)[idx], "unknown")


def emitter_delta_f(tp: pd.DataFrame, tracks: dict) -> np.ndarray:
    """Distance of the true emitter's track to the nearest other live track
    at the label's centre (NaN if the emitter or every other track is not live)."""
    out = np.full(len(tp), np.nan)
    rec = tp["recording"].astype(str).to_numpy()
    t = tp["label_t_center_s"].to_numpy(float)
    emitter = tp["true_emitter_id"].astype("Int64").fillna(-1).to_numpy(int)
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        trk = tracks[r]
        freqs = {int(i): trk.track(int(i)).frequency_at(t[idx], TRACK_MAX_GAP_S) for i in trk.ids}
        for j, k in enumerate(idx):
            if emitter[k] not in freqs or not np.isfinite(freqs[emitter[k]][j]):
                continue
            fe = freqs[emitter[k]][j]
            others = [f[j] for i, f in freqs.items() if i != emitter[k] and np.isfinite(f[j])]
            if others:
                out[k] = float(np.min(np.abs(np.asarray(others) - fe)))
    return out


def assign_baseline() -> None:
    from chirpdetector.data import load_dataset
    from chirpdetector.evaluation import assign_by_frequency
    from chirpdetector.postprocess import COLUMNS
    from gridtools.io import load_tracks

    dataset = load_dataset(SIM_DATASET, check_files=False)
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
    out["emitter_df_hz"] = emitter_delta_f(tp, tracks)
    out["emitter_df_bin"] = df_bin_names(out["emitter_df_hz"].to_numpy(float))
    log.info("emitter ΔF unknown (emitter not live or no other live track) for %d of %d TPs",
             int(np.isnan(out["emitter_df_hz"]).sum()), len(out))
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
    print(out.groupby("emitter_df_bin")[["head_correct", "edge_correct",
                                         "edge_offset_correct"]].agg(["mean", "size"]))
    print(out[["head_correct", "edge_correct", "edge_offset_correct"]].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="+",
                        choices=["snippets", "traces", "raw", "case-heights", "assign-baseline"])
    parser.add_argument("--only", nargs="*", help="snippet names (default: all)")
    parser.add_argument(
        "--window", nargs="+", action="append", default=[],
        metavar="NAME RECORDING T0 T1 [F0 F1]",
        help="snippets: set (or add) snippet NAME to this window instead of the "
        "SNIPPETS table, e.g. to preview candidates (repeatable; only these are "
        "extracted unless --only names more). For the figure, write the chosen "
        "window into SNIPPETS so that the run is reproducible",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    only = list(args.only or [])
    for values in args.window:
        name, entry = parse_window(values)
        if name in SNIPPETS and SNIPPETS[name] != entry:
            log.warning("snippet %s: --window %s overrides the SNIPPETS table %s",
                        name, entry, SNIPPETS[name])
        SNIPPETS[name] = entry
        only.append(name)
    for step in args.steps:
        if step == "snippets":
            snippets(only or None)
        elif step == "traces":
            traces()
        elif step == "raw":
            raw()
        elif step == "case-heights":
            case_heights()
        else:
            assign_baseline()


if __name__ == "__main__":
    main()
