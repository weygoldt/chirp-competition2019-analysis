"""Build a chase-review session: every BORIS chase of the trials behind Fig. 6, ready for the browser.

    paper/chase_review/build_session.py NAME [--detections rtdetr_gapfill_t080] [--seed 42]

Writes ``$CHIRP_DATA/labels/chase_review_2019/NAME/session.json`` (``--root`` for another
place); ``server.py`` serves it. Nothing in the dataset is written: the BORIS files, the
videos and the LED files are only read.

The chases are exactly the ones the paper's analyses use: the trials that pass
``analyze_event_time_rates_trial.is_good_trial`` (group 5, no draw, not in
``bad_recordings``, LED files present), the BORIS pairing of ``load_boris`` (each onset
with the first offset after it), video frames ``round(t * FPS)`` and the LED map of
``led_alignment.fit_led_alignment``. The builder checks its recording times against
``load_and_convert_boris_events`` and stops on any difference.

Per recording the session holds the video (the file BORIS was scored on, its frame
count and its stream delay, see :func:`probe_video`), the LED map's segments (so the
browser converts any video frame to recording time with the same piecewise-linear map),
the contacts, and the winner's and loser's chirps (recording time, from
``chirp_times_<detections>.npy``). Per chase: its id ``<recording>_c<k>`` (k = index in
the BORIS pairing), BORIS onset / offset (video s, frame), recording time, and whether its
offset is shared with another chase (one BORIS offset is missing in 2020-05-13).
``order`` is a seeded random order of all chases: reviewing it from the start gives an
unbiased sample at every point.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from analyze_event_time_rates_trial import (  # noqa: E402
    ACTIVE_PHASE_S,
    is_good_trial,
    load_and_convert_boris_events,
    load_boris,
)
from chirps import load_chirps  # noqa: E402
from led_alignment import check_led_alignment, fit_led_alignment  # noqa: E402
from paper_config import CHIRP_DATA, DATA_ROOT, PRIMARY  # noqa: E402

SR = 20_000  # grid sampling rate (Hz), as in analyze_event_time_rates_trial.load_data
SESSIONS = CHIRP_DATA / "labels" / "chase_review_2019"
PAD_BEFORE_S = 10.0  # default window: this long before the BORIS onset ...
PAD_AFTER_S = 10.0  # ... to this long after the BORIS offset (recording seconds)


def probe_video(path: Path) -> dict:
    """Frame count, frame rate and stream delay of a video.

    The delay is the timestamp (in frames) ffmpeg gives the first frame. The ``_mod.avi``
    files are h264 with B-frames in AVI, which has no presentation timestamps, so ffmpeg
    stamps frame k with k + 2: seeking to t = f / fps lands on frame f - 2. The clips
    seek to (f + delay) / fps, so clip frame i is video frame f + i, the numbering of
    ``LED_frames.npy`` (checked on the LED, ``check_clip_leds.py``).
    """
    def run(*args: str) -> str:
        return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", *args, str(path)],
                              capture_output=True, text=True, check=True).stdout.strip()

    stream = json.loads(run("-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json"))["streams"][0]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    # read a few packets: the decoder stamps frames only once its B-frame queue is full
    frames = json.loads(run("-show_frames", "-read_intervals", "%+#10", "-show_entries",
                            "frame=best_effort_timestamp", "-of", "json"))["frames"]
    ts = [f.get("best_effort_timestamp") for f in frames]
    if ts[0] is None or ts[1] != ts[0] + 1:
        raise ValueError(f"{path}: cannot read the first frames' timestamps ({ts})")
    return {"n_frames": int(stream["nb_frames"]), "fps": num / den, "delay_frames": int(ts[0])}


def boris_media(trial_path: Path, recording: str) -> str:
    """The video file BORIS was scored on (basename of its ``Media file`` column, in the trial folder)."""
    boris_file = "-".join(recording.split("-")[:3]) + ".csv"
    media = pd.read_csv(trial_path / boris_file)["Media file"].unique()
    if len(media) != 1:
        raise ValueError(f"{recording}: BORIS file names {len(media)} media files: {media}")
    name = str(media[0]).replace("\\", "/").rsplit("/", 1)[-1]
    if not (trial_path / name).exists():
        raise FileNotFoundError(f"{recording}: BORIS video {name} not in {trial_path}")
    return name


def shared_offsets(off_frames: np.ndarray, ids: list[str]) -> list[list[str]]:
    """For each chase, the other chases paired with the same BORIS offset."""
    return [[ids[j] for j in np.flatnonzero(off_frames == f) if j != i] for i, f in enumerate(off_frames)]


def build_recording(trial_path: Path, row: pd.Series, detections: str) -> tuple[dict, list[dict]]:
    rec = row["recording"]
    _, _, on_off_s, contact_s, fps = load_boris(trial_path, rec)
    on_off_s = np.asarray(on_off_s, dtype=float).reshape(-1, 2)
    on_off_f = np.array(np.round(on_off_s * fps), dtype=int)  # as load_and_convert_boris_events
    contact_f = np.array(np.round(np.asarray(contact_s, dtype=float) * fps), dtype=int)

    led_idx = pd.read_csv(trial_path / "led_idxs.csv", header=None).iloc[:, 0].to_numpy()
    led_frames = np.load(trial_path / "LED_frames.npy")
    align = fit_led_alignment(led_frames, led_idx, SR)
    check_led_alignment(align, rec)
    on_off_g = align.to_grid(on_off_f)
    contact_g = align.to_grid(contact_f)

    # the analysis' own conversion must give the same recording times
    ref_contact, ref_on_off, _, _ = load_and_convert_boris_events(trial_path, rec, sr=SR)
    if not (np.array_equal(ref_on_off.reshape(-1, 2), on_off_g) and np.array_equal(ref_contact, contact_g)):
        raise AssertionError(f"{rec}: chase times differ from load_and_convert_boris_events")

    media = boris_media(trial_path, rec)
    video = probe_video(trial_path / media)
    if abs(video["fps"] - fps) > 1e-9:
        raise ValueError(f"{rec}: video runs at {video['fps']} fps, BORIS at {fps}")
    if led_frames.max() >= video["n_frames"]:
        raise ValueError(f"{rec}: LED frame {led_frames.max()} beyond the video's {video['n_frames']} frames")

    chirp_t, chirp_id = load_chirps(trial_path, detections)
    ids = [f"{rec}_c{k:03d}" for k in range(len(on_off_f))]
    shared = shared_offsets(on_off_f[:, 1], ids)
    chases = [{
        "id": ids[k], "recording": rec, "k": k,
        "boris_onset_s": round(float(on_off_s[k, 0]), 3), "boris_offset_s": round(float(on_off_s[k, 1]), 3),
        "onset_frame": int(on_off_f[k, 0]), "offset_frame": int(on_off_f[k, 1]),
        "onset_rec_s": float(on_off_g[k, 0]), "offset_rec_s": float(on_off_g[k, 1]),
        "shares_offset_with": shared[k],
    } for k in range(len(on_off_f))]
    info = {
        "recording": rec,
        "video": media, **video, "boris_fps": float(fps),
        "led_map": {"bounds": align.seg_bounds.tolist(), "slope": align.seg_slope.tolist(),
                    "icpt": align.seg_icpt.tolist(), "method": align.method,
                    "resid_med_s": align.resid_med, "resid_p95_s": align.resid_p95},
        "win_id": float(row["win_ID"]), "lose_id": float(row["lose_ID"]),
        "sex_win": str(row["sex_win"]), "sex_lose": str(row["sex_lose"]),
        "chirps": {"winner": np.round(np.sort(chirp_t[chirp_id == row["win_ID"]]), 4).tolist(),
                   "loser": np.round(np.sort(chirp_t[chirp_id == row["lose_ID"]]), 4).tolist()},
        "contacts": [{"frame": int(f), "rec_s": float(g)} for f, g in zip(contact_f, contact_g, strict=True)],
    }
    return info, chases


def git_commit() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", ".") else "")


def build_session(detections: str, seed: int) -> dict:
    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    recordings, chases = {}, []
    for _, row in ts.iterrows():
        trial_path = DATA_ROOT / row["recording"]
        if not is_good_trial(trial_path, row):
            continue
        info, rec_chases = build_recording(trial_path, row, detections)
        recordings[info["recording"]] = info
        chases += rec_chases
    order = [chases[i]["id"] for i in np.random.default_rng(seed).permutation(len(chases))]
    return {
        "meta": {"created": datetime.now(UTC).isoformat(timespec="seconds"), "data_root": str(DATA_ROOT),
                 "detections": detections, "seed": seed, "commit": git_commit(),
                 "pad_before_s": PAD_BEFORE_S, "pad_after_s": PAD_AFTER_S,
                 "active_phase_s": list(ACTIVE_PHASE_S), "n_chases": len(chases),
                 "n_recordings": len(recordings)},
        "recordings": recordings, "chases": chases, "order": order,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", help="session name, e.g. offsets_v1")
    ap.add_argument("--detections", default=PRIMARY, help=f"chirp detections to show (default {PRIMARY})")
    ap.add_argument("--seed", type=int, default=42, help="seed of the review order")
    ap.add_argument("--root", type=Path, default=SESSIONS, help=f"session root (default {SESSIONS})")
    ap.add_argument("--force", action="store_true", help="rebuild session.json of an existing session")
    args = ap.parse_args()
    out = args.root / args.name
    if (out / "session.json").exists() and not args.force:
        sys.exit(f"{out}/session.json exists; --force rebuilds it (revisions.jsonl is kept)")
    session = build_session(args.detections, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "session.json.tmp"
    tmp.write_text(json.dumps(session))
    tmp.replace(out / "session.json")
    m = session["meta"]
    print(f"{out}/session.json: {m['n_chases']} chases in {m['n_recordings']} recordings, chirps {args.detections}")


if __name__ == "__main__":
    main()
