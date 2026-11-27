#!/usr/bin/env python3
"""Build the derived-data package of the chirp competition paper.

The package holds every input of the paper's analyses except the raw grid
recordings, the videos and caches that can be regenerated from them (about
1.7 TB): per recording the frequency tracks, the chirp detections of the three
threshold runs, the rises, the grid configuration and, for the annotated
trials, the behavioral annotations and the video synchronization; the trial and
fish metadata; the re-labeled chases; and the data behind the chirp-detection
pipeline (chirp templates, expert labels, evaluation outputs). Its
``recordings/`` folder has the layout of the original data folder, so the
analysis code runs on it with ``CHIRP_REC_2019=<package>/recordings``.

Standard library only. Re-runnable: a file whose checksum already matches is
not copied again, so the package can be rebuilt whenever the data change.
Sources are only read, never changed. Before copying, the detections in the
trial folders are checked against the run outputs they were copied from; any
difference stops the build.

Usage::

    python3 paper/scripts/build_data_package.py OUT [--dry-run]
    python3 paper/scripts/build_data_package.py OUT --verify     # re-hash the package
    python3 paper/scripts/build_data_package.py OUT --prune      # also delete stale files

Writes ``OUT/README.md``, ``OUT/MANIFEST.tsv`` (path, bytes, sha256, source),
``OUT/package.json`` (build date, source paths, git commits, counts) and
``OUT/recordings/recordings.csv`` (duration, sampling rate and electrodes of
every recording, from the detection metadata, so that nothing needs the raw
files).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CHIRP_REC_2019", "/mnt/data2/2019_tube_competition"))
CHIRP_DATA = Path(os.environ.get("CHIRP_DATA", "/mnt/data1/chirp_v2"))
INPUTS = Path(os.environ.get("PAPER_INPUTS", REPO / "paper" / "inputs"))  # git-ignored data inputs

# detection threshold -> (name in the trial folders = PAPER_DETECTIONS, run folder under CHIRP_DATA/detections)
RUNS = {
    "0.5": ("rtdetr_gapfill", "competition2019_v2_20261113_gapfill_recal1"),
    "0.8": ("rtdetr_gapfill_t080", "competition2019_v2_20261113_gapfill_t080_recal1"),
    "0.9": ("rtdetr_gapfill_t090", "competition2019_v2_20261113_gapfill_t090_recal1"),
}
RECORDING = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}_\d{2}$")

# top-level files of the data folder
TOP_LEVEL = ["trial_summary.csv", "chirp_notes.csv", "id_meta.csv", "order_meta.csv", "README.txt"]

# per recording: (file name or pattern, required). "{date}" is YYYY-MM-DD of the recording.
TRIAL_FILES = [
    ("fund_v.npy", True),  # wavetracker: frequency of every track point (Hz)
    ("ident_v.npy", True),  # wavetracker: fish identity of every track point (NaN = none)
    ("idx_v.npy", True),  # wavetracker: time index of every track point (into times.npy)
    ("times.npy", True),  # wavetracker: time of every index (s)
    ("fishgrid.cfg", True),  # grid configuration (electrodes, sampling)
    ("chirp_times_rcnn.npy", True),  # v1 detections (published analysis), for comparison
    ("chirp_ids_rcnn.npy", True),
    ("analysis/ids.npy", False),  # fish identities of the rise detection
    ("analysis/rise_idx.npy", False),  # rises (frequency rises) per fish
    ("{date}.csv", False),  # BORIS annotations (chases, contacts), annotated trials only
    ("LED_frames.npy", False),  # LED flashes in the video (frames)
    ("led_idxs.csv", False),  # LED flashes in the recording (samples)
    ("{date}_0_times.dat", False),  # video frame times
]
EXCLUDED = ("sign_v.npy",)  # per-electrode signal of every track point; not used by the analyses (1.9 GB)


@dataclass(frozen=True)
class Item:
    """One file of the package: where it comes from and where it goes."""

    src: Path
    dst: str  # relative to the package root
    kind: str


# Globs relative to a root -> copied to the same relative path under the package folder
# (CHIRP_DATA -> pipeline/, paper/inputs -> inputs/). A glob that matches nothing stops the build.
_SOURCE_RUN = "detections/competition2019_v2_20261113"
_SIM_TEST = "scenes/train_v4/test/*"
PIPELINE = [  # (glob under CHIRP_DATA, kind)
    ("templates/v4/**/*", "chirp templates"),
    (f"{_SOURCE_RUN}/*/meta.json", "detection metadata (source run)"),
    (f"{_SOURCE_RUN}/track_gaps/gaps.csv", "track gaps"),
    (f"{_SOURCE_RUN}_gapfill/*/tracks_gapfill.npz", "gap-filled tracks"),
    *[(f"detections/{run}/*/meta.json", "detection metadata") for _, run in RUNS.values()],
    ("runs/dualhead_r18_sim_v4_r1/final/bundle/bundle.json", "detector configuration"),
    ("runs/dualhead_r18_sim_v4_r1/final/bundle/metrics.json", "detector configuration"),
    ("runs/dualhead_r18_sim_v4_longonly_r1/final/bundle/bundle.json", "front-end ablation"),
    ("runs/dualhead_r18_sim_v4_longonly_r1c/bundles/epoch_16/bundle.json", "front-end ablation"),
    ("runs/ablation_frontend/scenarios/paper/bench_labels.parquet", "front-end ablation"),
    ("runs/dualhead_r18_sim_v4_r1/eval_postprocess/sim_test_predictions_flagged.parquet", "detector evaluation"),
    ("runs/dualhead_r18_sim_v4_r1/eval_postprocess/sim_test_after/**/*", "detector evaluation"),
    ("datasets/sim_v4/dataset.json", "simulated dataset"),
    ("datasets/sim_v4/boxes.parquet", "simulated dataset"),
    ("datasets/sim_v4/diagnostics/counts.json", "simulated dataset"),
    *[(f"{_SIM_TEST}/{f}", "simulated test scenes") for f in (
        "times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy",
        "scene.json", "paths.npz", "chirp_traces.npz", "chirps.csv")],
    ("scenes/train_v4/train/*/chirps.parquet", "simulated training chirps"),
    ("params/runs/v4-dualhead-sim-v4-r2-recal1/bundle/bundle.json", "parameter network"),
    ("params/runs/v4-dualhead-sim-v4-r2-recal1/bundle/eval/predictions.parquet", "parameter network"),
    ("labels/benchmark/report_v2/**/*", "expert labels: blind benchmark"),
    ("labels/benchmark/human_benchmark_v*_truth.*", "expert labels: blind benchmark"),
    *[(f"labels/real_v1/{g}", "expert labels: real recordings")  # not tracks_gapfill/ (display copies)
      for g in ("README.md", "report/**/*", "truth/**/*", "review/**/*")],
    ("eval/real_labels_365/labels/**/*", "expert labels: real recordings"),
    ("background/v2/manifest.csv", "background segments (list)"),
]
INPUT_FILES = [  # (glob under paper/inputs, kind); git-ignored, not regenerable
    ("chase_validation/*", "re-labeled chases"),
    ("new_chirpextractions/*", "Fig. 1 chirp extractions"),
    ("README.md", "Fig. 1 chirp extractions"),
]


def sha256(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def recordings() -> list[Path]:
    recs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and RECORDING.match(p.name))
    if not recs:
        sys.exit(f"no recordings in {DATA_ROOT}")
    return recs


def check_detections(recs: list[Path]) -> None:
    """The trial folders' detections must equal the run outputs (fail loudly otherwise)."""
    bad = []
    for rec in recs:
        for name, run in RUNS.values():
            for f in (f"chirps_{name}.parquet", f"chirp_times_{name}.npy", f"chirp_ids_{name}.npy"):
                a, b = rec / f, CHIRP_DATA / "detections" / run / rec.name / f
                if not b.exists():
                    bad.append(f"{b}: missing")
                elif not a.exists():
                    bad.append(f"{a}: missing")
                elif a.stat().st_size != b.stat().st_size or sha256(a) != sha256(b):
                    bad.append(f"{a} differs from {b}")
    if bad:
        sys.exit("detections in the trial folders do not match the runs:\n  " + "\n  ".join(bad[:20]))


def items(recs: list[Path]) -> tuple[list[Item], list[str]]:
    out, missing = [], []
    for name in TOP_LEVEL:
        src = DATA_ROOT / name
        if src.exists():
            out.append(Item(src, f"recordings/{name}", "metadata"))
        else:
            missing.append(str(src))
    for rec in recs:
        date = rec.name[:10]
        for pattern, required in TRIAL_FILES:
            name = pattern.format(date=date)
            src = rec / name
            if src.exists():
                out.append(Item(src, f"recordings/{rec.name}/{name}", "trial"))
            elif required:
                missing.append(str(src))
        for name, run in RUNS.values():
            run_dir = CHIRP_DATA / "detections" / run / rec.name
            for f, dst in (
                (f"chirps_{name}.parquet", f"chirps_{name}.parquet"),
                (f"chirp_times_{name}.npy", f"chirp_times_{name}.npy"),
                (f"chirp_ids_{name}.npy", f"chirp_ids_{name}.npy"),
                ("meta.json", f"chirps_{name}_meta.json"),
            ):
                out.append(Item(run_dir / f, f"recordings/{rec.name}/{dst}", "detections"))
    for root, prefix, globs in ((CHIRP_DATA, "pipeline", PIPELINE), (INPUTS, "inputs", INPUT_FILES)):
        for pattern, kind in globs:
            files = sorted(f for f in root.glob(pattern) if f.is_file())
            if not files:
                missing.append(f"{root}/{pattern}")
            out.extend(Item(f, f"{prefix}/{f.relative_to(root).as_posix()}", kind) for f in files)
    return out, missing


def recordings_table(recs: list[Path]) -> list[dict]:
    """Duration, sampling rate and electrodes of every recording, from the primary run's metadata."""
    rows = []
    run = RUNS["0.8"][1]
    for rec in recs:
        meta = json.loads((CHIRP_DATA / "detections" / run / rec.name / "meta.json").read_text())
        inp = meta["inputs"]
        rows.append({
            "recording": rec.name,
            "duration_s": inp["duration_s"],
            "samplerate_hz": inp["samplerate_hz"],
            "n_samples": inp["n_samples"],
            "n_electrodes": inp["n_electrodes"],
        })
    return rows


README = """\
# Chirp competition 2019/2020: derived data

Derived data of the paper "{title}": everything its analyses read except the
raw grid recordings, the videos and the caches computed from them (about
1.7 TB; see the paper's data availability statement). Built {built} by
`paper/scripts/build_data_package.py` of the analysis repository (commit
`{commit}`). {n_rec} recordings, {n_files} files, {size}.
`MANIFEST.tsv` lists every file with its size, sha256 and source;
`package.json` the sources, code versions and counts.

## Layout

```
recordings/                    the data folder of the competitions, as the code expects it
  trial_summary.csv            one row per recording: fish, status (winner/loser), sex, size,
                               EODf, experience, counts, draw
  id_meta.csv                  fish metadata (group, weights, lengths)
  order_meta.csv               competitions in order: group, fish, winner, experience
  recordings.csv               duration, sampling rate, samples and electrodes per recording
  README.txt                   the original data README (Raab et al. 2021); see BORIS below
  YYYY-MM-DD-HH_MM/            one folder per recording
    fund_v.npy, ident_v.npy,   frequency tracks (wavetracker): frequency (Hz), fish identity
    idx_v.npy, times.npy       (NaN = unassigned) and time index of every track point, and
                               the time (s) of every index
    fishgrid.cfg               grid configuration (electrodes, sampling)
    chirps_<det>.parquet       chirp detections, one row per detection, with score, box,
                               predicted emitter EODf, assignment to a track, post-processing
                               flags and chirp parameters (the columns are documented in
                               chirpdetector, `chirpdetector.pipeline.outputs`)
    chirps_<det>_meta.json     provenance of the detections (models, configuration, code)
    chirp_times_<det>.npy,     the analyzed chirps (score at or above the operating threshold,
    chirp_ids_<det>.npy        not flagged, assigned): time of the box center (s) and fish
                               identity (as in ident_v)
    chirp_times_rcnn.npy,      the chirps of the published v1 analysis, for comparison
    chirp_ids_rcnn.npy
    analysis/ids.npy,          frequency rises: fish identities and rise indices
    analysis/rise_idx.npy
    YYYY-MM-DD.csv             behavioral annotations (BORIS), annotated trials only
    LED_frames.npy,            synchronization: frames of the LED flashes in the video and
    led_idxs.csv               samples of the same flashes in the recording
    YYYY-MM-DD_0_times.dat     time stamps of the video frames
inputs/chase_validation/       the authors' re-labeling of the annotated chases
                               (session.json: settings and per-recording LED maps;
                               revisions.jsonl: one record per reviewed chase)
inputs/new_chirpextractions/   chirp extractions shown in Fig. 1
pipeline/                      data of the chirp-detection pipeline, at the paths the code
                               expects under CHIRP_DATA
  templates/v4/                the chirp template library (frequency excursions of real
                               chirps) used for the hybrid simulations; see its report.md
{pipeline_lines}```

`<det>` names the detection run by its operating threshold: `rtdetr_gapfill` (0.5),
`rtdetr_gapfill_t080` (0.8, the primary analysis) and `rtdetr_gapfill_t090` (0.9).
All three hold the same detections (stored down to a score of 0.05, assigned
with gap-filled tracks); only the operating threshold, and with it the
`operating` and `chirp` columns and the chirp arrays, differ.

## Notes

- **BORIS codes.** In the annotation files, `Behavior` is 0 for a chase onset,
  1 for a chase offset and 2 for a physical contact (the original README.txt says
  1--3; the files use 0--2). `Start (s)` is video time; the analysis maps it to
  recording time with the LED flashes (`LED_frames.npy`, `led_idxs.csv`), as
  described in the paper's Methods.
- **Not included.** {excluded}.
- **Running the analyses on this package.** Point the analysis code at it:
  `CHIRP_REC_2019=<package>/recordings`, `CHIRP_DATA=<package>/pipeline` and
  `PAPER_INPUTS=<package>/inputs`, then follow the analysis repository's README.

## License and citation

{license}
"""


def write_readme(root: Path, info: dict, n_files: int) -> None:
    pipeline_lines = "".join(
        f"  {pattern:<29}{kind}\n" for pattern, kind in PIPELINE if not pattern.startswith("templates/")
    )
    text = README.format(
        title=os.environ.get("PAPER_TITLE", "Signals That Reflect but Do Not Shape Social Competition"),
        built=info["built"], commit=info["code"]["analysis"][:10], n_rec=info["counts"]["recordings"],
        n_files=n_files, size=human(info["counts"]["bytes"]), pipeline_lines=pipeline_lines,
        excluded="; ".join(info["excluded"]),
        license=os.environ.get("PACKAGE_LICENSE", "To be decided by the authors (e.g. CC BY 4.0); cite the paper."),
    )
    (root / "README.md").write_text(text)


def git_commit(path: Path) -> str:
    """HEAD of the repository at `path` ("unknown" unless `path` is a checked-out repository's root)."""
    top = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if not path.is_dir() or Path(top.stdout.strip() or "/nonexistent").resolve() != path.resolve():
        return "unknown"
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def copy(item: Item, root: Path) -> tuple[bool, str]:
    """Copy unless the destination already has the same content; returns (copied, sha256)."""
    dst = root / item.dst
    digest = sha256(item.src)
    if dst.exists() and dst.stat().st_size == item.src.stat().st_size and sha256(dst) == digest:
        return False, digest
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(item.src, tmp)
    tmp.replace(dst)
    return True, digest


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t" if path.suffix == ".tsv" else ",")
        w.writeheader()
        w.writerows(rows)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000:
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} PB"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path, help="package folder (created if missing)")
    p.add_argument("--dry-run", action="store_true", help="list what would be copied, copy nothing")
    p.add_argument("--verify", action="store_true", help="re-hash the package against its manifest")
    p.add_argument("--prune", action="store_true", help="delete files in the package that are not in it any more")
    args = p.parse_args()
    root: Path = args.out

    if args.verify:
        rows = list(csv.DictReader((root / "MANIFEST.tsv").open(), delimiter="\t"))
        bad = [r["path"] for r in rows if not (root / r["path"]).exists() or sha256(root / r["path"]) != r["sha256"]]
        print(f"{len(rows)} files, {len(bad)} missing or changed", *bad[:20], sep="\n  ")
        sys.exit(1 if bad else 0)

    recs = recordings()
    check_detections(recs)
    todo, missing = items(recs)
    if missing:
        sys.exit("missing required inputs:\n  " + "\n  ".join(missing))

    sizes: dict[str, int] = {}
    for it in todo:
        sizes[it.kind] = sizes.get(it.kind, 0) + it.src.stat().st_size
    print(f"{len(recs)} recordings, {len(todo)} files, {human(sum(sizes.values()))}")
    for kind, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {human(n):>10}  {kind}")
    if args.dry_run:
        return

    manifest, copied = [], 0
    for i, it in enumerate(todo, 1):
        did, digest = copy(it, root)
        copied += did
        manifest.append({"path": it.dst, "bytes": it.src.stat().st_size, "sha256": digest, "source": str(it.src)})
        if i % 500 == 0:
            print(f"  {i}/{len(todo)} files ({copied} copied)", flush=True)

    rec_rows = recordings_table(recs)
    write_table(root / "recordings" / "recordings.csv", rec_rows)
    generated = root / "recordings" / "recordings.csv"
    manifest.append({"path": "recordings/recordings.csv", "bytes": generated.stat().st_size,
                     "sha256": sha256(generated), "source": "generated (detection metadata)"})

    listed = {m["path"] for m in manifest} | {"MANIFEST.tsv", "README.md", "package.json"}
    stale = [p for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in listed]
    for p in stale:
        if args.prune:
            p.unlink()
    write_table(root / "MANIFEST.tsv", manifest)
    info = {
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": {"recordings": str(DATA_ROOT), "chirp_data": str(CHIRP_DATA), "inputs": str(INPUTS)},
        "runs": {t: {"detections": name, "run": run} for t, (name, run) in RUNS.items()},
        "excluded": list(EXCLUDED) + ["raw recordings (traces-grid1.raw)", "videos (*.avi)", "spectrogram caches"],
        "code": {
            "analysis": git_commit(REPO),
            "chirpdetector": git_commit(REPO / "packages" / "chirpdetector"),
            "gridtools": git_commit(REPO / "packages" / "gridtools"),
        },
        "counts": {"recordings": len(recs), "files": len(manifest), "bytes": sum(m["bytes"] for m in manifest),
                   "by_kind": sizes},
    }
    (root / "package.json").write_text(json.dumps(info, indent=1) + "\n")
    write_readme(root, info, len(manifest))
    print(f"{root}: {len(manifest)} files ({copied} copied, {len(manifest) - copied} unchanged), "
          f"{len(stale)} stale{' (deleted)' if args.prune and stale else ''}")


if __name__ == "__main__":
    main()
