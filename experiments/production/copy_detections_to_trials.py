"""Copy the v2 detections of a production run into the recording directories.

The run directory stays the provenance archive; each recording directory of
the dataset gets a copy of the files the analysis reads:

    <run>/<rec>/chirps_rtdetr.parquet   -> <dest>/<rec>/chirps_rtdetr.parquet
    <run>/<rec>/chirp_times_rtdetr.npy  -> <dest>/<rec>/chirp_times_rtdetr.npy
    <run>/<rec>/chirp_ids_rtdetr.npy    -> <dest>/<rec>/chirp_ids_rtdetr.npy
    <run>/<rec>/meta.json               -> <dest>/<rec>/chirps_rtdetr_meta.json

(``meta.json`` is renamed so it cannot be confused with the recording's
``meta.npy`` or a later ``meta.json``.) ``qc/`` and ``windows/`` stay in the
run directory.

``--run`` names the source run; ``--suffix`` the suffix of its file names
(default none). The gap-filled patch of the production run
(``gapfill_reassign.py``) is copied with ``--run
$CHIRP_DATA/detections/competition2019_v2_20261113_gapfill --suffix
_gapfill``:

    <run>/<rec>/chirps_rtdetr_gapfill.parquet  -> <dest>/<rec>/chirps_rtdetr_gapfill.parquet
    <run>/<rec>/chirp_times_rtdetr_gapfill.npy -> <dest>/<rec>/chirp_times_rtdetr_gapfill.npy
    <run>/<rec>/chirp_ids_rtdetr_gapfill.npy   -> <dest>/<rec>/chirp_ids_rtdetr_gapfill.npy
    <run>/<rec>/meta.json                      -> <dest>/<rec>/chirps_rtdetr_gapfill_meta.json

Idempotent and conservative, because the recording directories hold raw data:

- a destination file that already exists with the same sha256 is skipped;
- a destination file that exists with different content is never touched,
  the script reports it and exits non-zero;
- files are created with exclusive create (``open(..., "xb")``), so nothing
  is ever overwritten, moved or deleted in a recording directory;
- recording directories are never created; a missing one is an error;
- before writing, the free space of the destination must exceed the bytes
  still to copy plus 1 GiB, otherwise the script stops before the first write.

After copying, every destination is re-hashed against its source, and the
counts are checked against ``meta.json``: the legacy arrays hold
``legacy.n`` rows each, the parquet holds ``counts.stored`` rows, and its
``chirp & track_id.notna()`` rows equal ``legacy.n``. A manifest
(recording, file, sha256, bytes, source, destination, action, copied_at;
the copy time of earlier runs is kept) is written to
``<run>/trial_copies_manifest.csv``.

Usage (repository root, workspace environment):

    uv run python experiments/production/copy_detections_to_trials.py \
        [--run $CHIRP_DATA/detections/competition2019_v2_20261113] \
        [--suffix SUFFIX] [--dest $CHIRP_REC_2019] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2019  # noqa: E402


def files(suffix: str = "") -> tuple[tuple[str, str], ...]:
    """(source, destination) file names of one recording."""
    return (
        (f"chirps_rtdetr{suffix}.parquet", f"chirps_rtdetr{suffix}.parquet"),
        (f"chirp_times_rtdetr{suffix}.npy", f"chirp_times_rtdetr{suffix}.npy"),
        (f"chirp_ids_rtdetr{suffix}.npy", f"chirp_ids_rtdetr{suffix}.npy"),
        ("meta.json", f"chirps_rtdetr{suffix}_meta.json"),
    )


FILES = files()
MARGIN_BYTES = 1 << 30
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def copy_exclusive(src: Path, dst: Path) -> None:
    with src.open("rb") as fi, dst.open("xb") as fo:
        shutil.copyfileobj(fi, fo, CHUNK)
        fo.flush()
    shutil.copystat(src, dst)


def check_counts(rec_dir: Path, meta: dict, suffix: str = "") -> list[str]:
    """Consistency of the copied files with meta.json; returns problems."""
    problems = []
    n_legacy = int(meta["legacy"]["n"])
    times = np.load(rec_dir / f"chirp_times_rtdetr{suffix}.npy")
    ids = np.load(rec_dir / f"chirp_ids_rtdetr{suffix}.npy")
    if not (len(times) == len(ids) == n_legacy):
        problems.append(f"legacy arrays {len(times)}/{len(ids)} != legacy.n {n_legacy}")
    table = pq.read_table(rec_dir / f"chirps_rtdetr{suffix}.parquet", columns=["chirp", "track_id"])
    if table.num_rows != int(meta["counts"]["stored"]):
        problems.append(f"parquet rows {table.num_rows} != counts.stored {meta['counts']['stored']}")
    df = table.to_pandas()
    n_keep = int((df["chirp"] & df["track_id"].notna()).sum())
    if n_keep != n_legacy:
        problems.append(f"parquet chirp & assigned {n_keep} != legacy.n {n_legacy}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--run", type=Path, default=DATA / "detections" / "competition2019_v2_20261113",
        help="source run directory",
    )
    ap.add_argument(
        "--suffix", default="",
        help="suffix of the run's file names, e.g. _gapfill (default: none)",
    )
    ap.add_argument("--dest", type=Path, default=REC_2019)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]*", args.suffix):
        print(f"ERROR: suffix {args.suffix!r}: letters, digits, '_' and '-' only")
        return 2
    names = files(args.suffix)

    recs = sorted(
        p.name
        for p in args.run.iterdir()
        if p.is_dir() and (p / "meta.json").is_file()
    )
    plan, conflicts, missing = [], [], []
    for rec in recs:
        meta = json.loads((args.run / rec / "meta.json").read_text())
        if meta.get("status") != "complete":
            conflicts.append(f"{rec}: run status {meta.get('status')!r}, not complete")
            continue
        dest_dir = args.dest / rec
        if not dest_dir.is_dir():
            missing.append(rec)
            continue
        for src_name, dst_name in names:
            src, dst = args.run / rec / src_name, dest_dir / dst_name
            digest = sha256(src)
            if dst.exists():
                if sha256(dst) == digest:
                    plan.append((rec, src, dst, digest, "present"))
                else:
                    conflicts.append(f"{dst} exists with different content")
            else:
                plan.append((rec, src, dst, digest, "copy"))

    for m in missing:
        print(f"ERROR: no recording directory {args.dest / m}")
    for c in conflicts:
        print(f"ERROR: {c}")
    if missing or conflicts:
        print("nothing written")
        return 1

    to_copy = [p for p in plan if p[4] == "copy"]
    need = sum(p[1].stat().st_size for p in to_copy)
    free = shutil.disk_usage(args.dest).free
    print(
        f"{len(recs)} recordings, {len(plan)} files: {len(to_copy)} to copy "
        f"({need / 1e6:.1f} MB), {len(plan) - len(to_copy)} already present; "
        f"{free / 1e9:.0f} GB free on {args.dest}"
    )
    if free < need + MARGIN_BYTES:
        print("ERROR: not enough free space; stopping before the first write")
        return 2
    if args.dry_run:
        return 0

    manifest = args.run / "trial_copies_manifest.csv"
    previous = {}
    if manifest.is_file():
        with manifest.open() as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        previous = {
            (r["destination"], r["sha256"]): r["copied_at"]
            for r in csv.DictReader(lines)
            if r.get("copied_at")
        }

    rows, failed = [], []
    now = datetime.now().isoformat(timespec="seconds")
    for rec, src, dst, digest, action in plan:
        if action == "copy":
            copy_exclusive(src, dst)
            action, copied_at = "copied", now
        else:
            # not in an earlier manifest: the inode change time is the copy time
            # (copystat keeps the source mtime; nothing else touches these files)
            ctime = datetime.fromtimestamp(dst.stat().st_ctime).isoformat(timespec="seconds")
            copied_at = previous.get((str(dst), digest), ctime)
        got = sha256(dst)
        if got != digest:
            failed.append(f"{dst}: sha256 {got} != source {digest}")
        rows.append(
            {
                "recording": rec,
                "file": dst.name,
                "sha256": got,
                "bytes": dst.stat().st_size,
                "source": str(src),
                "destination": str(dst),
                "action": action,
                "copied_at": copied_at,
            }
        )
    for rec in recs:
        meta = json.loads((args.run / rec / "meta.json").read_text())
        failed += [f"{rec}: {p}" for p in check_counts(args.dest / rec, meta, args.suffix)]

    with manifest.open("w", newline="") as f:
        f.write(f"# written {now} by {Path(__file__).name}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_copied = sum(r["action"] == "copied" for r in rows)
    print(f"copied {n_copied}, already present {len(rows) - n_copied}; manifest {manifest}")
    for msg in failed:
        print(f"ERROR: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
