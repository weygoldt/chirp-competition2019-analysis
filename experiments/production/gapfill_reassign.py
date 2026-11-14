"""Fill the gaps of the trial fish's tracks and redo post-processing and assignment on the stored detections.

A dataset patch for the 2019/2020 tube competition recordings, not a change
of chirpdetector: the package assigns a chirp only to a track that is live
at its onset (``pipeline/assign.py``: between the track's first and last
detection and not inside a gap longer than ``max_gap_s``), and stays that
way (no live track, no assignment). The wavetracker tracks of this dataset
have more gaps than expected (``track_gaps.py``: 1,328 interior gaps > 10 s
in 56 of the 66 fish of the 33 analysed trials), so here, and only here,
the two trial fish get their gaps filled (author's decision, 2026-11-14).

Patch rule, per trial (``trial_summary.csv``) and per trial fish (``win_ID``,
``lose_ID``) only:

- every interior gap (consecutive detections more than ``max_gap_s`` = 10 s
  apart) with duration <= ``MAX_FILL_S`` (300 s) whose EODf jump is at most
  ``MAX_JUMP_HZ`` (15 Hz) is filled by linear interpolation between the
  median EODf of the fish's detections in the 1 s before the gap and in the
  1 s after it (``EDGE_S``; the edges of ``track_gaps.py``), with one sample
  on every frame of the tracker's own time grid (``times.npy``) strictly
  inside the gap;
- never before the first or after the last detection; longer gaps and gaps
  with a larger jump stay as they are (counted, ``action`` in the gap table);
- other tracks and the recordings that are not trials are not changed.

Then chirpdetector's own steps run on the stored detections of every
recording, exactly as `chirpdetector.pipeline.run.process_recording` runs
them after detection, with the run's configuration (``config.toml`` of the
source run; its resolved config and detection hash are checked against
``run.json``): `assign_detections`, `_postprocess` (harmonic copies,
duplicates, re-attribution at the operating threshold), `reattributed_assignment`,
`build_table`, `legacy_arrays`. Nothing is re-detected; the detector and
decoder columns are taken from the stored table. The detections are
rebuilt from ``chirps_rtdetr.parquet``: rows in ``chirp_id`` order (the
order of the detector output, which is sorted by centre time), the
detector and decoder columns, and for re-attributed rows the detected
``emitter_hz``, ``f_low_hz`` and ``f_high_hz`` from ``pp_harmonic_*``. The
decoder values of a re-attributed row are NaN in the stored table (the
package discards them), so a row that is re-attributed in the source run
but not after the patch would keep NaN decoder values; the script counts
such rows (``decoder_values_lost``).

Equivalence test first: with the unpatched tracks every recording must
reproduce the stored ``chirps_rtdetr.parquet`` (frame read back from
parquet, all columns and rows, exact), the legacy arrays (bytes), the
post-processing summary and the counts in ``meta.json``, and the track
files must hash to the run's ``tracks.sha256``. Only if all recordings
pass is anything written (``equivalence.csv`` is written either way).

Outputs, in OUT (default ``<source run>_gapfill``):

- ``<rec>/chirps_rtdetr_gapfill.parquet``: the table, same columns plus
  ``gapfill_changed``: the row differs from the source run's row, i.e. its
  assignment or post-processing outcome changed (any column; float columns
  by more than ``CHANGE_ATOL`` = 1e-6, because a track with more samples
  changes the baselines' cumulative sums, `track_baselines`, by floating-point
  rounding everywhere after a filled gap: <= 7.5e-9 Hz in this run);
- ``<rec>/chirp_{times,ids}_rtdetr_gapfill.npy``: legacy arrays (same
  selection as the source run's);
- ``<rec>/meta.json``: provenance, patch rule and parameters, gaps filled
  and left per fish, counts before and after, row transitions;
- ``<rec>/tracks_gapfill.npz`` (trials only): the trial fish's patched
  tracks in wavetracker layout (``times``, ``fund_v``, ``ident_v``,
  ``idx_v``, ``filled`` = interpolated sample) and the gap table;
- ``equivalence.csv``, ``gaps_gapfill.csv`` (every interior gap of a trial
  fish), ``transitions.csv`` (chirp state before -> after per recording),
  ``overview_trials.csv``, ``run.json`` and ``README.md``.

Usage (repository root of chirp_competition_upgrade, root environment; CPU
only, single-threaded, a few minutes):

    python experiments/production/gapfill_reassign.py [--source RUN] [--out DIR]
        [--data $CHIRP_REC_2019] [--check-only] [--overwrite]
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only: nothing here needs a GPU

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from chirpdetector.params.inference import ESTIMATE_COLUMNS
from chirpdetector.pipeline import run as pipeline
from chirpdetector.pipeline.assign import (
    ATTRIBUTED,
    SIM_V4_DUALHEAD,
    assign_detections,
)
from chirpdetector.pipeline.config import PipelineConfig, load_config
from chirpdetector.pipeline.outputs import (
    atomic_dir,
    legacy_arrays,
    write_json,
    write_parquet,
)
from chirpdetector.training.provenance import code_state, git_state, uv_lock, versions
from gridtools.io import Tracks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2019  # noqa: E402

SOURCE_RUN = DATA / "detections" / "competition2019_v2_20261113"
TAG = "rtdetr"
SUFFIX = "_gapfill"
MAX_FILL_S = 300.0  # longest gap that is filled
MAX_JUMP_HZ = 15.0  # largest EODf jump across a filled gap
EDGE_S = 1.0  # edge EODf: median over this long before / after the gap
ACTIVE_S = 3 * 3600  # "first 3 h": legacy time (box centre) before this
CHANGE_ATOL = 1e-6  # gapfill_changed: float differences up to this are rounding (Hz, s, probability)
LOCAL_PAD_S = 1.5  # a changed row's box should lie within this of a filled gap (baseline window 1 s)
BAD_RECORDINGS = ("2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00")

# columns of the detector output (`stream.detect_recording`), i.e. what the
# pipeline assigns and post-processes; everything else is computed here
DETECTION_COLUMNS = (
    "recording",
    "tile",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "score",
    "model_sha256",
    "emitter_hz",
    *ESTIMATE_COLUMNS,
    "span_id",
    "click_fraction_box",
)
DETECTED = {"emitter_hz": "pp_harmonic_emitter_hz", "f_low_hz": "pp_harmonic_f_low_hz", "f_high_hz": "pp_harmonic_f_high_hz"}
# columns no patch may change (detector output and derived from it)
FIXED_COLUMNS = (
    "recording", "chirp_id", "t_start_s", "t_end_s", "t_center_s", "score", "operating",
    "click_fraction_box", "span_click_fraction", "span_n_click_runs", "tile", "span_id", "model_sha256",
)  # fmt: skip


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def parquet_bytes(table: pd.DataFrame, metadata: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    write_parquet(buf, table, metadata)  # type: ignore[arg-type]  # pq.write_table takes a file object
    return buf.getvalue()


def read_parquet_bytes(data: bytes) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(data)).to_pandas()


def jsonable(value: Any) -> Any:
    """A JSON round trip (what meta.json holds)."""
    return json.loads(json.dumps(value, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def differing_rows(
    a: pd.DataFrame, b: pd.DataFrame, columns: list[str] | None = None, atol: float = 0.0
) -> np.ndarray:
    """Per row: does any of ``columns`` differ (NaN equals NaN, NA equals NA)?

    Float columns differ where ``|a - b| > atol`` (or only one is NaN);
    every other column must be equal.
    """
    out = np.zeros(len(a), dtype=bool)
    numeric = pd.api.types.is_numeric_dtype
    for c in columns if columns is not None else list(a.columns):
        x, y = a[c], b[c]
        if numeric(x.dtype) and numeric(y.dtype):  # float, int, Int64 (small ints: exact), bool
            xv = x.to_numpy(dtype=np.float64, na_value=np.nan)
            yv = y.to_numpy(dtype=np.float64, na_value=np.nan)
            tol = atol if pd.api.types.is_float_dtype(x.dtype) and pd.api.types.is_float_dtype(y.dtype) else 0.0
            with np.errstate(invalid="ignore"):
                close = np.abs(xv - yv) <= tol
            out |= ~(close | (np.isnan(xv) & np.isnan(yv)))
        else:
            xn, yn = x.isna().to_numpy(), y.isna().to_numpy()
            xs = x.astype(object).to_numpy()
            ys = y.astype(object).to_numpy()
            same = np.array([bool(u == v) for u, v in zip(xs[~xn & ~yn], ys[~xn & ~yn], strict=True)], dtype=bool)
            eq = xn & yn
            eq[~xn & ~yn] = same
            out |= ~eq
    return out


# ------------------------------------------------------------------ source


@dataclass
class Source:
    """A recording's outputs in the source run."""

    name: str
    dir: Path
    table: pd.DataFrame  # chirps_rtdetr.parquet, read back
    file_meta: dict[str, Any]  # the parquet's ``chirpdetector`` metadata
    numpy_types: dict[str, str]  # the parquet's pandas metadata: dtype of each column when written
    meta: dict[str, Any]
    chunks: pd.DataFrame
    parquet_sha256: str
    times_bytes: bytes
    ids_bytes: bytes

    @classmethod
    def load(cls, run_dir: Path, name: str) -> Source:
        d = run_dir / name
        path = d / f"chirps_{TAG}.parquet"
        arrow = pq.read_table(path)
        file_meta = json.loads((arrow.schema.metadata or {})[b"chirpdetector"])
        pandas_meta = json.loads((arrow.schema.metadata or {})[b"pandas"])
        return cls(
            name=name,
            dir=d,
            table=arrow.to_pandas(),
            file_meta=file_meta,
            numpy_types={c["name"]: c["numpy_type"] for c in pandas_meta["columns"]},
            meta=json.loads((d / "meta.json").read_text()),
            chunks=pd.read_parquet(d / f"chunks_{TAG}.parquet"),
            parquet_sha256=sha256_file(path),
            times_bytes=(d / f"chirp_times_{TAG}.npy").read_bytes(),
            ids_bytes=(d / f"chirp_ids_{TAG}.npy").read_bytes(),
        )

    def detections(self) -> pd.DataFrame:
        """The detector output the pipeline assigned and post-processed."""
        t = self.table
        if not np.array_equal(t["chirp_id"].to_numpy(), np.arange(len(t))):
            msg = f"{self.name}: chirp_id is not the row number"
            raise ValueError(msg)
        det = t.loc[:, list(DETECTION_COLUMNS)].copy()
        # object columns (e.g. the decoder's params_flag) come back as str
        for column in DETECTION_COLUMNS:
            if self.numpy_types[column] == "object":
                det[column] = det[column].astype(object)
        again = t["pp_reattributed"].to_numpy(dtype=object) == "harmonic"
        for column, saved in DETECTED.items():
            det.loc[again, column] = t.loc[again, saved].to_numpy(dtype=np.float64)
        return det.reset_index(drop=True)


def reprocess(
    det: pd.DataFrame, tracks: Tracks, config: PipelineConfig, name: str, chunks: pd.DataFrame, n_owned: int
) -> dict[str, Any]:
    """`process_recording` after detection, on stored detections."""
    operating = config.detection.operating_threshold
    model = SIM_V4_DUALHEAD
    assignment = assign_detections(det, tracks, config.assignment, model)
    dets, pp_info = pipeline._postprocess(det, assignment, tracks, name, config, operating)
    assignment = pipeline.reattributed_assignment(dets, assignment, tracks, config, model)
    table = pipeline.build_table(name, dets, assignment, chunks, operating)
    times, ids = legacy_arrays(table)
    counts = pipeline._counts(table, SimpleNamespace(n_owned=n_owned))
    return {"table": table, "times": times, "ids": ids, "postprocess": pp_info, "counts": counts}


# ------------------------------------------------------------------- tracks


@dataclass
class Gap:
    """An interior gap of a trial fish's track."""

    fish_id: float
    role: str
    start_s: float  # last detection before the gap
    end_s: float  # first detection after it
    f_before_hz: float
    f_after_hz: float
    action: str  # filled | left_long | left_jump | left_long_jump
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    freqs_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def jump_hz(self) -> float:
        return self.f_after_hz - self.f_before_hz

    def row(self) -> dict[str, Any]:
        return {
            "fish": self.role, "fish_id": self.fish_id, "start_s": self.start_s, "end_s": self.end_s,
            "duration_s": self.duration_s, "f_before_hz": self.f_before_hz, "f_after_hz": self.f_after_hz,
            "f_jump_hz": self.jump_hz, "action": self.action, "n_samples": len(self.frames),
        }  # fmt: skip


def interior_gaps(tracks: Tracks, fish_id: float, role: str, max_gap_s: float) -> list[Gap]:
    """Interior gaps > ``max_gap_s`` of one fish, with the patch decision."""
    sel = tracks.idents == fish_id
    if not sel.any():
        msg = f"no detections of fish {fish_id}"
        raise ValueError(msg)
    t = tracks.times_s[tracks.time_indices[sel]]
    f = tracks.freqs_hz[sel]
    order = np.argsort(t, kind="stable")
    t, f = t[order], f[order]
    gaps = []
    for i in np.flatnonzero(np.diff(t) > max_gap_s):
        a, b = float(t[i]), float(t[i + 1])
        fa = float(np.median(f[(t >= a - EDGE_S) & (t <= a)]))
        fb = float(np.median(f[(t >= b) & (t <= b + EDGE_S)]))
        reasons = [r for r, bad in (("long", b - a > MAX_FILL_S), ("jump", abs(fb - fa) > MAX_JUMP_HZ)) if bad]
        gap = Gap(fish_id, role, a, b, fa, fb, "filled" if not reasons else "left_" + "_".join(reasons))
        if gap.action == "filled":
            gap.frames = np.flatnonzero((tracks.times_s > a) & (tracks.times_s < b)).astype(np.int64)
            ts = tracks.times_s[gap.frames]
            gap.freqs_hz = fa + (fb - fa) * (ts - a) / (b - a)
        gaps.append(gap)
    return gaps


def with_fills(tracks: Tracks, gaps: list[Gap]) -> Tracks:
    """The tracks plus the samples of the filled gaps (no powers: not used)."""
    filled = [g for g in gaps if g.action == "filled"]
    return Tracks(
        times_s=tracks.times_s,
        freqs_hz=np.concatenate([tracks.freqs_hz, *(g.freqs_hz for g in filled)]),
        idents=np.concatenate([tracks.idents, *(np.full(len(g.frames), g.fish_id) for g in filled)]),
        time_indices=np.concatenate([tracks.time_indices, *(g.frames for g in filled)]),
        powers_db=None,
    )


def check_patch(base: Tracks, patched: Tracks, gaps: list[Gap], max_gap_s: float) -> None:
    """The patch changes the trial fish inside filled gaps only."""
    for ident in base.ids:
        old, new = base.track(int(ident)), patched.track(int(ident))
        if not np.array_equal(new.frequency_at(old.times_s, max_gap_s), old.freqs_hz):
            msg = f"track {ident}: patched track differs at its own detections"
            raise AssertionError(msg)
        if (old.start_s, old.stop_s) != (new.start_s, new.stop_s):
            msg = f"track {ident}: patched track starts or ends elsewhere"
            raise AssertionError(msg)
        mine = [g for g in gaps if g.fish_id == ident]
        n_fill = sum(len(g.frames) for g in mine if g.action == "filled")
        if len(new.times_s) != len(old.times_s) + n_fill:
            msg = f"track {ident}: {len(new.times_s)} samples, expected {len(old.times_s)} + {n_fill}"
            raise AssertionError(msg)
        for g in mine:
            probe = np.linspace(g.start_s, g.end_s, 101)
            live = np.isfinite(new.frequency_at(probe, max_gap_s))
            if (g.action == "filled") != bool(live.all()) or (g.action != "filled" and live[1:-1].any()):
                msg = f"track {ident}: gap {g.start_s:.2f}-{g.end_s:.2f} s ({g.action}) has the wrong liveness"
                raise AssertionError(msg)


def tracks_npz(path: Path, base: Tracks, gaps: list[Gap], fish: dict[str, float]) -> None:
    """The trial fish's patched tracks in wavetracker layout, and the gap table."""
    filled = [g for g in gaps if g.action == "filled"]
    sel = np.isin(base.idents, list(fish.values()))
    idx = np.concatenate([base.time_indices[sel], *(g.frames for g in filled)])
    f = np.concatenate([base.freqs_hz[sel], *(g.freqs_hz for g in filled)])
    ident = np.concatenate([base.idents[sel], *(np.full(len(g.frames), g.fish_id) for g in filled)])
    is_fill = np.concatenate([np.zeros(int(sel.sum()), bool), *(np.ones(len(g.frames), bool) for g in filled)])
    order = np.lexsort((idx, ident))
    g = pd.DataFrame([x.row() for x in gaps]) if gaps else pd.DataFrame(columns=list(Gap(0, "", 0, 0, 0, 0, "").row()))
    np.savez_compressed(
        path,
        times=np.asarray(base.times_s),
        fund_v=f[order],
        ident_v=ident[order],
        idx_v=idx[order].astype(np.int64),
        filled=is_fill[order],
        gap_fish_id=g["fish_id"].to_numpy(dtype=np.float64),
        gap_start_s=g["start_s"].to_numpy(dtype=np.float64),
        gap_end_s=g["end_s"].to_numpy(dtype=np.float64),
        gap_f_before_hz=g["f_before_hz"].to_numpy(dtype=np.float64),
        gap_f_after_hz=g["f_after_hz"].to_numpy(dtype=np.float64),
        gap_action=g["action"].to_numpy(dtype="U16"),
        fish_role=np.array(list(fish), dtype="U8"),
        fish_id=np.array(list(fish.values()), dtype=np.float64),
    )


# ----------------------------------------------------------------- changes


def chirp_states(table: pd.DataFrame, fish: dict[str, float]) -> np.ndarray:
    """State of every row at the operating threshold (others: '')."""
    role = {v: k for k, v in fish.items()}
    op = table["operating"].to_numpy(dtype=bool)
    kept = table["kept"].to_numpy(dtype=bool)
    reason = table["pp_drop_reason"].astype(str).to_numpy()
    attributed = np.isin(table["assign_status"].to_numpy(dtype=object), ATTRIBUTED)
    tid = table["track_id"].to_numpy(dtype=np.float64)
    out = np.full(len(table), "", dtype=object)
    for i in np.flatnonzero(op):
        if not kept[i]:
            out[i] = f"flagged_{reason[i]}"
        elif attributed[i]:
            out[i] = role.get(tid[i], f"track_{tid[i]:g}")
        else:
            out[i] = "unassigned"
    return out


def transitions(before: pd.DataFrame, after: pd.DataFrame, fish: dict[str, float]) -> pd.DataFrame:
    """Rows at the operating threshold by state before and after."""
    a, b = chirp_states(before, fish), chirp_states(after, fish)
    early = before["t_center_s"].to_numpy() < ACTIVE_S
    df = pd.DataFrame({"before": a, "after": b, "first_3h": early})
    df = df[df["before"] != ""]
    out = df.groupby(["before", "after"]).agg(n=("first_3h", "size"), n_first_3h=("first_3h", "sum")).reset_index()
    return out.astype({"n": np.int64, "n_first_3h": np.int64})


def per_fish(table: pd.DataFrame, fish_id: float, early: bool) -> int:
    sel = table["chirp"].to_numpy(dtype=bool) & np.isin(table["assign_status"].to_numpy(dtype=object), ATTRIBUTED)
    sel &= table["track_id"].to_numpy(dtype=np.float64) == fish_id
    if early:
        sel &= table["t_center_s"].to_numpy() < ACTIVE_S
    return int(sel.sum())


def unassigned(table: pd.DataFrame, early: bool) -> int:
    sel = table["chirp"].to_numpy(dtype=bool) & ~np.isin(table["assign_status"].to_numpy(dtype=object), ATTRIBUTED)
    if early:
        sel &= table["t_center_s"].to_numpy() < ACTIVE_S
    return int(sel.sum())


# ------------------------------------------------------------- equivalence


def equivalence(source: Source, config: PipelineConfig, data: Path) -> tuple[dict[str, Any], Tracks, dict[str, Any]]:
    """Reproduce the source run's outputs with the unpatched tracks."""
    name = source.name
    tracks, info = pipeline._load_tracks(pipeline.tracks_dir(data / name, config.assignment.tracks))
    if tracks is None:
        msg = f"{name}: no tracks ({info.get('status')})"
        raise RuntimeError(msg)
    base = with_fills(tracks, [])
    res = reprocess(source.detections(), base, config, name, source.chunks, source.meta["counts"]["owned_before_merge"])
    data_bytes = parquet_bytes(res["table"], source.file_meta)
    again = read_parquet_bytes(data_bytes)
    same_columns = list(again.columns) == list(source.table.columns)
    same_dtypes = same_columns and list(again.dtypes.astype(str)) == list(source.table.dtypes.astype(str))
    cells = (
        int(differing_rows(again, source.table).sum()) if same_columns and len(again) == len(source.table) else -1
    )
    try:
        pd.testing.assert_frame_equal(again, source.table, check_exact=True)
        frame_equal = True
    except AssertionError:
        frame_equal = False
    row = {
        "recording": name,
        "rows": len(source.table),
        "tracks_sha256_equal": info["sha256"] == source.meta["tracks"]["sha256"],
        "columns_equal": same_columns,
        "dtypes_equal": same_dtypes,
        "rows_differing": cells,
        "frame_equal": frame_equal,
        "parquet_bytes_equal": sha256_bytes(data_bytes) == source.parquet_sha256,
        "times_bytes_equal": npy_bytes(res["times"]) == source.times_bytes,
        "ids_bytes_equal": npy_bytes(res["ids"]) == source.ids_bytes,
        "postprocess_meta_equal": jsonable(res["postprocess"]) == source.meta["postprocess"],
        "counts_meta_equal": jsonable(res["counts"]) == source.meta["counts"],
        "legacy_n_equal": len(res["times"]) == source.meta["legacy"]["n"],
    }
    row["pass"] = all(
        row[k] for k in row if k.endswith("_equal") and k != "parquet_bytes_equal"
    ) and row["rows_differing"] == 0
    return row, tracks, info


# --------------------------------------------------------------------- main


def load_trials(data: Path) -> pd.DataFrame:
    ts = pd.read_csv(data / "trial_summary.csv", index_col=0)
    ts["analysed"] = (ts["draw"] == 0) & ~ts["recording"].isin(BAD_RECORDINGS)
    return ts.set_index("recording")


def check_config(config: PipelineConfig, run: dict[str, Any]) -> None:
    if config.to_dict() != run["config"]:
        msg = "config.toml does not resolve to run.json's config"
        raise RuntimeError(msg)
    if config.detection_hash() != run["detection_hash"]:
        msg = "detection hash differs from run.json's"
        raise RuntimeError(msg)


def process(
    source: Source, tracks: Tracks, track_info: dict[str, Any], config: PipelineConfig,
    trial: pd.Series | None, out_dir: Path, provenance: dict[str, Any], equiv: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """Patch, reprocess and write one recording; returns its summary."""
    name = source.name
    max_gap = float(config.assignment.max_gap_s)
    base = with_fills(tracks, [])
    fish: dict[str, float] = {}
    gaps: list[Gap] = []
    if trial is not None:
        fish = {"winner": float(trial["win_ID"]), "loser": float(trial["lose_ID"])}
        for role, ident in fish.items():
            gaps += interior_gaps(base, ident, role, max_gap)
    patched = with_fills(base, gaps)
    check_patch(base, patched, gaps, max_gap)
    res = reprocess(
        source.detections(), patched, config, name, source.chunks, source.meta["counts"]["owned_before_merge"]
    )
    table = res["table"]
    if list(table.columns) != list(source.table.columns) or len(table) != len(source.table):
        msg = f"{name}: table shape differs from the source"
        raise AssertionError(msg)
    if differing_rows(table, source.table, list(FIXED_COLUMNS)).any():
        msg = f"{name}: detector columns changed"
        raise AssertionError(msg)
    exact = differing_rows(table, source.table)
    changed = differing_rows(table, source.table, atol=CHANGE_ATOL)
    if not any(g.action == "filled" for g in gaps) and exact.any():
        msg = f"{name}: rows changed without a patch"
        raise AssertionError(msg)
    # changed rows whose box lies further than LOCAL_PAD_S from every filled gap
    t0, t1 = table["t_start_s"].to_numpy(), table["t_end_s"].to_numpy()
    local = np.zeros(len(table), dtype=bool)
    for g in gaps:
        if g.action == "filled":
            local |= (t1 > g.start_s - LOCAL_PAD_S) & (t0 <= g.end_s + LOCAL_PAD_S)
    # re-attributed in the source, not any more: the decoder values were discarded
    was = source.table["pp_reattributed"].to_numpy(dtype=object) == "harmonic"
    now = table["pp_reattributed"].to_numpy(dtype=object) == "harmonic"
    lost = was & ~now
    table = table.assign(gapfill_changed=changed)

    trans = transitions(source.table, table, fish)
    trans.insert(0, "recording", name)
    op = table["operating"].to_numpy(dtype=bool)
    reason_changed = source.table["pp_drop_reason"].astype(str).to_numpy() != table["pp_drop_reason"].astype(str).to_numpy()
    decision = differing_rows(
        table, source.table, ["track_id", "assign_status", "pp_drop_reason", "pp_reattributed", "chirp"]
    )
    fill_s = {
        str(v): float(sum(g.duration_s for g in gaps if g.fish_id == v and g.action == "filled")) for v in fish.values()
    }
    fish_summary = {}
    for role, ident in fish.items():
        mine = [g for g in gaps if g.fish_id == ident]
        actions = pd.Series([g.action for g in mine], dtype=object).value_counts().to_dict()
        fish_summary[str(ident)] = {
            "role": role,
            "interior_gaps": len(mine),
            "filled": int(actions.get("filled", 0)),
            "left_long": int(actions.get("left_long", 0)),
            "left_jump": int(actions.get("left_jump", 0)),
            "left_long_jump": int(actions.get("left_long_jump", 0)),
            "filled_s": fill_s[str(ident)],
            "left_s": float(sum(g.duration_s for g in mine if g.action != "filled")),
            "fill_samples": int(sum(len(g.frames) for g in mine)),
            "gaps": [{k: v for k, v in g.row().items() if k not in ("fish", "fish_id")} for g in mine],
        }
    changes = {
        "change_atol": CHANGE_ATOL,
        "rows_changed": int(changed.sum()),
        "rows_changed_operating": int((changed & op).sum()),
        "rows_differing_exact": int(exact.sum()),
        "rows_rounding_only": int((exact & ~changed).sum()),
        "rows_changed_far_from_gaps": int((changed & ~local).sum()),
        "decision_changed": int(decision.sum()),
        "decision_changed_operating": int((decision & op).sum()),
        "value_only_changed": int((changed & ~decision).sum()),
        "pp_flag_changed": int(reason_changed.sum()),
        "pp_flag_changed_operating": int((reason_changed & op).sum()),
        "reattributed_before": int(was.sum()),
        "reattributed_after": int(now.sum()),
        "decoder_values_lost": int(lost.sum()),
        "transitions": trans.drop(columns="recording").to_dict(orient="records"),
    }

    file_meta = dict(source.file_meta)
    file_meta |= {
        "run": out_dir.parent.name,
        "source_run": source.meta["run"],
        "patch": "trial-fish track gaps filled (gapfill_reassign.py); assignment and post-processing redone",
    }
    patch_meta = {
        "applied": bool(any(g.action == "filled" for g in gaps)),
        "trial": trial is not None,
        "rule": (
            "trial fish only (win_ID, lose_ID): every interior gap (consecutive detections > max_gap_s apart) "
            "with duration <= max_fill_s and |EODf jump| <= max_jump_hz is filled by linear interpolation "
            "between the median EODf of the fish's detections in edge_s before and after the gap, one sample "
            "per frame of times.npy strictly inside the gap; no extrapolation before the first or after the "
            "last detection; other tracks unchanged"
        ),
        "parameters": {
            "max_gap_s": max_gap,
            "max_fill_s": MAX_FILL_S,
            "max_jump_hz": MAX_JUMP_HZ,
            "edge_s": EDGE_S,
        },
        "fish": fish_summary,
    }
    if trial is not None:
        patch_meta["trial_summary"] = {
            "win_ID": fish["winner"], "lose_ID": fish["loser"], "draw": int(trial["draw"]),
            "analysed": bool(trial["analysed"]),
        }  # fmt: skip
    with atomic_dir(out_dir) as tmp:
        files = {
            "chirps": tmp / f"chirps_{TAG}{SUFFIX}.parquet",
            "times": tmp / f"chirp_times_{TAG}{SUFFIX}.npy",
            "ids": tmp / f"chirp_ids_{TAG}{SUFFIX}.npy",
        }
        write_parquet(files["chirps"], table, file_meta)
        np.save(files["times"], res["times"])
        np.save(files["ids"], res["ids"])
        if trial is not None:
            files["tracks"] = tmp / "tracks_gapfill.npz"
            tracks_npz(files["tracks"], base, gaps, fish)
        meta = {
            "status": "complete",
            "recording": name,
            "run": out_dir.parent.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "kind": "dataset patch of the source run: gap-filled trial-fish tracks; chirpdetector's "
            "post-processing and assignment redone on the stored detections (no re-detection)",
            "source": {
                "run": source.meta["run"],
                "dir": str(source.dir),
                "detection_hash": source.meta["detection_hash"],
                "meta_sha256": sha256_file(source.dir / "meta.json"),
                "parquet_sha256": source.parquet_sha256,
                "times_sha256": sha256_bytes(source.times_bytes),
                "ids_sha256": sha256_bytes(source.ids_bytes),
                "code": source.meta["code"],
            },
            "equivalence": equiv,
            "patch": patch_meta,
            "inputs": source.meta["inputs"],
            "tracks": {
                "source": track_info,
                "patched": {
                    "file": "tracks_gapfill.npz" if trial is not None else None,
                    "added_samples": int(sum(len(g.frames) for g in gaps)),
                    "n_detections": len(patched.freqs_hz),
                },
            },
            "models": source.meta["models"],
            "config": config.to_dict(),
            "postprocess": res["postprocess"],
            "assignment": source.meta["assignment"],
            "operating_threshold": config.detection.operating_threshold,
            "counts": res["counts"],
            "counts_source": source.meta["counts"],
            "changes": changes,
            "code": provenance,
            "outputs": {},
            "legacy": {
                **source.meta["legacy"],
                "times": files["times"].name,
                "ids": files["ids"].name,
                "n": len(res["times"]),
            },
        }
        meta["outputs"] = {
            k: {"file": v.name, "bytes": v.stat().st_size, "sha256": sha256_file(v)} for k, v in files.items()
        }
        write_json(tmp / "meta.json", meta)

    summary: dict[str, Any] = {"recording": name, "trial": trial is not None, "gaps": [], "transitions": trans}
    for g in gaps:
        summary["gaps"].append({"recording": name, "analysed": bool(trial["analysed"]), **g.row()})
    if trial is not None:
        row: dict[str, Any] = {
            "recording": name, "win_ID": fish["winner"], "lose_ID": fish["loser"], "draw": int(trial["draw"]),
            "analysed": bool(trial["analysed"]),
        }  # fmt: skip
        for role, ident in fish.items():
            s = fish_summary[str(ident)]
            short = "win" if role == "winner" else "lose"
            row |= {
                f"{short}_gaps": s["interior_gaps"],
                f"{short}_gaps_filled": s["filled"],
                f"{short}_gaps_left": s["interior_gaps"] - s["filled"],
                f"{short}_filled_s": s["filled_s"],
            }
        for role, ident in fish.items():
            short = "win" if role == "winner" else "lose"
            for early, tag in ((False, ""), (True, "_3h")):
                row[f"chirps_{short}_before{tag}"] = per_fish(source.table, ident, early)
                row[f"chirps_{short}_after{tag}"] = per_fish(table, ident, early)
        for early, tag in ((False, ""), (True, "_3h")):
            row[f"unassigned_before{tag}"] = unassigned(source.table, early)
            row[f"unassigned_after{tag}"] = unassigned(table, early)

        def moved(a: str, b: str, early: bool) -> int:
            sel = (trans["before"] == a) & (trans["after"] == b)
            return int(trans.loc[sel, "n_first_3h" if early else "n"].sum())

        for early, tag in ((False, ""), (True, "_3h")):
            row[f"moved_win_to_lose{tag}"] = moved("winner", "loser", early)
            row[f"moved_lose_to_win{tag}"] = moved("loser", "winner", early)
            row[f"unassigned_to_win{tag}"] = moved("unassigned", "winner", early)
            row[f"unassigned_to_lose{tag}"] = moved("unassigned", "loser", early)
            flagged_b = trans["before"].str.startswith("flagged_")
            flagged_a = trans["after"].str.startswith("flagged_")
            col = "n_first_3h" if early else "n"
            row[f"newly_flagged{tag}"] = int(trans.loc[~flagged_b & flagged_a, col].sum())
            row[f"unflagged{tag}"] = int(trans.loc[flagged_b & ~flagged_a, col].sum())
        row["rows_changed"] = changes["rows_changed"]
        row["decoder_values_lost"] = changes["decoder_values_lost"]
        summary["overview"] = row
    summary["changes"] = {k: v for k, v in changes.items() if k != "transitions"}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, default=SOURCE_RUN, help="source run directory")
    ap.add_argument("--out", type=Path, default=None, help="output run directory (default: <source>_gapfill)")
    ap.add_argument("--data", type=Path, default=REC_2019, help="recording directories (tracks, trial_summary.csv)")
    ap.add_argument("--recordings", nargs="*", default=None, help="only these (testing; no run-level outputs)")
    ap.add_argument("--check-only", action="store_true", help="run the equivalence test only")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs in --out")
    args = ap.parse_args()
    out = args.out or args.source.with_name(args.source.name + SUFFIX)
    if out.resolve() == args.source.resolve():
        print("ERROR: --out must not be the source run")
        return 2

    run = json.loads((args.source / "run.json").read_text())
    config = load_config(args.source / "config.toml")
    check_config(config, run)
    if SIM_V4_DUALHEAD.to_dict() != json.loads((args.source / sorted(run["recordings"])[0] / "meta.json").read_text())["assignment"]["error_model"]:
        print("ERROR: error model differs from the run's")
        return 2
    trials = load_trials(args.data)
    recs = sorted(run["recordings"]) if args.recordings is None else list(args.recordings)
    full = args.recordings is None
    if not args.check_only:
        existing = [r for r in recs if (out / r).exists()]
        if existing and not args.overwrite:
            print(f"ERROR: {len(existing)} recordings already in {out} (--overwrite replaces them)")
            return 2
    out.mkdir(parents=True, exist_ok=True)

    # 1. equivalence test on every recording, with the unpatched tracks
    t0 = time.perf_counter()
    rows = []
    for i, rec in enumerate(recs, 1):
        row, _, _ = equivalence(Source.load(args.source, rec), config, args.data)
        rows.append(row)
        print(f"[equivalence {i}/{len(recs)}] {rec}: {'pass' if row['pass'] else 'FAIL'}"
              f" (parquet bytes {'identical' if row['parquet_bytes_equal'] else 'differ'})", flush=True)
    eq = pd.DataFrame(rows)
    eq_name = "equivalence.csv" if full else "equivalence_subset.csv"
    eq.to_csv(out / eq_name, index=False)
    n_pass = int(eq["pass"].sum())
    print(f"equivalence: {n_pass}/{len(eq)} recordings reproduced exactly "
          f"({int(eq['parquet_bytes_equal'].sum())} byte-identical parquet files) "
          f"in {time.perf_counter() - t0:.0f} s; {out / eq_name}")
    if n_pass != len(eq):
        print("ERROR: equivalence test failed; nothing patched")
        print(eq.loc[~eq["pass"]].to_string())
        return 1
    if args.check_only:
        return 0

    # 2. the patched run
    script = Path(__file__).resolve()
    provenance = {
        "script": {"path": str(script), "sha256": sha256_file(script), "git": git_state(script.parent)},
        "git": code_state(),
        "versions": versions(),
        "uv_lock": uv_lock(),
    }
    summaries = []
    for i, rec in enumerate(recs, 1):
        source = Source.load(args.source, rec)
        tracks, info = pipeline._load_tracks(pipeline.tracks_dir(args.data / rec, config.assignment.tracks))
        if tracks is None or info["sha256"] != source.meta["tracks"]["sha256"]:
            print(f"ERROR: {rec}: tracks differ from the source run's")
            return 1
        equiv = {k: (v.item() if hasattr(v, "item") else v) for k, v in eq.set_index("recording").loc[rec].items()}
        trial = trials.loc[rec] if rec in trials.index else None
        s = process(source, tracks, info, config, trial, out / rec, provenance, equiv)
        summaries.append(s)
        c = s["changes"]
        print(f"[patch {i}/{len(recs)}] {rec}: {'trial' if s['trial'] else 'not a trial'}, "
              f"{sum(g['action'] == 'filled' for g in s['gaps'])} gaps filled, "
              f"{c['rows_changed']} rows changed ({c['decision_changed_operating']} decisions at the operating threshold)",
              flush=True)
    if not full:
        return 0

    gaps = pd.DataFrame([g for s in summaries for g in s["gaps"]])
    gaps.to_csv(out / "gaps_gapfill.csv", index=False)
    trans = pd.concat([s["transitions"] for s in summaries], ignore_index=True)
    trans.to_csv(out / "transitions.csv", index=False)
    overview = pd.DataFrame([s["overview"] for s in summaries if "overview" in s])
    overview.to_csv(out / "overview_trials.csv", index=False)
    run_meta = {
        "run": out.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "source": {"run": run["run"], "dir": str(args.source), "detection_hash": run["detection_hash"]},
        "data": str(args.data),
        "parameters": {
            "max_gap_s": float(config.assignment.max_gap_s),
            "max_fill_s": MAX_FILL_S,
            "max_jump_hz": MAX_JUMP_HZ,
            "edge_s": EDGE_S,
            "first_hours_s": ACTIVE_S,
        },
        "bad_recordings": list(BAD_RECORDINGS),
        "equivalence": {"file": "equivalence.csv", "recordings": len(eq), "passed": n_pass,
                        "parquet_bytes_identical": int(eq["parquet_bytes_equal"].sum())},
        "recordings": {s["recording"]: {"trial": s["trial"], **s["changes"]} for s in summaries},
        "code": provenance,
    }
    write_json(out / "run.json", run_meta)
    (out / "README.md").write_text(readme(out, args.source, eq, gaps, overview, trans, summaries, float(config.assignment.max_gap_s)))
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------- README


def readme(
    out: Path, source: Path, eq: pd.DataFrame, gaps: pd.DataFrame, overview: pd.DataFrame,
    trans: pd.DataFrame, summaries: list[dict[str, Any]], max_gap_s: float,
) -> str:  # fmt: skip
    an = overview[overview["analysed"]]
    an_recs = set(an["recording"])
    g_an = gaps[gaps["analysed"]]
    g_counts = g_an["action"].value_counts()
    t_an = trans[trans["recording"].isin(an_recs)]
    lost = sum(s["changes"]["decoder_values_lost"] for s in summaries)
    rounding = sum(s["changes"]["rows_rounding_only"] for s in summaries)
    far = sum(s["changes"]["rows_changed_far_from_gaps"] for s in summaries)

    def n(a: str, b: str, col: str = "n") -> int:
        return int(t_an.loc[(t_an["before"] == a) & (t_an["after"] == b), col].sum())

    def fmt(x: float) -> str:
        return f"{int(x):,}"

    lines = [
        "# Gap-filled trial-fish tracks: chirp assignment redone (dataset patch)",
        "",
        f"Script: `experiments/production/gapfill_reassign.py` (chirp_competition_upgrade). Source run: `{source}`.",
        "Author's decision (2026-11-14): chirpdetector stays as it is (no live track, no assignment);",
        "only this dataset is patched, because its wavetracker tracks have more gaps than expected",
        "(`../competition2019_v2_20261113/track_gaps/README.md`).",
        "",
        "## Rule",
        "",
        f"For the two trial fish of each of the {len(overview)} trials (`trial_summary.csv`: `win_ID`, `lose_ID`):",
        f"every interior gap (consecutive detections > {max_gap_s:g} s apart, the assignment's `max_gap_s`) of",
        f"at most {MAX_FILL_S:g} s whose EODf jump (median EODf over {EDGE_S:g} s before vs. after the gap)",
        f"is at most {MAX_JUMP_HZ:g} Hz is filled by linear interpolation between those two medians, one",
        "sample per frame of `times.npy` inside the gap. No extrapolation before the first or after the",
        f"last detection; other tracks and the {sum(not x['trial'] for x in summaries)} recordings that are not "
        "trials are unchanged.",
        "chirpdetector's post-processing (harmonic copies, duplicates, re-attribution) and assignment",
        "then ran on the stored detections with the source run's configuration (no re-detection).",
        "",
        "## Equivalence test",
        "",
        f"With the unpatched tracks the script reproduced {int(eq['pass'].sum())} of {len(eq)} recordings exactly",
        "(parquet table read back, all rows and columns; legacy arrays byte for byte; post-processing",
        f"summary and counts of `meta.json`; track files hash to the run's): `equivalence.csv`. "
        f"{int(eq['parquet_bytes_equal'].sum())} of the {len(eq)} parquet files were byte-identical.",
        "",
        f"## Gaps ({len(an)} analysed trials: no draws, without {', '.join(BAD_RECORDINGS)})",
        "",
        f"- interior gaps of the trial fish: {fmt(len(g_an))}; filled {fmt(g_counts.get('filled', 0))}",
        f"  ({g_an.loc[g_an['action'] == 'filled', 'duration_s'].sum() / 3600:.2f} h);",
        f"  left: EODf jump > {MAX_JUMP_HZ:g} Hz {fmt(g_counts.get('left_jump', 0))}, longer than {MAX_FILL_S:g} s "
        f"{fmt(g_counts.get('left_long', 0))}, both {fmt(g_counts.get('left_long_jump', 0))}.",
        f"- all {len(overview)} trials: {fmt(len(gaps))} interior gaps, filled "
        f"{fmt((gaps['action'] == 'filled').sum())}, left {fmt((gaps['action'] != 'filled').sum())}"
        " (`gaps_gapfill.csv`).",
        "",
        "## Chirps (analysed trials; chirp = score >= 0.5 and not flagged; first 3 h by box centre)",
        "",
        "| | whole recording | first 3 h |",
        "|---|---|---|",
    ]
    for label, b, a in (
        ("winner chirps", "chirps_win_before", "chirps_win_after"),
        ("loser chirps", "chirps_lose_before", "chirps_lose_after"),
        ("unassigned chirps", "unassigned_before", "unassigned_after"),
    ):
        lines.append(
            f"| {label} before -> after | {fmt(an[b].sum())} -> {fmt(an[a].sum())} | "
            f"{fmt(an[b + '_3h'].sum())} -> {fmt(an[a + '_3h'].sum())} |"
        )
    for label, a, b in (
        ("unassigned -> winner", "unassigned", "winner"),
        ("unassigned -> loser", "unassigned", "loser"),
        ("winner -> loser", "winner", "loser"),
        ("loser -> winner", "loser", "winner"),
        ("winner -> unassigned", "winner", "unassigned"),
        ("loser -> unassigned", "loser", "unassigned"),
    ):
        lines.append(f"| {label} | {fmt(n(a, b))} | {fmt(n(a, b, 'n_first_3h'))} |")
    fb = t_an["before"].str.startswith("flagged_")
    fa = t_an["after"].str.startswith("flagged_")
    for label, sel in (("newly flagged by post-processing", ~fb & fa), ("unflagged by post-processing", fb & ~fa)):
        lines.append(
            f"| {label} | {fmt(t_an.loc[sel, 'n'].sum())} | {fmt(t_an.loc[sel, 'n_first_3h'].sum())} |"
        )
    lines += [
        "",
        "Detail: `transitions.csv` (rows at score >= 0.5 by state before -> after: `winner`, `loser`,",
        "`unassigned`, `flagged_harmonic`, `flagged_duplicate`), `overview_trials.csv` (per trial).",
        f"Rows re-attributed in the source run but not after the patch (decoder values lost): {lost}.",
        "",
        "## Files",
        "",
        "- `<rec>/chirps_rtdetr_gapfill.parquet`: columns of `../competition2019_v2_20261113/columns.md`",
        "  plus `gapfill_changed` (the row differs from the source run's row: any column, float columns",
        f"  by more than {CHANGE_ATOL:g}; another {fmt(rounding)} rows of the {len(overview)} trials differ only by",
        "  floating-point rounding of the baselines' cumulative sums and are not flagged); changed rows",
        f"  further than {LOCAL_PAD_S:g} s from every filled gap: {fmt(far)};",
        "- `<rec>/chirp_{times,ids}_rtdetr_gapfill.npy`: legacy arrays (chirp and assigned or re-attributed);",
        "- `<rec>/meta.json`: provenance, patch parameters, gaps filled and left per fish, counts before",
        "  (`counts_source`) and after (`counts`), changes and transitions;",
        "- `<rec>/tracks_gapfill.npz` (trials): the trial fish's patched tracks in wavetracker layout",
        "  (`times`, `fund_v`, `ident_v`, `idx_v`, `filled`) and the gap table (`gap_*`);",
        "- `equivalence.csv`, `gaps_gapfill.csv`, `transitions.csv`, `overview_trials.csv`, `run.json`.",
        "",
        "Copies in the recording directories (`chirps_rtdetr_gapfill.parquet`,",
        "`chirp_{times,ids}_rtdetr_gapfill.npy`, `chirps_rtdetr_gapfill_meta.json`):",
        "`copy_detections_to_trials.py --run <this run> --suffix _gapfill`, manifest `trial_copies_manifest.csv`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
