"""The chase-review tool (paper/chase_review): clips, time map, saved revisions, session builder, export."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOL = Path(__file__).resolve().parents[1] / "chase_review"
sys.path.insert(0, str(TOOL))

import build_session as bs  # noqa: E402
import export_revisions as ex  # noqa: E402
import server as srv  # noqa: E402
from led_alignment import fit_led_alignment  # noqa: E402
from paper_config import DATA_ROOT  # noqa: E402
from test_led_alignment import SR, leds  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
needs_data = pytest.mark.skipif(not (DATA_ROOT / "trial_summary.csv").exists(), reason="recordings not mounted")

LUMA_STEP = 4  # synthetic video: frame k has luma 16 + 4 k


@pytest.fixture(scope="module")
def bframe_avi(tmp_path_factory) -> Path:
    """h264 with B-frames in AVI, like the ``_mod.avi`` files; frame k has luma 16 + 4 k (50 frames)."""
    out = tmp_path_factory.mktemp("video") / "synthetic_mod.avi"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=black:s=64x48:r=25:d=2,format=gray,geq=lum='16+{LUMA_STEP}*N'",
                    "-c:v", "libx264", "-x264-params", "bframes=2:b-adapt=0", "-g", "10", "-crf", "1",
                    "-pix_fmt", "yuv420p", str(out)], check=True)  # b-adapt=0: B-frames even in flat frames
    return out


def frame_indices(path: Path) -> list[int]:
    """Decode a synthetic clip and read each frame's index from its brightness."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 48, 64)
    return [round(float(f.mean() - 16) / LUMA_STEP) for f in fr]


@needs_ffmpeg
def test_probe_sees_the_bframe_delay(bframe_avi):
    info = bs.probe_video(bframe_avi)
    assert info == {"n_frames": 50, "fps": 25.0, "delay_frames": 2}


@needs_ffmpeg
def test_clip_holds_exactly_the_requested_frames(bframe_avi, tmp_path):
    delay = bs.probe_video(bframe_avi)["delay_frames"]
    out = srv.make_clip(bframe_avi, 17, 12, 25.0, delay, tmp_path / "c.mp4")
    assert frame_indices(out) == list(range(17, 29))
    # without the delay correction the clip starts two frames early: the bug the correction fixes
    early = srv.make_clip(bframe_avi, 17, 12, 25.0, 0, tmp_path / "early.mp4")
    assert frame_indices(early) == list(range(15, 27))


@needs_ffmpeg
def test_clip_with_too_few_frames_raises(bframe_avi, tmp_path):
    with pytest.raises(RuntimeError, match="frames, expected"):
        srv.make_clip(bframe_avi, 45, 10, 25.0, 2, tmp_path / "short.mp4")
    assert not (tmp_path / "short.mp4").exists()


def two_segment_map() -> dict:
    """The LED map of a video that lost 4.6 s at 4.9 h (as 2020-05-13), as stored in session.json."""
    frames, idx = leds(lost=(4.9 * 3600, 4.6))
    align = fit_led_alignment(frames, idx, SR)
    assert len(align.seg_slope) == 2
    return align, {"bounds": align.seg_bounds.tolist(), "slope": align.seg_slope.tolist(),
                   "icpt": align.seg_icpt.tolist()}


def test_python_time_map_is_the_led_map():
    align, m = two_segment_map()
    f = np.linspace(0, 6 * 3600 * 25.86, 1001)
    assert np.allclose([srv.to_rec_s(m, x) for x in f], align.to_grid(f), rtol=0, atol=1e-9)


@needs_node
def test_js_time_map_matches_python(tmp_path):
    align, m = two_segment_map()
    b = int(m["bounds"][0])
    frames = [0, 1, 1000, b - 1, b, b + 1, 400_000, 500_000, 558_000]
    (tmp_path / "m.json").write_text(json.dumps({"map": m, "frames": frames}))
    js = (f"const tm = require({json.dumps(str(TOOL / 'timemap.js'))});"
          f"const d = require({json.dumps(str(tmp_path / 'm.json'))});"
          "const rec = d.frames.map(f => tm.toRec(d.map, f));"
          "const back = rec.map(t => tm.toFrame(d.map, t, 0, 600000));"
          "const w = tm.chaseWindow(d.map, 100000, 100300, 10, 10, 560000);"
          "console.log(JSON.stringify({rec, back, w, w0: tm.toRec(d.map, w.frame0), w1: tm.toRec(d.map, w.frame0 + w.n - 1)}));")
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert np.allclose(out["rec"], align.to_grid(np.array(frames, float)), rtol=0, atol=1e-9)
    assert out["back"] == frames
    on, off = align.to_grid(np.array([100000.0, 100300.0]))
    frame_s = float(align.seg_slope[0])
    assert abs(out["w0"] - (on - 10)) <= frame_s / 2 + 1e-9
    assert abs(out["w1"] - (off + 10)) <= frame_s / 2 + 1e-9


def tiny_session() -> dict:
    m = {"bounds": [], "slope": [0.04], "icpt": [-1.0]}
    rec = {"recording": "r1", "video": "v.avi", "n_frames": 10_000, "fps": 25.0, "delay_frames": 2, "boris_fps": 25.0,
           "led_map": m, "chirps": {"winner": [], "loser": []}, "contacts": []}
    chases = [{"id": "r1_c000", "recording": "r1", "k": 0, "onset_frame": 1000, "offset_frame": 1100,
               "onset_rec_s": 39.0, "offset_rec_s": 43.0, "shares_offset_with": []},
              {"id": "r1_c001", "recording": "r1", "k": 1, "onset_frame": 9000, "offset_frame": 9500,
               "onset_rec_s": 359.0, "offset_rec_s": 379.0, "shares_offset_with": []}]
    return {"meta": {"data_root": "/nonexistent", "active_phase_s": [0, 300]}, "recordings": {"r1": rec},
            "chases": chases, "order": ["r1_c001", "r1_c000"]}


def test_check_revision_validates_and_computes_the_shift():
    s = tiny_session()
    chases, recs = {c["id"]: c for c in s["chases"]}, s["recordings"]
    rev = srv.check_revision({"chase_id": "r1_c000", "onset_frame": 1010, "offset_frame": 1075,
                              "blind": True, "chirps_hidden": True, "note": "x" * 2000}, chases, recs)
    assert rev["d_onset_s"] == pytest.approx(0.4)
    assert rev["d_offset_s"] == pytest.approx(-1.0)
    assert rev["blind"]
    assert rev["chirps_hidden"]
    assert len(rev["note"]) == 1000
    assert rev["boris_offset_frame"] == 1100
    nc = srv.check_revision({"chase_id": "r1_c000", "not_a_chase": True}, chases, recs)
    assert nc["onset_frame"] is None
    assert "d_offset_s" not in nc
    with pytest.raises(ValueError, match="unknown chase"):
        srv.check_revision({"chase_id": "r1_c009"}, chases, recs)
    with pytest.raises(ValueError, match="mark both"):
        srv.check_revision({"chase_id": "r1_c000", "onset_frame": 5}, chases, recs)
    with pytest.raises(ValueError, match="before offset"):
        srv.check_revision({"chase_id": "r1_c000", "onset_frame": 1100, "offset_frame": 1100}, chases, recs)
    with pytest.raises(ValueError, match="video has"):
        srv.check_revision({"chase_id": "r1_c000", "onset_frame": 9000, "offset_frame": 10_000}, chases, recs)


def test_saves_append_and_the_latest_wins(tmp_path):
    (tmp_path / "session.json").write_text(json.dumps(tiny_session()))
    s = srv.Session(tmp_path, "me")
    s.save({"chase_id": "r1_c000", "onset_frame": 1000, "offset_frame": 1100})
    s.save({"chase_id": "r1_c000", "onset_frame": 1000, "offset_frame": 1090})
    s.save({"chase_id": "r1_c001", "not_a_chase": True})
    assert len((tmp_path / "revisions.jsonl").read_text().splitlines()) == 3
    got = srv.latest(tmp_path / "revisions.jsonl")
    assert got["r1_c000"]["offset_frame"] == 1090
    assert got["r1_c000"]["reviewer"] == "me"
    with pytest.raises(ValueError, match="outside the video"):
        s.clip_path("r1", 9990, 20)
    with pytest.raises(KeyError):
        s.clip_path("r9", 0, 20)
    assert s.clip_path("r1", 12, 30).name == "0000012_00030.mp4"

    df = ex.revision_table(tiny_session(), got)
    assert list(df["chase_id"]) == ["r1_c000", "r1_c001"]
    assert df.loc[0, "d_offset_frames"] == -10
    assert bool(df.loc[0, "in_fig6"])
    assert not bool(df.loc[1, "in_fig6"])
    assert bool(df.loc[1, "not_a_chase"])
    assert "1 moved, 1 not a chase" in ex.summary(df)


def test_parse_range():
    assert srv.parse_range(None, 100) is None
    assert srv.parse_range("bytes=0-", 100) == (0, 99)
    assert srv.parse_range("bytes=10-19", 100) == (10, 19)
    assert srv.parse_range("bytes=-10", 100) == (90, 99)
    with pytest.raises(ValueError, match="after the end"):
        srv.parse_range("bytes=100-", 100)


def test_shared_offsets():
    ids = ["a", "b", "c", "d"]
    assert bs.shared_offsets(np.array([10, 30, 30, 50]), ids) == [[], ["c"], ["b"], []]


@needs_data
@needs_ffmpeg
def test_build_recording_on_2020_05_13():
    """The trial with the missing BORIS offset: c019 has no offset of its own and takes c020's."""
    rec = "2020-05-13-10_00"
    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    row = ts[ts["recording"] == rec].iloc[0]
    info, chases = bs.build_recording(DATA_ROOT / rec, row, "rtdetr_gapfill_t080")  # checks the analysis' times
    assert info["video"] == "2020-05-13_0_mod.avi"
    assert info["delay_frames"] == 2
    assert len(chases) == 141
    assert len(info["led_map"]["slope"]) == 2
    shared = {c["id"]: c["shares_offset_with"] for c in chases if c["shares_offset_with"]}
    assert shared == {f"{rec}_c019": [f"{rec}_c020"], f"{rec}_c020": [f"{rec}_c019"]}
    assert all(np.diff(info["chirps"]["loser"]) >= 0)
