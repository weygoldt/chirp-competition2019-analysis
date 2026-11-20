"""Tests of the chase labeller server's helpers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "chase_labeller"))
import server as srv


def test_parse_range():
    assert srv.parse_range(None, 100) is None
    assert srv.parse_range("bytes=0-", 100) == (0, 99)
    assert srv.parse_range("bytes=10-19", 100) == (10, 19)
    assert srv.parse_range("bytes=90-200", 100) == (90, 99)
    assert srv.parse_range("bytes=-10", 100) == (90, 99)
    with pytest.raises(ValueError, match="after the end"):
        srv.parse_range("bytes=100-", 100)


def test_check_label_orders_chases_and_rejects_bad_ones():
    ids = {"a"}
    lab = srv.check_label({"clip_id": "a", "chases": [{"onset_s": 5, "offset_s": 6}, {"onset_s": 1, "offset_s": 2}]},
                          ids)
    assert [c["onset_s"] for c in lab["chases"]] == [1.0, 5.0]
    assert [c["contact"] for c in lab["chases"]] == [False, False]
    lab = srv.check_label({"clip_id": "a", "chases": [{"onset_s": 1, "offset_s": 2, "contact": True}]}, ids)
    assert lab["chases"][0]["contact"] is True
    with pytest.raises(ValueError, match="unknown clip"):
        srv.check_label({"clip_id": "b", "chases": []}, ids)
    with pytest.raises(ValueError, match="before offset"):
        srv.check_label({"clip_id": "a", "chases": [{"onset_s": 3, "offset_s": 2}]}, ids)
    with pytest.raises(ValueError, match="no chase"):
        srv.check_label({"clip_id": "a", "no_chase": True, "chases": [{"onset_s": 1, "offset_s": 2}]}, ids)


def test_latest_label_wins(tmp_path):
    (tmp_path / "clips.json").write_text(json.dumps([{"clip_id": "a", "video": "clips/a.mp4",
                                                      "video_marked": "clips/a_m.mp4", "fps": 25, "n_frames": 10}]))
    s = srv.Session(tmp_path, "me")
    s.save({"clip_id": "a", "no_chase": True})
    s.save({"clip_id": "a", "chases": [{"onset_s": 0.2, "offset_s": 0.3}]})
    got = srv.latest(tmp_path / "labels.jsonl")
    assert got["a"]["chases"][0]["onset_s"] == 0.2
    assert got["a"]["labeller"] == "me"
