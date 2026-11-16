"""manuscript_stats: what an analysis module writes, and how collect and check treat it."""

from __future__ import annotations

import json

import numpy as np
import pytest

import manuscript_stats as ms
import paper_config


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A repository in tmp_path with one run, ``testrun``; the writing script is ``analysis.py``."""
    out = tmp_path / "paper" / "output" / "testrun"
    monkeypatch.setattr(ms, "PAPER", tmp_path / "paper")
    monkeypatch.setattr(ms, "REPO", tmp_path)
    monkeypatch.setattr(paper_config, "OUT", out)
    monkeypatch.setattr(paper_config, "DETECTIONS", "testrun")
    monkeypatch.setattr(ms.sys, "argv", ["analysis.py"])
    monkeypatch.setattr(ms, "_git", lambda *args: "" if args[0] == "status" else "c0ffee")
    return out


def test_write_converts_numpy(run):
    path = ms.write_stats("demo", {"n": np.int64(1943), "fit": {"median": np.float64(1.2),
                                                                 "hdi": np.array([1.07, 1.34])}})
    record = json.loads(path.read_text())
    assert path == run / "stats" / "demo.json"
    assert record["values"] == {"n": 1943, "fit": {"median": 1.2, "hdi": [1.07, 1.34]}}
    assert record["script"] == "analysis.py"
    assert record["commit"] == "c0ffee"
    assert record["code_modified"] is False


@pytest.mark.parametrize("values", [{"x": float("nan")}, {"x": "1.2"}, {"x": None}, {"x": True},
                                    {"a.b": 1}, {"x": [1.0, float("inf")]}])
def test_write_refuses_what_is_not_a_number(run, values):
    with pytest.raises(ValueError):
        ms.write_stats("demo", values)


def test_namespace_belongs_to_one_script(run, monkeypatch):
    ms.write_stats("demo", {"x": 1})
    ms.write_stats("demo", {"x": 2})  # the same script may rewrite it
    monkeypatch.setattr(ms.sys, "argv", ["other.py"])
    with pytest.raises(ValueError, match="belongs to analysis.py"):
        ms.write_stats("demo", {"x": 3})


def test_collect_merges_namespaces_with_provenance(run, tmp_path):
    a = ms.write_stats("a", {"x": 1})
    ms.write_stats("b", {"y": {"median": 0.5}})
    stats = ms.build("testrun")
    assert stats["values"] == {"a": {"x": 1}, "b": {"y": {"median": 0.5}}}
    assert stats["detections"] == "testrun"
    entry = stats["inputs"]["paper/output/testrun/stats/a.json"]
    assert entry["script"] == "analysis.py" and entry["commit"] == "c0ffee"
    assert len(entry["sha256"]) == 64 and a.exists()


def test_collect_refuses_uncommitted_code(run, monkeypatch):
    monkeypatch.setattr(ms, "_git", lambda *args: " M x.py" if args[0] == "status" else "c0ffee")
    ms.write_stats("a", {"x": 1})
    with pytest.raises(ValueError, match="uncommitted"):
        ms.build("testrun")


def test_check_finds_every_difference(run, tmp_path):
    ms.write_stats("a", {"x": 1})
    ms.write_stats("b", {"y": 2})
    out = ms.collect("testrun", tmp_path / "stats.json")
    assert ms.check(out) == []

    ms.write_stats("a", {"x": 5})  # a module was rerun and its number changed
    ms.write_stats("c", {"z": 3})  # a new module
    (run / "stats" / "b.json").unlink()  # a module's file is gone
    assert ms.check(out) == ["changed: paper/output/testrun/stats/a.json",
                             "missing: paper/output/testrun/stats/b.json",
                             "new: paper/output/testrun/stats/c.json",
                             "value a.x: 1 -> 5", "value b.y: 2 -> None", "value c.z: None -> 3"]


def test_check_finds_hand_edits(run, tmp_path):
    ms.write_stats("a", {"x": 1})
    out = ms.collect("testrun", tmp_path / "stats.json")
    stats = json.loads(out.read_text())
    stats["values"]["a"]["x"] = 2
    out.write_text(json.dumps(stats))
    assert ms.check(out) == ["value a.x: 2 -> 1"]
