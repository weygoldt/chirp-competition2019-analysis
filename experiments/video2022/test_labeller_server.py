"""Tests of the keypoint labeller's server on a temporary session (no browser)."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pose_labeller"))
import pose_labels as pl  # noqa: E402
import server  # noqa: E402


@pytest.fixture
def running(tmp_path: Path):  # noqa: ANN201
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"\xff\xd8fake")
    items = [{"item_id": "a", "image": "images/a.jpg", "width": 100, "height": 80, "prefill": []},
             {"item_id": "b", "image": "images/b.jpg", "width": 100, "height": 80,
              "prefill": [{"head": [1, 2], "center": None, "tail": [3, 4]}]}]
    (tmp_path / "items.json").write_text(json.dumps(items))
    (tmp_path / "truth.csv").write_text("item_id,recording\na,secret\n")
    httpd = server.serve(tmp_path, 0, "tester")
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield tmp_path, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(url: str, obj: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_session_state_hides_truth(running) -> None:  # noqa: ANN001
    _, url = running
    code, body = get(url + "/api/session")
    state = json.loads(body)
    assert code == 200
    assert [it["item_id"] for it in state["items"]] == ["a", "b"]
    assert "secret" not in body.decode()
    assert get(url + "/images/a.jpg") == (200, b"\xff\xd8fake")
    assert get(url + "/images/../truth.csv")[0] == 404
    assert get(url + "/truth.csv")[0] == 404


def test_save_appends_and_validates(running) -> None:  # noqa: ANN001
    root, url = running
    code, r = post(url + "/api/label", {"item_id": "a", "status": "done", "elapsed_s": 2.0,
                                        "fish": [{"head": [10, 11], "center": None, "tail": [30, 31], "origin": "new"}]})
    assert code == 200 and r["ok"]
    labels = pl.load_labels(root / "labels.jsonl")
    assert labels["a"].labeller == "tester"
    assert labels["a"].fish[0].points["tail"] == (30.0, 31.0)
    assert post(url + "/api/label", {"item_id": "zzz", "status": "done"})[0] == 400
    assert post(url + "/api/label", {"item_id": "a", "status": "perhaps"})[0] == 400
    state = json.loads(get(url + "/api/session")[1])
    assert state["labels"]["a"]["status"] == "done"
