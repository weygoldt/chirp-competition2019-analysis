"""Runs the labeller page's logic test (``pose_labeller/test_labeller.mjs``) under node, if node is installed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TEST = Path(__file__).resolve().parent / "pose_labeller" / "test_labeller.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_labeller_page_logic() -> None:
    r = subprocess.run(["node", str(TEST)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "labeller logic ok" in r.stdout
