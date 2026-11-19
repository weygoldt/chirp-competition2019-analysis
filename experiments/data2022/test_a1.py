"""Tests of the pure functions behind the A1 diagnosis (no data files needed).

    .venv/bin/python -m pytest experiments/data2022/test_a1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_excursion
import a1_influence
import a1_signal


def items(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["dataset", "score_bin", "recording", "w", "y"])


def test_agreement_is_the_weighted_share() -> None:
    t = items([("competition2022", "0.7-0.9", "a", 3.0, 1.0), ("competition2022", "0.7-0.9", "b", 1.0, 0.0)])
    assert a1_influence.agreement(t) == pytest.approx(0.75)


def test_by_bin_reports_the_heaviest_item_and_leave_one_out() -> None:
    t = items([
        ("competition2022", "0.7-0.9", "heavy", 9.0, 0.0),
        ("competition2022", "0.7-0.9", "b", 1.0, 1.0),
        ("competition2022", "0.7-0.9", "c", 1.0, 1.0),
    ])
    row = a1_influence.by_bin(t).iloc[0]
    assert row["agreement"] == pytest.approx(2 / 11)
    assert row["heaviest_item_weight_share"] == pytest.approx(9 / 11)
    assert row["heaviest_recording"] == "heavy"
    assert row["agreement_without_heaviest_recording"] == pytest.approx(1.0)
    assert row["leave_one_recording_out_min"] == pytest.approx(0.1)


def test_explained_covers_harmonics_and_intermodulation() -> None:
    fish = [640.0, 760.0]
    assert a1_signal.explained(1280.0, fish)  # 2 f1
    assert a1_signal.explained(520.0, fish)  # 2 f1 - f2
    assert a1_signal.explained(1400.0 + 4.0, fish)  # f1 + f2, within 6 Hz
    assert not a1_signal.explained(950.0, fish)
    assert not a1_signal.explained(950.0, [])


def test_transients_counts_broadband_clicks() -> None:
    fs = 20_000.0
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e-3, (int(2 * fs), 4))
    for t in (0.5, 1.0, 1.5):  # three clicks on every electrode
        i = int(t * fs)
        x[i:i + 10] += 0.5
    rate, wide = a1_signal.transients(x, fs)
    assert rate == pytest.approx(1.5)
    assert wide == pytest.approx(1.0)


def test_auc() -> None:
    assert a1_excursion.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert a1_excursion.auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert np.isnan(a1_excursion.auc(np.array([]), np.array([1.0])))


def test_best_electrode_prefers_the_fish_relative_to_others() -> None:
    fs = 20_000.0
    t = np.arange(int(0.5 * fs)) / fs
    a, b = np.sin(2 * np.pi * 640 * t), np.sin(2 * np.pi * 760 * t)
    # electrode 0: fish A loud but fish B louder; electrode 1: A alone, weaker
    x = np.stack([1.0 * a + 3.0 * b, 0.5 * a + 0.01 * b], axis=1)
    assert a1_excursion.best_electrode(x, fs, 640.0, [760.0]) == 1
