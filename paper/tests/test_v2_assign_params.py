"""The join of the simulated assignment true positives to their chirp's truth (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import v2_assign_params as ap


def _labels() -> pd.DataFrame:
    """A label table in the evaluation's row order: two tiles of one scene and one of another."""
    return pd.DataFrame({
        "recording": ["s1_f0", "s1_f0", "s1_f440", "s2_f0"],
        "label": ["chirp", "ignore", "chirp", "chirp"],
        "t_center_s": [0.5, 1.2, 0.3, 2.0],
        "emitter_id": pd.array([2, 0, 2, 1], dtype="Int64"),
        "chirp_id": [7, 8, 7, 3],  # chirp 7 of scene s1 lies in both tiles
    })


SOURCES = {"s1_f0": "s1", "s1_f440": "s1", "s2_f0": "s2"}


def _tp() -> pd.DataFrame:
    return pd.DataFrame({
        "recording": ["s1_f0", "s1_f440", "s2_f0"],
        "label_row": [0.0, 2.0, 3.0],  # floats, as read from assign_baseline.csv
        "pred_row": [5.0, 6.0, 9.0],
        "true_emitter_id": [2.0, 2.0, 1.0],
    })


def _items() -> pd.DataFrame:
    return pd.DataFrame({
        "recording": ["s1_f0", "s1_f0", "s1_f440", "s2_f0", "s2_f0"],
        "label_row": [0, 1, 2, 3, np.nan],  # the last row: a false positive, no label
        "label": ["chirp", "ignore", "chirp", "chirp", None],
        "label_t_center_s": [0.5, 1.2, 0.3, 2.0, np.nan],
        "true_emitter_id": pd.array([2, 0, 2, 1, pd.NA], dtype="Int64"),
    })


def test_check_items_accepts_the_evaluation_order_and_counts_label_rows():
    assert ap.check_items(_items(), _labels()) == 4


@pytest.mark.parametrize(("column", "value", "what"), [
    ("label_t_center_s", 0.6, "box midpoint"),
    ("true_emitter_id", 1, "emitter"),
    ("label", "ignore", "label kind"),
    ("recording", "s2_f0", "tile"),
])
def test_check_items_refuses_a_ground_truth_in_another_order(column, value, what):
    items = _items()
    items.loc[0, column] = value
    with pytest.raises(ValueError, match=what):
        ap.check_items(items, _labels())


def test_tp_chirps_maps_label_rows_to_scene_and_chirp():
    out = ap.tp_chirps(_tp(), _labels(), SOURCES)
    assert out["source_id"].tolist() == ["s1", "s1", "s2"]
    assert out["chirp_id"].tolist() == [7, 7, 3]
    assert out[ap.KEYS].equals(_tp()[ap.KEYS])


def test_tp_chirps_refuses_a_true_positive_on_an_ignore_label_or_the_wrong_emitter():
    tp = _tp()
    tp.loc[0, "label_row"] = 1.0  # the ignore label of the same tile
    with pytest.raises(ValueError, match="do not point to a chirp label"):
        ap.tp_chirps(tp, _labels(), SOURCES)
    tp = _tp()
    tp.loc[2, "true_emitter_id"] = 0.0
    with pytest.raises(ValueError, match="do not point to a chirp label"):
        ap.tp_chirps(tp, _labels(), SOURCES)
    with pytest.raises(ValueError, match="tiles without a scene"):
        ap.tp_chirps(_tp(), _labels(), {"s1_f0": "s1", "s1_f440": "s1"})


def _truth() -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": ["s1", "s1", "s2"],
        "chirp_id": [7, 8, 3],
        "height_hz": [350.0, 60.0, 45.0],
        "fwhm_s": [0.15, 0.010, 0.008],
        "width10_s": [0.29, 0.018, 0.014],
        "undershoot_hz": [3.0, 0.0, 0.0],
        "edge_clipped": [False, False, False],
        "t_peak_s": [1.4, 2.0, 3.0],
    })


def test_join_truth_gives_every_true_positive_its_chirps_truth():
    out = ap.join_truth(ap.tp_chirps(_tp(), _labels(), SOURCES), _truth())
    # the chirp in two tiles appears twice, with the same truth
    assert out["height_hz"].tolist() == [350.0, 350.0, 45.0]
    assert out["width10_s"].tolist() == [0.29, 0.29, 0.014]
    assert len(out) == 3


def test_join_truth_refuses_missing_or_duplicate_truth():
    chirps = ap.tp_chirps(_tp(), _labels(), SOURCES)
    with pytest.raises(ValueError, match="without a truth row"):
        ap.join_truth(chirps, _truth().iloc[:2])
    with pytest.raises(ValueError, match="duplicate"):
        ap.join_truth(chirps, pd.concat([_truth(), _truth().iloc[:1]]))


def _decoder(truth: pd.DataFrame) -> pd.DataFrame:
    d = truth.iloc[[0, 2]].rename(columns={p: f"true_{p}" for p in ap.TRUTH})
    return d.assign(score=0.9)


def test_check_decoder_counts_shared_chirps_and_refuses_a_different_definition():
    truth = _truth()
    check = ap.check_decoder(truth, _decoder(truth))
    assert check["n_shared"] == 2
    assert all(check[f"max_abs_diff_{p}"] == 0.0 for p in ap.TRUTH)
    off = _decoder(truth)
    off.loc[off.index[0], "true_width10_s"] += 1e-3
    with pytest.raises(ValueError, match="width10_s"):
        ap.check_decoder(truth, off)
