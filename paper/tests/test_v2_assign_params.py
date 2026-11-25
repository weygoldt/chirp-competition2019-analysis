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


# ---------------------------------------------------------------------------
# detection: every chirp label, re-matched at the operating threshold
# ---------------------------------------------------------------------------
def _eval_items(outcomes=("tp", "tp", "fn"), pred_rows=(5.0, 6.0, np.nan),
                scores=(0.95, 0.6, np.nan)) -> pd.DataFrame:
    """Item rows of an evaluation: the three chirp labels (rows 0, 2, 3), the
    ignore label (row 1) and a false positive without a label."""
    return pd.DataFrame({
        "recording": ["s1_f0", "s1_f0", "s1_f440", "s2_f0", "s2_f0"],
        "outcome": [outcomes[0], "ignored", outcomes[1], outcomes[2], "fp"],
        "label_row": [0, 1, 2, 3, np.nan],
        "pred_row": [pred_rows[0], 4.0, pred_rows[1], pred_rows[2], 9.0],
        "label": ["chirp", "ignore", "chirp", "chirp", None],
        "score": [scores[0], 0.7, scores[1], scores[2], 0.55],
    })


def test_visible_chirps_gives_every_chirp_label_its_outcome_scene_and_chirp():
    out = ap.visible_chirps(_eval_items(), _labels(), SOURCES)
    assert out["label_row"].tolist() == [0, 2, 3]
    assert out["outcome"].tolist() == ["tp", "tp", "fn"]
    assert out["source_id"].tolist() == ["s1", "s1", "s2"]
    assert out["chirp_id"].tolist() == [7, 7, 3]
    assert out["pred_row"].isna().tolist() == [False, False, True]


def test_visible_chirps_refuses_a_missing_or_doubled_chirp_label():
    items = _eval_items()
    with pytest.raises(ValueError, match="miss 1 chirp labels"):
        ap.visible_chirps(items.drop(index=3), _labels(), SOURCES)
    with pytest.raises(ValueError, match="more than one item row"):
        ap.visible_chirps(pd.concat([items, items.iloc[[0]]]), _labels(), SOURCES)
    items.loc[0, "recording"] = "s1_f440"
    with pytest.raises(ValueError, match="tile"):
        ap.visible_chirps(items, _labels(), SOURCES)


def test_check_rematch_accepts_a_reproduction_and_refuses_a_different_match():
    ref = ap.visible_chirps(_eval_items(), _labels(), SOURCES)
    assert ap.check_rematch(ref, ref.copy()) == 3
    # the evaluation's own item table (Int64 rows, as in memory) against items.csv (floats)
    in_memory = _eval_items()
    in_memory["label_row"] = pd.array([0, 1, 2, 3, pd.NA], dtype="Int64")
    in_memory["pred_row"] = pd.array([5, 4, 6, pd.NA, 9], dtype="Int64")
    assert ap.check_rematch(ref, ap.visible_chirps(in_memory, _labels(), SOURCES)) == 3
    other = ap.visible_chirps(_eval_items(pred_rows=(5.0, 7.0, np.nan)), _labels(), SOURCES)
    with pytest.raises(ValueError, match="pred_row of 1"):
        ap.check_rematch(ref, other)
    other = ap.visible_chirps(_eval_items(outcomes=("tp", "fn", "fn"),
                                          pred_rows=(5.0, np.nan, np.nan)), _labels(), SOURCES)
    with pytest.raises(ValueError, match="outcome of 1"):
        ap.check_rematch(ref, other)


def test_detection_table_takes_the_outcome_of_the_rematch_not_the_filtered_reference():
    ref = ap.visible_chirps(_eval_items(), _labels(), SOURCES)
    # at 0.8 the label of row 2 (matched at 0.5 to a box of score 0.6) is
    # matched to another box of score 0.85: detected, although its
    # reference match scores below the threshold
    at_op = ap.visible_chirps(_eval_items(pred_rows=(5.0, 8.0, np.nan),
                                          scores=(0.95, 0.85, np.nan)), _labels(), SOURCES)
    out = ap.detection_table(ref, at_op, 0.8)
    assert out["detected"].tolist() == [True, True, False]
    assert out["outcome_ref"].tolist() == ["tp", "tp", "fn"]
    assert out["pred_row"].tolist()[:2] == [5, 8]
    assert out["source_id"].tolist() == ["s1", "s1", "s2"]


def test_detection_table_refuses_a_box_below_the_threshold_or_swapped_thresholds():
    ref = ap.visible_chirps(_eval_items(), _labels(), SOURCES)
    with pytest.raises(ValueError, match="score below"):
        ap.detection_table(ref, ref, 0.8)  # row 2's box scores 0.6
    found = ap.visible_chirps(_eval_items(outcomes=("tp", "fn", "tp"),
                                          pred_rows=(5.0, np.nan, 9.0),
                                          scores=(0.95, np.nan, 0.9)), _labels(), SOURCES)
    with pytest.raises(ValueError, match="swapped"):
        ap.detection_table(ref, found, 0.8)


def test_join_truth_gives_every_chirp_label_its_truth():
    missed_at_op = ap.visible_chirps(_eval_items(outcomes=("tp", "fn", "fn"),
                                                 pred_rows=(5.0, np.nan, np.nan),
                                                 scores=(0.95, np.nan, np.nan)),
                                     _labels(), SOURCES)
    det = ap.detection_table(ap.visible_chirps(_eval_items(), _labels(), SOURCES),
                             missed_at_op, 0.8)
    out = ap.join_truth(det, _truth())
    assert out["width10_s"].tolist() == [0.29, 0.29, 0.014]
    assert out["detected"].tolist() == [True, False, False]
