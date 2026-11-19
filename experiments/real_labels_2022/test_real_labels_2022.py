"""Tests of the design and scoring functions of real_v2_2022 (no data files needed).

    .venv/bin/python -m pytest experiments/real_labels_2022/test_real_labels_2022.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_2022 as b
import build_session as v1
import design2022 as d
import fake_labels_2022
import score_2022 as sc


def test_score_bins_and_groups() -> None:
    s = [0.49, 0.5, 0.6999, 0.7, 0.8, 0.8999, 0.9, 1.0]
    assert d.score_bin(s).tolist() == ["below", "0.5-0.7", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.8-0.9",
                                       "0.9-1", "0.9-1"]
    assert d.group_of(["2022-06-20-10_00", "2022-06-21-10_00", "2022-06-24-10_00"]).tolist() == [
        "pair3", "rest", "pair3"]


@pytest.mark.parametrize(("pool", "expected"), [
    ((2490, 22), (12, 3)),   # big chirps rare: at least MIN_BIG
    ((119, 58), (10, 5)),    # big chirps common: their share
    ((12, 2), (12, 2)),      # small pool: taken whole
    ((503, 0), (15, 0)),     # no big chirps
    ((4, 40), (3, 12)),      # big chirps the majority: small keeps its floor
    ((2, 400), (2, 13)),     # a size with fewer than the floor: all of it
])
def test_split_sizes(pool: tuple[int, int], expected: tuple[int, int]) -> None:
    assert d.split_sizes(*pool, 15, 3) == expected


def test_n_eff() -> None:
    assert d.kish_n_eff([2.0] * 7) == pytest.approx(7.0)
    assert d.cell_n_eff(2490, 22, 12, 3) < 15
    assert d.cell_n_eff(2490, 22, 12, 3) > d.cell_n_eff(2490, 22, 8, 7)  # even split loses more


def pool_frame(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for g, n in (("pair3", 300), ("rest", 40)):
        for k in range(n):
            rows.append({"score_bin": "0.8-0.9", "status": "loser", "group": g,
                         "size": "big" if k % 20 == 0 else "small", "recording": f"r{k % 3}",
                         "score": rng.uniform(0.8, 0.9)})
    return pd.DataFrame(rows)


def test_allocation_and_inclusion_probabilities() -> None:
    rng = np.random.default_rng(0)
    pool = pool_frame(rng)
    alloc = b.allocate(pool)
    cell = alloc[(alloc["score_bin"] == "0.8-0.9") & (alloc["status"] == "loser")]
    assert cell.groupby("group")["n"].sum().to_dict() == {"pair3": 15, "rest": 15}
    assert alloc.loc[alloc["status"] == "winner", "n"].sum() == 0
    s = b.sample_strata(pool, alloc, rng)
    assert len(s) == 30
    # every item's pi is n / N of its stratum, and the weights add up to the pool
    for _, g in s.groupby("stratum"):
        assert np.allclose(g["pi"], len(g) / g["cell_N"].iloc[0])
    assert s.groupby("group")["weight"].sum().to_dict() == pytest.approx({"pair3": 300.0, "rest": 40.0})


def test_repeats_come_at_least_the_gap_later() -> None:
    rng = np.random.default_rng(3)
    n_base, originals = 300, [0, 5, 250, 299, 120]
    v1.MIN_REPEAT_GAP = b.MIN_REPEAT_GAP
    order = v1.order_items(n_base, [(n_base + k, o) for k, o in enumerate(originals)], rng)
    pos = {item: k for k, item in enumerate(order)}
    assert all(pos[n_base + k] - pos[o] >= 100 for k, o in enumerate(originals))


def candidates_table() -> pd.DataFrame:
    # two strata: A (N=100, 4 labelled of 5), B (N=10, 2 labelled)
    rows = [("A", 0.85, "chirp"), ("A", 0.86, "not_chirp"), ("A", 0.95, "chirp"), ("A", 0.81, "chirp"),
            ("A", 0.82, None), ("B", 0.6, "not_chirp"), ("B", 0.65, "chirp_emitter_unclear")]
    t = pd.DataFrame(rows, columns=["stratum", "score", "label"])
    t["stratum"] = "main/" + t["stratum"]
    t["cell_N"] = np.where(t["stratum"] == "main/A", 100, 10)
    t["labelled"] = t["label"].notna()
    t["y"] = np.where(t["labelled"], t["label"].isin(d.POSITIVE), np.nan)
    t["score_bin"] = d.score_bin(t["score"])
    t["status"], t["group"], t["size"] = "loser", "pair3", "small"
    return t


def test_weights_and_agreement() -> None:
    t = candidates_table()
    n_lab = t[t["labelled"]].groupby("stratum").size()
    t["w"] = np.where(t["labelled"], t["cell_N"] / t["stratum"].map(n_lab), 0.0)
    main = t[t["labelled"]].reset_index(drop=True)
    a = sc.agreement_table(main, np.random.default_rng(0))
    row = a[(a["by"] == "all") & (a["scores"] == ">= 0.5")].iloc[0]
    # A: 3 of 4 chirps, weight 25 each; B: 1 of 2, weight 5 each
    assert row["agreement"] == pytest.approx((3 * 25 + 5) / (100 + 10))
    assert row["N_hat"] == pytest.approx(110)
    assert row["ci_low"] <= row["agreement"] <= row["ci_high"]
    row = a[(a["by"] == "all") & (a["scores"] == ">= 0.9")].iloc[0]
    assert row["agreement"] == 1.0
    assert row["n"] == 1


def test_candidate_items_weights_from_labels() -> None:
    truth = candidates_table().drop(columns=["label", "labelled", "y"])
    truth["candidate_id"] = [f"c{k}" for k in range(len(truth))]
    truth["item_kind"] = "box"
    lab = candidates_table()
    labels = pd.DataFrame({"candidate_id": truth["candidate_id"], "label": lab["label"],
                           "parent_item_id": "", "labeled_at": pd.Timestamp("2026-11-19"),
                           "emitter_id": np.nan, "emitter_baseline_hz": np.nan, "delta_f_hz": np.nan,
                           "elapsed_s": 1.0, "via": "mouse"}).dropna(subset=["label"])
    t = sc.candidate_items(truth, labels)
    assert t.loc[t["candidate_id"] == "c0", "w"].iloc[0] == pytest.approx(25.0)
    assert t.loc[t["candidate_id"] == "c4", "w"].iloc[0] == 0.0  # unlabelled
    assert t.loc[t["candidate_id"] == "c5", "w"].iloc[0] == pytest.approx(5.0)


def test_composite_weights_by_effective_sample_size() -> None:
    idx = np.tile(np.arange(4), (sc.s1.N_BOOT, 1))  # a degenerate bootstrap: the sample itself
    new = (np.array([1.0, 1, 1, 0]), np.ones(4), idx, 3.0)  # 0.75
    old = (np.array([0.0, 0, 0, 0]), np.ones(4), idx, 1.0)  # 0.0
    out = sc.composite([new, old])
    assert out["lambda_new"] == pytest.approx(0.75)
    assert out["pooled"] == pytest.approx(0.75 * 0.75)
    assert sc.composite([new, (np.zeros(4), np.zeros(4), idx, 0.0)])["pooled"] == pytest.approx(0.75)


def test_window_tables() -> None:
    per = pd.DataFrame({"group": ["pair3", "pair3", "rest", "rest"], "w": [10.0, 10.0, 50.0, 50.0],
                        "marks": [2, 0, 1, 0]})
    for t in d.THRESHOLDS:
        per[f"found_{t}"] = [2 if t < 0.9 else 1, 0, 0, 0]
        per[f"found_time_only_{t}"] = per[f"found_{t}"]
        per[f"det_zone_{t}"] = [2 if t < 0.9 else 1, 1, 0, 0]
        per[f"det_zone_matched_{t}"] = per[f"found_{t}"]
    found, rates = sc.window_tables(per, np.random.default_rng(0))
    f = found.set_index(["group", "threshold"])
    assert f.loc[("all", 0.8), "found_share"] == pytest.approx(20 / (20 + 50))
    assert f.loc[("pair3", 0.9), "found_share"] == pytest.approx(0.5)
    assert f.loc[("pair3", 0.8), "zone_agreement"] == pytest.approx(20 / 30)
    r = rates.set_index("group")
    assert r.loc["all", "chirps_per_window"] == pytest.approx((20 + 50) / 120)
    assert r.loc["rest", "chirps_per_minute"] == pytest.approx(0.5 * 30)


def test_fake_labels_refuse_the_session_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fake", "--root", str(d.ROOT), "--candidates", str(d.CANDIDATES),
                                      "--out", str(d.LABELS_OUT)])
    with pytest.raises(SystemExit, match="refusing"):
        fake_labels_2022.main()
