"""The example chirps of Figs. 9 and S4, named by their chirp_id in pipeline v2.

The author chose these examples among the detections of pipeline v2
(``paper_config.PIPELINES["v2"]``, run competition2019_v2_20261113). Chirp
ids are stable across one pipeline's run variants (gap-fill, thresholds,
recalibration) but not between pipelines. ``example_id`` therefore maps a v2
id to the box of the selected run (PAPER_DETECTIONS) at the same place: the
id itself under pipeline v2, otherwise the stored row with the highest
time-frequency IoU with the v2 box (``V2_BOXES``, copied from the v2 table so
that no v2 data is needed) among the rows that play the example's role:

- ``"chirp"``: kept at the operating threshold (``chirp``);
- ``"harmonic"`` / ``"duplicate"``: removed by post-processing for that
  reason (``pp_drop_reason``), at the operating threshold;
- ``None``: any stored row (the below-threshold chirp of Fig. S4A; the figure
  checks its score itself).

``example_id`` raises if no such row overlaps the v2 box by at least
``MIN_IOU``, and lists the rows nearby: the example then has to be chosen
anew under the new pipeline (by the author).
"""

from __future__ import annotations

import pandas as pd

import paper_config

MIN_IOU = 0.3

# (recording, v2 chirp_id) -> (t_start_s, t_end_s, f_low_hz, f_high_hz, role) of
# the v2 box (competition2019_v2_20261113_recal1, chirps_rtdetr.parquet)
V2_BOXES: dict[tuple[str, int], tuple[float, float, float, float, str | None]] = {
    # Fig. 9 detection and assignment (snippet f1_detect)
    ("2020-06-02-10_00", 2003): (2247.5552, 2247.7998, 655.66, 1045.31, "chirp"),  # score 0.912
    ("2020-06-02-10_00", 2004): (2247.9204, 2248.1557, 724.33, 1254.57, "chirp"),  # score 0.962
    ("2020-06-02-10_00", 2014): (2252.9272, 2253.1642, 748.00, 962.58, "chirp"),  # score 0.958
    # Fig. 9 harmonic copy and decoder examples (snippet f1_tile)
    ("2020-05-29-10_00", 2084): (10850.9200, 10851.1657, 1260.58, 1501.46, "harmonic"),  # 0.841
    ("2020-05-29-10_00", 2085): (10850.9209, 10851.1660, 629.08, 764.17, "chirp"),  # score 0.952
    ("2020-05-29-10_00", 2086): (10852.4728, 10852.7727, 675.29, 1166.80, "chirp"),  # score 0.954
    # Fig. 9 duplicate (snippet f1_dup)
    ("2019-12-02-10_00", 244): (1196.9109, 1197.1563, 697.67, 802.48, "chirp"),  # score 0.907
    ("2019-12-02-10_00", 245): (1196.9576, 1197.1921, 688.09, 895.65, "duplicate"),  # score 0.868
    # Fig. S4A: the labelled chirp below the operating threshold (v2 score 0.325)
    ("2019-12-09-10_00", 4080): (4320.1829, 4320.4293, 707.39, 802.48, None),
    # Fig. S4 stats: assignments between close fish
    ("2019-12-09-10_00", 14783): (15388.4446, 15388.6905, 676.85, 847.80, "chirp"),  # 0.955
    ("2020-03-13-10_00", 2974): (9394.3242, 9394.5665, 643.70, 778.87, "chirp"),  # score 0.933
    ("2020-03-13-10_00", 2975): (9394.4020, 9394.6418, 648.63, 795.63, "chirp"),  # score 0.941
}


def box_iou(rows: pd.DataFrame, box: tuple[float, float, float, float]) -> pd.Series:
    """Time-frequency IoU of every row's box with ``box`` (t0, t1, f0, f1)."""
    t0, t1, f0, f1 = box
    dt = (rows.t_end_s.clip(upper=t1) - rows.t_start_s.clip(lower=t0)).clip(lower=0)
    df = (rows.f_high_hz.clip(upper=f1) - rows.f_low_hz.clip(lower=f0)).clip(lower=0)
    inter = dt * df
    union = ((rows.t_end_s - rows.t_start_s) * (rows.f_high_hz - rows.f_low_hz)
             + (t1 - t0) * (f1 - f0) - inter)
    return inter / union


def _role_mask(rows: pd.DataFrame, role: str | None) -> pd.Series:
    if role is None:
        return pd.Series(True, index=rows.index)
    if role == "chirp":
        return rows.chirp.astype(bool)
    return rows.operating.astype(bool) & (rows.pp_drop_reason == role)


def example_id(rows: pd.DataFrame, recording: str, v2_id: int) -> int:
    """The chirp_id in ``rows`` (stored rows of ``recording`` in the selected
    run) of the box that pipeline v2's chirp ``v2_id`` marks (module docstring)."""
    if paper_config.PIPELINE_NAME == "v2":
        if v2_id not in set(rows.chirp_id):
            msg = f"{recording}: v2 chirp {v2_id} not among the rows"
            raise KeyError(msg)
        return int(v2_id)
    *box, role = V2_BOXES[(recording, v2_id)]
    iou = box_iou(rows, tuple(box))
    ok = _role_mask(rows, role) & (iou >= MIN_IOU)
    if not ok.any():
        near = rows[iou > 0].assign(iou=iou[iou > 0]).sort_values("iou", ascending=False)
        cols = ["chirp_id", "t_center_s", "f_low_hz", "f_high_hz", "score", "operating",
                "chirp", "pp_drop_reason", "iou"]
        msg = (f"{recording}: no row of role {role!r} overlaps v2 chirp {v2_id} "
               f"(IoU >= {MIN_IOU}) in {paper_config.DETECTIONS}; rows nearby:\n"
               f"{near[cols].round(3).to_string() if len(near) else '(none)'}")
        raise LookupError(msg)
    best = rows[ok].assign(iou=iou[ok]).sort_values(["iou", "score"], ascending=False).iloc[0]
    return int(best.chirp_id)


def example_ids(rows: pd.DataFrame, recording: str, v2_ids) -> list[int]:
    """``example_id`` of several v2 ids; distinct ids required."""
    out = [example_id(rows, recording, i) for i in v2_ids]
    if len(set(out)) != len(out):
        msg = f"{recording}: v2 chirps {list(v2_ids)} map onto the same rows {out}"
        raise LookupError(msg)
    return out
