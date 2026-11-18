"""Does the paper's BORIS -> grid converter carry the LED misalignment that hz_data repairs?

New in the paper port (a check, not part of the hazard models). Every paper
script that uses chases or contacts converts BORIS video times to grid time
with ``analyze_event_time_rates_trial.load_and_convert_boris_events``: frame =
round(t_video * FPS), then a line through the first and the last LED
(``converte_video_frames_to_grid_idx``). ``hz_data.fit_led_alignment`` pairs
video and grid LEDs by the gap fingerprint of the LED train and fits the map
piecewise; in 2020-05-13 and 2020-06-09 the first/last-LED line is wrong (a
grid LED is missing from the video, or the video starts LEDs late; see the
hazard README, Data contradictions 4).

For every chase onset, chase offset and contact of every trial with BORIS
and LED files, this converts the same video time with both maps and reports
the difference (paper minus hazard data layer, s). The LED residual of each
map (median distance of the mapped video LEDs to the nearest grid LED) shows
which map is right. Chases are paired as the paper's ``load_boris`` pairs them.

Outputs in ``$PAPER_OUT/data/processed/chirp_chase_hazard/led_check/``:
``led_check_events.csv`` (one row per event), ``led_check_trials.csv`` (one
row per trial), and the numbers in ``write_stats("boris_led_check")``.

    paper/run.sh hazard/hz_led_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/scripts
import hz_data  # noqa: E402
from analyze_event_time_rates_trial import (  # noqa: E402
    bad_recordings,
    load_and_convert_boris_events,
    load_boris,
)
from hz_config import Config  # noqa: E402
from manuscript_stats import write_stats  # noqa: E402
from paper_config import DATA_ROOT, processed_dir  # noqa: E402

SR = 20_000  # grid sampling rate (Hz), as the paper scripts pass it
WINDOW_S = 3 * 3600.0  # the hazard analysis window (first 3 h)


def trial_events(rec_dir: Path, rec: str, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Both conversions of every BORIS event of one trial, and the LED residuals."""
    contact_t, ag_on_off_t, led_idx, led_frames = load_and_convert_boris_events(rec_dir, rec, SR)
    _, _, t_ag_on_off, t_contact, fps = load_boris(rec_dir, rec)
    align = hz_data.fit_led_alignment(led_frames, led_idx, SR, cfg)
    t_ag_on_off = np.asarray(t_ag_on_off, float).reshape(-1, 2)
    paper_on_off = np.asarray(ag_on_off_t, float).reshape(-1, 2)
    parts = []
    for event, video, paper in (("onset", t_ag_on_off[:, 0], paper_on_off[:, 0]),
                                ("offset", t_ag_on_off[:, 1], paper_on_off[:, 1]),
                                ("contact", np.asarray(t_contact, float), np.asarray(contact_t))):
        hazard = align.to_grid(np.round(video * fps))
        parts.append(pd.DataFrame(dict(recording=rec, event=event, t_video_s=video,
                                       t_paper_s=paper, t_hazard_s=hazard,
                                       diff_s=paper - hazard)))
    led = dict(ref_led_resid_med_ms=1e3 * align.ref_resid_med,
               robust_led_resid_med_ms=1e3 * align.resid_med, led_method=align.method)
    return pd.concat(parts, ignore_index=True), led


def summarize(ev: pd.DataFrame, led: dict, rec: str, analysed: bool) -> dict:
    a = ev["diff_s"].abs()
    w = ev["t_hazard_s"] < WINDOW_S
    big = ev[w & (a > 0.5)]
    return dict(
        recording=rec, analysed_by_paper=analysed, n_events=len(ev), n_events_3h=int(w.sum()),
        max_abs_s=a.max(), median_abs_s=a.median(), max_abs_3h_s=a[w].max(),
        median_abs_3h_s=a[w].median(), median_diff_3h_s=ev.loc[w, "diff_s"].median(),
        n_gt_0_1s_3h=int((w & (a > 0.1)).sum()),
        n_gt_0_5s_3h=len(big), first_gt_0_5s_t=big["t_hazard_s"].min() if len(big) else np.nan,
        **led,
    )


def main() -> None:
    cfg = Config()
    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    events, rows = [], []
    for r in ts.itertuples():
        d = DATA_ROOT / r.recording
        if not (hz_data.boris_file(d, r.recording).exists() and (d / "LED_frames.npy").exists()
                and (d / "led_idxs.csv").exists()):
            continue
        ev, led = trial_events(d, r.recording, cfg)
        # the paper's is_good_trial for analyses with BORIS events
        analysed = r.group >= 5 and r.draw != 1 and r.recording not in bad_recordings
        events.append(ev)
        rows.append(summarize(ev, led, r.recording, analysed))
    ev, tab = pd.concat(events, ignore_index=True), pd.DataFrame(rows)
    out = processed_dir("chirp_chase_hazard", "led_check")
    ev.to_csv(out / "led_check_events.csv", index=False)
    tab.to_csv(out / "led_check_trials.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.drop(columns=["led_method"]).round(3).to_string(index=False))

    def num(row: pd.Series) -> dict:
        keys = ("n_events", "n_events_3h", "max_abs_s", "median_abs_s", "max_abs_3h_s",
                "median_abs_3h_s", "median_diff_3h_s", "n_gt_0_1s_3h", "n_gt_0_5s_3h",
                "first_gt_0_5s_t", "ref_led_resid_med_ms", "robust_led_resid_med_ms")
        d = {k: float(row[k]) for k in keys if np.isfinite(row[k])}
        d["analysed_by_paper"] = int(bool(row["analysed_by_paper"]))
        return d

    off = tab[tab["max_abs_3h_s"] > 0.1]
    write_stats("boris_led_check", dict(
        n_trials=len(tab), n_trials_analysed=int(tab["analysed_by_paper"].sum()),
        n_trials_max_gt_0_1s_3h=len(off),
        max_abs_3h_s_other_trials=tab.loc[tab["max_abs_3h_s"] <= 0.1, "max_abs_3h_s"].max(),
        trials={"_".join(row.recording.split("-")[:3]): num(row) for _, row in tab.iterrows()},
    ))
    print(f"trials with a conversion difference > 0.1 s in the first 3 h: "
          f"{', '.join(off['recording']) or 'none'}")


if __name__ == "__main__":
    main()
