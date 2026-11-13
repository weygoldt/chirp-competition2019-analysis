# New script (rebuilds an input of chirp_competition d040123 whose writer is not in git).
# Changes: none to logic; it re-creates data/processed/competition_overview_<rec>.json, the input of
# plot_competition_trial_overview.py, following src/summarize_trial_summary.py (make_nice_example_plot).
"""Collect the data of the example trial of Fig. 2 into one JSON file.

The old ``data/processed/competition_overview_2020-06-09-10_00.json`` was
written by a variant of ``make_nice_example_plot`` in
``src/summarize_trial_summary.py`` (its JSON dump is commented out there;
the writer itself was never committed). This script rebuilds the same 12
keys, with the same structure and types, from the trial folder:

- ``fund_v``, ``ident_v``, ``idx_v``, ``times``, ``spec``: the wavetracker
  files of the recording;
- ``win_id``, ``lose_id``: ``win_ID`` / ``lose_ID`` of ``trial_summary.csv``;
- ``contact_t_GRID``, ``ag_on_off_t_GRID``: BORIS contacts and chases in grid
  time (``load_and_convert_boris_events``);
- ``rise_idx_int``: ``analysis/rise_idx.npy`` ordered [winner, loser] as in
  ``summarize_trial_summary.py``;
- ``chirp_times``: [winner, loser] chirp times of the detections named by
  ``PAPER_DETECTIONS`` (``chirps.load_chirps``); the old file held the 2023
  ``chirp_times_cnn.npy`` detections.

Output: ``$PAPER_OUT/data/processed/competition_overview_<recording>.json``.
"""

import argparse
import json

import numpy as np
import pandas as pd

from analyze_event_time_rates_trial import load_and_convert_boris_events
from chirps import load_chirps
from paper_config import DATA_ROOT, DETECTIONS, processed

SR = 20_000


def collect(recording: str) -> dict:
    trial_path = DATA_ROOT / recording
    summary = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    row = summary[summary["recording"] == recording]
    if len(row) != 1:
        raise ValueError(f"{recording}: {len(row)} rows in trial_summary.csv")
    row = row.iloc[0]
    win_id, lose_id = row["win_ID"], row["lose_ID"]

    fund_v = np.load(trial_path / "fund_v.npy")
    ident_v = np.load(trial_path / "ident_v.npy")
    idx_v = np.load(trial_path / "idx_v.npy")
    times = np.load(trial_path / "times.npy")
    spec = np.load(trial_path / "spec.npy")

    # rises ordered [winner, loser] (summarize_trial_summary.py)
    ids = np.load(trial_path / "analysis" / "ids.npy")
    sorter = -1 if win_id != ids[0] else 1
    rise_idx = np.load(trial_path / "analysis" / "rise_idx.npy")[::sorter]
    rise_idx_int = [
        np.array(rise_idx[i][~np.isnan(rise_idx[i])], dtype=int)
        for i in range(len(rise_idx))
    ]

    contact_t_GRID, ag_on_off_t_GRID, _, _ = load_and_convert_boris_events(
        trial_path, recording, SR
    )

    chirp_t, chirp_ids = load_chirps(trial_path)
    chirp_times = [chirp_t[chirp_ids == win_id], chirp_t[chirp_ids == lose_id]]

    plotdata = {
        "fund_v": fund_v,
        "ident_v": ident_v,
        "idx_v": idx_v,
        "times": times,
        "spec": spec,
        "win_id": win_id,
        "lose_id": lose_id,
        "contact_t_GRID": contact_t_GRID,
        "ag_on_off_t_GRID": ag_on_off_t_GRID,
        "rise_idx_int": rise_idx_int,
        "chirp_times": chirp_times,
        "recording": recording,
    }
    # make everything into a list (as in make_nice_example_plot)
    for key in plotdata:
        if isinstance(plotdata[key], np.ndarray):
            plotdata[key] = plotdata[key].tolist()
        elif isinstance(plotdata[key], list):
            for i in range(len(plotdata[key])):
                if isinstance(plotdata[key][i], np.ndarray):
                    plotdata[key][i] = plotdata[key][i].tolist()
        elif isinstance(plotdata[key], np.generic):
            plotdata[key] = plotdata[key].item()
    return plotdata


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recording", default="2020-06-09-10_00")
    args = parser.parse_args()

    plotdata = collect(args.recording)
    out = processed(f"competition_overview_{args.recording}.json")
    with open(out, "w") as f:
        json.dump(plotdata, f)
    print(f"win_id {plotdata['win_id']}, lose_id {plotdata['lose_id']}")
    print(
        f"chirps ({DETECTIONS}): winner {len(plotdata['chirp_times'][0])}, "
        f"loser {len(plotdata['chirp_times'][1])}"
    )
    print(
        f"chases {len(plotdata['ag_on_off_t_GRID'])}, "
        f"contacts {len(plotdata['contact_t_GRID'])}, "
        f"rises winner {len(plotdata['rise_idx_int'][0])}, "
        f"loser {len(plotdata['rise_idx_int'][1])}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
