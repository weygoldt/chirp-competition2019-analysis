# Ported from chirp_competition d040123 (src/collect_chirp_overview_plot_data.py).
# Changes: inputs read from paper/inputs/new_chirpextractions (verbatim
# copies of the old data/interrim arrays, see README.md there); output under PROCESSED.
from pathlib import Path

import numpy as np

from paper_config import processed

INPUTS = Path(__file__).resolve().parents[1] / "inputs" / "new_chirpextractions"


def save_data():
    chirppath = INPUTS / "flattened_chirps.npy"
    heightspath = INPUTS / "flattened_heights.npy"
    widthspath = INPUTS / "flattened_widths.npy"
    outputpath = processed("chirp_overview_plot_data.npz")

    samplerate = 20000
    chirps = np.load(chirppath)
    heights = np.load(heightspath)
    widths = np.load(widthspath)
    troughs = np.min(chirps, axis=1)
    time = np.linspace(0, len(chirps[0]) / samplerate, len(chirps[0]))
    time = time * 1000
    time = time - time[len(time) // 2]

    print("Chirps shape: ", chirps.shape)
    print("Heights shape: ", heights.shape)
    print("Widths shape: ", widths.shape)
    print("Troughs shape: ", troughs.shape)
    print("Time shape: ", time.shape)

    output = {
        "chirps": chirps,
        "heights": heights,
        "widths": widths,
        "troughs": troughs,
        "time": time,
    }
    np.savez(outputpath, **output)


if __name__ == "__main__":
    save_data()
