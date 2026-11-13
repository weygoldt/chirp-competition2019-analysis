import numpy as np


def save_data():
    chirppath = "data/interrim/new_chirpextractions/flattened_chirps.npy"
    heightspath = "data/interrim/new_chirpextractions/flattened_heights.npy"
    widthspath = "data/interrim/new_chirpextractions/flattened_widths.npy"
    outputpath = "data/processed/chirp_overview_plot_data.npz"

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
