"""The frozen v1 production detector on simulated scenes (recording.wav).

Wraps the repository's scripts/v1_screen.py unchanged except for how the
recording is opened: the scenes are float32 WAVE_FORMAT_EXTENSIBLE files,
which the frozen environment's audioio cannot read (no soundfile), so
thunderlab's DataLoader is replaced by a memory-mapped scipy.io.wavfile
reader with the same interface (``samplerate``, ``shape``, ``[a:b, :]``,
``close``).
Detection itself (windows, spectrograms, model, NMS, peak check) is v1's.
Same arguments as v1_screen.py. Run with the v1 environment
(the frozen v1 environment of scripts/v1_screen.py) through scripts/guarded --vram 0.2.
"""

import importlib.util
import pathlib

import numpy as np
import thunderlab.dataloader
from scipy.io import wavfile


class WavLoader:
    """What v1_screen.run_recording uses of thunderlab's DataLoader."""

    def __init__(self, path: str) -> None:
        rate, data = wavfile.read(path, mmap=True)
        self.samplerate = float(rate)
        self._data = data if data.ndim == 2 else data[:, None]
        self.shape = self._data.shape

    def __getitem__(self, key):
        def whole(k):  # v1 slices with float sample indices
            if isinstance(k, slice):
                return slice(
                    None if k.start is None else int(k.start),
                    None if k.stop is None else int(k.stop),
                    k.step,
                )
            return k

        key = tuple(whole(k) for k in key) if isinstance(key, tuple) else whole(key)
        return np.asarray(self._data[key], dtype=np.float64)

    def close(self) -> None:
        self._data = None


thunderlab.dataloader.DataLoader = WavLoader
# the repository's scripts/v1_screen.py (this file is experiments/benchmark/run_v1_scenes.py)
SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "v1_screen.py"
spec = importlib.util.spec_from_file_location("v1_screen", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.RAW_FILE = "recording.wav"
module.main()
