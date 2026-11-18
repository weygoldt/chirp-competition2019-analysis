# Ported from chirp_competition b65108e (src_hazard/hz_config.py).
# Changes: data root, output directory and chirp detections from paper_config
# (CHIRP_REC_2019, $PAPER_OUT/data/processed/chirp_chase_hazard, PAPER_DETECTIONS).
"""Configuration for the event-level chirp -> chase hazard analysis.

All times are in seconds (grid/recording time), all frequencies in Hz.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/scripts
import paper_config  # noqa: E402

DEFAULT_DATA_ROOT = paper_config.DATA_ROOT
DEFAULT_OUT_DIR = paper_config.PROCESSED / "chirp_chase_hazard"

# Recordings where the two EODfs are so close that chirp-to-sender assignment is
# unreliable (inherited from src/analyze_event_time_rates_trial.py).
BAD_RECORDINGS: tuple[str, ...] = (
    "2020-05-21-10_00",
    "2020-05-25-10_00",
    "2020-05-28-10_00",
    "2020-06-10-10_00",
)


@dataclass
class Config:
    """Analysis configuration shared by all hz_* modules."""

    data_root: Path = DEFAULT_DATA_ROOT
    out_dir: Path = DEFAULT_OUT_DIR

    # --- binning and history bases -------------------------------------------
    bin_width: float = 0.05  # s
    hist_max: float = 2.0  # s, extent of raised-cosine history basis
    n_basis: int = 5  # number of raised-cosine basis functions
    tau_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)  # s

    # --- trial selection -----------------------------------------------------
    exclude_draws: bool = True
    exclude_bad_recordings: bool = True
    bad_recordings: tuple[str, ...] = BAD_RECORDINGS
    # Analysis window end in grid seconds from recording start (None = whole
    # recording). Default 3 h, the 'active period' of earlier analyses. The
    # light schedule is not recorded in the dataset, but the data support the
    # cut: in included trials chase onsets drop to 0-6 in 3.0-3.5 h and none
    # follow (last BORIS event at 2.95-3.52 h; the draw 2020-06-08 is annotated
    # to the end), while some subordinates keep chirping, so later bins would
    # mix a different behavioural regime into the onset risk set.
    max_time: float | None = 3 * 3600.0
    use_rises: bool = False  # rise history covariates (analysis/rise_idx.npy)

    # --- model fitting / resampling -----------------------------------------
    onset_control_fraction: float = 1.0
    n_jitter: int = 200
    n_sims: int = 50
    seed: int = 42
    quick: bool = False

    # --- data layer specifics -----------------------------------------------
    sr: float = 20_000.0  # grid sampling rate, Hz (fishgrid.cfg AISampleRate)
    # rtdetr*: chirps_<det>.parquet (rows with keep); else chirp_times/ids_<det>.npy
    chirp_detector: str = paper_config.DETECTIONS
    # BORIS video-time -> grid-time mapping: "robust" (anchor search + least
    # squares on nearest-LED inliers) or "reference" (first/last LED as in
    # src/analyze_event_time_rates_trial.py).
    led_alignment: str = "robust"
    led_match_tol: float = 1.0  # s, max |residual| to pair a video LED with a grid LED
    led_inlier_tol: float = 0.1  # s, max |residual| for a pair to enter the fit
    led_step_tol: float = 0.1  # s, residual step that splits the map (frame drop)
    led_max_anchor_shift: int = 3  # LEDs dropped at either end in anchor search
    led_bad_resid: float = 0.05  # s, median residual above which a mapping is flagged
    # Same-sender detections closer than this are merged (first kept, flagged).
    chirp_dedup_window: float = 0.05  # s
    # Dom/sub chirps closer than this are flagged as possible double assignment.
    cross_sender_window: float = 0.05  # s
    eodf_smooth_window: float = 60.0  # s, rolling median for baseline EODf
    eodf_grid_dt: float = 1.0  # s, regular grid for baseline EODf
    eodf_max_gap: float = 60.0  # s, track gaps longer than this are flagged
    eodf_min_coverage: float = 0.8  # fraction of spectrogram frames with a value
    # Flag (not exclude) trials outside bad_recordings whose baseline EODfs are
    # closer than close_eodf_hz for more than close_eodf_frac of the window.
    close_eodf_hz: float = 30.0
    close_eodf_frac: float = 0.25
    max_plausible_chase: float = 120.0  # s, longer chases are flagged (kept)
    # If True, the analysis window also ends at the last annotated BORIS event.
    clip_to_last_annotation: bool = False

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.out_dir = Path(self.out_dir)
        self.tau_grid = tuple(float(x) for x in self.tau_grid)
        self.bad_recordings = tuple(self.bad_recordings)
        if self.max_time is not None and self.max_time <= 0:
            self.max_time = None  # max_time <= 0 disables the cap
        if self.bin_width <= 0:
            raise ValueError("bin_width must be > 0")
        if self.hist_max < self.bin_width:
            raise ValueError("hist_max must be >= bin_width")
        if self.led_alignment not in ("robust", "reference"):
            raise ValueError("led_alignment must be 'robust' or 'reference'")

    def updated(self, **kwargs: Any) -> Config:
        """Return a copy with some fields replaced (None values are ignored)."""
        valid = {f.name for f in fields(self)}
        unknown = set(kwargs) - valid
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for logging (paths as str)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else v
        return out
