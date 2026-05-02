from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """Just a container for per-frame data."""

    raw: np.ndarray
    frame_idx: int
    altitude_m: float
    pitch_rad: float
    roll_rad: float
    yaw_rad: float

    # these fields are populated by VisualOdometry.step():
    rectified_gray: np.ndarray | None = None
    H: np.ndarray | None = None
    mask: np.ndarray | None = None
    pts: np.ndarray | None = None
    prev_pts: np.ndarray | None = None
    tx_px: float = 0.0
    ty_px: float = 0.0
    distance_m: float = 0.0
    skipped: bool = False
