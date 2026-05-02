from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Assumptions:
    1. Pinhole camera model.
    2. Principal point is at the center of the image.
    """

    K: np.ndarray
    K_inv: np.ndarray
    hfov_rad: float
    vfov_rad: float
    f_x: float
    f_y: float

    @classmethod
    def from_dimensions(
        cls, width: int, height: int, hfov_deg: float, vfov_deg: float
    ) -> "CameraIntrinsics":
        hfov_rad = float(np.radians(hfov_deg))
        vfov_rad = float(np.radians(vfov_deg))
        f_x = width / (2 * np.tan(hfov_rad / 2))
        f_y = height / (2 * np.tan(vfov_rad / 2))
        K = np.array(
            [
                [f_x, 0, width / 2],
                [0, f_y, height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        K_inv = np.linalg.inv(K)
        return cls(K=K, K_inv=K_inv, hfov_rad=hfov_rad, vfov_rad=vfov_rad, f_x=f_x, f_y=f_y)

    def mpp(self, altitude_m: float, width_px: int, height_px: int) -> tuple[float, float]:
        mpp_x = (2 * altitude_m * np.tan(self.hfov_rad / 2)) / width_px
        mpp_y = (2 * altitude_m * np.tan(self.vfov_rad / 2)) / height_px
        return float(mpp_x), float(mpp_y)
