from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.consts import AVG_EARTH_RADIUS, FRAME, FT_TO_M, HEIGHT, LAT, LON, PITCH, ROLL, YAW


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(a))


@dataclass
class FlightLogs:
    frame: np.ndarray
    altitude_m: np.ndarray
    pitch_rad: np.ndarray
    roll_rad: np.ndarray
    yaw_rad: np.ndarray
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None

    @classmethod
    def from_path(cls, log_path: Path) -> "FlightLogs":
        data = np.genfromtxt(
            log_path,
            delimiter=",",
            names=True,
            encoding="utf-8",
        )
        return cls(
            frame=data[FRAME].astype(int),
            altitude_m=data[HEIGHT] * FT_TO_M,
            pitch_rad=np.radians(data[PITCH]),
            roll_rad=np.radians(data[ROLL]),
            yaw_rad=np.unwrap(np.radians(data[YAW])),
            latitude=data[LAT],
            longitude=data[LON],
        )

    def interpolate(self, num_frames: int) -> "FlightLogs":
        target = np.arange(num_frames, dtype=int)
        interpolate = lambda v: np.interp(target, self.frame, v)
        return FlightLogs(
            frame=target,
            altitude_m=interpolate(self.altitude_m),
            pitch_rad=interpolate(self.pitch_rad),
            roll_rad=interpolate(self.roll_rad),
            yaw_rad=interpolate(self.yaw_rad),
        )

    def gps_distance_m(self) -> float:
        if self.latitude is None or self.longitude is None:
            raise ValueError("Should not be called on interpolated logs")

        lat, lon = self.latitude, self.longitude
        return float(haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())

    def gps_cumulative_at_frames(self, frame_indices: np.ndarray) -> np.ndarray:
        if self.latitude is None or self.longitude is None:
            raise ValueError("Should not be called on interpolated logs")
        per_log_step = haversine(
            self.latitude[:-1],
            self.longitude[:-1],
            self.latitude[1:],
            self.longitude[1:],
        )
        cumulative_at_log_frames = np.concatenate([[0.0], np.cumsum(per_log_step)])
        return np.interp(frame_indices, self.frame, cumulative_at_log_frames)
