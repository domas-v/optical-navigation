"""Plot GPS vs odometry cumulative distance.
Loads ``odometry.npz`` (from ``odometry.py``)

Usage:
    python plot_dist.py --log path/to/log.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.logs import FlightLogs, haversine


NPZ_PATH = "odometry.npz"


def main(log_path: Path, npz_path: Path) -> None:
    logs = FlightLogs.from_path(log_path)
    if logs.latitude is None or logs.longitude is None:
        raise ValueError("Log must include latitude and longitude")

    frames = logs.frame

    lat, lon = logs.latitude, logs.longitude
    step_m = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum_m = np.concatenate([[0.0], np.cumsum(step_m)])
    total_m = logs.gps_distance_m()

    data = np.load(npz_path)
    tx_px = data["tx_px"]
    ty_px = data["ty_px"]
    width = int(data["width"])
    height = int(data["height"])
    hfov_deg = float(data["hfov_deg"])
    vfov_deg = float(data["vfov_deg"])
    frame_idx = data["frame_idx"]

    altitude_m = np.interp(frame_idx, logs.frame, logs.altitude_m)
    mpx = 2 * altitude_m * np.tan(np.radians(hfov_deg / 2)) / width
    mpy = 2 * altitude_m * np.tan(np.radians(vfov_deg / 2)) / height
    tx_m = tx_px * mpx
    ty_m = ty_px * mpy

    odom_step = np.hypot(tx_m, ty_m)  # per-frame magnitude
    odom_cum = np.cumsum(odom_step)  # cumulative path length

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames, cum_m, "g-", lw=2)
    ax.plot(frame_idx, odom_cum, "r-", lw=2)
    ax.scatter(
        [0],
        [0],
        c="black",
        marker="o",
        label="start (0, 0) m",
    )
    ax.scatter(
        [frames[-1]],
        [odom_cum[-1]],
        c="green",
        marker="s",
        label=f"end ({frames[-1]:.0f}, {odom_cum[-1]:.0f}) m",
    )
    ax.set_xlabel("frame idx")
    ax.set_ylabel("m")
    ax.set_title("Cumulative GPS distance")
    ax.grid()

    fig.suptitle(f"GPS total path ≈ {total_m:.0f} m")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPS vs odometry from odometry.npz")
    parser.add_argument("--log", type=str, required=True, help="Flight log")
    parser.add_argument("--npz", type=str, required=True, help="NPZ file")
    args = parser.parse_args()

    log_path = Path(args.log)
    npz_path = Path(args.npz)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    main(log_path, npz_path)
