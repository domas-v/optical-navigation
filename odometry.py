import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from src.frame import Frame
from src.intrinsics import CameraIntrinsics
from src.logs import FlightLogs
from src.tracker import VisualOdometry, make_mask
from src.video_helpers import VideoParameters
from src.visualizer import draw_overlay

WINDOW_NAME = "odometry"


def init_writer(params: VideoParameters) -> cv.VideoWriter:
    save_path = Path("output.mp4")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    size = (params.width, params.height)

    path = save_path.with_suffix(".mp4")
    fourcc = cv.VideoWriter.fourcc(*"avc1")
    writer = cv.VideoWriter(str(path), fourcc, params.fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {save_path}")
    return writer


def main(
    video_path: Path,
    log_path: Path,
    display: bool,
) -> None:
    logs = FlightLogs.from_path(log_path)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    params = VideoParameters.from_cap(cap)
    per_frame = logs.interpolate(params.num_frames)
    cap.release()

    # harcoded values for the camera
    # found by brute force sweep, minimizing
    # the error between the odometry and the gps.
    # I assumed that due to software stabilization and cropping,
    # the real FOV is different from the one given.
    hfov = 55
    vfov = 42.7

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    intrinsics = CameraIntrinsics.from_dimensions(params.width, params.height, hfov, vfov)

    mask = make_mask(params.width, params.height)
    vo = VisualOdometry(intrinsics, mask)

    gps_cum_per_frame = logs.gps_cumulative_at_frames(np.arange(params.num_frames))

    print(
        f"Video: {params.width}x{params.height} @ {params.fps:.2f} fps, {params.num_frames} frames"
    )
    print(f"Using: HFOV={hfov}° VFOV={vfov}°")

    if display:
        print("Displaying annotated video")
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    writer = init_writer(params)

    t_frame_idx: list[int] = []
    tx_px: list[float] = []
    ty_px: list[float] = []

    total_odom_m, last_processed_idx = 0.0, -1
    for frame_idx in range(params.num_frames):
        ok, raw = cap.read()
        if not ok:
            print(f"Failed to read frame {frame_idx}")
            break

        processed = vo.step(
            Frame(
                raw=raw,
                frame_idx=frame_idx,
                altitude_m=float(per_frame.altitude_m[frame_idx]),
                pitch_rad=float(per_frame.pitch_rad[frame_idx]),
                roll_rad=float(per_frame.roll_rad[frame_idx]),
                yaw_rad=float(per_frame.yaw_rad[frame_idx]),
            )
        )

        if not processed.skipped:
            t_frame_idx.append(frame_idx)
            tx_px.append(processed.tx_px)
            ty_px.append(processed.ty_px)

        total_odom_m += processed.distance_m
        last_processed_idx = frame_idx

        gps_so_far_m = float(gps_cum_per_frame[frame_idx])
        canvas = draw_overlay(processed, gps_so_far_m, total_odom_m, params.num_frames)
        writer.write(canvas)
        if display:
            cv.imshow(WINDOW_NAME, canvas)
            if cv.waitKey(1) == ord("q"):
                break
        elif not display and (frame_idx % 30 == 0 or frame_idx == params.num_frames - 1):
            print(f"\r{frame_idx + 1}/{params.num_frames}", end="", flush=True, file=sys.stderr)

    # save for plotting
    # with python plot_dist.py --log path/to/log.csv
    np.savez(
        "odometry.npz",
        frame_idx=np.asarray(t_frame_idx, dtype=int),
        tx_px=np.asarray(tx_px, dtype=float),
        ty_px=np.asarray(ty_px, dtype=float),
        width=params.width,
        height=params.height,
        hfov_deg=hfov,
        vfov_deg=vfov,
    )

    writer.release()
    cap.release()
    cv.destroyAllWindows()

    if last_processed_idx == params.num_frames - 1:
        gps_distance_m = logs.gps_distance_m()
    else:
        gps_distance_m = float(gps_cum_per_frame[max(last_processed_idx, 0)])

    print(f"Odometry distance: {total_odom_m:.2f} m")
    print(f"Real distance:     {gps_distance_m:.2f} m")
    print(f"Error:             {abs(total_odom_m - gps_distance_m) / gps_distance_m * 100:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to the log file",
    )
    parser.add_argument(
        "--no-display",
        help="Don't display the annotated video while processing",
        action="store_true",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    log_path = Path(args.log)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    main(video_path, log_path, not args.no_display)
