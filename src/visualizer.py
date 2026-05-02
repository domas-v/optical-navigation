"""Live visualization of the visual-odometry pipeline.

Renders the rectified frame, tracked points, optical-flow vectors, and a
status overlay. The rectified (not raw) frame is shown because that is the
image LK is actually operating on — drawing tracked points in raw-pixel
space would put them in the wrong location.
"""

import cv2 as cv
import numpy as np

from src.frame import Frame

_FONT = cv.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_LINE_HEIGHT = 22


def draw_overlay(
    frame: Frame,
    gps_so_far_m: float,
    odom_so_far_m: float,
    num_frames: int,
) -> np.ndarray:
    """Render the tracking state and return a BGR image for `cv.imshow`."""
    assert frame.rectified_gray is not None
    canvas = cv.cvtColor(frame.rectified_gray, cv.COLOR_GRAY2BGR)

    if frame.prev_pts is not None and frame.pts is not None:
        n = min(len(frame.prev_pts), len(frame.pts))
        for i in range(n):
            p0 = tuple(frame.prev_pts[i, 0].astype(int))
            p1 = tuple(frame.pts[i, 0].astype(int))
            cv.line(canvas, p0, p1, (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(canvas, p1, 2, (0, 255, 255), -1, cv.LINE_AA)
    elif frame.pts is not None:
        for pt in frame.pts.reshape(-1, 2).astype(int):
            cv.circle(canvas, tuple(pt), 2, (0, 255, 255), -1, cv.LINE_AA)

    err_pct = (
        (odom_so_far_m - gps_so_far_m) / gps_so_far_m * 100 if gps_so_far_m > 0.5 else 0.0
    )
    n_tracked = len(frame.pts) if frame.pts is not None else 0
    status = "SKIPPED" if frame.skipped else f"tracked: {n_tracked}"
    pitch_deg = np.degrees(frame.pitch_rad)
    roll_deg = np.degrees(frame.roll_rad)

    left = [
        f"frame {frame.frame_idx + 1}/{num_frames}  {status}",
        f"alt {frame.altitude_m:6.1f} m   pitch {pitch_deg:+5.1f}   roll {roll_deg:+5.1f}",
        f"GPS  {gps_so_far_m:8.1f} m",
        f"odom {odom_so_far_m:8.1f} m",
        f"err  {err_pct:+7.2f} %",
    ]
    _draw_text_block(canvas, left, x=10, y=20)
    _draw_text_block(canvas, ["q: quit"], x=canvas.shape[1] - 130, y=20)

    return canvas


def _draw_text_block(canvas: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for i, line in enumerate(lines):
        py = y + i * _LINE_HEIGHT
        cv.putText(canvas, line, (x + 1, py + 1), _FONT, _FONT_SCALE, (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(canvas, line, (x, py), _FONT, _FONT_SCALE, (255, 255, 255), 1, cv.LINE_AA)
