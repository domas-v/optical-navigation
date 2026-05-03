import cv2 as cv
import numpy as np

from src.frame import Frame
from src.intrinsics import CameraIntrinsics


def make_mask(width: int, height: int) -> np.ndarray:
    """The IDLE xx:yy is in the top left corner of the image.
    Mask it out."""

    mask = np.full((height, width), 255, dtype=np.uint8)
    mx = int(width * 0.20)
    my = int(height * 0.20)
    mask[:my, :mx] = 0
    return mask


def warp_mask(
    mask: np.ndarray,
    H: np.ndarray,
    dsize: tuple[int, int],
    erode_px: int = 20,
) -> np.ndarray:
    """Warp `mask` through the same homography as the frame, then erode the
    boundary."""

    warped = cv.warpPerspective(mask, H, dsize, flags=cv.INTER_NEAREST)
    if erode_px > 0:
        k = 2 * erode_px + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        warped = cv.erode(warped, kernel)
    return warped


def grid_detect(
    gray: np.ndarray,
    mask: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    per_cell: int = 25,
    quality_level: float = 0.005,
    min_distance: int = 10,
    block_size: int = 7,
) -> np.ndarray | None:
    """Detect `goodFeaturesToTrack` corners per grid cell."""

    h, w = gray.shape
    pts: list[np.ndarray] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0, y1 = row * h // grid_rows, (row + 1) * h // grid_rows
            x0, x1 = col * w // grid_cols, (col + 1) * w // grid_cols
            cell_mask = np.zeros_like(mask)
            cell_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            if not cell_mask.any():
                continue
            cell_pts = cv.goodFeaturesToTrack(
                gray,
                mask=cell_mask,
                maxCorners=per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance,
                blockSize=block_size,
            )
            if cell_pts is not None:
                pts.append(cell_pts)
    if not pts:
        return None
    return np.concatenate(pts)


def build_rotation_matrix(pitch: float, roll: float) -> np.ndarray:
    # only rectifying pitch and roll, not yaw
    # as yaw orientatio is magnetic, I assumed
    # that it is with the respect to the ground
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R_y = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return R_x @ R_y


class VisualOdometry:
    """Owns the inter-frame state (`_prev`, `_frames_since_redetect`)"""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        mask: np.ndarray,
        *,
        redetect_every: int = 30,
        min_pts: int = 50,
        min_tracked: int = 30,
        erode_px: int = 20,
        grid_rows: int = 4,
        grid_cols: int = 4,
        per_cell: int = 25,
        lk_win_size: tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
    ) -> None:

        self.intrinsics: CameraIntrinsics = intrinsics
        self.mask: np.ndarray = mask
        self.redetect_every: int = redetect_every
        self.min_pts: int = min_pts
        self.min_tracked: int = min_tracked
        self.erode_px: int = erode_px
        self.grid_rows: int = grid_rows
        self.grid_cols: int = grid_cols
        self.per_cell: int = per_cell
        self.lk_win_size: tuple[int, int] = lk_win_size
        self.lk_max_level: int = lk_max_level

        self._prev: Frame | None = None
        self._frames_since_redetect: int = 0

    def step(self, frame: Frame) -> Frame:
        self._rectify(frame)
        self._build_mask(frame)

        if self._prev is None:
            self._detect_features(frame)
            self._prev = frame
            return frame

        if self._needs_redetect():
            self._detect_features(self._prev)

        self._track_features(frame)
        self._compute_distance(frame)

        self._prev = frame
        self._frames_since_redetect += 1

        return frame

    def _rectify(self, frame: Frame) -> None:
        R = build_rotation_matrix(-frame.pitch_rad, -frame.roll_rad)
        H = self.intrinsics.K @ R @ self.intrinsics.K_inv
        h, w = frame.raw.shape[:2]
        rectified = cv.warpPerspective(frame.raw, H, (w, h))

        frame.H = H
        frame.rectified_gray = cv.cvtColor(rectified, cv.COLOR_BGR2GRAY)

    def _build_mask(self, frame: Frame) -> None:
        assert frame.H is not None
        h, w = frame.raw.shape[:2]
        frame.mask = warp_mask(self.mask, frame.H, (w, h), self.erode_px)

    def _needs_redetect(self) -> bool:
        assert self._prev is not None
        if self._prev.pts is None or len(self._prev.pts) < self.min_pts:
            return True
        if self._frames_since_redetect >= self.redetect_every:
            return True
        return False

    def _detect_features(self, frame: Frame) -> None:
        assert frame.rectified_gray is not None and frame.mask is not None

        frame.pts = grid_detect(
            frame.rectified_gray,
            frame.mask,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            per_cell=self.per_cell,
        )

        self._frames_since_redetect = 0

    def _track_features(self, frame: Frame) -> None:
        assert self._prev is not None
        assert self._prev.rectified_gray is not None and frame.rectified_gray is not None

        if self._prev.pts is None or len(self._prev.pts) == 0:
            frame.skipped = True
            return

        next_pts, status, _err = cv.calcOpticalFlowPyrLK(
            self._prev.rectified_gray,
            frame.rectified_gray,
            self._prev.pts,
            None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
        )

        if next_pts is None or status is None:
            frame.skipped = True
            frame.pts = None
            frame.prev_pts = None
            return

        ok = status.flatten() == 1
        good_old = self._prev.pts[ok]
        good_new = next_pts[ok]

        if len(good_new) < self.min_tracked:
            frame.skipped = True
            frame.pts = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
            frame.prev_pts = good_old.reshape(-1, 1, 2) if len(good_old) > 0 else None
            return

        disp = np.median(good_new - good_old, axis=0).flatten()
        frame.tx_px = float(disp[0])
        frame.ty_px = float(disp[1])
        frame.pts = good_new.reshape(-1, 1, 2)
        frame.prev_pts = good_old.reshape(-1, 1, 2)

    def _compute_distance(self, frame: Frame) -> None:
        if frame.skipped:
            return
        h, w = frame.raw.shape[:2]
        mpp_x, mpp_y = self.intrinsics.mpp(frame.altitude_m, w, h)
        frame.distance_m = float(np.hypot(frame.tx_px * mpp_x, frame.ty_px * mpp_y))
