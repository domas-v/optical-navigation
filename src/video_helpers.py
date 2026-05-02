from dataclasses import dataclass

import cv2 as cv


@dataclass
class VideoParameters:
    num_frames: int
    fps: float
    width: int
    height: int

    @staticmethod
    def from_cap(cap: cv.VideoCapture) -> "VideoParameters":
        return VideoParameters(
            num_frames=int(cap.get(cv.CAP_PROP_FRAME_COUNT)),
            fps=cap.get(cv.CAP_PROP_FPS),
            width=int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
