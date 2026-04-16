import cv2
from collections.abc import Generator
import numpy as np


class VideoReader:
    def __init__(self, stride: int, video_path: str):
        self.video_path: str = video_path
        self.stride: int = stride

        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video at {self.video_path}")

    def extract_frames(self) -> Generator[np.ndarray, None, None]:
        frame_count = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success:
                break

            if frame_count % self.stride == 0:
                yield frame

            frame_count += 1

        self.cap.release()
