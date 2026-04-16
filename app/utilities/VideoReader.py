import cv2
from collections.abc import Generator
import numpy as np


class VideoReader:
    def __init__(self, stride: int, video_path: str):
        self.video_path: str = video_path
        self.stride: int = stride

        # Open the video file
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"🚨 Error: Could not open video at {self.video_path}")

    def extract_frames(self) -> Generator[np.ndarray, None, None]:
        """Reads the video and yields frames one by one based on the stride."""
        frame_count = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()

            # If we reach the end of the video, break the loop
            if not success:
                break

            # Only yield the frame if it matches our stride (e.g., every 2nd frame)
            if frame_count % self.stride == 0:
                yield frame

            frame_count += 1

        # Clean up when the video finishes
        self.cap.release()
