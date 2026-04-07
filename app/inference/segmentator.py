from ultralytics.models import YOLO
import numpy as np
import cv2
from typing import Any


class YoloSegmentator:
    def __init__(self, segmentator: str, videoPath: str, frameCount: int) -> None:
        self._segmentator_path = segmentator
        self._video_path = videoPath
        self._frame_count = frameCount
        # Initialize the model using the path
        self._model = YOLO(segmentator)

    # --- Getters and Setters for Segmentator Path ---
    @property
    def segmentator(self) -> str:
        return self._segmentator_path

    @segmentator.setter
    def segmentator(self, path: str) -> None:
        self._segmentator_path = path
        # Re-initialize the model if the path changes
        self._model = YOLO(path)

    # --- Getters and Setters for Video Path ---
    @property
    def video_path(self) -> str:
        return self._video_path

    @video_path.setter
    def video_path(self, path: str) -> None:
        self._video_path = path

    # --- Getters and Setters for Frame Count ---
    @property
    def frame_count(self) -> int:
        return self._frame_count

    @frame_count.setter
    def frame_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("Frame count cannot be negative.")
        self._frame_count = value

    # --- Core Methods ---
    def get_frame_detections(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self._model(frame, verbose=False)
        detections_list = []
        result = results[0]

        if result.masks is not None:
            for i in range(len(result.masks)):
                detection_data = {
                    "mask": result.masks.data[i],
                    "box": result.boxes.xyxy[i],
                    "class_id": int(result.boxes.cls[i]),
                }
                detections_list.append(detection_data)

        return detections_list

    def isolate_object(
        self, frame: np.ndarray, detections: list[dict], target_name: str
    ) -> np.ndarray:
        black_bg = np.zeros_like(frame)
        target_id = None

        # Accessing model names via the internal model
        for k, v in self._model.names.items():
            if v.lower() == target_name.lower():
                target_id = k
                break

        if target_id is None:
            return black_bg

        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for det in detections:
            if det["class_id"] == target_id:
                mask_data = det["mask"].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                combined_mask = cv2.bitwise_or(
                    combined_mask, (mask_resized > 0.5).astype(np.uint8)
                )

        return cv2.bitwise_and(frame, frame, mask=combined_mask)
