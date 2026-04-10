from dataclasses import dataclass
from ultralytics.models import YOLO
import numpy as np
import cv2
from typing import Any


@dataclass
class Object:
    polygon: np.ndarray
    box: np.ndarray
    conf: float
    class_id: int


class YoloSegmentator:
    def __init__(
        self, segmentator_path: str, video_path: str, frame_count: int
    ) -> None:
        self.__segmentator_path = segmentator_path # should be model path
        self.__video_path = video_path
        self.deneme = frame_count
        self.__model = YOLO(segmentator_path)

    @property
    def segmentator(self) -> str:
        return self.__segmentator_path

    @segmentator.setter
    def segmentator(self, path: str) -> None:
        self.__segmentator_path = path
        self.__model = YOLO(path)

    @property
    def video_path(self) -> str:
        return self.__video_path

    @video_path.setter
    def video_path(self, path: str) -> None:
        self.__video_path = path

    @property
    def frame_count(self) -> int:
        return self.deneme

    @frame_count.setter
    def frame_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("Frame count cannot be negative.")
        self.deneme = value

    def get_frame_detections(self, frame: np.ndarray) -> list[Object]:
        results = self.__model(frame, verbose=False)
        detections_list = []
        result = results[0]

        if result.masks is not None:
            for i in range(len(result.masks)):
                detection_data = Object(
                    polygon=result.masks.xy[i],
                    box=result.boxes.xyxy[i].cpu().numpy(),
                    conf=float(result.boxes.conf[i]),
                    class_id=int(result.boxes.cls[i]),
                )
                detections_list.append(detection_data)
        return detections_list

    def draw_masks(
        self, frame: np.ndarray, polygon: np.ndarray, color=(0, 255, 0)
    ) -> np.ndarray:
        overlay = frame.copy()
        points = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(overlay, points, color)
        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    def draw_bounding_box(
        self, frame: np.ndarray, box: np.ndarray, color=(0, 255, 0)
    ) -> None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def draw_label(
        self, frame: np.ndarray, box: np.ndarray, label: str, color=(0, 255, 0)
    ) -> None:
        x1, y1, _, _ = box.astype(int)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    def isolate_object( # do you really need ?
        self, frame: np.ndarray, detections: list[Object], target_name: str
    ) -> np.ndarray:
        target_id = None
        for k, v in self.__model.names.items():
            if v.lower() == target_name.lower():
                target_id = k
                break

        annotated_frame = frame.copy()

        for det in detections:
            if det.class_id == target_id:
                # 1. Apply Mask
                annotated_frame = self.draw_masks(annotated_frame, det.polygon)

                # 2. Draw Box
                self.draw_bounding_box(annotated_frame, det.box)

                # 3. Draw Label
                full_label = f"{target_name} {det.conf:.2f}"
                self.draw_label(annotated_frame, det.box, full_label)

        return annotated_frame
