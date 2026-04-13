from dataclasses import dataclass
from ultralytics.models import YOLO
import numpy as np
from utilities.VideoProcessor import VideoProcessor
import cv2


@dataclass
class Object:
    masks: np.ndarray | None
    box: np.ndarray
    confidence: float
    class_identifier: int
    label_name: str


class YoloSegmentator:
    def __init__(
        self, model_path: str, video_path: str, initial_frame_count: int
    ) -> None:
        self.__model_path = model_path
        self.__video_path = video_path
        self.__frame_count = initial_frame_count
        self.__model = YOLO(model_path)
        self.processor = VideoProcessor()

    def get_frame_detections(self, frame: np.ndarray) -> list[Object]:
        results = self.__model(frame, verbose=False)
        detections_list = []
        result = results[0]

        if result.masks is not None:
            # frame.shape returns (height, width, channels)
            # For 1080p: height=1080, width=1920
            frame_height, frame_width = frame.shape[:2]

            for index in range(len(result.masks)):
                class_identifier = int(result.boxes.cls[index])

                # binary_mask.shape is usually (640, 640) from the model
                binary_mask = result.masks.data[index].cpu().numpy()

                # OpenCV resize takes (WIDTH, HEIGHT)
                # This must be (1920, 1080) for a standard 1080p frame
                resized_mask = cv2.resize(
                    binary_mask,
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST,
                )

                detection_data = Object(
                    masks=resized_mask,
                    box=result.boxes.xyxy[index].cpu().numpy(),
                    confidence=float(result.boxes.conf[index]),
                    class_identifier=class_identifier,
                    label_name=self.__model.names[class_identifier],
                )
                detections_list.append(detection_data)

        return detections_list

    def isolate_object(
        self, frame: np.ndarray, detections: list[Object], target_name: list[str]
    ) -> np.ndarray:
        self.processor.start_annotator(frame)

        for detection in detections:
            if detection.label_name.lower() in target_name:
                # Layer 1: Mask
                self.processor.draw_mask(
                    masks=detection.masks[None],
                    class_identifier=detection.class_identifier,
                )

                # Layer 2: Bounding Box
                self.processor.draw_bounding_box(
                    box_coordinates=detection.box,
                    class_identifier=detection.class_identifier,
                )

                # Layer 3: Label
                label_text = f"{detection.label_name} {detection.confidence:.2f}"
                self.processor.draw_label(
                    box_coordinates=detection.box,
                    display_text=label_text,
                    class_identifier=detection.class_identifier,
                )

        return self.processor.get_final_frame()

    @property
    def model_path(self) -> str:
        return self.__model_path

    @model_path.setter
    def model_path(self, path: str) -> None:
        self.__model_path = path
        self.__model = YOLO(path)

    @property
    def frame_count(self) -> int:
        return self.__frame_count

    @frame_count.setter
    def frame_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("Frame count cannot be negative.")
        self.__frame_count = value
