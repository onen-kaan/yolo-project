from dataclasses import dataclass
from ultralytics.models import YOLO
import numpy as np
from app.utilities.VideoProcessor import VideoProcessor
import app.utilities.utils as utils
import cv2
import time


@dataclass
class Object:
    masks: np.ndarray | None
    box: np.ndarray
    confidence: float
    class_identifier: int
    label_name: str


class YoloSegmentator:
    def __init__(self, config_path: str) -> None:
        self.config = config_path
        self.model = YOLO(utils.get_from_config(self.config, "model"))
        self.processor = VideoProcessor()

    def segment(self) -> None:
        # 1. Setup variables from config
        video_path = utils.get_from_config(self.config, "data")
        stride = utils.get_from_config(self.config, "frame_count")
        targets = utils.get_from_config(self.config, "target_class")

        results = self.model(video_path, stream=True, vid_stride=stride)

        for i, result in enumerate(results):
            detections = (
                [
                    Object(
                        masks=cv2.resize(
                            m.cpu().numpy(),
                            (result.orig_shape[1], result.orig_shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ),
                        box=result.boxes.xyxy[idx].cpu().numpy(),
                        confidence=float(result.boxes.conf[idx]),
                        class_identifier=int(result.boxes.cls[idx]),
                        label_name=self.model.names[int(result.boxes.cls[idx])],
                    )
                    for idx, m in enumerate(result.masks.data)
                ]
                if result.masks is not None
                else []
            )

            processed_frame = self.isolate_object(result.orig_img, detections, targets)

            cv2.imshow("window_name", processed_frame)
            time.sleep(2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User interrupted the stream.")
                break

    def isolate_object(
        self, frame: np.ndarray, detections: list[Object], target_name: list[str]
    ) -> np.ndarray:
        self.processor.start_annotator(frame)
        for detection in detections:
            if detection.label_name.lower() in target_name:
                self.processor.draw_mask(
                    detection.masks[None], detection.class_identifier
                )
                self.processor.draw_bounding_box(
                    detection.box, detection.class_identifier
                )
                self.processor.draw_label(
                    detection.box,
                    f"{detection.label_name} {detection.confidence:.2f}",
                    detection.class_identifier,
                )
        return self.processor.get_final_frame()
