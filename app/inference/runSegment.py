from app.utilities.VideoReader import VideoReader
from app.utilities.utils import get_from_config
from app.inference.YoloAnnotator import YoloAnnotator
from app.inference.segmentation import YoloSegmentModel
import cv2


class runSegment:
    def __init__(self, config_path: str) -> None:
        self.config_path: str = config_path

        self.frame_skip: int = get_from_config(config_path, "frame_count")
        self.target_class: list[str] = get_from_config(config_path, "target_class")
        self.data: str = get_from_config(config_path, "data")
        self.model: str = get_from_config(config_path, "model")

    def run(self) -> None:
        video_reader = VideoReader(stride=self.frame_skip, video_path=self.data)
        annotator = YoloAnnotator(targets=self.target_class)
        segment_model = YoloSegmentModel(model_path=self.model)

        for frame in video_reader.extract_frames():
            detections = segment_model.process_frame(frame)

            annotated_frame = annotator.draw_detections(frame, detections)

            cv2.imshow("YOLO Stream", annotated_frame)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
