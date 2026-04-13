# ~/projects/work/yolo_project/app/utilities/VideoProcessor.py

from ultralytics.utils.plotting import Annotator, colors
import numpy as np


class VideoProcessor:
    def __init__(self):
        self.annotator = None

    def start_annotator(self, frame: np.ndarray):
        self.annotator = Annotator(frame, line_width=2)

    def draw_mask(self, masks: np.ndarray | None, class_identifier: int):
        if self.annotator is not None and masks is not None:
            self.annotator.masks(masks, colors(class_identifier, True), alpha=0.5)

    def draw_bounding_box(self, box_coordinates: np.ndarray, class_identifier: int):
        if self.annotator is not None:
            self.annotator.box_label(
                box_coordinates, label="", color=colors(class_identifier, True)
            )

    def draw_label(
        self, box_coordinates: np.ndarray, display_text: str, class_identifier: int
    ):
        if self.annotator is not None:
            self.annotator.box_label(
                box_coordinates,
                label=display_text,
                color=colors(class_identifier, True),
            )

    def get_final_frame(self) -> np.ndarray:
        if self.annotator is not None:
            final_image = self.annotator.result()
            if isinstance(final_image, np.ndarray):
                return final_image
            return np.array(final_image)
        return np.array([], dtype=np.uint8)
