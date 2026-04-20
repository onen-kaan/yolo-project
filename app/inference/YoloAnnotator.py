import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from app.inference.Detection import Detection


class YoloAnnotator:
    def __init__(self, targets: list[str]) -> None:
        self.targets: list[str] = targets

    def draw_detections(
        self,
        image_frame: np.ndarray,
        detections_list: list[Detection],
    ) -> np.ndarray:

        annotator = Annotator(image_frame, line_width=2)

        safe_targets = [target.lower() for target in self.targets]

        for detection in detections_list:
            if detection.class_label_name.lower() not in safe_targets:
                continue

            assigned_color = colors(detection.class_identifier, True)
            display_text = (
                f"{detection.class_label_name} {detection.confidence_score:.2f}"
            )

            annotator.box_label(
                box=detection.bounding_box_coordinates,
                label=display_text,
                color=assigned_color,
            )

            if detection.segmentation_mask is not None:
                annotator.masks(
                    detection.segmentation_mask[None],
                    colors=[assigned_color],
                    alpha=0.5,
                )

        return annotator.result()
