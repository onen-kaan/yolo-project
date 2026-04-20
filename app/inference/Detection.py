import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    bounding_box_coordinates: np.ndarray
    confidence_score: float
    class_identifier: int
    class_label_name: str
    segmentation_mask: np.ndarray | None
