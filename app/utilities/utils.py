import yaml
import os
from typing import Any
import numpy as np
import cv2


def get_from_config(configPath: str, key: str) -> Any:
    if not os.path.exists(configPath):
        return None

    with open(configPath, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config.get(key)


def get_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
    return cap
