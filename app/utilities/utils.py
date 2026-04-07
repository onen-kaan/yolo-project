import yaml
import os
from typing import Any
import numpy as np
import cv2


def getFromConfig(configPath: str, key: str) -> Any | None:
    if not os.path.exists(configPath):
        return None

    with open(configPath, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config.get(key)


def getVideoImages(videoPath: str, frameCount: int) -> list[np.ndarray]:

    cap = cv2.VideoCapture(videoPath)
    frames: list[np.ndarray] = []

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for targetIdx in range(0, totalFrames, frameCount):
        cap.set(cv2.CAP_PROP_POS_FRAMES, targetIdx)

        success, frame = cap.read()

        if success:
            frames.append(frame)

    cap.release()
    return frames
