from ultralytics import YOLO
from app.utilities.utils import getFromConfig
from typing import Any


class YoloTrainer:
    def __init__(self, config_path: str = "default.yaml") -> None:
        self.__config_path = config_path

        self.__model = self._setup_model()

    def _setup_model(self) -> YOLO:
        model_val = get_from_config(self.__config_path, "model")
        return YOLO(str(model_val) if model_val else "yolo26n.pt")

    @property
    def config_path(self) -> str:
        return self.__config_path

    @property
    def model_name(self) -> str:
        return self.__model.overrides.get("model", "unknown")

    def train(self) -> Any:
        print(f"Starting training with config: {self.__config_path}")
        results = self.__model.train(cfg=self.__config_path)
        return results
