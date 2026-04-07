from ultralytics import YOLO
from app.utilities.utils import getFromConfig


class YoloPredictor:
    def __init__(self, configPath: str = "predict.yaml") -> None:
        self.__configPath = configPath
        self.__model = self.__setupModel()

    def __setupModel(self) -> YOLO:
        modelVal = getFromConfig(self.__configPath, "model")
        return YOLO(str(modelVal) if modelVal else "yolo26n.pt")

    @property
    def configPath(self) -> str:
        return self.__configPath

    @property
    def modelName(self) -> str:
        return self.__model.overrides.get("model", "Unknown")

    def predict(self) -> list:
        print(f"Starting inference with config: {self.__configPath}")
        results = self.__model.predict(cfg=self.__configPath)
        return results
