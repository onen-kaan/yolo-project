import fire
from app.inference.predictor import YoloPredictor
from app.inference.segmentator import YoloSegmentator
from app.train.trainer import YoloTrainer


class YoloCLI:
    def train(self, config: str = "train.yaml") -> None:
        trainer = YoloTrainer(config_path=config)
        trainer.train()

    def predict(self, config: str = "predict.yaml") -> None:
        predictor = YoloPredictor(configPath=config)
        predictor.reuslt(predictor.predict())

    def segment(self, config: str = "segmentator.yaml") -> None:
        segment = YoloSegmentator(config_path=config)
        segment.segment()


def main() -> None:
    fire.Fire(YoloCLI)


if __name__ == "__main__":
    main()
