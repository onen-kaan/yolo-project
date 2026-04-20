import fire
from app.inference.predictor import YoloPredictor
from app.train.trainer import YoloTrainer
from app.inference.runSegment import runSegment


class YoloCLI:
    def train(self, config: str = "train.yaml") -> None:
        trainer = YoloTrainer(config_path=config)
        trainer.train()

    def predict(self, config: str = "predict.yaml") -> None:
        predictor = YoloPredictor(configPath=config)
        predictor.reuslt(predictor.predict())

    def segmentation(self, config: str = "segmentator.yaml") -> None:
        segment = runSegment(config_path=config)
        segment.run()


def main() -> None:
    fire.Fire(YoloCLI)


if __name__ == "__main__":
    main()
