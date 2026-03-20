import argparse
from app.train.trainer import YoloTrainer
from app.inference.predictor import YoloPredictor


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO Train & Inference CLI")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-t", "--train",     action="store_true", help="Run training")
    mode.add_argument("-i", "--inference", action="store_true", help="Run inference")

    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: train.yaml for -t, predict.yaml for -i)",
    )

    return parser.parse_args()


def main() -> None:
    args = parseArgs()

    if args.train:
        config = args.config or "train.yaml"
        trainer = YoloTrainer(configPath=config)
        trainer.train()

    elif args.inference:
        config = args.config or "predict.yaml"
        predictor = YoloPredictor(configPath=config)
        results = predictor.predict()

        for r in results:
            print(r)


if __name__ == "__main__":
    main()