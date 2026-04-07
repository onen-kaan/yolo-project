import argparse
from app.train.trainer import YoloTrainer
from app.inference.predictor import YoloPredictor
from app.inference.segmentator import YoloSegmentator
import app.utilities.utils as util


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO CLI with Subcommands")

    # Create the subparser handler
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Working mode"
    )

    # --- TRAIN Subcommand ---
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "-c", "--config", type=str, default="train.yaml", help="Path to train config"
    )

    # --- INFERENCE Subcommand ---
    predict_parser = subparsers.add_parser("predict", help="Run standard inference")
    predict_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="predict.yaml",
        help="Path to predict config",
    )

    # --- SEGMENT Subcommand ---
    seg_parser = subparsers.add_parser(
        "segment", help="Run segmentation with black background"
    )
    # These arguments ONLY exist under the 'segment' command
    seg_parser.add_argument(
        "-v", "--video", type=str, required=True, help="Path to input video"
    )
    seg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Path to -seg.pt model",
    )
    seg_parser.add_argument(
        "-f", "--frames", type=int, default=30, help="Apply operation per 'N'th frame"
    )
    seg_parser.add_argument(
        "--target", type=str, default="car", help="Object class to isolate"
    )

    return parser.parse_args()


def main() -> None:
    args = parseArgs()

    # The 'args.command' will contain the name of the subparser used
    if args.train:
        config = args.config or "train.yaml"

        trainer = YoloTrainer(config_path=config)

        trainer.train()

    elif args.command == "predict":
        config = args.config or "predict.yaml"

        predictor = YoloPredictor(configPath=config)

        results = predictor.predict()

        for r in results:
            print(r)

    elif args.command == "segment":
        # Now these are safely scoped to only the 'segment' command
        seg = YoloSegmentator(
            segmentator=args.model, videoPath=args.video, frameCount=args.frames
        )

        print(f"Starting isolation for target: {args.target}")
        frames = util.getVideoImages(videoPath=seg.video_path, seg.frameCount)


if __name__ == "__main__":
    main()
