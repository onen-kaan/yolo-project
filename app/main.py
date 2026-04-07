import argparse

import cv2
import utilities.utils as util
from inference.predictor import YoloPredictor
from inference.segmentator import YoloSegmentator
from train.trainer import YoloTrainer


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO CLI with Subcommands")

    # Create the subparser handler
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Working mode"
    )

    # --- TRAIN Subcommand ---
    train_parser = subparsers.add_parser("train", help="Run model training")
    _ = train_parser.add_argument(
        "-c", "--config", type=str, default="train.yaml", help="Path to train config"
    )

    # --- INFERENCE Subcommand ---
    predict_parser = subparsers.add_parser("predict", help="Run standard inference")
    _ = predict_parser.add_argument(
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
    _ = seg_parser.add_argument(
        "-v", "--video", type=str, required=True, help="Path to input video"
    )
    _ = seg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolov26n-seg.pt",
        help="Path to -seg.pt model",
    )
    _ = seg_parser.add_argument(
        "-f", "--frames", type=int, default=30, help="Apply operation per 'N'th frame"
    )
    _ = seg_parser.add_argument(
        "--target", type=str, default="car", help="Object class to isolate"
    )

    return parser.parse_args()


def main() -> None:
    args = parseArgs()

    # The 'args.command' will contain the name of the subparser used
    if args.command == "train":
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
        # 1. Initialize the segmentator
        seg = YoloSegmentator(
            segmentator=args.model, videoPath=args.video, frameCount=args.frames
        )

        print(f"Starting isolation for target: {args.target}")

        # 2. Get frames from the utility (Ensure these names match your YoloSegmentator property names!)
        # Using snake_case as defined in our previous property discussion
        frames = util.getVideoImages(
            videoPath=seg.video_path, frameCount=seg.frame_count
        )

        if not frames:
            print("No frames found or video failed to open.")
            return

        # 3. Process the frames
        processed_count = 0
        for i, frame in enumerate(frames):
            # A. Get raw detection data (masks, boxes, etc.)
            detections = seg.get_frame_detections(frame)

            # B. Create the black background version
            isolated_frame = seg.isolate_object(frame, detections, args.target)

            # C. Handle the result (Save or Show)
            # Creating a simple naming convention for output
            output_name = f"output_{args.target}_{i}.jpg"
            cv2.imwrite(output_name, isolated_frame)
            processed_count += 1

        print(
            f"Finished! Processed {processed_count} frames. Results saved to current directory."
        )


if __name__ == "__main__":
    main()
