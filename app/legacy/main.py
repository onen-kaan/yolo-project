import argparse
import cv2
import utilities.utils as util
import os
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
    segmentation_parser = subparsers.add_parser(
        "segment", help="Run model segmentation"
    )
    _ = segmentation_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="segmentation.yaml",
        help="Path to segmentation config",
    )

    return parser.parse_args()


def main() -> None:
    args = parseArgs()

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
        cfg_file = args.config
        model_path: str = util.get_from_config(cfg_file, "model")
        video_path: str = util.get_from_config(cfg_file, "data")
        frame_step: int = util.get_from_config(cfg_file, "frame_count")
        target: str = util.get_from_config(cfg_file, "target_class")

        project = util.get_from_config(cfg_file, "project") or "runs"
        name = util.get_from_config(cfg_file, "name") or "exp"
        output_dir = os.path.join(project, name)
        os.makedirs(output_dir, exist_ok=True)

        segmentator = YoloSegmentator(model_path, video_path, frame_step)
        capturer = util.get_video_capture(video_path)
        total_frames = int(capturer.get(cv2.CAP_PROP_FRAME_COUNT))

        # if frame_count % 5 == 0 didn't used that because of readability and the % 5 logic is not here

        processed_count = 0
        while True:
            current_frame_index = processed_count * segmentator.frame_count
            if current_frame_index > total_frames:
                break

            capturer.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

            success, frame = capturer.read()
            if not success:
                break

            # Process
            detections = segmentator.get_frame_detections(frame)
            processed_frame = segmentator.isolate_object(frame, detections, target)

            # Save
            file_name = f"frame_{processed_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, file_name), processed_frame)

            processed_count += 1

        capturer.release()


if __name__ == "__main__":
    main()
