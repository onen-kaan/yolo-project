# ~/projects/work/yolo_project/app/segmentation_main.py

import argparse
import cv2
import os
import utilities.utils as utils
from inference.segmentator import YoloSegmentator


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="YOLO Video Segmentation Tool"
    )

    subparser_handler = argument_parser.add_subparsers(
        dest="command", required=True, help="Mode of operation"
    )

    # Segmentation Sub-command
    segmentation_parser = subparser_handler.add_parser(
        "segment", help="Execute segmentation on a video file"
    )

    segmentation_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="segmentation.yaml",
        help="Path to the segmentation configuration file",
    )

    return argument_parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.command == "segment":
        configuration_path = arguments.config

        model_file_path: str = utils.get_from_config(configuration_path, "model")
        video_file_path: str = utils.get_from_config(configuration_path, "data")
        frame_skip_interval: int = utils.get_from_config(
            configuration_path, "frame_count"
        )
        target_object_names: list[str] = utils.get_from_config(
            configuration_path, "target_class"
        )

        project_folder = utils.get_from_config(configuration_path, "project") or "runs"
        experiment_folder = utils.get_from_config(configuration_path, "name") or "exp"
        output_directory = os.path.join(project_folder, experiment_folder)
        os.makedirs(output_directory, exist_ok=True)

        segmentator_instance = YoloSegmentator(
            model_path=model_file_path,
            video_path=video_file_path,
            initial_frame_count=frame_skip_interval,
        )

        video_capturer = utils.get_video_capture(video_file_path)
        total_video_frames = int(video_capturer.get(cv2.CAP_PROP_FRAME_COUNT))

        processed_iteration_count = 0

        while True:
            target_frame_index = (
                processed_iteration_count * segmentator_instance.frame_count
            )

            if target_frame_index >= total_video_frames:
                break

            video_capturer.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            reading_success, current_frame = video_capturer.read()

            if not reading_success:
                break

            detections = segmentator_instance.get_frame_detections(current_frame)

            processed_frame = segmentator_instance.isolate_object(
                frame=current_frame,
                detections=detections,
                target_name=target_object_names,
            )

            output_filename = f"frame_{target_frame_index:06d}.jpg"
            full_save_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(full_save_path, processed_frame)

            processed_iteration_count += 1

        video_capturer.release()


if __name__ == "__main__":
    main()
