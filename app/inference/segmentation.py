import numpy as np
import cv2
from ultralytics import YOLO
from app.inference.Detection import Detection


class YoloSegmentModel:
    def __init__(self, model_path: str) -> None:
        self.model: YOLO = YOLO(model_path)

    def process_frame(self, image_frame: np.ndarray) -> list[Detection]:
        inference_results = self.model(image_frame, verbose=False)

        if not inference_results or inference_results[0].boxes is None:
            return []

        parsed_results = self.__parse_inference_results(inference_results[0])
        return parsed_results

    def __parse_inference_results(self, single_frame_result) -> list[Detection]:
        parsed_detections_list = []
        class_names_dictionary = self.model.names
        all_detected_boxes = single_frame_result.boxes

        for index in range(len(all_detected_boxes)):
            class_identifier = int(all_detected_boxes.cls[index])
            class_label_name = class_names_dictionary[class_identifier]

            detection_object = self.__build_single_detection_object(
                index, single_frame_result, class_label_name, class_identifier
            )
            parsed_detections_list.append(detection_object)

        return parsed_detections_list

    def __build_single_detection_object(
        self,
        index: int,
        single_frame_result,
        class_label_name: str,
        class_identifier: int,
    ) -> Detection:

        bounding_box_coordinates = single_frame_result.boxes.xyxy[index].cpu().numpy()
        confidence_score = float(single_frame_result.boxes.conf[index])

        segmentation_mask = None
        if single_frame_result.masks is not None:
            raw_mask_data = single_frame_result.masks.data[index]
            segmentation_mask = self.__resize_segmentation_mask(
                raw_mask_data, single_frame_result.orig_shape
            )

        return Detection(
            bounding_box_coordinates=bounding_box_coordinates,
            confidence_score=confidence_score,
            class_identifier=class_identifier,
            class_label_name=class_label_name,
            segmentation_mask=segmentation_mask,
        )

    def __resize_segmentation_mask(
        self, raw_mask_data, original_image_shape
    ) -> np.ndarray:

        mask_numpy_array = raw_mask_data.cpu().numpy()
        return cv2.resize(
            mask_numpy_array,
            (original_image_shape[1], original_image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
