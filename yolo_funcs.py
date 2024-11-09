import cv2
import numpy as np
from os import path
import urllib.request
from ultralytics import YOLO

from constants import (
    CUR_DIR,
    YOLO_VERSION,
    YOLO_HUMAN_CONFIDENCE_THRESHOLD,
)



# Load YOLO model
def load_yolo_model(version: int = YOLO_VERSION) -> YOLO:
    if version == 8:
        yolo_model_name = "yolov8n.pt"
    elif version == 11:
        yolo_model_name = "yolo11n.pt"
    else:
        raise ValueError("Invalid YOLO version. Supported versions are 8 and 11.")

    # create the path to the model
    yolo_model_path = path.join(CUR_DIR, "yolo", yolo_model_name)

    if path.exists(yolo_model_path):
        # Load the local copy of model
        model = YOLO(yolo_model_path, verbose = True)
    else:
        # Download it and save a local copy
        model = YOLO(yolo_model_name, verbose = True)
        model.save(yolo_model_path)

    return model


# Run inference on video frames using YOLOv8
def get_human_bounding_boxes(frame, model: YOLO):
    detection_threshold = YOLO_HUMAN_CONFIDENCE_THRESHOLD
    boxes = []
    class_ids = []
    confidences = []

    # run inference on the frame
    objects_detected = model(frame)

    for item in objects_detected:
        for detection in item.boxes:
            class_id = int(detection.cls)
            confidence = float(detection.conf)

            if confidence > detection_threshold and class_id == 0:  # Class ID 0 is for 'person' in COCO dataset
                x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
                boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                confidences.append(confidence)
                class_ids.append(class_id)

    return boxes, class_ids, confidences


def draw_bounding_boxes(frame, boxes, label, color=(0, 255, 0)):
    boxes = list(boxes)
    if len(boxes) > 0:
        for i in range(len(boxes)):
            tl_x, tl_y, w, h = boxes[i]
            cv2.rectangle(frame, (tl_x, tl_y), (tl_x + w, tl_y + h), color, 2)
            cv2.putText(frame, label, (tl_x, tl_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
