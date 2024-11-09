import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

from constants import (
    CUR_DIR,
    YOLO_VERSION,
)



# Load YOLO model
def load_yolo_model(version: int = YOLO_VERSION):
    if version == 8:
        v8_path = os.path.join(CUR_DIR, "yolo", "yolov8n.pt")
        if os.path.exists(v8_path):
            model = YOLO(v8_path)
        else:
            model = YOLO("yolov8n.pt")
            # save the model locally
            model.save(v8_path)

    return model


# Run inference on video frames using YOLOv8
def get_human_bounding_boxes(frame, model):
    results = model(frame)
    boxes = []
    class_ids = []
    confidences = []

    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls)
            confidence = float(detection.conf)
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is for 'person' in COCO dataset
                x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
                boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                confidences.append(confidence)
                class_ids.append(class_id)

    return boxes, class_ids, confidences


def draw_bounding_boxes(frame, boxes, class_ids, classes, color=(0, 255, 0)):
    boxes = list(boxes)
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
