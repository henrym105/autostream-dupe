import cv2
import numpy as np
from os import path
from ultralytics import YOLO

from src.constants import (
    CUR_DIR, 
    DRAW_COURT_BOX, 
    YOLO_HUMAN_CONFIDENCE_THRESHOLD, 
    YOLO_VERSION, 
    TEMP_CORNERS_COORDS_PATH
)

import json

# Load YOLO model
def load_yolo_model(version: int = YOLO_VERSION) -> YOLO:
    """Load the YOLO model for object detection.

    Args:
        version (int, optional): The version of YOLO model to load. Defaults to YOLO_VERSION.

    Returns:
        YOLO: The YOLO model for object detection.
    """
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

# Run inference on video frames using YOLO
def get_all_yolo_bounding_boxes(frame, model: YOLO, class_id=0, court_coords: np.ndarray = None) -> tuple:
    """Get the bounding boxes of humans detected in the frame using YOLO model.

    Args:
        frame (numpy.ndarray): The video frame.
        model (YOLO): The YOLO model for object detection.

    Returns:
        tuple: Bounding boxes, class IDs, and confidences.
    """
    detection_threshold = YOLO_HUMAN_CONFIDENCE_THRESHOLD
    boxes = []

    # Load court coordinates
    if court_coords is None:
        with open(TEMP_CORNERS_COORDS_PATH, 'r') as f:
            court_coords = json.load(f)
    court_polygon = np.array(court_coords, dtype=np.int32)

    # Run inference on the frame
    objects_detected = model(frame)

    for item in objects_detected:
        for detection in item.boxes:
            if (detection.conf > detection_threshold) and (detection.cls == class_id):
                xyxy = np.array(detection.xyxy[class_id]).astype(int).tolist()
                # Filter boxes based on court coordinates
                if cv2.pointPolygonTest(court_polygon, (xyxy[2], xyxy[3]), False) >= 0:
                    boxes.append(xyxy)

    return boxes


def draw_bounding_boxes(frame: np.ndarray, boxes: list, label: str = "", color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes on the frame."""
    for box in boxes:
        tl_point, br_point = box[:2], box[2:]
        cv2.rectangle(frame, tl_point, br_point, color, 1)
        if label:
            cv2.putText(frame, label, (tl_point[0], tl_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def draw_court_outline(frame, court_coords: list = None) -> np.ndarray:
    """Draw the outline of the court on the frame."""
    if not court_coords:
        with open(TEMP_CORNERS_COORDS_PATH, 'r') as f:
            court_coords = json.load(f)
    court_polygon = np.array(court_coords, dtype=np.int32)
    cv2.polylines(frame, [court_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    return frame