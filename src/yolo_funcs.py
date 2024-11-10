import cv2
import numpy as np
from os import path
from ultralytics import YOLO

from src.constants import CUR_DIR, YOLO_HUMAN_CONFIDENCE_THRESHOLD, YOLO_VERSION


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
def get_all_yolo_bounding_boxes(frame, model: YOLO, class_id=0) -> tuple:
    """Get the bounding boxes of humans detected in the frame using YOLO model.

    Args:
        frame (numpy.ndarray): The video frame.
        model (YOLO): The YOLO model for object detection.

    Returns:
        tuple: Bounding boxes, class IDs, and confidences.
    """
    detection_threshold = YOLO_HUMAN_CONFIDENCE_THRESHOLD
    boxes = []

    # Run inference on the frame
    objects_detected = model(frame)

    for item in objects_detected:
        for detection in item.boxes:
            if (detection.conf > detection_threshold) and (detection.cls == class_id):
                # xywh = np.array(detection.xywh[class_id]).astype(int).tolist()
                # boxes.append(xywh)
                xyxy = np.array(detection.xyxy[class_id]).astype(int).tolist()
                boxes.append(xyxy)

    return boxes


def draw_bounding_boxes(frame, boxes, label, color=(0, 255, 0)):
    # boxes = [boxes]
    for i in range(len(boxes)):
        # Unpack the bounding box coordinates
        # tl_x, tl_y, w, h = boxes[i]
        # tl_point = (tl_x, tl_y)
        # br_point = (tl_x + w, tl_y + h)
        tl_x, tl_y, br_x, br_y = boxes[i]
        tl_point = (tl_x, tl_y)
        br_point = (br_x, br_y)

        # draw the bounding box on the frame
        cv2.rectangle(frame, tl_point, br_point, color, 2)
        # add the label to this bounding box
        cv2.putText(frame, label, (tl_x, tl_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
