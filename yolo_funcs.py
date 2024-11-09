import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

from constants import CUR_DIR, YOLO_3_FILE_DOWNLOAD_PATHS, YOLO_VERSION


# Download YOLO model files if they do not exist
def download_yolo_files(files = YOLO_3_FILE_DOWNLOAD_PATHS):
    yolo_dir = "/Users/Henry/Desktop/github/cv-soccer/yolo"
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    for file_name, url in files.items():
        file_path = os.path.join(yolo_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {file_name}")


# Load YOLO model
def load_yolo_model(version: int = YOLO_VERSION):
    if version == 3:
        net = cv2.dnn.readNet(os.path.join(CUR_DIR, "yolo/yolov3.weights"), os.path.join(CUR_DIR, "yolo/yolov3.cfg"))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open(os.path.join(CUR_DIR, "yolo/coco.names"), "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers
    
    elif version == 8:
        v8_path = os.path.join(CUR_DIR, "yolo", "yolov8n.pt")
        if os.path.exists(v8_path):
            model = YOLO(v8_path)
        else:
            model = YOLO("yolov8n.pt")
            # save the model locally
            model.save(v8_path)
        return model, None, None



# Run inference on video frames
def get_human_bounding_boxes(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is for 'person' in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    filtered_boxes = [boxes[i] for i in range(len(boxes)) if i in indexes]
    return filtered_boxes, class_ids, confidences


def draw_bounding_boxes(frame, boxes, class_ids, classes, color=(0, 255, 0)):
    boxes = list(boxes)
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
