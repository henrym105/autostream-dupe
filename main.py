# main.py

import cv2
import os
import numpy as np
from constants import CUR_DIR
from yolo_funcs import download_yolo_files, load_yolo_model, get_human_bounding_boxes, draw_bounding_boxes


def read_video(video_path, net, classes, output_layers):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        # Press Q on keyboard to exit
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break

        # Get human bounding boxes
        boxes, class_ids, confidences = get_human_bounding_boxes(frame, net, classes, output_layers)
        frame = draw_bounding_boxes(frame, boxes, class_ids, classes)
        

        # Display the resulting frame
        cv2.imshow('Frame', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    download_yolo_files()
    video_path = os.path.join(CUR_DIR, 'data/raw/trimmed_video_path_go pro 12 full court view.mp4')
    net, classes, output_layers = load_yolo_model()
    read_video(video_path, net, classes, output_layers)