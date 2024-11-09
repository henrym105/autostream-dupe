# main.py

import cv2
import numpy as np
import os
import urllib.request
from constants import CUR_DIR
from yolo_funcs import download_yolo_files, load_yolo_model, detect_humans


def read_video(video_path, net, classes, output_layers):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_humans(frame, net, classes, output_layers)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    download_yolo_files()
    video_path = os.path.join(CUR_DIR, 'data/raw/trimmed_video_path_go pro 12 full court view.mp4')
    net, classes, output_layers = load_yolo_model()
    read_video(video_path, net, classes, output_layers)