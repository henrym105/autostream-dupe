# main.py

import cv2
import os
import numpy as np
from constants import CUR_DIR
from yolo_funcs import (
    download_yolo_files, 
    load_yolo_model, 
    get_human_bounding_boxes, 
    draw_bounding_boxes
)
from camera_utils import (
    calculate_optimal_zoom_area, 
    zoom_frame, 
    smooth_transition
)


def read_video(video_path, net, classes, output_layers):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()

    while cap.isOpened():
        ret, frame = cap.read()
        # Press Q on keyboard to exit
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break

        # Get human bounding boxes
        boxes, class_ids, confidences = get_human_bounding_boxes(frame, net, output_layers)
        frame = draw_bounding_boxes(frame, boxes, class_ids, classes)

        zoom_box = calculate_optimal_zoom_area(boxes)
        # frame = draw_bounding_boxes(frame, [zoom_box], [0], classes, color=(0, 255, 0))
        
        frame = zoom_frame(frame, zoom_box)
        frame = smooth_transition(prev_frame, frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # save this frame as reference for the next iteration
        prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    download_yolo_files()
    video_path = os.path.join(CUR_DIR, 'data/raw/trimmed_video_path_go pro 12 full court view.mp4')
    net, classes, output_layers = load_yolo_model()
    read_video(video_path, net, classes, output_layers)