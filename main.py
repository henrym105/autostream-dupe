# main.py

import cv2
import os
import time
import numpy as np
from constants import (
    CUR_DIR, 
    ZOOM_SMOOTHING_ALPHA,
    ZOOM_SMOOTHING_FRAME_COUNT,
)
from yolo_funcs import (
    # download_yolo_files, 
    load_yolo_model, 
    get_human_bounding_boxes, 
    draw_bounding_boxes,
)
from camera_utils import (
    calculate_optimal_zoom_area, 
)


def read_video(video_path, yolo_model, n=ZOOM_SMOOTHING_FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)

    current_frame_num = 0
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev_frame = cap.read()
    current_zoom_area = [0, 0, prev_frame.shape[1], prev_frame.shape[0]]
    target_zoom_area = current_zoom_area.copy()
    aspect_ratio_h_w = prev_frame.shape[0] / prev_frame.shape[1]


    while cap.isOpened():
        # read the next frame from the video file
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break

        # Only run inference on the odd numbered frames bc inference is too slow
        # if int(source_fps) >= 30 and current_frame_num % 2 == 1:
        # if current_frame_num % 2 == 1:
        #     cv2.imshow('Frame', frame)
        #     current_frame_num += 1
        #     continue

        # ------------------------------------------------
        # Process the Frame
        # ------------------------------------------------
        # find the human bounding boxes on the 
        if current_frame_num % 3 == 0:
            # run inference on the frame
            boxes, class_ids, confidences = get_human_bounding_boxes(frame, yolo_model)
            # save the bounding boxes for the next frame
            prev_boxes = boxes
        else:
            # this is an even frame, so use the bounding boxes from the previous frame
            boxes = prev_boxes
        frame = draw_bounding_boxes(frame, boxes, class_ids, classes=["person"])


        zoom_box = calculate_optimal_zoom_area(boxes, aspect_ratio_h_w)
        frame = draw_bounding_boxes(frame, [zoom_box], class_ids=[0], classes=["zoom_box"], color=(0, 0, 255))
        # frame = zoom_frame(frame, zoom_box)
        # frame = smooth_transition(prev_frame, frame)

        # ------------------------------------------------
        # Display the resulting frame
        # ------------------------------------------------
        cv2.imshow('Frame', frame)

        # ------------------------------------------------
        # save this frame as reference for the next one
        # ------------------------------------------------
        prev_frame = frame
        current_frame_num += 1

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = os.path.join(CUR_DIR, 'data/raw/trimmed_video_path_go pro 12 full court view.mp4')
    yolo_model = load_yolo_model()
    read_video(video_path, yolo_model)