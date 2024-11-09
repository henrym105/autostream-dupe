# main.py

import cv2
import os
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
    zoom_frame, 
    smooth_transition,
    interpolate_zoom_area,
)


def read_video(video_path, yolo_model, n=ZOOM_SMOOTHING_FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Source FPS: {source_fps}")
    ret, prev_frame = cap.read()
    frame_count = 0
    current_zoom_area = [0, 0, prev_frame.shape[1], prev_frame.shape[0]]
    target_zoom_area = current_zoom_area.copy()
    aspect_ratio_h_w = prev_frame.shape[0] / prev_frame.shape[1]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break
        elif int(source_fps) >= 30 and frame_count % 2 == 1:
            frame_count += 1
            continue

        # ------------------------------------------------
        # Process the Frame
        # ------------------------------------------------
        # Get human bounding boxes
        boxes, class_ids, confidences = get_human_bounding_boxes(frame, yolo_model)
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
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = os.path.join(CUR_DIR, 'data/raw/trimmed_video_path_go pro 12 full court view.mp4')
    yolo_model = load_yolo_model()
    read_video(video_path, yolo_model)