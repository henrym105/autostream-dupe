# main.py

import cv2
import os
import time
import numpy as np
from src.constants import (
    CUR_DIR, 
    ZOOM_SMOOTHING_ALPHA,
    ZOOM_SMOOTHING_FRAME_COUNT,
)
from src.yolo_funcs import (
    # download_yolo_files, 
    load_yolo_model, 
    get_all_yolo_bounding_boxes, 
    draw_bounding_boxes,
)
from src.camera_utils import (
    calculate_optimal_zoom_area, 
    zoom_frame,
)


def read_video(video_path, yolo_model, draw_player_boxes=True, crop_video=True, n=ZOOM_SMOOTHING_FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    # read the first frame
    ret, prev_frame = cap.read()

    # initialize variables
    current_zoom_box = [0,0, prev_frame.shape[0], prev_frame.shape[1]]
    target_zoom_box = current_zoom_box.copy()
    frame_display_size_h_w = prev_frame.shape[:2]
    frame_ratio_h_w = frame_display_size_h_w[0] / frame_display_size_h_w[1]

    print(f"{frame_display_size_h_w = }")
    print(f"{prev_frame.shape = }")
    print(f"{frame_ratio_h_w = }")
    
    # input("waiting...")
    prev_boxes = []
    current_frame_num = 0

    while cap.isOpened():
        # read the next frame from the video file
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break
        
        # ------------------------------------------------
        # Process the Frame
        # ------------------------------------------------
        # find the human bounding boxes on the 
        if current_frame_num % 2 == 0:
            # run inference on the frame
            boxes = get_all_yolo_bounding_boxes(frame, yolo_model)
            # save the bounding boxes for the next frame
            prev_boxes = boxes
        else:
            # this is an even frame, so use the bounding boxes from the previous frame
            boxes = prev_boxes
        
        if boxes and draw_player_boxes:
            frame = draw_bounding_boxes(frame, boxes, label="player")

        # zoom_box in format [tl_x, tl_y, w, h]
        zoom_box: list = calculate_optimal_zoom_area(frame, boxes, frame_display_size_h_w)
        if zoom_box and draw_player_boxes:
            frame = draw_bounding_boxes(frame, [zoom_box], label="zoom_box", color=(0, 0, 255))
        
        if crop_video:
            frame = zoom_frame(frame, zoom_box)
        # frame = smooth_transition(prev_frame, frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # save this frame as reference for the next one
        prev_frame = frame
        current_frame_num += 1

    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = os.path.join(CUR_DIR, "data", "raw", "example_video.mp4")
    draw_player_boxes = True
    crop_video = True

    yolo_model = load_yolo_model()

    read_video(video_path, yolo_model, draw_player_boxes, crop_video)