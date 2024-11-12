# main.py

import cv2
import os
import time
import numpy as np
from src.constants import (
    CROP_VIDEO,
    CUR_DIR, 
    DRAW_PLAYER_BOXES,
    DRAW_COURT_BOX,
    SAVE_VIDEO_LOCAL,
    ZOOM_SMOOTHING_ALPHA,
    ZOOM_SMOOTHING_FRAME_COUNT,
    TEMP_CORNERS_COORDS_PATH,
)
from src.yolo_funcs import (
    load_yolo_model, 
    draw_court_outline,
    get_all_yolo_bounding_boxes, 
    draw_bounding_boxes,
)
from src.camera_utils import (
    calculate_optimal_zoom_area, 
    linear_smooth_zoom_box_shift,
    smooth_transition,
    zoom_frame,
)
from ultralytics import YOLO
from select_court_corners import select_court_corners

# coordinates = []

def read_video(
    video_path: str, 
    yolo_model: YOLO, 
    save_to_path: str = None,
) -> None:
    """Read a video file and process each frame to detect players and zoom in on them.

    Args:
        video_path (str): The path to the video file.
        yolo_model (YOLO): The YOLO model for object detection.
        save_to_path (str, optional): The path to save the output video. Defaults to None.
    """
    # open the video file and read the first frame
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    # initialize variables
    frame_display_size_h_w = prev_frame.shape[:2]
    prev_boxes = []
    current_frame_num = 0
    prev_zoom_box = None

    # Define the output video path
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_to_path, fourcc, source_fps, (frame_display_size_h_w[1], frame_display_size_h_w[0]))

    while cap.isOpened():
        # read the next frame from the video file
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break
        
        # pause on the first frame and select the corner points for the court, save them to a file
        if current_frame_num == 0:
            select_court_corners(frame)

        # ------------------------------------------------
        # Process the Frame
        # ------------------------------------------------
        if current_frame_num % 1 == 0: # run inference on the frame
            boxes = get_all_yolo_bounding_boxes(frame, yolo_model)
            # save the bounding boxes for the next frame
            prev_boxes = boxes
        else:
            # this is an even frame, so use the bounding boxes from the previous frame
            boxes = prev_boxes

        if boxes and DRAW_PLAYER_BOXES:
            frame = draw_bounding_boxes(frame, boxes, color=(0, 255, 255))  # Yellow color for human bounding boxes
        
        if DRAW_COURT_BOX:
            frame = draw_court_outline(frame)

        # zoom_box in format [tl_x, tl_y, w, h]
        zoom_box: list = calculate_optimal_zoom_area(frame, boxes, frame_display_size_h_w)        
        
        # skip the smoothing algo on the first frame
        if prev_zoom_box is not None: 
            zoom_box = linear_smooth_zoom_box_shift(frame, prev_zoom_box, zoom_box)
        prev_zoom_box = zoom_box.copy()

        # frame = smooth_transition(prev_frame, frame, alpha=ZOOM_SMOOTHING_ALPHA)

        if zoom_box and not CROP_VIDEO:
            frame = draw_bounding_boxes(frame, [zoom_box], label="zoom_box", color=(0, 0, 255))
        
        elif CROP_VIDEO:
            frame = zoom_frame(frame, zoom_box)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Write the frame to the output video
        if SAVE_VIDEO_LOCAL:
            out.write(frame)

        # save this frame as reference for the next one
        prev_frame = frame
        current_frame_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # src_path = os.path.join(CUR_DIR, "data", "raw", "example_video.mp4")
    # save_path = os.path.join(CUR_DIR, "data", "processed", "example_video_autozoom.mp4")
    src_path = os.path.join(CUR_DIR, "data", "raw", "example_video_2.mp4")
    save_path = os.path.join(CUR_DIR, "data", "processed", "example_video_2_autozoom.mp4")

    yolo_model = load_yolo_model()

    # read_video(src_path, yolo_model, DRAW_PLAYER_BOXES, save_path)
    read_video(src_path, yolo_model, save_path)