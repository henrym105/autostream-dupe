# main.py

import cv2
import os
import time
import numpy as np
from src.constants import (
    CROP_VIDEO,
    CUR_DIR, 
    DRAW_PLAYER_BOXES,
    SAVE_VIDEO_LOCAL,
    ZOOM_SMOOTHING_ALPHA,
    ZOOM_SMOOTHING_FRAME_COUNT,
    TEMP_CORNERS_COORDS_PATH,
)
from src.yolo_funcs import (
    load_yolo_model, 
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


def read_video(
    video_path: str, 
    yolo_model: YOLO, 
    draw_player_boxes: bool = True, 
    crop_video: bool = True, 
    save_video_local: bool = SAVE_VIDEO_LOCAL,
    n: int = ZOOM_SMOOTHING_FRAME_COUNT,
) -> None:
    """Read a video file and process each frame to detect players and zoom in on them.

    Args:
        video_path (str): The path to the video file.
        yolo_model (YOLO): The YOLO model for object detection.
        draw_player_boxes (bool, optional): Whether to draw bounding boxes around players. Defaults to True.
        crop_video (bool, optional): Whether to crop the video to the zoom box. Defaults to True.
        n (int, optional): The number of frames to smooth the zoom box over. Defaults to ZOOM_SMOOTHING_FRAME_COUNT.
    """
    select_court_corners(video_path)

    # open the video file
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    # read the first frame
    ret, prev_frame = cap.read()

    # initialize variables
    frame_display_size_h_w = prev_frame.shape[:2]
    
    prev_boxes = []
    current_frame_num = 0

    # Define the output video path
    output_video_path = os.path.join(CUR_DIR, "data", "processed", os.path.basename(video_path))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(output_video_path, fourcc, source_fps, (frame_display_size_h_w[1], frame_display_size_h_w[0]))

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
        
        # skip the smoothing algo on the first frame
        if current_frame_num != 0: 
            zoom_box = linear_smooth_zoom_box_shift(frame, prev_zoom_box, zoom_box)
        prev_zoom_box = zoom_box.copy()

        # frame = smooth_transition(prev_frame, frame, alpha=ZOOM_SMOOTHING_ALPHA)

        if zoom_box and draw_player_boxes:
            frame = draw_bounding_boxes(frame, [zoom_box], label="zoom_box", color=(0, 0, 255))
        
        if crop_video:
            frame = zoom_frame(frame, zoom_box)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Write the frame to the output video
        if save_video_local:
            out.write(frame)

        # save this frame as reference for the next one
        prev_frame = frame
        current_frame_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    src_path = os.path.join(CUR_DIR, "data", "raw", "example_video.mp4")
    save_path = os.path.join(CUR_DIR, "data", "processed", "example_video_autozoom.mp4")

    yolo_model = load_yolo_model()

    read_video(src_path, yolo_model, DRAW_PLAYER_BOXES, CROP_VIDEO, SAVE_VIDEO_LOCAL, save_path)