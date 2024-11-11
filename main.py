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
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    # Read the first frame
    ret, prev_frame = cap.read()
    frame_display_size_h_w = prev_frame.shape[:2]
        
    prev_boxes = []
    current_frame_num = 0
    prev_zoom_box = None

    # Define the output video path
    output_video_path = os.path.join(CUR_DIR, "data", "processed", os.path.basename(video_path))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(output_video_path, fourcc, source_fps, (frame_display_size_h_w[1], frame_display_size_h_w[0]))

    while cap.isOpened():
        # Read the next frame from the video file
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break

        # ------------------------------------------------
        # Process the Frame
        # ------------------------------------------------
        if current_frame_num % 2 == 0:
            # Run inference on the frame
            boxes = get_all_yolo_bounding_boxes(frame, yolo_model)
            prev_boxes = boxes
        else:
            # Use the bounding boxes from the previous frame
            boxes = prev_boxes

        if boxes and draw_player_boxes:
            frame = draw_bounding_boxes(frame, boxes, label="player")

        # Calculate zoom_box
        zoom_box = calculate_optimal_zoom_area(frame, boxes, frame_display_size_h_w)        

        # Apply smoothing if not the first frame
        if prev_zoom_box is not None:
            # zoom_box = linear_smooth_zoom_box_shift(prev_zoom_box, zoom_box, alpha=0.3)  # Adjust alpha as needed
            zoom_box = linear_smooth_zoom_box_shift(frame, prev_zoom_box, zoom_box, ZOOM_SMOOTHING_ALPHA)
        prev_zoom_box = zoom_box.copy()

        if zoom_box and draw_player_boxes:
            frame = draw_bounding_boxes(frame, [zoom_box], label="zoom_box", color=(0, 0, 255))
        
        if crop_video:
            frame = zoom_frame(frame, zoom_box)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Write the frame to the output video
        if save_video_local:
            out.write(frame)

        # Save this frame as reference for the next one
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