# main.py

import cv2
import os
from ultralytics import YOLO
import numpy as np

from src.constants import (
    CROP_VIDEO,
    CUR_DIR, 
    DRAW_PLAYER_BOXES,
    DRAW_COURT_BOX,
    DRAW_MINIMAP,
    MINIMAP_POSITION,
    SAVE_VIDEO_LOCAL,
)
from src.yolo_funcs import (
    load_yolo_model, 
    draw_bounding_boxes,
    draw_court_outline,
    get_all_yolo_bounding_boxes,
)
from src.camera_utils import (
    calculate_optimal_zoom_area, 
    linear_smooth_zoom_box_shift,
    zoom_frame,
)
from src.select_court_corners import (
    select_court_corners, 
    infer_4_corners,
)
from src.minimap import (
    add_minimap_to_frame,
    create_minimap,
    get_perspective_transform_matrix,
)



def read_video(
    video_path: str, 
    yolo_model: YOLO, 
    save_to_path: str = None,
    minimap_position: str = MINIMAP_POSITION
) -> None:
    """Read a video file and process each frame to detect players and zoom in on them.

    Args:
        video_path (str): The path to the video file.
        yolo_model (YOLO): The YOLO model for object detection.
        save_to_path (str, optional): The path to save the output video. Defaults to None.
        minimap_position (str, optional): Position of the minimap on the frame. Defaults to "top_left".
    """
    # open the video file and read the first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    ret, prev_frame = cap.read()

    # initialize variables
    current_frame_num = 0
    prev_zoom_bbox = None

    # Define the output video path
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_to_path, fourcc, source_fps, (prev_frame.shape[1], prev_frame.shape[0]))

    while cap.isOpened():
        # read the next frame from the video file
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(25) & 0xFF == ord('q')):
            break
        
        # pause on the first frame and select the corner points for the court, save them to a file
        if current_frame_num == 0:
            all_edge_points_xy = select_court_corners(frame)
            # placeholder for future func to handle when >4 court points are input. 
            four_corner_points_xy = all_edge_points_xy.copy() 

            # Homography matrix for creating the minimap, camera is fixed so only need this on first frame. 
            M = get_perspective_transform_matrix(four_corner_points_xy)

        # Update the human bounding boxes and zoom-area bounding box every frame
        player_bboxes = get_all_yolo_bounding_boxes(frame, yolo_model)
        zoom_bbox = calculate_optimal_zoom_area(frame, player_bboxes) 
        zoom_bbox = linear_smooth_zoom_box_shift(frame, prev_zoom_bbox, zoom_bbox)
        
        if DRAW_PLAYER_BOXES:
            frame = draw_bounding_boxes(frame, player_bboxes, color=(0, 255, 255))  # Yellow color for human bounding boxes

        if DRAW_COURT_BOX:
            frame = draw_court_outline(frame)

        if DRAW_MINIMAP:
            minimap = create_minimap(player_bboxes, M)

        if CROP_VIDEO:
            frame = zoom_frame(frame, zoom_bbox)
        else:
            frame = draw_bounding_boxes(frame, [zoom_bbox], label="zoom_box", color=(0, 0, 255))

        if DRAW_MINIMAP:
            frame = add_minimap_to_frame(frame, minimap, position=minimap_position)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Write the frame to the output video
        if SAVE_VIDEO_LOCAL:
            out.write(frame)

        # save this frame as reference for the next one
        prev_frame = frame
        current_frame_num += 1
        prev_zoom_bbox = zoom_bbox.copy()

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
    read_video(src_path, yolo_model, save_path, minimap_position="bottom_right")