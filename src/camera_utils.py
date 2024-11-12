import cv2
import numpy as np

from src.constants import (
    CAMERA_ZOOM_PADDING_PCT, 
    ZOOM_MIN_WIDTH_PCT, 
    ZOOM_SMOOTHING_ALPHA,
)

def convert_elements_to_int(input_list: list | tuple | np.ndarray | int) -> list | tuple | np.ndarray | int:
    """Convert all elements in a list-like object to integers, maintaining the same shape and format.

    Args:
        input_list (list-like): The input list-like object.

    Returns:
        list-like: The list-like object with all elements converted to integers.
    """
    if isinstance(input_list, list):
        return [convert_elements_to_int(item) for item in input_list]
    elif isinstance(input_list, tuple):
        return tuple(convert_elements_to_int(item) for item in input_list)
    elif isinstance(input_list, np.ndarray):
        return input_list.astype(int)
    else:
        return int(input_list)


def keep_zoom_box_inside_frame(tl_point: tuple, br_point: tuple, frame) -> tuple:
    """Adjust the zoom box to ensure it stays within the frame boundaries.
    
    Args:
        tl_point (tuple): The top-left corner of the zoom box.
        br_point (tuple): The bottom-right corner of the zoom box.
        img_width (int): The width of the frame.
        img_height (int): The height of the frame.
        
    Returns:
        tuple: ((tl_x, tl_y), (br_x, br_y)) The adjusted top-left and bottom-right corners of the zoom box.
    """
    img_height, img_width = frame.shape[:2]

    # Ensure the top and left edges are within the frame boundaries
    if tl_point[0] < 0:
        tl_point[0] = 0
    
    if tl_point[1] < 0:
        tl_point[1] = 0

    # Ensure the bottom and right edges are within the frame boundaries
    if br_point[0] > img_width:
        tl_point[0] -= (br_point[0] - img_width)
        br_point[0] = img_width
    if br_point[1] > img_height:
        tl_point[1] -= (br_point[1] - img_height)
        br_point[1] = img_height

    # Ensure the top-left corner is still within the frame boundaries after adjustment
    if tl_point[0] < 0:
        tl_point[0] = 0
    if tl_point[1] < 0:
        tl_point[1] = 0

    tl_point = convert_elements_to_int(tl_point)
    br_point = convert_elements_to_int(br_point)

    return tl_point, br_point


def calculate_optimal_zoom_area(frame: np.ndarray, player_positions_xyxy: list, frame_display_size_h_w: tuple) -> list:
    """Calculate the optimal zoom area based on player positions. 
    The resulting box will have the same aspect ratio as the `frame`, centered on the middle of player_positions,
    and will be large enough to include all player positions with at least `padding` pixels inside the outer dimensions of the current frame.

    Args:
        frame (np.ndarray): The original video frame.
        player_positions_xyxy (list): List of [x,y] coordinates for the top left and bottom right corners of bounding box surrounding the players.
        frame_display_size_h_w (tuple): The height and width of the current frame. Used to return crop box with constant aspect ratio.

    Returns:
        list: Coordinates of the zoom area (x_min, y_min, x_max, y_max).
    """
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    x_padding = frame_width * CAMERA_ZOOM_PADDING_PCT
    y_padding = frame_height * CAMERA_ZOOM_PADDING_PCT
    
    # Convert the player positions to xywh format
    player_positions_xyxy = np.array(player_positions_xyxy)

    # If there are no player positions, return the full frame
    if len(player_positions_xyxy) == 0:
        return [0, 0, frame_display_size_h_w[1], frame_display_size_h_w[0]]

    # Calculate the aspect ratio of the frame
    frame_height, frame_width = frame_display_size_h_w
    aspect_ratio = frame_width / frame_height

    # Separate the x and y coordinates
    x_coords = player_positions_xyxy[:, [0, 2]].flatten()
    y_coords = player_positions_xyxy[:, [1, 3]].flatten()
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # ---------------------------------------------------------
    # This is the minimum width and height needed for a bounding 
    # box to fully encompass all players bboxes 
    min_width_zoom_box = (x_max - x_min) + (2 * x_padding)
    min_height_zoom_box = (y_max - y_min) + (2 * y_padding)

    # But also ensure the width and height used are not less than ZOOM_MIN_WIDTH_PCT of the frame dimensions
    min_width_zoom_box = max(min_width_zoom_box, frame_width * ZOOM_MIN_WIDTH_PCT)
    min_height_zoom_box = max(min_height_zoom_box, frame_height * ZOOM_MIN_WIDTH_PCT) 

    # Calculate the width and height of the smallest rectangle that is at least min_width and min_height and has the same w/h aspect_ratio
    if min_width_zoom_box / min_height_zoom_box > aspect_ratio:
        zoom_width = min_width_zoom_box
        zoom_height = int(zoom_width / aspect_ratio)
    else:
        zoom_height = min_height_zoom_box
        zoom_width = int(zoom_height * aspect_ratio)

    # ---------------------------------------------------------
    # Move the zoom-box to be centered around the players

    # find middle point between all of the person bboxes
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate the top-left and bottom-right corners of the zoom box
    tl_x = max(0, center_x - zoom_width // 2)
    tl_y = max(0, center_y - zoom_height // 2)
    br_x = min(frame_width, center_x + zoom_width // 2)
    br_y = min(frame_height, center_y + zoom_height // 2)

    # Ensure the zoom box stays within the frame boundaries
    tl_point, br_point = keep_zoom_box_inside_frame((tl_x, tl_y), (br_x, br_y), frame)

    return [tl_point[0], tl_point[1], br_point[0], br_point[1]]


def zoom_frame(frame: np.ndarray, zoom_area: list) -> np.ndarray:
    """Zooms into the specified area of the frame.
    
    Args:
        frame (np.ndarray): The original video frame.
        zoom_area (list): A list [x, y, w, h] specifying the area to zoom into.
    
    Returns:
        np.ndarray: The zoomed frame.
    """
    tlx, tly, brx, bry = zoom_area
    zoomed_frame = frame[tly:bry, tlx:brx]
    zoomed_frame = cv2.resize(zoomed_frame, (frame.shape[1], frame.shape[0]))
    return zoomed_frame



def smooth_transition(prev_frame: np.ndarray, curr_frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Smooths the transition between two frames.
    
    Args:
        prev_frame (np.ndarray): The previous video frame.
        curr_frame (np.ndarray): The current video frame.
        alpha (float): The blending factor (0.0 to 1.0).
    
    Returns:
        np.ndarray: The blended frame.
    """
    return cv2.addWeighted(prev_frame, 1 - alpha, curr_frame, alpha, 0)


def linear_smooth_zoom_box_shift(
    frame: np.ndarray,
    prev_zoom_box_xyxy: np.ndarray, 
    new_zoom_box_xyxy: np.ndarray, 
    max_shift_pct: float = ZOOM_SMOOTHING_ALPHA
) -> np.ndarray:
    """Shift the zoom box linearly towards the new zoom box coordinates, but no more than the given max_speed (% of wdith/height)

    Args:
        frame (np.ndarray): The video frame.
        prev_zoom_box_xyxy (np.ndarray): The previous zoom box coordinates (tl xy, br xy) in xyxy format.
        new_zoom_box_xyxy (np.ndarray): The new zoom box coordinates in xyxy format.
        max_shift_pct (float): The maximum shift distance as a percentage of the frame dimensions.

    Returns:
        np.ndarray: The updated zoom box coordinates in xyxy format.
    """
    # Calculate the difference between the previous and new zoom box coordinates
    resulting_shift_xy = np.array([])
    for i in range(len(prev_zoom_box_xyxy)):
        coord_xy = i % 2
        desired_shift = new_zoom_box_xyxy[i] - prev_zoom_box_xyxy[i]
        max_shift = max_shift_pct * frame.shape[coord_xy]
        result_this_axis = np.clip(desired_shift, -max_shift, max_shift)
        resulting_shift_xy = np.append(resulting_shift_xy, result_this_axis)

    updated_zoom_box_xyxy = prev_zoom_box_xyxy + resulting_shift_xy

    updated_coords = keep_zoom_box_inside_frame(updated_zoom_box_xyxy[:2], updated_zoom_box_xyxy[2:], frame)
    updated_zoom_box_xyxy = np.array(updated_coords).flatten()

    return updated_zoom_box_xyxy.astype(int).tolist()