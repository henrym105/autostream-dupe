import cv2
import numpy as np

from src.constants import (
    CAMERA_ZOOM_PADDING_PCT, 
    FRAME_MAX_SHIFT_PCT,
    FRAME_MAX_ZOOM_CHANGE_PCT,
    ZOOM_MIN_WIDTH_PCT, 
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


def calculate_optimal_zoom_area(frame: np.ndarray, player_positions_xyxy: list) -> list:
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
    # Get the frame dimensions and aspect ratio
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    # If no player positions are provided, return the full frame dimensions
    if not player_positions_xyxy:
        return [0, 0, frame_width, frame_height]

    # Convert player positions to a numpy array for easier manipulation
    player_positions_xyxy = np.array(player_positions_xyxy)
    x_coords = player_positions_xyxy[:, [0, 2]].flatten()
    y_coords = player_positions_xyxy[:, [1, 3]].flatten()
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Calculate padding based on the frame dimensions and predefined percentage
    x_padding = frame_width * CAMERA_ZOOM_PADDING_PCT
    y_padding = frame_height * CAMERA_ZOOM_PADDING_PCT

    # Calculate the minimum width and height of the zoom box, ensuring it is not smaller than a predefined percentage of the frame dimensions
    min_width_zoom_box = max((x_max - x_min) + 2 * x_padding, frame_width * ZOOM_MIN_WIDTH_PCT)
    min_height_zoom_box = max((y_max - y_min) + 2 * y_padding, frame_height * ZOOM_MIN_WIDTH_PCT)

    # Adjust the zoom box dimensions to maintain the aspect ratio of the frame
    if min_width_zoom_box / min_height_zoom_box > aspect_ratio:
        zoom_width = min_width_zoom_box
        zoom_height = int(zoom_width / aspect_ratio)
    else:
        zoom_height = min_height_zoom_box
        zoom_width = int(zoom_height * aspect_ratio)

    # Calculate the center of the zoom box based on the player positions
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate the top-left and bottom-right corners of the zoom box
    tl_x = max(0, center_x - zoom_width // 2)
    tl_y = max(0, center_y - zoom_height // 2)
    br_x = min(frame_width, center_x + zoom_width // 2)
    br_y = min(frame_height, center_y + zoom_height // 2)

    # Ensure the zoom box stays within the frame boundaries
    tl_point, br_point = keep_zoom_box_inside_frame((tl_x, tl_y), (br_x, br_y), frame)

    # Return the coordinates of the zoom area
    return [tl_point[0], tl_point[1], br_point[0], br_point[1]]


def zoom_frame(frame: np.ndarray, zoom_area: list) -> np.ndarray:
    """Zooms into the specified area of the frame.
    
    Args:
        frame (np.ndarray): The video frame.
        zoom_area (list): The zoom area coordinates (tlx, tly, brx, bry).
    
    Returns:
        np.ndarray: The zoomed frame.
    """
    tlx, tly, brx, bry = zoom_area
    # Crop the frame to the specified zoom area and resize it to the original frame size
    zoomed_frame = frame[tly:bry, tlx:brx]
    # Upscale the zoomed frame to keep the original frame's pixel size
    zoomed_frame = cv2.resize(zoomed_frame, (frame.shape[1], frame.shape[0]))
    return zoomed_frame



def linear_smooth_zoom_box_shift(
    frame: np.ndarray,
    prev_zoom_box_xyxy: np.ndarray, 
    new_zoom_box_xyxy: np.ndarray, 
    max_shift_pct: float = FRAME_MAX_SHIFT_PCT,
    max_zoom_change_pct: float = FRAME_MAX_ZOOM_CHANGE_PCT
) -> np.ndarray:
    """Shift the zoom box linearly towards the new zoom box coordinates, but no more than the given max_speed (% of width/height)

    Args:
        frame (np.ndarray): The video frame.
        prev_zoom_box_xyxy (np.ndarray): The previous zoom box coordinates (tl xy, br xy) in xyxy format.
        new_zoom_box_xyxy (np.ndarray): The new zoom box coordinates in xyxy format.
        max_shift_pct (float): The maximum shift distance as a percentage of the frame dimensions.

    Returns:
        np.ndarray: The updated zoom box coordinates in xyxy format.
    """
    if not prev_zoom_box_xyxy:
        return [0, 0, frame.shape[1], frame.shape[0]]

    # Convert previous and new zoom box coordinates from xyxy to centerxywh format
    prev_centerxywh = convert_xyxy_to_centerxy(prev_zoom_box_xyxy)
    new_centerxywh = convert_xyxy_to_centerxy(new_zoom_box_xyxy)

    # Calculate the actual shift in x and y directions, capped by the maximum allowed shift
    max_shift_x = max_shift_pct * frame.shape[1]
    max_shift_y = max_shift_pct * frame.shape[0]
    capped_shift_x = np.clip(new_centerxywh[0] - prev_centerxywh[0], -max_shift_x, max_shift_x)
    capped_shift_y = np.clip(new_centerxywh[1] - prev_centerxywh[1], -max_shift_y, max_shift_y)
    
    # Calculate the maximum allowed zoom change (given by height and width of zoom box) in x and
    # y directions based on the previous zoom box dimensions
    max_width_change_px = max_zoom_change_pct * prev_centerxywh[2]
    max_height_change_px = max_zoom_change_pct * prev_centerxywh[3]
    change_amt_w = np.clip(new_centerxywh[2] - prev_centerxywh[2], -max_width_change_px, max_width_change_px)
    change_amt_h = np.clip(new_centerxywh[3] - prev_centerxywh[3], -max_height_change_px, max_height_change_px)

    # Update the center coordinates of the zoom box by adding the capped shift
    updated_centerxywh = [
        prev_centerxywh[0] + capped_shift_x,
        prev_centerxywh[1] + capped_shift_y,
        prev_centerxywh[2] + change_amt_w,
        prev_centerxywh[3] + change_amt_h,
    ]

    # Convert the updated centerxywh coordinates back to xyxy format
    updated_zoom_box_xyxy = convert_centrxywh_to_xyxy(updated_centerxywh)

    # Adjust the zoom box to maintain the aspect ratio of the frame
    updated_zoom_box_xyxy = adjust_zoom_box_aspect_ratio(frame, updated_zoom_box_xyxy)

    # Ensure the zoom box stays within the frame boundaries
    updated_zoom_box_xyxy = keep_zoom_box_inside_frame(updated_zoom_box_xyxy[:2], updated_zoom_box_xyxy[2:], frame)

    # Return the updated zoom box coordinates as a flattened list of integers
    return np.array(updated_zoom_box_xyxy).flatten().astype(int).tolist()


def adjust_zoom_box_aspect_ratio(frame: np.ndarray, zoom_box_xyxy: list) -> list:
    """Adjust the zoom box to have the same center but with the same height-width ratio as the frame.

    Args:
        frame (np.ndarray): The video frame.
        zoom_box_xyxy (list): The original zoom box coordinates in xyxy format.

    Returns:
        list: The adjusted zoom box coordinates in xyxy format.
    """
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    cx, cy, w, h = convert_xyxy_to_centerxy(zoom_box_xyxy)

    if w / h > aspect_ratio:
        new_w = w
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = h
        new_w = int(new_h * aspect_ratio)

    new_zoom_box_xyxy = convert_centrxywh_to_xyxy([cx, cy, new_w, new_h])
    new_zoom_box_xyxy = keep_zoom_box_inside_frame(new_zoom_box_xyxy[:2], new_zoom_box_xyxy[2:], frame)
    return np.array(new_zoom_box_xyxy).flatten().astype(int).tolist()



def convert_xyxy_to_centerxy(box: list) -> list:
    """Convert bounding box coordinates from xyxy to centerxy format.
    
    Args:
        box (list): The bounding box coordinates in xyxy format.
    
    Returns:
        list: The bounding box coordinates in centerxy format.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def convert_centrxywh_to_xyxy(box: list) -> list:
    """Convert bounding box coordinates from centerxy to xyxy format.

    Args:
        box (list): The bounding box coordinates in centerxy format.

    Returns:
        list: The bounding box coordinates in xyxy format.
    """
    cx, cy, w, h = box
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2
    return [x1, y1, x2, y2]


def get_bbox_bottom_center_xy(box: list) -> tuple:
    """Get the bottom center coordinates of the bounding box.
    
    Args:
        box (list): List of bounding box coordinates.

    Returns:
        tuple: (x, y) coordinates for the middle point of the bottom line, representing 
        where the player is standing on the court/field. 
    """
    return (int((box[0] + box[2]) / 2), box[3])
