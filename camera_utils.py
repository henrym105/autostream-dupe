import cv2
import numpy as np
from constants import CAMERA_ZOOM_PADDING


def calculate_optimal_zoom_area(player_positions, aspect_ratio_h_w: float, padding=CAMERA_ZOOM_PADDING) -> list:
    """Calculate the optimal zoom area based on player positions.

    Args:
        player_positions (list of tuples): List of (x, y) coordinates of detected players.
        aspect_ratio_h_w (float): Aspect ratio of the video frame (height / width).
        padding (int): Padding to add around the bounding box.

    Returns:
        list: Coordinates of the zoom area (x_min, y_min, x_max, y_max).
    """
    if not player_positions:
        return None

    # Convert player positions to a numpy array for easier manipulation
    player_positions = np.array(player_positions)

    # Calculate the bounding box
    x_min = np.min(player_positions[:, 0])
    y_min = np.min(player_positions[:, 1])
    x_max = np.max(player_positions[:, 0])
    y_max = np.max(player_positions[:, 1])

    # Add padding to the bounding box
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = x_max + padding
    y_max = y_max + padding

    # Adjust the bounding box to maintain the aspect ratio
    box_width = x_max - x_min
    box_height = y_max - y_min
    box_center_x = x_min + box_width / 2
    box_center_y = y_min + box_height / 2

    if box_height > aspect_ratio_h_w * box_width:
        # Adjust height
        new_height = aspect_ratio_h_w * box_width
        y_min = int(box_center_y - new_height / 2)
        y_max = int(box_center_y + new_height / 2)
    else:
        # Adjust width
        new_width = box_height / aspect_ratio_h_w
        x_min = int(box_center_x - new_width / 2)
        x_max = int(box_center_x + new_width / 2)

    return [x_min, y_min, x_max, y_max]


def zoom_frame(frame, zoom_area):
    """Zooms into the specified area of the frame.
    
    Args:
        frame (numpy.ndarray): The original video frame.
        zoom_area (tuple): A tuple (x, y, w, h) specifying the area to zoom into.
    
    Returns:
        numpy.ndarray: The zoomed frame.
    """
    x, y, w, h = zoom_area
    zoomed_frame = frame[y:y+h, x:x+w]
    zoomed_frame = cv2.resize(zoomed_frame, (frame.shape[1], frame.shape[0]))
    return zoomed_frame


def smooth_transition(prev_frame, curr_frame, alpha=0.5):
    """Smooths the transition between two frames.
    
    Args:
        prev_frame (numpy.ndarray): The previous video frame.
        curr_frame (numpy.ndarray): The current video frame.
        alpha (float): The blending factor (0.0 to 1.0).
    
    Returns:
        numpy.ndarray: The blended frame.
    """
    return cv2.addWeighted(prev_frame, 1 - alpha, curr_frame, alpha, 0)


def interpolate_zoom_area(start_area, end_area, step, total_steps):
    """Interpolate between two zoom areas.
    
    Args:
        start_area (list): Starting zoom area [x_min, y_min, x_max, y_max].
        end_area (list): Ending zoom area [x_min, y_min, x_max, y_max].
        step (int): Current step in the interpolation.
        total_steps (int): Total number of steps for interpolation.
    
    Returns:
        list: Interpolated zoom area [x_min, y_min, x_max, y_max].
    """
    return [
        int(start_area[i] + (end_area[i] - start_area[i]) * step / total_steps)
        for i in range(4)
    ]
