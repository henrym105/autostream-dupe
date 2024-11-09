import cv2
import numpy as np
from constants import CAMERA_ZOOM_PADDING
import time



def clip_coordinates(x, y, w, h, img_width, img_height):
    x1 = max(0, min(x, img_width - 1))
    y1 = max(0, min(y, img_height - 1))
    x2 = max(0, min(x + w, img_width - 1))
    y2 = max(0, min(y + h, img_height - 1))
    return x1, y1, x2, y2



def calculate_optimal_zoom_area(frame, player_positions, frame_display_size_h_w, padding=CAMERA_ZOOM_PADDING) -> list:
    """Calculate the optimal zoom area based on player positions. 
    the resulting box will have the same aspect ratio as the `frame`, centered on the player_positions
    must be at least `padding` pixels inside of the outer dimensions of the current frame.

    Args:
        frame (numpy.ndarray): The original video frame.
        player_positions [[tl_x, tl_y], [br_x, br_y]]: list of [x,y] coordinates for the top left and bottom right corners of bounding box surroinding the players.
        frame_display_size_h_w (tuple): The height and width of the current frame. Used to return crop box with constant aspect ratio.
        padding (int): The padding to add around the zoom area.

    Returns:
        list: Coordinates of the zoom area (x_min, y_min, x_max, y_max).
    """
    if not player_positions:
        return None
    player_positions = np.array(player_positions)
    
    x_coords = [pos[0] for pos in player_positions]
    y_coords = [pos[1] for pos in player_positions]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Calculate the center of the bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate the width and height of the bounding box
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Maintain the aspect ratio of the frame
    frame_height, frame_width = frame_display_size_h_w
    aspect_ratio = frame_width / frame_height

    if box_width / box_height > aspect_ratio:
        zoom_width = box_width + 2 * padding
        zoom_height = int(zoom_width / aspect_ratio)
    else:
        zoom_height = box_height + 2 * padding
        zoom_width = int(zoom_height * aspect_ratio)

    # Calculate the top-left and bottom-right coordinates of the zoom area
    x1 = center_x - zoom_width // 2
    y1 = center_y - zoom_height // 2
    x2 = center_x + zoom_width // 2
    y2 = center_y + zoom_height // 2

    # Clip the coordinates to ensure they are within the frame boundaries
    x1, y1, x2, y2 = clip_coordinates(x1, y1, zoom_width, zoom_height, frame_width, frame_height)

    return [x1, y1, x2, y2]



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



