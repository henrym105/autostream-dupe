import cv2
import numpy as np
from constants import CAMERA_ZOOM_PADDING
import time


def keep_zoom_box_inside_frame(tl_point, br_point, img_width, img_height):
    """Adjust the zoom box to ensure it stays within the frame boundaries.
    
    Args:
        tl_point (tuple): The top-left corner of the zoom box.
        br_point (tuple): The bottom-right corner of the zoom box.
        img_width (int): The width of the frame.
        img_height (int): The height of the frame.
        
    Returns:
        tuple: ((tl_x, tl_y), (br_x, br_y)) The adjusted top-left and bottom-right corners of the zoom box.
        """
    # Ensure the top-left corner is within the frame boundaries
    if tl_point[0] < 0:
        tl_point[0] = 0
    if tl_point[1] < 0:
        tl_point[1] = 0

    # Ensure the bottom-right corner is within the frame boundaries
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

    return tl_point, br_point


def calculate_optimal_zoom_area(frame, player_positions_xyxy, frame_display_size_h_w, padding=CAMERA_ZOOM_PADDING) -> list:
    """Calculate the optimal zoom area based on player positions. 
    The resulting box will have the same aspect ratio as the `frame`, centered on the middle of player_positions,
    and will be large enough to include all player positions with at least `padding` pixels inside the outer dimensions of the current frame.

    Args:
        frame (numpy.ndarray): The original video frame.
        player_positions [[tl_x, tl_y], [br_x, br_y]]: list of [x,y] coordinates for the top left and bottom right corners of bounding box surrounding the players.
        frame_display_size_h_w (tuple): The height and width of the current frame. Used to return crop box with constant aspect ratio.
        padding (int): The padding to add around the zoom area.

    Returns:
        list: Coordinates of the zoom area (x_min, y_min, x_max, y_max).
    """
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

    # AT THIS POINT: [x_min, y_min, x_max, y_max] will perfectly encompass all player bboxes

    # this is the minimum width and height needed for a bounding box to fully encompass all players bboxes
    min_width_zoom_box = (x_max - x_min) + (2*padding)
    min_height_zoom_box = (y_max - y_min) + (2*padding)
    # Ensure the minimum width and height are at least 50% of the frame dimensions
    min_width_zoom_box = max(min_width_zoom_box, frame_width // 2)
    min_height_zoom_box = max(min_height_zoom_box, frame_height // 2)
    tl_point = [x_min-padding, y_min-padding]

    # AT THIS POINT: [tl[0], tl[1], tl[0]+min_width_zoom_box, tl[1]+min_height_zoom_box] will perfectly encompass all 
    # player bboxes with padding around the edges

    # Calculate the width and height of the smallest rectangle that is at least min_width and min_height and has the same w/h aspect_ratio
    if min_width_zoom_box / min_height_zoom_box > aspect_ratio:
        zoom_width = min_width_zoom_box
        zoom_height = int(zoom_width / aspect_ratio)
    else:
        zoom_height = min_height_zoom_box
        zoom_width = int(zoom_height * aspect_ratio)

    # Define the bottom right coordinates for the zoom box
    br_point = [tl_point[0]+zoom_width, tl_point[1]+zoom_height]

    # Ensure the zoom box stays within the frame boundaries
    # tl_point, br_point = keep_zoom_box_inside_frame(tl_point, br_point, frame_width, frame_height)

    # AT THIS POINT: [tl_point + br_point] will be the zoom box that has the same aspect ratio as 
    # the frame and will not extend beyond the image edges

    # center of the person bboxes
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate the height of the zoom box to keep the aspect ratio with this width value
    zoom_width = min_width_zoom_box
    zoom_height = int(min_width_zoom_box / aspect_ratio)

    # Define the bottom right coordinates for the zoom box
    br_point = [tl_point[0]+zoom_width, tl_point[1]+zoom_height]

    # Ensure the zoom box stays within the frame boundaries
    # tl_point, br_point = keep_zoom_box_inside_frame(tl_point, br_point, frame_width, frame_height)

    # top left corner
    tl_x = max(0, int(center_x - zoom_width // 2))
    tl_y = max(0, int(center_y - zoom_height // 2))

    br_x = min(frame_width, int(center_x + zoom_width // 2))
    br_y = min(frame_height, int(center_y + zoom_height // 2))

    tl_point, br_point = keep_zoom_box_inside_frame((tl_x, tl_y), (br_x, br_y), frame_width, frame_height)

    return [tl_point[0], tl_point[1], br_point[0], br_point[1]]
    # return [tl_x, tl_y, br_x, br_y]


def zoom_frame(frame, zoom_area):
    """Zooms into the specified area of the frame.
    
    Args:
        frame (numpy.ndarray): The original video frame.
        zoom_area (tuple): A tuple (x, y, w, h) specifying the area to zoom into.
    
    Returns:
        numpy.ndarray: The zoomed frame.
    """
    tlx, tly, brx, bry = zoom_area
    zoomed_frame = frame[tly:bry, tlx:brx]
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



