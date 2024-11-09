import cv2
import numpy as np
from constants import CAMERA_ZOOM_PADDING


def calculate_optimal_zoom_area(player_positions, padding=CAMERA_ZOOM_PADDING) -> list:
    """Calculate the optimal zoom area based on player positions.

    Args:
        player_positions (list of tuples): List of (x, y) coordinates of detected players.
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


# Example usage
cap = cv2.VideoCapture('input_video.mp4')
ret, prev_frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Example zoom area (x, y, width, height)
    zoom_area = (100, 100, 300, 300)
    
    zoomed_frame = zoom_frame(frame, zoom_area)
    smooth_frame = smooth_transition(prev_frame, zoomed_frame)
    
    cv2.imshow('Zoomed Frame', smooth_frame)
    
    prev_frame = zoomed_frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()