import cv2
import os
import json
import math
import numpy as np
from src.constants import TEMP_CORNERS_COORDS_PATH

# Initialize a list to store the coordinates
coordinates = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        if len(coordinates) >= 3:
            overlay = param.copy()
            cv2.fillPoly(overlay, [np.array(coordinates)], (0, 255, 0))
            cv2.imshow("Select Court Corners", overlay)
        else:
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Court Corners", param)


def rearrange_corner_coords(coordinates) -> list:
    """Rearranges points in clockwise order: 
    [top_left, top_right, bottom_right, bottom_left]
    
    Args:
        coordinates (list): List of (x, y) coordinates.

    Returns:
        dict: Dictionary containing the rearranged coordinates
    """
    top_left = min(coordinates, key=lambda x: (x[1], x[0]))
    
    def angle(point):
        return math.atan2(point[1] - top_left[1], point[0] - top_left[0])
    
    # Sort points based on angle from top-left
    sorted_points = sorted(
        [pt for pt in coordinates if pt != top_left],
        key=angle,
        reverse=True
    )
    
    return [top_left] + sorted_points


def select_court_corners(frame):
    global coordinates

    cv2.imshow("Select Court Corners", frame)
    cv2.setMouseCallback("Select Court Corners", click_event, frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or len(coordinates) == 20:  # Enter key or 20 points selected
            break
    cv2.destroyAllWindows()

    coordinates = rearrange_corner_coords(coordinates)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(TEMP_CORNERS_COORDS_PATH), exist_ok=True)

    with open(TEMP_CORNERS_COORDS_PATH, 'w') as f:
        json.dump(coordinates, f)
        
    print(f"Coordinates saved to {TEMP_CORNERS_COORDS_PATH}")
    return None
    # return coordinates


if __name__ == "__main__":
    video_path = os.path.join(os.getcwd(), "data", "raw", "example_video_2.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    select_court_corners(frame)

    cap.release()
    cv2.destroyAllWindows()