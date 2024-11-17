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
        list: List containing the rearranged coordinates
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


def select_court_corners(frame) -> list:
    """Allows the user to manually select the corners of a court in a given frame.
    This function displays the provided frame in a window and allows the user to 
    click on the corners of the court. The coordinates of the selected points are 
    stored globally. The function waits until the user presses the Enter key or 
    selects 20 points, whichever comes first. 

    Args:
        frame (numpy.ndarray): The image frame in which the court corners are to be selected.
    """

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
    return coordinates


def infer_4_corners(all_points: list) -> list[int]:
    """Infer the 4 corner points of the court from up to 20 selected points.
    Need the 4 corner points for the perspective transform that creates the minimap. 
    
    Args:
        all_points (list): List of up to 20 (x, y) coordinates.
    
    Returns:
        list: List of 4 (x, y) coordinates.
    """
    # Rearrange the points in clockwise order
    all_points = rearrange_corner_coords(all_points)
    
    # Calculate the center of the court
    center_x = sum([pt[0] for pt in all_points]) / 20
    center_y = sum([pt[1] for pt in all_points]) / 20
    
    # Sort the points based on their distance from the center
    all_points.sort(key=lambda pt: math.sqrt((pt[0] - center_x) ** 2 + (pt[1] - center_y) ** 2))
    
    # Return the 4 corner points
    return all_points[:4]



if __name__ == "__main__":
    video_path = os.path.join(os.getcwd(), "data", "raw", "example_video_2.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    select_court_corners(frame)

    cap.release()
    cv2.destroyAllWindows()