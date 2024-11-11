import cv2
import os
import json
import time
import numpy as np
from src.constants import TEMP_CORNERS_COORDS_PATH

# Initialize a list to store the coordinates
coordinates = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        print(f"Point selected: ({x}, {y})")
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)  # Draw a bright green dot

        if len(coordinates) > 1:
            cv2.line(param, coordinates[-2], coordinates[-1], (0, 255, 0), 2)
        
        if len(coordinates) == 4:
            cv2.line(param, coordinates[3], coordinates[0], (0, 255, 0), 2)
            cv2.imshow("Select Court Corners", param)
            cv2.waitKey(2000)  # Wait for 2 seconds before exiting
            cv2.destroyAllWindows()

        cv2.imshow("Select Court Corners", param)


def rearrange_corner_coords(coordinates) -> list:
    """Rearranges points in clockwise order: 
    [top_left, top_right, bottom_right, bottom_left]
    
    Args:
        coordinates (list): List of (x, y) coordinates.

    Returns:
        dict: Dictionary containing the rearranged coordinates
    """
    sorted_by_y = sorted(coordinates, key=lambda x: x[1])
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    top_left = min(top_two, key=lambda x: x[0])
    top_right = max(top_two, key=lambda x: x[0])
    bottom_left = min(bottom_two, key=lambda x: x[0])
    bottom_right = max(bottom_two, key=lambda x: x[0])

    return [top_left, top_right, bottom_right, bottom_left]



def select_court_corners(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return

    cv2.imshow("Select Court Corners", frame)
    cv2.setMouseCallback("Select Court Corners", click_event, frame)
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27 or len(coordinates) == 4:
            break

    cv2.destroyAllWindows()  # Ensure all windows are closed after selection
    cap.release()

    corners = rearrange_corner_coords(coordinates)
    print(f"Determined corners: {corners}")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(TEMP_CORNERS_COORDS_PATH), exist_ok=True)

    with open(TEMP_CORNERS_COORDS_PATH, 'w') as f:
        json.dump(corners, f)
    print(f"Coordinates saved to {TEMP_CORNERS_COORDS_PATH}")


if __name__ == "__main__":
    video_path = os.path.join(os.getcwd(), "data", "raw", "example_video.mp4")
    
    select_court_corners(video_path)