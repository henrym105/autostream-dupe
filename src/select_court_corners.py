import cv2
import os
import json
import math
import numpy as np
from src.constants import TEMP_COURT_OUTLINE_COORDS_PATH, TEMP_4_CORNERS_COORDS_PATH

# Initialize a list to store the coordinates
coordinates = []

def mouse_event(event, x, y, flags, param):
    coordinates = param['coordinates']
    border_size = param['border_size']
    frame = param['frame']
    overlay = frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust coordinates to account for the border
        adjusted_x = x - border_size
        adjusted_y = y - border_size
        coordinates.append((adjusted_x, adjusted_y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Court Corners", frame)

    if event == cv2.EVENT_MOUSEMOVE and len(coordinates) > 0:
        cv2.line(overlay, (coordinates[-1][0] + border_size, coordinates[-1][1] + border_size), (x, y), (0, 255, 0), 2)
        cv2.imshow("Select Court Corners", overlay)

    if len(coordinates) >= 2:
        cv2.line(overlay, (coordinates[0][0] + border_size, coordinates[0][1] + border_size), 
                 (coordinates[1][0] + border_size, coordinates[1][1] + border_size), (0, 255, 0), 2)
        cv2.imshow("Select Court Corners", overlay)

    if len(coordinates) >= 3:
        cv2.fillPoly(overlay, [np.array(coordinates) + border_size], (0, 255, 0))
        cv2.imshow("Select Court Corners", overlay)


def rearrange_corner_coords(coordinates) -> list:
    """Rearranges points in clockwise order: 
    [top_left, top_right, bottom_right, bottom_left]
    
    Args:
        coordinates (list): List of (x, y) coordinates.

    Returns:
        list: List containing the rearranged coordinates
    """
    # Find the top-left point
    top_left = min(coordinates, key=lambda x: (x[1], x[0]))
    
    # Calculate the angle of each point from the top-left point
    def angle(point):
        return math.atan2(point[1] - top_left[1], point[0] - top_left[0])
    
    # Sort points based on angle from top-left
    sorted_points = sorted(coordinates, key=angle)
    
    # Ensure the top-left point is first
    sorted_points.remove(top_left)
    sorted_points.insert(0, top_left)
    
    return sorted_points


def add_border_to_frame(frame, border_ratio=0.2):
    """Add a black border around the frame.
    
    Args:
        frame (numpy.ndarray): The original frame.
        border_ratio (float): The ratio of the border size to the width of the frame.
    
    Returns:
        numpy.ndarray: The frame with the added border.
    """
    border_size = int(frame.shape[1] * border_ratio)
    return cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])


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
    
    border_ratio = 0.2
    frame_with_border = add_border_to_frame(frame, border_ratio)
    border_size = int(frame.shape[1] * border_ratio)
    print(f"{border_size = }")

    # Display instructions on the frame
    instructions = [
        "Click on the court corners in clockwise order starting from the far-left corner.",
        "Click enter to finish."
    ]
    y0, dy = 30, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame_with_border, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Select Court Corners", frame_with_border)

    # cv2.imshow("Select Court Corners", frame_with_border)
    click_params = {
        'frame': frame_with_border, 
        'border_size': border_size, 
        'coordinates': coordinates
    }
    cv2.setMouseCallback("Select Court Corners", mouse_event, click_params)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or len(coordinates) == 20:  # Enter key or 20 points selected
            break
    cv2.destroyAllWindows()

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(TEMP_COURT_OUTLINE_COORDS_PATH), exist_ok=True)

    with open(TEMP_COURT_OUTLINE_COORDS_PATH, 'w') as f:
        json.dump(coordinates, f)
        
    print(f"Coordinates saved to {TEMP_COURT_OUTLINE_COORDS_PATH}")
    return coordinates


def infer_4_corners(all_points: list) -> list[int]:
    """Infer the 4 corner points of the court from up to 20 selected points.
    Need the 4 corner points for the perspective transform that creates the minimap. 
    Returns the 4 points that are farthest from the geometric center of the image.
    
    Args:
        all_points (list): List of up to 20 (x, y) coordinates.
    
    Returns:
        list: List of 4 (x, y) coordinates.
    """
    
    # Calculate the center of the court
    center_x = sum([pt[0] for pt in all_points]) / 20
    center_y = sum([pt[1] for pt in all_points]) / 20
    
    # Sort the points based on their distance from the center, return 4 farthest points
    all_points.sort(key=lambda pt: math.sqrt((pt[0] - center_x)**2 + (pt[1] - center_y)**2), reverse=True)
    corners = all_points[:4]
    
    # Rearrange the points in clockwise order
    corners = rearrange_corner_coords(corners)

    # Save the 4 corner points to a file
    with open(TEMP_4_CORNERS_COORDS_PATH, 'w') as f:
        json.dump(all_points[:4], f)
    
    return corners



if __name__ == "__main__":
    video_path = os.path.join(os.getcwd(), "data", "raw", "example_video_2.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    select_court_corners(frame)

    cap.release()
    cv2.destroyAllWindows()