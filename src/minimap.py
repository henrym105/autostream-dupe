import cv2
import numpy as np
from src.camera_utils import get_bbox_bottom_center_xy


def create_minimap(
    frame: np.ndarray, 
    court_coords: list, 
    player_bboxes: list, 
    minimap_width: float = 0.25
) -> np.ndarray:
    """Create a minimap showing the bird's eye view of the court and player positions.

    Args:
        frame (np.ndarray): The original video frame.
        court_coords (list): List of court corner coordinates.
        player_positions (list): List of player bounding box coordinates.
        minimap_width (int, optional): Width of the minimap as ratio of zoomed frame width

    Returns:
        np.ndarray: The minimap image.
    """
    # Calculate the minimap dimensions
    frame_height, frame_width = frame.shape[:2]
    minimap_height = int(minimap_width * frame_height)
    minimap_width = int(minimap_width * frame_width)

    # Create a blank minimap
    minimap = np.zeros((minimap_height, minimap_width, 3), dtype=np.uint8)

    # Scale court coordinates to minimap size
    # tl tr br bl
    court_coords = np.array(court_coords)  # Convert list to NumPy array
    court_coords[:, 0] = court_coords[:, 0] * minimap_width / frame_width
    court_coords[:, 1] = court_coords[:, 1] * minimap_height / frame_height

    # Draw the court on the minimap
    cv2.polylines(minimap, [court_coords.astype(np.int32)], isClosed=True, color=(255, 255, 255), thickness=1)

    # fill court with white color
    cv2.fillPoly(minimap, [court_coords.astype(np.int32)], (150, 150, 150))

    # Get player locations and scale them to minimap size
    player_locations_xy = [get_bbox_bottom_center_xy(box) for box in player_bboxes]
    player_locations_xy = np.array(player_locations_xy, dtype=np.float32)
    player_locations_xy[:, 0] = player_locations_xy[:, 0] * minimap_width / frame_width
    player_locations_xy[:, 1] = player_locations_xy[:, 1] * minimap_height / frame_height

    # Draw players on the minimap
    for (x, y) in player_locations_xy.astype(np.int32):
        cv2.circle(minimap, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    return minimap



def add_minimap_to_frame(frame, minimap) -> np.ndarray:
    """Add a minimap to the frame showing the court and player positions.
    Args:
        frame (np.ndarray): The original video frame.
        minimap (np.ndarray): The minimap image.
    Returns:
        np.ndarray: The frame with the minimap overlay.
    """
    overlay = frame.copy()
    minimap_height, minimap_width = minimap.shape[:2]
    overlay[0:minimap_height, 0:minimap_width] = minimap[..., :3]  # Only take RGB channels
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)  # Blend the minimap with the frame
    return frame


