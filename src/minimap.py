import cv2
import numpy as np
from src.camera_utils import get_bbox_bottom_center_xy
from src.constants import MINIMAP_SIZE


def create_minimap(
    frame: np.ndarray, 
    court_coords: list, 
    player_positions: list, 
    minimap_width: float = MINIMAP_SIZE
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
    player_locations_xy = [get_bbox_bottom_center_xy(box) for box in player_positions]

    # Calculate the height to maintain the aspect ratio of a basketball court (94:50 ~ 1.88)
    minimap_width = int(minimap_width * frame.shape[1])  # Convert to integer
    minimap_height = int(minimap_width / 1.88)
    minimap = np.zeros((minimap_height, minimap_width, 4), dtype=np.uint8)  # 4 channels for RGBA

    # Transform court coordinates to fit the minimap
    court_polygon = np.array(court_coords, dtype=np.float32)
    m = cv2.getPerspectiveTransform(
            court_polygon, 
            np.array([[0, 0], [minimap_width, 0], [minimap_width, minimap_height], [0, minimap_height]], dtype=np.float32)
        )

    minimap_court_polygon = cv2.perspectiveTransform(court_polygon.reshape(-1, 1, 2), m)

    # Draw the court outline on the minimap
    cv2.polylines(minimap, [minimap_court_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0, 255), thickness=1)

    # Draw the half-court line
    half_court_x = minimap_width // 2
    cv2.line(minimap, (half_court_x, 0), (half_court_x, minimap_height), (0, 255, 0, 255), 2)

    # Draw the three-point arcs
    three_point_radius = int(minimap_width * 0.237)  # Approximate radius of three-point arc
    cv2.ellipse(minimap, (0, minimap_height // 2), (three_point_radius, three_point_radius), 0, -90, 90, (0, 255, 0, 255), 2)
    cv2.ellipse(minimap, (minimap_width, minimap_height // 2), (three_point_radius, three_point_radius), 0, 90, 270, (0, 255, 0, 255), 2)

    # Draw the box (paint area)
    box_width = int(minimap_width * 0.16)  # Approximate width of the box
    box_height = int(minimap_height * 0.19)  # Approximate height of the box
    cv2.rectangle(minimap, (0, minimap_height // 2 - box_height // 2), (box_width, minimap_height // 2 + box_height // 2), (0, 255, 0, 255), 2)
    cv2.rectangle(minimap, (minimap_width - box_width, minimap_height // 2 - box_height // 2), (minimap_width, minimap_height // 2 + box_height // 2), (0, 255, 0, 255), 2)

    # Transform and draw player positions on the minimap
    for player_center in player_locations_xy:
        minimap_player_center = cv2.perspectiveTransform(np.array(player_center, dtype=np.float32).reshape(-1, 1, 2), m)
        cv2.circle(minimap, tuple(minimap_player_center[0][0].astype(int)), 5, (0, 0, 255, 255), -1)

    return minimap



def add_minimap_to_frame(frame, player_bboxes, four_corner_points_xy) -> np.ndarray:
    """Add a minimap to the frame showing the court and player positions.
    Args:
        frame (np.ndarray): The original video frame.
        player_bboxes (list): List of player bounding boxes.
        four_corner_points_xy (list): List of 4 corner points of the court.
    Returns:
        np.ndarray: The frame with the minimap overlay.
    """
    minimap = create_minimap(frame, four_corner_points_xy, player_bboxes)
    overlay = frame.copy()
    overlay[0:minimap.shape[0], 0:minimap.shape[1]] = minimap[..., :3]  # Only take RGB channels
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)  # Blend the minimap with the frame
    return frame


