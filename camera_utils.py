import numpy as np
from constants import CAMERA_ZOOM_PADDING


def calculate_optimal_zoom_area(player_positions, padding=CAMERA_ZOOM_PADDING) -> list:
    """
    Calculate the optimal zoom area based on player positions.

    Args:
    - player_positions (list of tuples): List of (x, y) coordinates of detected players.
    - padding (int): Padding to add around the bounding box.

    Returns:
    - zoom_area (tuple): Coordinates of the zoom area (x_min, y_min, x_max, y_max).
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


