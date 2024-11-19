import os

# Runtime options
DRAW_PLAYER_BOXES = True
DRAW_COURT_BOX = True
DRAW_MINIMAP = True
MINIMAP_SIZE = 0.2
CROP_VIDEO = True
SAVE_VIDEO_LOCAL = True

# Define the current directory as a constant
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the temporary file storing the corner coordinates
TEMP_COURT_OUTLINE_COORDS_PATH = os.path.join(CUR_DIR, "data", "temp", "court_outline_coords_xy.txt")
TEMP_4_CORNERS_COORDS_PATH = os.path.join(CUR_DIR, "data", "temp", "court_4_corners_coords_xy.txt")

# Padding for camera zoom
CAMERA_ZOOM_PADDING = 10
CAMERA_ZOOM_PADDING_PCT = 0.10

# Smoothing parameters for zoom transition
FRAME_MAX_SHIFT_PCT = 0.003
FRAME_MAX_ZOOM_CHANGE_PCT = .01
ZOOM_SMOOTHING_FRAME_COUNT = 1

# Minimum width percentage for zoom
ZOOM_MIN_WIDTH_PCT = 0.5

# YOLO model version
YOLO_VERSION = 11

# Confidence threshold for detecting humans using YOLO
YOLO_HUMAN_CONFIDENCE_THRESHOLD = 0.15