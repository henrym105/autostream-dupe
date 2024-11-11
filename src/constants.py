import os

# Runtime options
DRAW_PLAYER_BOXES = True
DRAW_COURT_BOX = True
CROP_VIDEO = False
SAVE_VIDEO_LOCAL = True

# Define the current directory as a constant
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the temporary file storing the corner coordinates
TEMP_CORNERS_COORDS_PATH = os.path.join(CUR_DIR, "data", "temp", "court_corners_coords_xy.txt")

# Padding for camera zoom
CAMERA_ZOOM_PADDING = 30

# Smoothing parameters for zoom transition
ZOOM_SMOOTHING_ALPHA = 0.01
ZOOM_SMOOTHING_FRAME_COUNT = 1

# Minimum width percentage for zoom
ZOOM_MIN_WIDTH_PCT = 0.5

# YOLO model version
YOLO_VERSION = 11

# Confidence threshold for detecting humans using YOLO
YOLO_HUMAN_CONFIDENCE_THRESHOLD = 0.5