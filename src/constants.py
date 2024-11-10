import os

# Runtime options
DRAW_PLAYER_BOXES = False
CROP_VIDEO = True


# Define the current directory as a constant
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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