import os

# Define the current directory as a constant
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMERA_ZOOM_PADDING = 10

ZOOM_SMOOTHING_ALPHA = 0.9
ZOOM_SMOOTHING_FRAME_COUNT = 1
ZOOM_MIN_WIDTH_PCT = 0.5

YOLO_VERSION = 11

YOLO_HUMAN_CONFIDENCE_THRESHOLD = 0.3