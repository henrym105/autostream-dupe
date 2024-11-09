import os

# Define the current directory as a constant
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_ZOOM_PADDING = 50

ZOOM_SMOOTHING_ALPHA = 0.9
ZOOM_SMOOTHING_FRAME_COUNT = 1


YOLO_3_FILE_DOWNLOAD_PATHS = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

YOLO_VERSION = 8
