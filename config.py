"""
Configuration file for Smart Vehicle Traffic Sign Recognition System

This file contains all configurable parameters for the system.
Copy this file and modify the values according to your setup.
"""

# Network Configuration
VEHICLE_IP = "192.168.1.1"
SOCKET_PORT = 2001
CAMERA_STREAM_URL = "http://192.168.1.1:8080/?action=stream"

# Model Configuration
SVM_MODEL_PATH = "./svm_model.pkl"
MODEL_INPUT_SIZE = (64, 64)  # Width, Height for image resize

# Image Processing Parameters
# HSV Color Thresholds for Blue Signs
BLUE_HSV_MIN = (110, 43, 46)
BLUE_HSV_MAX = (124, 255, 255)

# HSV Color Thresholds for Red Signs
RED_HSV_MIN = (165, 43, 46)
RED_HSV_MAX = (180, 255, 255)

# Contour Detection Parameters
MIN_AREA_RATIO = 1 / (25 * 25)  # Minimum area relative to image size
MAX_AREA_RATIO = -1  # Maximum area (-1 means image size)
WIDTH_HEIGHT_RATIO = 2.0  # Maximum width/height or height/width ratio

# HOG Feature Extraction Parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2"
HOG_TRANSFORM_SQRT = True

# Traffic Sign Classes
CLS_NAMES = [
    "Speed_limit_15",
    "Speed_limit_30", 
    "Speed_limit_60",
    "Speed_limit_80",
    "No straight",
    "Turn left",
    "Turn right",
    "background"
]

IMG_LABELS = {
    "Speed_limit_15": 0,
    "Speed_limit_30": 1,
    "Speed_limit_60": 2,
    "Speed_limit_80": 3,
    "No straight": 4,
    "Turn left": 5,
    "Turn right": 6,
    "background": 7
}

# Vehicle Control Commands (Hexadecimal)
# Format: \xff\xXX\xYY\xZZ\xff
COMMANDS = {
    "STOP": {
        "right_track": b'\xff\x02\x01\x00\xff',
        "left_track": b'\xff\x02\x02\x00\xff'
    },
    "FORWARD": b'\xff\x00\x01\x00\xff',
    "TURN_LEFT": b'\xff\x00\x04\x00\xff',
    "TURN_RIGHT": b'\xff\x00\x03\x00\xff',
    "SPEED_15": {
        "right_track": b'\xff\x02\x01\x40\xff',
        "left_track": b'\xff\x02\x02\x40\xff'
    },
    "SPEED_30": {
        "right_track": b'\xff\x02\x01\x45\xff',
        "left_track": b'\xff\x02\x02\x47\xff'
    },
    "SPEED_60": {
        "right_track": b'\xff\x02\x01\x30\xff',
        "left_track": b'\xff\x02\x02\x30\xff'
    },
    "SPEED_80": {
        "right_track": b'\xff\x02\x01\x60\xff',
        "left_track": b'\xff\x02\x02\x60\xff'
    }
}

# Display Settings
WINDOW_NAME = "camera"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
FRAME_DELAY = 40  # milliseconds between frames

# Threading Settings
CAMERA_THREAD_DAEMON = True

# Debug Settings
SHOW_BINARY_IMAGE = True
SHOW_PROPOSAL = True
PRINT_CLASSIFICATION = True
