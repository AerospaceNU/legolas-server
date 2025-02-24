from enum import Enum, auto

import cv2


class CameraType(Enum):
    """Possible camera types"""

    WEBCAM = auto()
    """Standard default webcam capture"""
    NVARGUS = auto()
    """NVIDIA Argus gstreamer capture (the tracking camera connected with ribbon cable)"""


class CameraCapture:

    def __init__(self):
        self.cap = None

    def start(self, camera_type: CameraType):
        if camera_type == CameraType.NVARGUS:
            make_cap = lambda: cv2.VideoCapture(
                (
                    "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
                    "width=1920 height=1080 ! nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink"
                ),
                cv2.CAP_GSTREAMER,
            )
        elif camera_type == CameraType.WEBCAM:
            make_cap = lambda: cv2.VideoCapture(0)
        else:
            raise ValueError("Invalid camera type")
        self.cap = make_cap()
        if not self.cap.isOpened():
            raise ValueError("Error: Unable to access the webcam.")

    def get_frame(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to capture frame. Exiting...")

        # Return the frame
        return frame

    def shutdown(self):
        self.cap.release()
