import time
from enum import Enum, auto
from threading import Lock, Thread

import cv2

from video_input import VideoInput


class CameraType(Enum):
    """Possible camera types"""

    WEBCAM = auto()
    """Standard default webcam capture"""
    NVARGUS = auto()
    """NVIDIA Argus gstreamer capture (the tracking camera connected with ribbon cable)"""


class CameraCapture(VideoInput):

    def __init__(self, camera_type: CameraType):
        self.cap = None
        self.running = False
        self.thread = Thread(target=self._run)
        self.lock = Lock()
        self.camera_type = camera_type

    def start(self):
        if self.camera_type == CameraType.NVARGUS:
            make_cap = lambda: cv2.VideoCapture(
                (
                    "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
                    "width=1920 height=1080 ! nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink"
                ),
                cv2.CAP_GSTREAMER,
            )
        elif self.camera_type == CameraType.WEBCAM:
            make_cap = lambda: cv2.VideoCapture(0)
        else:
            raise ValueError("Invalid camera type")
        self.cap = make_cap()
        if not self.cap.isOpened():
            raise ValueError("Error: Unable to access the webcam.")
        _, self.frame = self.cap.read()
        self.running = True
        self.thread.start()

    def _run(self):
        while self.running:
            _, cap_frame = self.cap.read()
            if self.camera_type == CameraType.NVARGUS:
                cap_frame = cv2.flip(cap_frame, 0)
                cap_frame = cv2.flip(cap_frame, 1)
            with self.lock:
                self.frame = cap_frame
            time.sleep(0.00001)

    def get_frame(self):

        return self.frame

    def shutdown(self):
        self.running = False
        self.thread.join()
        self.cap.release()
