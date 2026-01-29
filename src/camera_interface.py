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

    def __init__(self, camera_type: CameraType, device_id=None):
        """
        Initialize camera capture
        
        Args:
            camera_type: Type of camera (WEBCAM or NVARGUS)
            device_id: Device index for webcam (default: 1 for USB cameras on Jetson)
        """
        self.cap = None
        self.running = False
        self.thread = Thread(target=self._run)
        self.lock = Lock()
        self.camera_type = camera_type
        # Default to video1 for WEBCAM (USB camera on Jetson Orin)
        # video0 is typically the CSI camera
        self.device_id = device_id if device_id is not None else (1 if camera_type == CameraType.WEBCAM else 0)

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
            # Use the specified device_id (defaults to 1 for USB camera)
            make_cap = lambda: cv2.VideoCapture(self.device_id)
        else:
            raise ValueError("Invalid camera type")
        
        print(f"Opening camera: {self.camera_type.name} at device {self.device_id if self.camera_type == CameraType.WEBCAM else 'GStreamer'}")
        self.cap = make_cap()
        
        if not self.cap.isOpened():
            raise ValueError(f"Error: Unable to access the webcam at /dev/video{self.device_id}. "
                           f"Check permissions: sudo chmod 666 /dev/video{self.device_id}")
        
        print(f"✓ Camera opened successfully")
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