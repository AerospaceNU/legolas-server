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
                       or sensor-id for NVARGUS (default: 0 for CSI port that worked)
        """
        self.cap = None
        self.frame = None  # initialize to avoid race before first read
        self.running = False
        self.thread = Thread(target=self._run, daemon=True)
        self.lock = Lock()
        self.camera_type = camera_type
        # WEBCAM defaults to video1 (USB); NVARGUS defaults to sensor-id=0 (CSI)
        self.device_id = device_id if device_id is not None else (1 if camera_type == CameraType.WEBCAM else 0)

    def _make_nvargus_pipeline(self, width=1920, height=1080, framerate=60):
        return (
            f"nvarguscamerasrc sensor-id={self.device_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={framerate}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "   # nvvidconv outputs BGRx to system memory
            f"videoconvert ! video/x-raw, format=BGR ! " # videoconvert handles BGRx -> BGR
            f"appsink drop=1"
        )

    def start(self):
        if self.camera_type == CameraType.NVARGUS:
            pipeline = self._make_nvargus_pipeline()
            print(f"Opening camera: NVARGUS sensor-id={self.device_id}")
            print(f"Pipeline: {pipeline}")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif self.camera_type == CameraType.WEBCAM:
            print(f"Opening camera: WEBCAM at device {self.device_id}")
            self.cap = cv2.VideoCapture(self.device_id)
        else:
            raise ValueError(f"Invalid camera type: {self.camera_type}")

        if not self.cap.isOpened():
            if self.camera_type == CameraType.NVARGUS:
                raise ValueError(
                    f"Error: Unable to open NVARGUS camera on sensor-id={self.device_id}. "
                    f"Ensure nvargus-daemon is running: sudo systemctl start nvargus-daemon"
                )
            else:
                raise ValueError(
                    f"Error: Unable to access webcam at /dev/video{self.device_id}. "
                    f"Check permissions: sudo chmod 666 /dev/video{self.device_id}"
                )

        # Grab first frame before starting thread
        ret, self.frame = self.cap.read()
        if not ret or self.frame is None:
            raise ValueError("Camera opened but failed to read first frame.")

        print(f"✓ Camera opened successfully — frame size: {self.frame.shape}")
        self.running = True
        self.thread.start()

    def _run(self):
        while self.running:
            ret, cap_frame = self.cap.read()
            if not ret or cap_frame is None:
                continue  # skip bad frames instead of crashing

            if self.camera_type == CameraType.NVARGUS:
                cap_frame = cv2.flip(cap_frame, 0)
                cap_frame = cv2.flip(cap_frame, 1)

            with self.lock:
                self.frame = cap_frame

            time.sleep(0.00001)

    def get_frame(self):
        with self.lock:  # was missing lock — fixes potential race condition
            return self.frame

    def shutdown(self):
        self.running = False
        self.thread.join(timeout=5)  # don't hang forever if thread is stuck
        if self.cap:
            self.cap.release()