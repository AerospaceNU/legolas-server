import threading
import time

import cv2

from video_input import VideoInput


class VideoReader(VideoInput):

    def __init__(self, video_path: str, loop: bool = True, rotation: int | None = None):
        self.video_path = video_path
        self.cap = None
        self.frame = None
        self.running = False
        self.loop = loop
        self.rotation = rotation

    def start(self):

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30  # fallback to 30 FPS
        self.frame_duration = 1.0 / self.fps

        _, self.frame = self.cap.read()
        if self.rotation is not None:
            self.frame = cv2.rotate(self.frame, self.rotation)
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                if self.loop:
                    self.cap = cv2.VideoCapture(self.video_path)
                    _, frame = self.cap.read()

                else:
                    self.running = False
                    break

            if self.rotation is not None:
                frame = cv2.rotate(frame, self.rotation)

            with self.lock:
                self.frame = frame

            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_duration - elapsed)
            time.sleep(sleep_time)

    def get_frame(self):
        return self.frame

    def shutdown(self):
        self.running = False
        self.thread.join()
        self.cap.release()
