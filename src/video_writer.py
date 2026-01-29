from dataclasses import asdict
import os

import json

import cv2

from legolas_common.src.frame_annotator import draw_tracked_object


class VideoWriter:
    def __init__(self, filename, fps=60, frame_size=(640, 480), include_annotations=True):
        prefix, ext = os.path.splitext(filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.standard_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size, True)
        if not self.standard_writer.isOpened():
            raise ValueError("Not opened")
        if include_annotations:
            self.annotations_log_name = f"{filename}.annotations"
        else:
            self.annotations_log_name = None
        self.frame_size = frame_size

    def write(self, frame, detections=None):
        if detections is None:
            detections = []
        # Resize if frame size mismatches
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self.standard_writer.write(frame)

        if self.annotations_log_name is not None:
            dict_list = [asdict(entry) for entry in detections]
            with open(self.annotations_log_name, "a") as f:
                f.write(json.dumps(dict_list) + '\n')

    def release(self):
        print("releasing")
        self.standard_writer.release()
