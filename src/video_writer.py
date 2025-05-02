import os

import cv2

from legolas_common.src.frame_annotator import draw_tracked_object


class VideoWriter:
    def __init__(self, filename, fps=60, frame_size=(640, 480), include_annotations=True):
        prefix, ext = os.path.splitext(filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.standard_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size, True)
        if include_annotations:
            annotated_name = f"{prefix}_annotated{ext}"
            self.annotated_writer = cv2.VideoWriter(annotated_name, fourcc, fps, frame_size, True)
        else:
            self.annotated_writer = None
        self.frame_size = frame_size

    def write(self, frame, detections=None):
        frame = frame.copy()
        if detections is None:
            detections = []
        # Resize if frame size mismatches
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self.standard_writer.write(frame)

        if self.annotated_writer is not None:
            for detection in detections:
                draw_tracked_object(frame, detection)
            self.annotated_writer.write(frame)

    def release(self):
        self.standard_writer.release()
        if self.annotated_writer is not None:
            self.annotated_writer.release()
