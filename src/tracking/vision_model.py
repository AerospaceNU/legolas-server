from abc import ABC, abstractmethod

import cv2

from legolas_common.src.tracked_object import DetectedObject, TrackerObject
from tracking.object_tracker import ObjectTracker


class VisionModel(ABC):

    @abstractmethod
    def __init__(self, object_tracker: ObjectTracker):
        """Initialize the vision model

        Args:
            object_tracker: Class to use for persistent object tracking
        """
        self.object_tracker = object_tracker

    @abstractmethod
    def _process_frame(self, frame: cv2.typing.MatLike) -> list[DetectedObject]:
        """Process an individual frame and output the detections.

        Note that this is just a simple list of detected objects with no knowledge
        of persistent tracking.

        Args:
            frame: The input raw frame to be processed

        Returns:
            The list of objects seen in this frame.
        """
        pass

    def update(self, frame: cv2.typing.MatLike) -> list[TrackerObject]:
        """Update the tracker with a new frame

        Args:
            frame: The input raw frame to be processed

        Returns:
            The list of persistent tracked objects that are currently known to the model.
        """
        frame_detections = self._process_frame(frame)
        return self.object_tracker.update_detections(frame_detections)
