from abc import ABC, abstractmethod

from legolas_common.src.tracked_object import DetectedObject, TrackerObject


class ObjectTracker(ABC):

    @abstractmethod
    def __init__(self):
        """Initialize the object tracker"""
        pass

    @abstractmethod
    def update_detections(self, objects: list[DetectedObject]) -> list[TrackerObject]:
        """Update the list of persistently tracked objects with new detections.

        Args:
            objects: A list of new objects that were detected.

        Returns:
            A list of persistently tracked objects
        """
        pass
