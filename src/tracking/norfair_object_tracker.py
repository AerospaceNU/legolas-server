import numpy as np
from norfair import Detection, Tracker  # type: ignore

from legolas_common.src.tracked_object import (
    BoundingBox,
    DetectedObject,
    Point2D,
    TrackerObject,
)
from tracking.object_tracker import ObjectTracker


class NorfairObjectTracker(ObjectTracker):

    def __init__(self):
        """Initialize the object tracker"""
        super().__init__()
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=300,
            hit_counter_max=15,
            initialization_delay=2,
        )

    def update_detections(self, objects: list[DetectedObject]) -> list[TrackerObject]:
        """Update the list of persistently tracked objects with new detections.

        Args:
            objects: A list of new objects that were detected.

        Returns:
            A list of persistently tracked objects
        """
        norfair_detections = []
        for obj in objects:
            bbox = np.array(
                [
                    [obj.bbox.top_left.x, obj.bbox.top_left.y],
                    [obj.bbox.bottom_right.x, obj.bbox.bottom_right.y],
                ],
            )
            scores = np.array([obj.confidence] * 2)
            norfair_detections.append(Detection(points=bbox, label=obj.class_name, scores=scores))

        tracked_objects = self.tracker.update(detections=norfair_detections)

        new_objects = []
        for tracked in tracked_objects:
            if tracked.id is None:
                continue
            ((x1, y1), (x2, y2)) = tracked.estimate
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            center = Point2D(x=cx, y=cy)
            bounding_box = BoundingBox(Point2D(x1, y1), Point2D(x2, y2))

            tracker_object = TrackerObject(
                class_name=tracked.label,
                persistent_id=tracked.id,
                confidence=tracked.last_detection.scores[0],
                age_ms=tracked.age,
                center=center,
                bbox=bounding_box,
            )
            new_objects.append(tracker_object)
        return new_objects
