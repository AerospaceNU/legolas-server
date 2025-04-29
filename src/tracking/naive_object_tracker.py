from legolas_common.src.tracked_object import DetectedObject, TrackerObject
from tracking.object_tracker import ObjectTracker


class NaiveObjectTracker(ObjectTracker):

    def __init__(self):
        """Initialize the object tracker"""
        super().__init__()
        self.current_id = 0

    def update_detections(self, objects: list[DetectedObject]) -> list[TrackerObject]:
        """Update the list of persistently tracked objects with new detections.

        Args:
            objects: A list of new objects that were detected.

        Returns:
            A list of persistently tracked objects
        """
        output_detections = []
        for obj in objects:
            output_detections.append(
                TrackerObject(
                    class_name=obj.class_name,
                    persistent_id=self.current_id,
                    confidence=obj.confidence,
                    age_ms=0,
                    center=obj.center,
                    bbox=obj.bbox,
                )
            )
            self.current_id += 1

        return output_detections
