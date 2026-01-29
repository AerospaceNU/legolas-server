import cv2
from ultralytics import YOLO  # type: ignore

from legolas_common.src.tracked_object import BoundingBox, DetectedObject, Point2D
from tracking.object_tracker import ObjectTracker
from tracking.vision_model import VisionModel


def yolo_output_to_detected_objects(results, class_names) -> list[DetectedObject]:
    detected_objects = []

    for result in results:
        for box in result.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bbox = BoundingBox(Point2D(x1, y1), Point2D(x2, y2))
            detected_objects.append(
                DetectedObject(
                    class_name=class_names[class_id],
                    confidence=conf,
                    center=Point2D(center_x, center_y),
                    bbox=bbox,
                )
            )

    return detected_objects


class YoloModel(VisionModel):

    def __init__(self, object_tracker: ObjectTracker, yolo_file: str = "weights.pt", image_size=320):
        super().__init__(object_tracker)
        self.image_size = image_size
        
        # Initialize YOLO model and force it to use GPU
        print("Loading YOLO model...")
        self.model = YOLO(yolo_file, verbose=False)
        self.model.to('cuda')  # Force GPU usage
        print(f"✓ YOLO model loaded on device: {next(self.model.model.parameters()).device}")
        
        self.class_names = self.model.names
        self.valid_names = set(
            (
                "rockets",
                "rocket",
                "trails",
                "boat",
                "airplane",
                "kite",
                "person",
                "parachute",
                "descending_rocket",
                "laptop",
            )
        )

    def _process_frame(self, frame) -> list[DetectedObject]:
        # Inference on GPU
        results = self.model(frame, conf=0.55, imgsz=self.image_size, device='cuda')
        detected_objects = yolo_output_to_detected_objects(results, self.class_names)
        valid_objects = filter(
            lambda obj: obj.class_name in self.valid_names or "rocket" in obj.class_name,
            detected_objects,
        )
        return valid_objects