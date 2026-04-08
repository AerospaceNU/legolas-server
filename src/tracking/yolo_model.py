from ultralytics import YOLO  # type: ignore

from legolas_common.src.tracked_object import BoundingBox, DetectedObject, Point2D
from tracking.object_tracker import ObjectTracker
from tracking.vision_model import VisionModel
import threading
import time


def yolo_output_to_detected_objects(results, class_names) -> list[DetectedObject]:
    detected_objects = []

    for result in results:
        for box in result.boxes:
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

        print("Loading YOLO model...")
        self.model = YOLO(yolo_file, verbose=False)
        self.model.to('cuda')
        print(f"YOLO model loaded on device: {next(self.model.model.parameters()).device}")

        self.class_names = self.model.names

    def _process_frame(self, frame) -> list[DetectedObject]:
        results = self.model(frame, conf=0.55, imgsz=self.image_size, device='cuda')
        return yolo_output_to_detected_objects(results, self.class_names)


class YoloThread:
    def __init__(self, model):
        self.model = model
        self.latest_frame = None
        self.latest_detections = []
        self.new_detections = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def update_frame(self, frame):
        """Called from main loop to give YOLO the latest frame"""
        with self.lock:
            self.latest_frame = frame

    def get_detections(self):
        """Called from main loop to get the latest detections"""
        with self.lock:
            if not self.new_detections:
                return []
            self.new_detections = False
            return self.latest_detections

    def get_detections_nowait(self):
        with self.lock:
            return self.latest_detections

    def _run(self):
        while not self.stop_event.is_set():
            with self.lock:
                frame = self.latest_frame

            if frame is None:
                time.sleep(0.001)
                continue

            try:
                detections = self.model.update(frame)
                with self.lock:
                    self.latest_detections = detections
                    self.new_detections = True
            except Exception as e:
                print(f"YOLO ERROR: {e}")
