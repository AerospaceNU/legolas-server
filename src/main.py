import math
import time
from queue import Queue

import cv2
import torch

from camera_interface import CameraCapture, CameraType
from gimbal.pid_controller import PIDGimbalController
from gimbal.ronin_controller import RoninController
from legolas_common.src.frame_annotator import draw_tracked_object
from legolas_common.src.packet_types import BROADCAST_DEST, Packet, PacketType
from legolas_common.src.socket_server import SocketServer
from legolas_common.src.tracked_object import BoundingBox, TrackerObject
from tracking.norfair_object_tracker import NorfairObjectTracker
from tracking.yolo_model import YoloModel
from video_input import VideoInput
from video_writer import VideoWriter

TRANSMIT_DOWNSCALE_FACTOR = 4

transferable_rocket_names = set(
    (
        "rockets",
        "rocket",
        "boat",
        "airplane",
        "kite",
        "parachute",
        "descending_rocket",
    )
)


def bounding_box_center(bbox: BoundingBox):
    cx = (bbox.bottom_right.x + bbox.top_left.x) / 2
    cy = (bbox.bottom_right.y + bbox.top_left.y) / 2
    return (cx, cy)


def bbox_distance(b1: BoundingBox, b2: BoundingBox):
    b1x, b1y = bounding_box_center(b1)
    b2x, b2y = bounding_box_center(b2)

    distance = math.sqrt((b2x - b1x) ** 2 + (b2y - b1y) ** 2)
    return distance


def get_object_with_id(objects: list[TrackerObject], search_id: int):
    result = list(filter(lambda detection: detection.persistent_id == search_id, objects))

    if len(result) == 0:
        return None
    else:
        return result[0]


RECORD_FILENAME = "recording.mp4"


def main() -> None:
    """Program entrypoint"""

    outgoing_data: Queue[Packet] = Queue()
    received_data: Queue[Packet] = Queue()

    yaw_Kp = 0.025
    yaw_Ki = 0.0002

    pitch_Kp = 0.025
    pitch_Ki = 0.0002

    ronin = RoninController("can0")
    yaw_controller = PIDGimbalController(
        Kp=yaw_Kp, Ki=yaw_Ki, Kd=0.00, control_callback=lambda val: ronin.set_yaw(val)
    )
    pitch_controller = PIDGimbalController(
        Kp=pitch_Kp,
        Ki=pitch_Ki,
        Kd=0,
        control_callback=lambda val: ronin.set_pitch(val),
    )

    server = SocketServer("0.0.0.0", 12345, outgoing_data, received_data)
    server.run()
    cam: VideoInput = CameraCapture(CameraType.WEBCAM)
    # cam = VideoReader("IMG_3061.MOV", rotation=cv2.ROTATE_90_CLOCKWISE)
    cam.start()

    norfair_model = NorfairObjectTracker()
    yolo_model = YoloModel(norfair_model, "yolo11l.pt")

    transmit_delay = 1 / 10
    previous_transmit_time = time.time()
    previous_tracked_object: TrackerObject | None = None
    currently_selected_id = 1

    video_recorder = None

    try:
        while True:
            frame_data = cam.get_frame()
            frame_height, frame_width = frame_data.shape[:2]

            center_x, center_y = frame_width / 2, frame_height / 2
            if yolo_model is not None:
                try:
                    detections = yolo_model.update(frame_data)
                except:
                    detections = []
            else:
                detections = []

            if video_recorder is None:
                video_recorder = VideoWriter("record.mp4", frame_size=(frame_width, frame_height))
            currently_selected: TrackerObject | None = get_object_with_id(
                detections, currently_selected_id
            )

            if currently_selected is not None:
                target = currently_selected
                print(target.bbox)
                cx, cy = bounding_box_center(target.bbox)

                error_x = cx - center_x
                error_y = cy - center_y
                print(f"Frame center is {center_x}, {center_y}")
                print(f"Object center is {cx}, {cy}")
                print(f"Error X: {error_x}, Error Y: {error_y}")
                yaw_controller.process(error_x)
                pitch_controller.process(-1 * error_y)

                previous_tracked_object = target
            elif currently_selected_id is not None:
                if len(detections) != 0:
                    # We want to pick the object that is closest to the previously tracked object
                    min_distance = 100000000
                    current_min = None
                    for detection in filter(
                        lambda detection: detection.class_name == previous_tracked_object.class_name
                        or (
                            previous_tracked_object.class_name in transferable_rocket_names
                            and detection.class_name in transferable_rocket_names
                        ),
                        detections,
                    ):
                        if (
                            bbox_distance(detection.bbox, previous_tracked_object.bbox)
                            < min_distance
                        ):
                            current_min = detection
                    if current_min:
                        previous_tracked_object = current_min
                        currently_selected_id = current_min.persistent_id

            for detection in detections:
                if detection.persistent_id == currently_selected_id:
                    detection.primary_track = True
                else:
                    detection.primary_track = False

            if video_recorder is not None:
                video_recorder.write(frame_data, detections)

            outgoing_data.put(
                Packet(PacketType.CONTROL, BROADCAST_DEST, {"frame_detections": detections})
            )
            if time.time() - previous_transmit_time > transmit_delay:
                previous_transmit_time = time.time()

                transmit_frame = cv2.resize(
                    frame_data,
                    (
                        frame_width // TRANSMIT_DOWNSCALE_FACTOR,
                        frame_height // TRANSMIT_DOWNSCALE_FACTOR,
                    ),
                )
                outgoing_data.put(Packet(PacketType.IMAGE, BROADCAST_DEST, transmit_frame))

            if not received_data.empty():
                msg = received_data.get()
                print(f"From {msg.packet_address}: ", end="")
                if msg.packet_type == PacketType.INTERNAL:
                    print(f"Received internal: {msg.payload}")
                elif msg.packet_type == PacketType.CONTROL:
                    payload = msg.payload
                    if "trackID" in payload:
                        selected_object = get_object_with_id(detections, payload["trackID"])
                        if selected_object is not None:
                            currently_selected_id = payload["trackID"]
                            previous_tracked_object = selected_object
                        if payload["trackID"] is None:
                            currently_selected_id = None
                            previous_tracked_object = None
                    if "record" in payload:
                        record = payload["record"]
                        if video_recorder is not None:
                            video_recorder.release()
                            video_recorder = None
                        if record is not None:
                            video_recorder = VideoWriter(
                                record, frame_size=(frame_width, frame_height)
                            )
                    if "gimbalParams" in payload:
                        params = payload["gimbalParams"]
                        if "yaw" in params:
                            yaw = params["yaw"]
                            if "kp" in yaw:
                                yaw_controller.set_Kp(yaw["kp"])
                            if "ki" in yaw:
                                yaw_controller.set_Ki(yaw["ki"])
                        if "pitch" in params:
                            pitch = params["pitch"]
                            if "kp" in pitch:
                                pitch_controller.set_Kp(pitch["kp"])
                            if "ki" in pitch:
                                pitch_controller.set_Ki(pitch["ki"])
                    if "yoloModel" in params:
                        desired_model_name = params["yoloModel"]
                        # Delete and free the current model if it exists
                        if yolo_model is not None:
                            del yolo_model
                            yolo_model = None
                            torch.cuda.empty_cache()
                        try:
                            yolo_model = YoloModel(norfair_model, desired_model_name)
                        except:
                            yolo_model = None

                elif msg.packet_type == PacketType.IMAGE:
                    print("Received image")
                else:
                    print(f"Received ack: {msg.payload}")
    except KeyboardInterrupt:
        cam.shutdown()
        server.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
