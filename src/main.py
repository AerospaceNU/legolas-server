import math
import time
from queue import Queue

import cv2
import torch

from camera_interface import CameraCapture, CameraType
from dashboard import Dashboard, DashboardState, EventLog
from gimbal.pid_controller import PIDGimbalController
from gimbal.ronin_controller import RoninController
from legolas_common.src.packet_types import BROADCAST_DEST, Packet, PacketType
from legolas_common.src.socket_server import SocketServer
from legolas_common.src.tracked_object import BoundingBox, TrackerObject
from tracking.norfair_object_tracker import NorfairObjectTracker
from tracking.yolo_model import YoloModel, YoloThread
from video_input import VideoInput
from video_writer import VideoWriter

TRANSMIT_DOWNSCALE_FACTOR = 1


def bounding_box_center(bbox: BoundingBox):
    cx = (bbox.bottom_right.x + bbox.top_left.x) / 2
    cy = (bbox.bottom_right.y + bbox.top_left.y) / 2
    return (cx, cy)


def bbox_distance(b1: BoundingBox, b2: BoundingBox):
    b1x, b1y = bounding_box_center(b1)
    b2x, b2y = bounding_box_center(b2)
    return math.sqrt((b2x - b1x) ** 2 + (b2y - b1y) ** 2)


def get_object_with_id(objects: list[TrackerObject], search_id: int):
    result = list(filter(lambda detection: detection.persistent_id == search_id, objects))
    if len(result) == 0:
        return None
    else:
        return result[0]


def main() -> None:
    """Program entrypoint"""

    # Dashboard state and event log
    state = DashboardState()
    event_log = EventLog()
    dashboard = Dashboard(state, event_log)

    event_log.log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        event_log.log(f"GPU: {torch.cuda.get_device_name(0)}")

    outgoing_data: Queue[Packet] = Queue()
    received_data: Queue[Packet] = Queue()

    yaw_Kp = 0.08
    yaw_Ki = 0.005
    yaw_Kd = 0.015

    pitch_Kp = 0.04
    pitch_Ki = 0.003
    pitch_Kd = 0.01

    ronin = RoninController("can0", event_log=event_log)
    state.gimbal_connected = ronin.jetson

    yaw_controller = PIDGimbalController(
        Kp=yaw_Kp, Ki=yaw_Ki, Kd=yaw_Kd, control_callback=lambda val: ronin.set_yaw_joystick(-val)
    )
    pitch_controller = PIDGimbalController(
        Kp=pitch_Kp,
        Ki=pitch_Ki,
        Kd=pitch_Kd,
        control_callback=lambda val: ronin.set_pitch_joystick(val),
    )

    server = SocketServer("0.0.0.0", 12345, outgoing_data, received_data)
    server.run()
    cam: VideoInput = CameraCapture(CameraType.WEBCAM, device_id=0)
    cam.start()
    state.camera_connected = True
    event_log.log("Camera started")

    norfair_model = NorfairObjectTracker()
    yolo_model = YoloModel(norfair_model, r"/ssd/legolas-nuli-launch/legolas-server/model_weights.pt", 320)
    yolo_thread = YoloThread(yolo_model)
    yolo_thread.start()
    event_log.log("YOLO thread started")

    transmit_delay = 1 / 7
    previous_transmit_time = time.time()
    previous_tracked_object: TrackerObject | None = None
    currently_selected_id = None

    video_recorder = None

    ronin.reset_to_zero()

    # Start TUI after all initialization
    dashboard.start()

    try:
        while True:
            start_time = time.time()

            frame_data = cam.get_frame()
            yolo_thread.update_frame(frame_data)

            frame_height, frame_width = frame_data.shape[:2]
            state.frame_size = (frame_width, frame_height)
            center_x, center_y = frame_width / 2, frame_height / 2

            try:
                detections = yolo_thread.get_detections()
            except Exception as e:
                event_log.log(f"YOLO ERROR: {e}")
                detections = []

            state.detection_count = len(detections)

            currently_selected: TrackerObject | None = get_object_with_id(
                detections, currently_selected_id
            )

            if currently_selected is None and len(detections) > 0:
                currently_selected = detections[0]
                event_log.log(
                    f"Auto-selected target ID={currently_selected.persistent_id} "
                    f"cls={currently_selected.class_name}"
                )

            if currently_selected is not None:
                target = currently_selected
                cx, cy = bounding_box_center(target.bbox)

                error_x = cx - center_x
                error_y = cy - center_y

                yaw_out = yaw_controller.process(error_x)
                pitch_out = pitch_controller.process(error_y)

                # Update dashboard state
                state.yaw_pid = yaw_out
                state.pitch_pid = pitch_out
                state.target_id = target.persistent_id
                state.target_class = target.class_name
                state.target_position = (cx, cy)
                bbox_w = target.bbox.bottom_right.x - target.bbox.top_left.x
                bbox_h = target.bbox.bottom_right.y - target.bbox.top_left.y
                state.target_bbox_size = (bbox_w, bbox_h)
                state.target_confidence = target.confidence
                state.yaw_speed = yaw_out.output
                state.pitch_speed = pitch_out.output

                previous_tracked_object = target
            else:
                state.target_id = None
                state.target_class = None
                state.target_position = None
                state.target_bbox_size = None
                state.target_confidence = None
                state.yaw_pid = None
                state.pitch_pid = None
                state.yaw_speed = 0.0
                state.pitch_speed = 0.0

            # Reacquisition: if target lost, pick closest detection by spatial proximity
            if currently_selected is None and currently_selected_id is not None:
                if len(detections) != 0 and previous_tracked_object is not None:
                    current_min = min(
                        detections,
                        key=lambda d: bbox_distance(d.bbox, previous_tracked_object.bbox),
                    )
                    old_id = currently_selected_id
                    previous_tracked_object = current_min
                    currently_selected_id = current_min.persistent_id
                    event_log.log(
                        f"Reacquired target ID={currently_selected_id} (was ID={old_id})"
                    )
                else:
                    event_log.log(
                        f"Target ID={currently_selected_id} lost, no detections to reacquire"
                    )
                    currently_selected_id = None

            for detection in detections:
                detection.primary_track = detection.persistent_id == currently_selected_id

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
                if msg.packet_type == PacketType.INTERNAL:
                    event_log.log(f"Internal: {msg.payload}")
                elif msg.packet_type == PacketType.CONTROL:
                    payload = msg.payload
                    if "trackID" in payload:
                        selected_object = get_object_with_id(detections, payload["trackID"])
                        if selected_object is not None:
                            currently_selected_id = payload["trackID"]
                            previous_tracked_object = selected_object
                            event_log.log(f"Manual track select ID={currently_selected_id}")
                        if payload["trackID"] is None:
                            currently_selected_id = None
                            previous_tracked_object = None
                            event_log.log("Track deselected")
                    if "record" in payload:
                        record = payload["record"]
                        if video_recorder is not None:
                            video_recorder.release()
                            video_recorder = None
                            event_log.log("Recording stopped")
                        if record is not None:
                            video_recorder = VideoWriter(
                                record, frame_size=(frame_width, frame_height)
                            )
                            event_log.log(f"Recording started: {record}")
                    if "yoloModel" in payload:
                        desired_model_name = payload["yoloModel"]
                        if yolo_model is not None:
                            del yolo_model
                            yolo_model = None
                            torch.cuda.empty_cache()
                        try:
                            yolo_model = YoloModel(norfair_model, desired_model_name)
                            event_log.log(f"YOLO model loaded: {desired_model_name}")
                        except Exception as e:
                            yolo_model = None
                            event_log.log(f"YOLO model load failed: {e}")

                elif msg.packet_type == PacketType.IMAGE:
                    event_log.log("Received image from client")
                else:
                    event_log.log(f"Received: {msg.payload}")

            state.loop_dt = time.time() - start_time
    finally:
        dashboard.stop()
        yolo_thread.stop()
        cam.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
