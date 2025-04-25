import time
from queue import Queue

import cv2

from camera_interface import CameraCapture, CameraType
from frame_annotator import draw_tracked_object
from gimbal.ronin_controller import RoninController
from legolas_common.src.packet_types import BROADCAST_DEST, Packet, PacketType
from legolas_common.src.socket_server import SocketServer
from legolas_tracking.naive_object_tracker import NaiveObjectTracker
from legolas_tracking.norfair_object_tracker import NorfairObjectTracker
from legolas_tracking.yolo_model import YoloModel
from video_input import VideoInput
from video_reader import VideoReader


def main() -> None:
    """Program entrypoint"""

    outgoing_data: Queue[Packet] = Queue()
    received_data: Queue[Packet] = Queue()
    # ronin = RoninController("can0")
    yaw = 0

    server = SocketServer("127.0.0.1", 12345, outgoing_data, received_data)
    server.run()
    # cam: VideoInput = CameraCapture(CameraType.WEBCAM)
    cam = VideoReader("IMG_3061.MOV", rotation=cv2.ROTATE_90_CLOCKWISE)
    cam.start()

    norfair_model = NaiveObjectTracker()
    yolo_model = YoloModel(norfair_model, "best_large_1024.pt")

    try:
        while True:

            frame_data = cam.get_frame()
            detections = yolo_model.update(frame_data)
            for obj in detections:
                draw_tracked_object(frame_data, obj)
            outgoing_data.put(Packet(PacketType.IMAGE, BROADCAST_DEST, frame_data))

            if not received_data.empty():
                msg = received_data.get()
                print(f"From {msg.packet_address}: ", end="")
                if msg.packet_type == PacketType.INTERNAL:
                    print(f"Received internal: {msg.payload}")
                elif msg.packet_type == PacketType.CONTROL:
                    print(f"Received control: {msg.payload}")
                elif msg.packet_type == PacketType.IMAGE:
                    print("Received image")
                else:
                    print("Received ack: {msg.payload}")
            yaw += 1
            if yaw == 360:
                yaw = 0
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        cam.shutdown()
        server.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
