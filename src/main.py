from queue import Queue

from camera_interface import CameraCapture, CameraType
from legolas_common.src.packet_types import BROADCAST_DEST, Packet, PacketType
from legolas_common.src.socket_server import SocketServer

import time
import cv2

def main() -> None:
    """Program entrypoint"""

    outgoing_data: Queue[Packet] = Queue()
    received_data: Queue[Packet] = Queue()

    server = SocketServer("127.0.0.1", 12345, outgoing_data, received_data)
    server.run()
    cam = CameraCapture()
    cam.start(CameraType.WEBCAM)

    color_off = (0,0,0)
    color_set = (255,255,0)

    color_0 = color_off
    color_1 = color_off

    try:
        while True:
            time.sleep(0.3)
            # time.sleep(10)
            frame_data = cam.get_frame()
            resized_img = cv2.resize(frame_data, (320, 180), interpolation= cv2.INTER_LINEAR)
            cv2.rectangle(resized_img, (10,10), (40,40), color_0)
            cv2.rectangle(resized_img, (50,10), (90,40), color_1)
            cv2.imshow("servercam", resized_img)
            cv2.waitKey(1)
            outgoing_data.put(Packet(PacketType.IMAGE, BROADCAST_DEST, resized_img))
            # outgoing_data.put(Packet(PacketType.CONTROL, BROADCAST_DEST, {"time": time.time()}))
            outgoing_data.put(Packet(PacketType.CONTROL, BROADCAST_DEST, {"bounding_box": 2, "bb0": ((10,10),(40,40)), "bb1": ((40,10), (80,40))}))

            if not received_data.empty():
                msg = received_data.get()
                print(f"From {msg.packet_address}: ", end="")
                if msg.packet_type == PacketType.INTERNAL:
                    print(f"Received internal: {msg.payload}")
                elif msg.packet_type == PacketType.CONTROL:
                    print(f"Received control: {msg.payload}")
                    if "chosen_box" in msg.payload.keys():
                        box_i = msg.payload["chosen_box"]
                        if box_i == 0:
                            color_0 = color_set
                            color_1 = color_off
                        else:
                            color_1 = color_set
                            color_0 = color_off


                elif msg.packet_type == PacketType.IMAGE:
                    print("Received image")
                else:
                    print("Received ack: {msg.payload}")
    except KeyboardInterrupt:
        cam.shutdown()
        server.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
