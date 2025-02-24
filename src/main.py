from queue import Queue

from camera_interface import CameraCapture, CameraType
from legolas_common.src.packet_types import BROADCAST_DEST, Packet, PacketType
from legolas_common.src.socket_server import SocketServer


def main() -> None:
    """Program entrypoint"""

    outgoing_data: Queue[Packet] = Queue()
    received_data: Queue[Packet] = Queue()

    server = SocketServer("127.0.0.1", 12345, outgoing_data, received_data)
    server.run()
    cam = CameraCapture()
    cam.start(CameraType.WEBCAM)
    try:
        while True:

            frame_data = cam.get_frame()
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
    except KeyboardInterrupt:
        cam.shutdown()
        server.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
