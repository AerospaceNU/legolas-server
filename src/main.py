import time
from queue import Queue

from legolas_common.src.packet_types import (
    BROADCAST_DEST,
    Packet,
    PacketAddress,
    PacketType,
)
from legolas_common.src.socket_server import SocketServer


def main() -> None:
    """Program entrypoint"""

    outgoing_data: Queue[Packet] = Queue()
    received_data: Queue[Packet] = Queue()

    server = SocketServer("127.0.0.1", 12348, outgoing_data, received_data)
    prev_update_time = time.time()
    addresses: set[PacketAddress] = set()
    server.run()
    try:
        while True:
            if time.time() - prev_update_time > 1:
                outgoing_data.put(Packet(PacketType.CONTROL, BROADCAST_DEST, {"hello": "world"}))
                prev_update_time = time.time()
                for i, address in enumerate(addresses):
                    outgoing_data.put(Packet(PacketType.CONTROL, address, {"hello": f"client {i}"}))

            if not received_data.empty():
                msg = received_data.get()
                addresses.add(msg.packet_address)
                print(f"From {msg.packet_address}: ", end="")
                if msg.packet_type == PacketType.INTERNAL:
                    print(f"Received internal: {msg.payload}")
                elif msg.packet_type == PacketType.CONTROL:
                    print(f"Received control: {msg.payload}")
                elif msg.packet_type == PacketType.IMAGE:
                    print("Received image")
                else:
                    print("Received ack: {msg.payload}")
            time.sleep(0.001)
    except KeyboardInterrupt:
        server.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
