import time
from queue import Queue

from legolas_common.src.packet_types import BROADCAST_DEST, Packet, PacketType
from legolas_common.src.socket_client import SocketClient

if __name__ == "__main__":
    outgoing_data: Queue[Packet] = Queue()
    received_data: Queue[Packet] = Queue()
    client = SocketClient("10.0.0.3", 12345, outgoing_data, received_data)
    prev_update_time = 0.0

    client.run()
    try:
        while True:
            if not received_data.empty():
                packet = received_data.get()
            if time.time() - prev_update_time > 10:
                outgoing_data.put(
                    Packet(
                        PacketType.CONTROL,
                        BROADCAST_DEST,  # This doesn't actually matter
                        {"hello": "server", "time": time.time()},
                    )
                )
                prev_update_time = time.time()
    except KeyboardInterrupt:
        client.shutdown()
        print("Shutting down client main")
