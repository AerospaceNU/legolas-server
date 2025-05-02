import struct
from enum import IntEnum

import can

from gimbal.crc import crc16, crc32


class ReplyType(IntEnum):
    NOT_REQUIRED = 0
    OPTIONAL = 1
    REQUIRED = 2


class FrameType(IntEnum):
    COMMAND = 0
    REPLY = 1


class CanMessage:
    sequence_num = 2

    def __init__(
        self, data: bytes, frame_type: FrameType = FrameType.COMMAND, reply_type=ReplyType.REQUIRED
    ):
        self.data = data
        self.full_type = ((reply_type & 0xF)) & ((frame_type & 0b1) << 5)

    HEADER_FORMAT = "<BHBB3sH"
    DATA_FORMAT = lambda _, data_len: f"<H{data_len}sI"

    def _make_seq_num(self):
        if CanMessage.sequence_num >= 0xFFFD:
            CanMessage.sequence_num = 2
        CanMessage.sequence_num += 1
        return CanMessage.sequence_num

    def to_bytes(self) -> bytes:
        """Convert the dataclass instance to bytes"""
        data_length = len(self.data)
        header_data = struct.pack(
            self.HEADER_FORMAT,
            0xAA,
            data_length + 16,
            self.full_type,
            0,
            bytes([0] * 3),
            self._make_seq_num(),
        )

        header_crc16 = crc16(header_data)
        data_crc32 = crc32(header_data + struct.pack("<H", header_crc16) + self.data)
        return header_data + struct.pack(
            self.DATA_FORMAT(data_length), header_crc16, self.data, data_crc32
        )

    @classmethod
    def from_bytes(cls, data: bytes):
        """Convert bytes to a dataclass instance"""
        header_bytes = data[: struct.calcsize(cls.HEADER_FORMAT)]
        data = data[struct.calcsize(cls.HEADER_FORMAT) :]
        unpacked = struct.unpack(cls.HEADER_FORMAT, header_bytes)
        length = unpacked[1]
        data_length = length - 16
        data_format = cls.DATA_FORMAT(data_length)  # type: ignore
        if struct.calcsize(data_format) != len(data):
            raise ValueError("Invalid binary data")
        header_crc16, packet_data, data_crc32 = struct.unpack(data_format, data)
        if crc16(header_bytes) != header_crc16:
            raise ValueError("Invalid header CRC")
        if crc32(packet_data) != data_crc32:
            raise ValueError("Invalid data CRC")
        command_type = FrameType((unpacked[2] >> 5) & 0b1)
        return cls(packet_data, command_type)


def is_jetson():
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "NVIDIA" in f.read()
    except FileNotFoundError:
        return False


class RoninController:
    def __init__(self, can_bus):

        self._send_id = 0x223
        self._recv_id = 0x222
        self.yaw = 0
        self.roll = 0
        self.pitch = 0
        if not is_jetson():
            self.jetson = False
            return
        self._bus = can.interface.Bus(bustype="socketcan", channel=can_bus, bitrate=1000000)

    def _send_cmd(self, payload: bytes):
        if not self.jetson:
            return
        message = CanMessage(payload)
        data_bytes = message.to_bytes()
        while len(data_bytes) != 0:
            msg = can.Message(
                arbitration_id=self._send_id,
                data=data_bytes[:8],
                is_extended_id=False,
            )
            try:
                self._bus.send(msg)
                data_bytes = data_bytes[8:]
            except can.CanError as e:
                print(f"CAN transmit error {e}")

    def set_yaw(self, value):
        self.yaw = value
        self.set_gimbal_speed(self.yaw, self.roll, self.pitch)

    def set_pitch(self, value):
        self.pitch = value
        self.set_gimbal_speed(self.yaw, self.roll, self.pitch)

    def set_position_control(self, yaw, roll, pitch, ctrl_byte=0x01, time_for_action=1):
        yaw = int(yaw * 10.0)
        roll = int(roll * 10.0)
        pitch = int(pitch * 10.0)
        cmd_data = struct.pack("<BBhhhBB", 0x0E, 0x00, yaw, roll, pitch, ctrl_byte, time_for_action)
        self._send_cmd(cmd_data)

    def set_gimbal_speed(self, yaw, roll, pitch):
        yaw = int(yaw * 10.0)
        roll = int(roll * 10.0)
        pitch = int(pitch * 10.0)

        ctrl_byte = 0b10001000
        cmd_data = struct.pack("<BBhhhB", 0x0E, 0x01, yaw, roll, pitch, ctrl_byte)
        self._send_cmd(cmd_data)
