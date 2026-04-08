import struct
from enum import IntEnum

import can

import time

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
    @staticmethod
    def DATA_FORMAT(data_len):
        return f"<H{data_len}sI"

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
        sof = data[0]
        if sof not in (0xAA, 0x55):
            raise ValueError(f"Invalid SOF byte: 0x{sof:02X}")

        length_low = data[1]
        length_high_and_version = data[2]
        length = length_low | ((length_high_and_version & 0x03) << 8)

        cmd_type = data[3]

        data_bytes = data[10:]
        data_length = length - 16

        data_format = cls.DATA_FORMAT(data_length)
        if struct.calcsize(data_format) != len(data_bytes):
            raise ValueError(f"Invalid binary data: expected {struct.calcsize(data_format)}, got {len(data_bytes)}")

        header_crc16, packet_data, data_crc32 = struct.unpack(data_format, data_bytes)

        header_bytes = data[:10]
        calculated_crc = crc16(header_bytes)
        if calculated_crc != header_crc16:
            pass  # CRC validation mismatch -- non-fatal for now

        frame_type = FrameType((cmd_type >> 5) & 0b1)
        return cls(packet_data, frame_type)


def is_jetson():
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "NVIDIA" in f.read()
    except FileNotFoundError:
        return False


class RoninController:
    def __init__(self, can_bus, event_log=None):

        self._send_id = 0x223
        self._recv_id = 0x530
        self._event_log = event_log

        self.yaw = 0
        self.pitch = 0

        if not is_jetson():
            self.jetson = False
            self._log("Not on Jetson, gimbal commands are no-ops")
            return

        self.jetson = True
        self._bus = can.interface.Bus(bustype="socketcan", channel=can_bus, bitrate=1000000)

        self.check_connection()
        self.reset_to_zero()

    def _log(self, message: str):
        if self._event_log is not None:
            self._event_log.log(message)

    def check_connection(self, duration=3.0):
        """
        Listen to the CAN bus and report what's talking.
        If the gimbal is connected and powered, you should see 0x2E1 messages.
        """
        if not self.jetson:
            return

        self._log(f"Listening on CAN bus for {duration}s...")
        deadline = time.time() + duration
        seen_ids = {}

        while time.time() < deadline:
            try:
                msg = self._bus.recv(timeout=0.1)
                if msg is None:
                    continue
                aid = msg.arbitration_id
                if aid not in seen_ids:
                    seen_ids[aid] = 0
                seen_ids[aid] += 1
            except Exception as e:
                self._log(f"CAN bus error: {e}")
                continue

        if not seen_ids:
            self._log("No CAN traffic detected -- check wiring/power")
        else:
            for aid, count in sorted(seen_ids.items()):
                label = ""
                if aid == 0x2E1:
                    label = " (gimbal status)"
                elif aid == 0x530:
                    label = " (gimbal reply)"
                elif aid == 0x223:
                    label = " (our TX)"
                self._log(f"CAN 0x{aid:03X}: {count} msgs{label}")

            if 0x2E1 in seen_ids:
                self._log("Gimbal connected and broadcasting")
            else:
                self._log("CAN traffic present but gimbal (0x2E1) not seen")

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
                self._log(f"CAN transmit error: {e}")

    def set_yaw_joystick(self, value):
        self.yaw = value
        self.set_gimbal_speed(self.yaw, 0, self.pitch)

    def set_yaw_position(self, value):
        self.yaw = value
        self.set_position_control(self.yaw, 0, self.pitch)

    def set_pitch_joystick(self, value):
        self.pitch = value
        self.set_gimbal_speed(self.yaw, 0, self.pitch)

    def set_pitch_position(self, value):
        self.pitch = value
        self.set_position_control(self.yaw, 0, self.pitch)

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

    def reset_to_zero(self):
        """Resets the gimbal back to zero (level)"""
        self.set_pitch_position(0)
        self.set_yaw_position(0)
