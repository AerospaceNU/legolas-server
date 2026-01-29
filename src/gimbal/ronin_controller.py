import struct
from enum import IntEnum

import can

import time
import math


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
        enc = data[4]
        res = data[5:8]
        seq = struct.unpack("<H", data[8:10])[0]
        
        data_bytes = data[10:]
        data_length = length - 16
        
        data_format = cls.DATA_FORMAT(data_length)
        if struct.calcsize(data_format) != len(data_bytes):
            raise ValueError(f"Invalid binary data: expected {struct.calcsize(data_format)}, got {len(data_bytes)}")
        
        header_crc16, packet_data, data_crc32 = struct.unpack(data_format, data_bytes)
        
        # TEMPORARILY SKIP CRC VALIDATION - just warn
        header_bytes = data[:10]
        calculated_crc = crc16(header_bytes)
        if calculated_crc != header_crc16:
            print(f"[WARNING] Header CRC mismatch: expected {header_crc16:04X}, got {calculated_crc:04X}")
            print(f"[WARNING] Header bytes: {header_bytes.hex()}")
            # Don't raise, just continue
        
        frame_type = FrameType((cmd_type >> 5) & 0b1)
        return cls(packet_data, frame_type)
        """Convert bytes to a dataclass instance"""
        # Parse header manually to handle the special length encoding
        sof = data[0]
        if sof not in (0xAA, 0x55):  # Accept both command (0xAA) and response (0x55)
            raise ValueError(f"Invalid SOF byte: 0x{sof:02X}")
        
        length_low = data[1]
        length_high_and_version = data[2]
        
        # Extract length: byte 1 + (byte 2 & 0x03) << 8
        length = length_low | ((length_high_and_version & 0x03) << 8)
        
        # Parse rest of header
        cmd_type = data[3]
        enc = data[4]
        res = data[5:8]
        seq = struct.unpack("<H", data[8:10])[0]
        
        # Now parse data section
        data_bytes = data[10:]  # Everything after header
        data_length = length - 16  # Data payload length
        
        data_format = cls.DATA_FORMAT(data_length)
        if struct.calcsize(data_format) != len(data_bytes):
            raise ValueError(f"Invalid binary data: expected {struct.calcsize(data_format)}, got {len(data_bytes)}")
        
        header_crc16, packet_data, data_crc32 = struct.unpack(data_format, data_bytes)
        
        # Verify CRCs using ORIGINAL data (with 0x55 or 0xAA intact)
        header_bytes = data[:10]
        if crc16(header_bytes) != header_crc16:
            raise ValueError(f"Invalid header CRC: expected {header_crc16:04X}, got {crc16(header_bytes):04X}")
        if crc32(header_bytes + struct.pack("<H", header_crc16) + packet_data) != data_crc32:
            raise ValueError("Invalid data CRC")
        
        frame_type = FrameType((cmd_type >> 5) & 0b1)
        return cls(packet_data, frame_type)
        """Convert bytes to a dataclass instance"""
        # Parse header manually to handle the special length encoding
        sof = data[0]
        length_low = data[1]
        length_high_and_version = data[2]
        
        # Extract length: byte 1 + (byte 2 & 0x03) << 8
        # DJI protocol only uses lower 2 bits of byte 2 for length!
        length = length_low | ((length_high_and_version & 0x03) << 8)
        
        # Parse rest of header
        cmd_type = data[3]
        enc = data[4]
        res = data[5:8]
        seq = struct.unpack("<H", data[8:10])[0]
        
        # Now parse data section
        data_bytes = data[10:]  # Everything after header
        data_length = length - 16  # Data payload length
        
        data_format = cls.DATA_FORMAT(data_length)
        if struct.calcsize(data_format) != len(data_bytes):
            raise ValueError(f"Invalid binary data: expected {struct.calcsize(data_format)}, got {len(data_bytes)}")
        
        header_crc16, packet_data, data_crc32 = struct.unpack(data_format, data_bytes)
        
        # Verify CRCs
        header_bytes = data[:10]
        if crc16(header_bytes) != header_crc16:
            raise ValueError("Invalid header CRC")
        if crc32(header_bytes + struct.pack("<H", header_crc16) + packet_data) != data_crc32:
            raise ValueError("Invalid data CRC")
        
        frame_type = FrameType((cmd_type >> 5) & 0b1)
        return cls(packet_data, frame_type)


def is_jetson():
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "NVIDIA" in f.read()
    except FileNotFoundError:
        return False


class RoninController:
    def __init__(self, can_bus):

        self.gimbal_values = "/ssd/legolas-nuli-launch/legolas-server/src/gimbal_values.txt"

        self._send_id = 0x223
        self._recv_id = 0x530

        self.yaw = 0
        self.roll = 0
        self.pitch = 0

        if not is_jetson():
            self.jetson = False
            return

        self.jetson = True
        self._bus = can.interface.Bus(bustype="socketcan", channel=can_bus, bitrate=1000000)

        # #sets the speed of the gimbal
        # self.set_yaw(300)
        # self.set_roll(300)
        # self.set_pitch(300)
        # time.sleep(2) #for some reason setting speed also moves the gimbal, so wait for that to 
        # #finish before resetting back to zero
        self.reset_to_zero()


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
                print("send")
                self._bus.send(msg)
                print("sent")
                data_bytes = data_bytes[8:]
            except can.CanError as e:
                print(f"CAN transmit error {e}")

    def set_yaw_joystick(self, value):
        self.yaw = value
        self.set_gimbal_speed(self.yaw, self.roll, self.pitch)

    def set_yaw_position(self, value):
        self.yaw = value
        self.set_position_control(self.yaw, self.roll, self.pitch)
        self.write_to_file(self.gimbal_values, "yaw", value)

    def set_pitch_joystick(self, value):
        self.pitch = value
        self.set_gimbal_speed(self.yaw, self.roll, self.pitch)

    def set_pitch_position(self, value):
        self.pitch = value
        self.set_position_control(self.yaw, self.roll, self.pitch)
        self.write_to_file(self.gimbal_values, "pitch", value)

    def set_roll_joystick(self, value):
        self.roll = value
        self.set_gimbal_speed(self.yaw, self.roll, self.pitch)

    def set_roll_position(self, value):
        self.roll = value
        self.set_position_control(self.yaw, self.roll, self.pitch)
        self.write_to_file(self.gimbal_values, "roll", value)

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

    #!untested
    def delta_to_gimbal_angles(self, delta_x: float, delta_y: float, frame_width: int, frame_height: int, 
                          fov_diagonal: float = 80.0):
        """
        Convert pixel deltas to gimbal angles (yaw, pitch, roll).
        
        Args:
            delta_x: Horizontal pixel offset from center (positive = right)
            delta_y: Vertical pixel offset from center (positive = down)
            frame_width: Width of the camera frame in pixels
            frame_height: Height of the camera frame in pixels
            fov_diagonal: Diagonal field of view in degrees
        
        Returns:
            tuple: (yaw_deg, pitch_deg, roll_deg)
        """        
        # Calculate aspect ratio and diagonal in pixels
        aspect_ratio = frame_width / frame_height
        diagonal_pixels = math.sqrt(frame_width**2 + frame_height**2)
        
        # Convert diagonal FOV to radians
        fov_diagonal_rad = math.radians(fov_diagonal)
        
        # Calculate horizontal and vertical FOV from diagonal FOV
        # Using the relationship: tan(fov_diag/2) = sqrt(tan(fov_h/2)^2 + tan(fov_v/2)^2)
        # With aspect ratio constraint: tan(fov_h/2) = aspect_ratio * tan(fov_v/2)
        
        tan_half_diag = math.tan(fov_diagonal_rad / 2)
        tan_half_vertical = tan_half_diag / math.sqrt(1 + aspect_ratio**2)
        tan_half_horizontal = aspect_ratio * tan_half_vertical
        
        fov_horizontal = 2 * math.degrees(math.atan(tan_half_horizontal))
        fov_vertical = 2 * math.degrees(math.atan(tan_half_vertical))
        
        # Calculate degrees per pixel
        deg_per_pixel_x = fov_horizontal / frame_width
        deg_per_pixel_y = fov_vertical / frame_height
        
        # Convert to angles
        yaw_deg = delta_x * deg_per_pixel_x
        pitch_deg = -delta_y * deg_per_pixel_y  # Negative because positive delta_y = pitch down
        roll_deg = 0  # Typically don't use roll for tracking
        
        return (yaw_deg, pitch_deg, roll_deg)
    #!end untested

    def read_from_file(self, file_path, arg):
        """
        Returns the value at a key in a file.

        Args:
            file_path (string): the path to the file
            arg (string): the key to get the value from
        
        Returns:
            value (string): the value at the key
        """
        file = open(file_path, "r")
        for line in file:
            line = line.strip()

            if not line: #skip empty lines
                continue

            if ':' in line:
                key, value = line.split(":", 1)
                if key.strip() == arg:
                    return value.strip() #returns the value at the key
        
        return None
    
    def write_to_file(self, file_path, key, value):
        """
        Writes a value to a file

        Args:
            file_path (string): the path to the file
            key (string): the key to write the value to
            value (string): the value to write.
        """
        file = open(file_path, "r")
        updated_lines = []
        key_found = False
        for line in file:
            line = line.strip()

            if not line: #skip empty lines
                updated_lines.append(line)
                continue

            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip() == key:
                    updated_lines.append(f"{key}:{value}") #add the new value to the lines to be updated
                    key_found = True
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        if not key_found:
            print(f"COULD NOT FIND KEY {key} TO WRITE {value} TO!")
            updated_lines.append(f"{key}:{value}")
        
        
        with open(file_path, 'w') as f: #update all lines
            for item in updated_lines:
                f.write('{}\n'.format(item))
        file.close()

    def reset_to_zero(self):
        """
        Resets the gimbal back to zero (level)
        """
        self.set_pitch_position(0)
        self.set_yaw_position(0)
        self.set_roll_position(0)
    

    def move_axis_to(self, axis: str, target_deg: float, max_speed_deg: float):
        """
        Smoothly moves a single axis toward a target using proportional speed with a speed cap.
        
        axis: "yaw", "pitch", or "roll"
        target_deg: target angle in degrees
        max_speed_deg: maximum speed in degrees/sec
        """

        # Read current angle
        current = getattr(self, axis)

        # Compute error
        error = target_deg - current

        # If already close, snap to final position
        if abs(error) < 0.3:     # 0.3 deg threshold
            setattr(self, axis, target_deg)
            self.set_position_control(self.yaw, self.roll, self.pitch)
            return

        # Proportional gain: tune this
        k = 3.0   # higher = snappier motion

        # Compute speed command
        speed = k * error

        # Apply speed limit
        speed = max(min(speed, max_speed_deg), -max_speed_deg)

        # Apply movement to correct axis
        if axis == "yaw":
            self.yaw = speed
        elif axis == "pitch":
            self.pitch = speed
        elif axis == "roll":
            self.roll = speed
        else:
            raise ValueError("Invalid axis. Use 'yaw', 'pitch', or 'roll'.")

        # Send speed update
        self.set_gimbal_speed(self.yaw, self.roll, self.pitch)


    

    def _receive_reply(self, timeout=2.0):
        """Receive and parse a reply from the gimbal"""
        if not self.jetson:
            return None
        
        print(f"[DEBUG] Waiting for reply on ID 0x{self._recv_id:03X}...")
        
        accumulated_data = bytearray()
        end_time = time.time() + timeout
        frame_count = 0
        packet_started = False
        expected_length = None
        
        while time.time() < end_time:
            try:
                msg = self._bus.recv(timeout=0.2)
                
                if msg.arbitration_id == self._recv_id:
                    frame_data = msg.data
                    
                    # Look for start marker
                    if not packet_started:
                        if frame_data[0] == 0x55:
                            packet_started = True
                            accumulated_data.extend(frame_data)
                            frame_count = 1
                            print(f"[DEBUG] Frame {frame_count} (START): {frame_data.hex()}")
                            
                            # Read length: byte 1 + (byte 2 & 0x3) << 8
                            if len(accumulated_data) >= 3:
                                length_low = accumulated_data[1]
                                length_high = (accumulated_data[2] & 0x03)  # ONLY lower 2 bits!
                                expected_length = length_low | (length_high << 8)
                                print(f"[DEBUG] Expected packet length: {expected_length} bytes")
                        continue
                    
                    # Accumulate more frames
                    accumulated_data.extend(frame_data)
                    frame_count += 1
                    print(f"[DEBUG] Frame {frame_count}: {frame_data.hex()}")
                    
                    # Check if we have a complete packet
                    if expected_length and len(accumulated_data) >= expected_length:
                        print(f"[DEBUG] Complete packet received: {len(accumulated_data)} bytes")
                        accumulated_data = accumulated_data[:expected_length]
                        break
                            
            except Exception as e:
                if packet_started and len(accumulated_data) >= 32:
                    break
                continue
        
        if len(accumulated_data) >= 20:
            print(f"[DEBUG] Trying to parse {len(accumulated_data)} bytes: {accumulated_data.hex()}")
            try:
                return CanMessage.from_bytes(bytes(accumulated_data))
            except Exception as e:
                print(f"[DEBUG] Failed to parse: {e}")
        
        return None

    def get_current_position(self):
        """
        Get current gimbal position from 0x2E1 broadcast channel
        """
        if not self.jetson:
            return None
        
        print("[DEBUG] Looking for position in broadcast messages...")
        
        for _ in range(30):
            try:
                msg = self._bus.recv(timeout=0.1)
                if msg.arbitration_id == 0x2E1:
                    # Try to find messages with repeating bytes (position data)
                    # Look for at least 3 consecutive similar bytes
                    for i in range(len(msg.data) - 2):
                        byte_val = msg.data[i]
                        if (msg.data[i] == msg.data[i+1] or 
                            abs(msg.data[i] - msg.data[i+1]) <= 2):
                            
                            # Found repeating pattern - this might be position
                            # Try treating bytes as signed int8
                            vals = [b if b < 128 else b - 256 for b in msg.data]
                            
                            print(f"[DEBUG] Message: {msg.data.hex()}")
                            print(f"[DEBUG] As signed bytes: {vals}")
                            print(f"[DEBUG] Divided by 10: {[v/10.0 for v in vals]}")
                            return None  # Just show data for now
                            
            except:
                continue
        
        return None
        """
        Get current gimbal position from 0x2E1 broadcast channel
        """
        if not self.jetson:
            return None
        
        print("[DEBUG] Dumping ALL 0x2E1 messages...")
        
        messages_seen = {}
        for _ in range(30):
            try:
                msg = self._bus.recv(timeout=0.1)
                if msg.arbitration_id == 0x2E1:
                    hex_msg = msg.data.hex()
                    if hex_msg not in messages_seen:
                        messages_seen[hex_msg] = True
                        print(f"[DEBUG] {hex_msg}")
            except:
                continue
        
        print(f"\n[DEBUG] Total unique messages: {len(messages_seen)}")
        
        return None
        """
        Get current gimbal position from 0x2E1 broadcast channel
        """
        if not self.jetson:
            return None
        
        print("[DEBUG] Scanning ALL 0x2E1 messages...")
        
        messages_seen = {}
        for _ in range(50):  # Scan more messages
            try:
                msg = self._bus.recv(timeout=0.1)
                if msg.arbitration_id == 0x2E1:
                    hex_msg = msg.data.hex()
                    
                    # Track unique messages
                    if hex_msg not in messages_seen:
                        messages_seen[hex_msg] = 0
                    messages_seen[hex_msg] += 1
                    
                    # Try parsing as position (int16 triplet at various offsets)
                    for offset in [0, 2]:
                        if len(msg.data) >= offset + 6:
                            yaw = struct.unpack("<h", msg.data[offset:offset+2])[0] / 10.0
                            roll = struct.unpack("<h", msg.data[offset+2:offset+4])[0] / 10.0
                            pitch = struct.unpack("<h", msg.data[offset+4:offset+6])[0] / 10.0
                            
                            # Check if values are in reasonable range (-180 to 180)
                            if -180 <= yaw <= 180 and -180 <= roll <= 180 and -180 <= pitch <= 180:
                                if abs(yaw) > 1 or abs(roll) > 1 or abs(pitch) > 1:  # Not all zeros
                                    print(f"[DEBUG] Possible position in {hex_msg} at offset {offset}:")
                                    print(f"        yaw={yaw:.1f}°, roll={roll:.1f}°, pitch={pitch:.1f}°")
            except:
                continue
        
        print(f"\n[DEBUG] Saw {len(messages_seen)} unique message types")
        
        return None
        """
        Get current gimbal position from 0x2E1 broadcast channel
        """
        if not self.jetson:
            return None
        
        print("[DEBUG] Reading position from 0x2E1 broadcast...")
        
        # Look for the stable repeating message on 0x2E1
        for _ in range(20):
            try:
                msg = self._bus.recv(timeout=0.1)
                if msg.arbitration_id == 0x2E1:
                    # Look for messages starting with 0x0000
                    if msg.data[0] == 0x00 and msg.data[1] == 0x00:
                        print(f"[DEBUG] Found position broadcast: {msg.data.hex()}")
                        
                        # Parse yaw from bytes 0-2
                        yaw_raw = struct.unpack("<h", msg.data[0:2])[0]
                        
                        # Try different offsets for roll/pitch
                        print(f"[DEBUG] Testing byte offsets...")
                        for i in range(1, 6):
                            if (i*2)+2 <= len(msg.data):
                                roll_raw = struct.unpack("<h", msg.data[i*2:(i*2)+2])[0]
                                print(f"  Offset {i*2}: {roll_raw / 10.0:.1f}°")
                        
                        return {'yaw': yaw_raw / 10.0, 'roll': 0, 'pitch': 0}
            except:
                continue
        
        return None
        """
        Get current gimbal position for yaw, roll, and pitch.
        Returns a dict with keys 'yaw', 'roll', 'pitch' in degrees, or None if failed.
        """
        if not self.jetson:
            return None
        
        # Send command: cmd_set=0x0E, cmd_id=0x02, data=0x01
        cmd_data = struct.pack("<BBB", 0x0E, 0x02, 0x01)
        print(f"[DEBUG] Sending position request: {cmd_data.hex()}")
        self._send_cmd(cmd_data)
        
        # Wait for multi-frame response on 0x530
        reply = self._receive_reply(timeout=2.0)
        
        if reply:
            print(f"[DEBUG] Received reply with {len(reply.data)} bytes")
            print(f"[DEBUG] Reply data: {reply.data.hex()}")
            
            # According to Handle.cpp:
            # Position data is at bytes 14, 16, 18 of the DATA section
            # But reply.data is the DATA section after parsing CanMessage
            # So position is at bytes 2, 4, 6 of reply.data (after cmd_set and cmd_id)
            
            if len(reply.data) >= 8:
                try:
                    # reply.data structure: [cmd_set, cmd_id, status?, yaw_lo, yaw_hi, roll_lo, roll_hi, pitch_lo, pitch_hi, ...]
                    # Extract position from bytes 2-8
                    yaw_raw = struct.unpack("<h", reply.data[2:4])[0]
                    roll_raw = struct.unpack("<h", reply.data[4:6])[0]
                    pitch_raw = struct.unpack("<h", reply.data[6:8])[0]
                    
                    print(f"[DEBUG] Raw values: yaw={yaw_raw}, roll={roll_raw}, pitch={pitch_raw}")
                    
                    return {
                        'yaw': yaw_raw / 10.0,
                        'roll': roll_raw / 10.0,
                        'pitch': pitch_raw / 10.0
                    }
                except Exception as e:
                    print(f"[DEBUG] Parse error: {e}")
            else:
                print(f"[DEBUG] Reply data too short: {len(reply.data)} bytes")
        else:
            print(f"[DEBUG] No reply received")
        
        return None
    