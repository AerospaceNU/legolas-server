import threading
import time
from collections import deque
from dataclasses import dataclass

from gimbal.pid_controller import PIDOutput


@dataclass
class DashboardState:
    # PID state
    yaw_pid: PIDOutput | None = None
    pitch_pid: PIDOutput | None = None

    # Target info
    target_id: int | None = None
    target_class: str | None = None
    target_position: tuple[float, float] | None = None
    target_bbox_size: tuple[float, float] | None = None
    target_confidence: float | None = None

    # Detection info
    detection_count: int = 0

    # System status
    camera_connected: bool = False
    gimbal_connected: bool = False
    frame_size: tuple[int, int] = (0, 0)
    loop_dt: float = 0.0

    # Gimbal outputs
    yaw_speed: float = 0.0
    pitch_speed: float = 0.0


class EventLog:
    def __init__(self, maxlen=100):
        self._entries: deque[tuple[float, str]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def log(self, message: str):
        with self._lock:
            self._entries.append((time.time(), message))

    def get_recent(self, n=20) -> list[tuple[float, str]]:
        with self._lock:
            return list(self._entries)[-n:]
