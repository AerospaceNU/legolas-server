import time
from typing import Callable


class PIDGimbalController:
    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        control_callback: Callable[[float], None],
        output_degree_min: int | None = None,
        output_degree_max: int | None = None,
    ):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.control_callback = control_callback
        self.setpoint = 0
        self.output_degree_min = output_degree_min
        self.output_degree_max = output_degree_max

        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def process(self, pixel_err):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:  # Prevent division by zero
            dt = 1e-6

        error = pixel_err

        p_out = self.Kp * error

        # Integral term
        self.integral += error * dt
        i_out = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_out = self.Kd * derivative

        # Compute total output
        self.setpoint = p_out + i_out + d_out

        # Apply output limits
        if self.output_degree_max is not None:
            self.setpoint = min(self.output_degree_max, self.setpoint)
        if self.output_degree_min is not None:
            self.setpoint = max(self.output_degree_min, self.setpoint)

        self.prev_error = error
        self.last_time = current_time

        self.control_callback(self.setpoint)
