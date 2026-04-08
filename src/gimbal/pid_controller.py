import time
from typing import Callable
from dataclasses import dataclass


@dataclass
class PIDOutput:
    error: float
    p: float
    i: float
    d: float
    d_raw: float
    integral: float
    output: float
    prev_output: float
    dt: float
    gain_scale: float

    def __str__(self):
        if self.gain_scale == 0.0:
            return f"err={self.error:+7.1f} [DEADBAND] output=0"
        return (
            f"err={self.error:+7.1f}  "
            f"P={self.p:+6.2f}  I={self.i:+6.3f}  "
            f"D={self.d:+6.3f}(raw={self.d_raw:+6.3f})  "
            f"out={self.output:+6.2f}  prev={self.prev_output:+6.2f}  "
            f"integ={self.integral:+6.1f}  dt={self.dt:.4f}  "
            f"scale={self.gain_scale:.2f}"
        )


class PIDGimbalController:
    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        control_callback: Callable[[float], None],
        output_degree_min: int | None = None,
        output_degree_max: int | None = None,
        derivative_filter_alpha: float = 0.3,
        max_output_delta: float | None = None,
    ):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.control_callback = control_callback
        self.output = 0
        self.output_degree_min = output_degree_min
        self.output_degree_max = output_degree_max
        self.derivative_filter_alpha = derivative_filter_alpha
        self.max_output_delta = max_output_delta

        self.prev_error = 0
        self.integral = 0
        self.filtered_derivative = 0
        self.last_time = time.time()

    def set_Kp(self, value):
        self.Kp = value

    def set_Ki(self, value):
        self.Ki = value

    def _compute_gain_scale(self, error, inner_deadband, outer_deadband):
        """Smooth quadratic ramp between inner and outer deadband."""
        abs_err = abs(error)
        if abs_err <= inner_deadband:
            return 0.0
        if abs_err >= outer_deadband:
            return 1.0
        t = (abs_err - inner_deadband) / (outer_deadband - inner_deadband)
        return t * t

    def process(self, pixel_err, inner_deadband=5, outer_deadband=25) -> PIDOutput:
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 1e-6

        prev_output = self.output
        error = pixel_err

        alpha = self._compute_gain_scale(error, inner_deadband, outer_deadband)

        # Inside inner deadband: decay integral and output zero
        if alpha == 0.0:
            decay = 0.8 ** (dt / 0.033)
            self.integral *= decay
            self.prev_error = error
            self.last_time = current_time
            self.output = 0
            self.control_callback(0)
            return PIDOutput(
                error=error, p=0, i=0, d=0, d_raw=0,
                integral=self.integral, output=0, prev_output=prev_output,
                dt=dt, gain_scale=0.0,
            )

        # P term
        p_out = self.Kp * error

        # Integral: reset on zero-crossing
        if error * self.prev_error < 0:
            self.integral = 0

        # Integral: accumulate scaled by proximity to center
        proximity_scale = min(1.0, abs(error) / outer_deadband)
        self.integral += error * dt * proximity_scale
        self.integral = max(-30, min(30, self.integral))
        i_out = self.Ki * self.integral

        # D term with low-pass filter
        raw_derivative = (error - self.prev_error) / dt
        a = self.derivative_filter_alpha
        self.filtered_derivative = (
            a * raw_derivative + (1 - a) * self.filtered_derivative
        )
        d_raw = self.Kd * raw_derivative
        d_out = self.Kd * self.filtered_derivative

        # Apply smooth gain scaling
        output = alpha * (p_out + i_out + d_out)

        # Rate-limit output change
        if self.max_output_delta is not None:
            delta = output - prev_output
            if abs(delta) > self.max_output_delta:
                output = prev_output + self.max_output_delta * (1 if delta > 0 else -1)

        # Apply output limits
        if self.output_degree_max is not None:
            output = min(self.output_degree_max, output)
        if self.output_degree_min is not None:
            output = max(self.output_degree_min, output)

        self.prev_error = error
        self.last_time = current_time
        self.output = output

        self.control_callback(output)

        return PIDOutput(
            error=error, p=p_out, i=i_out, d=d_out, d_raw=d_raw,
            integral=self.integral, output=output, prev_output=prev_output,
            dt=dt, gain_scale=alpha,
        )
