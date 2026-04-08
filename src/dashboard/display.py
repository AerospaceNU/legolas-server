import time
import threading
from datetime import datetime

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dashboard.state import DashboardState, EventLog
from gimbal.pid_controller import PIDOutput


def _pid_table(label: str, pid: PIDOutput | None) -> Table:
    """Build a compact table showing PID state for one axis."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="bold cyan", width=8)
    table.add_column("value", width=20)

    if pid is None:
        table.add_row(label, "[dim]no data[/dim]")
        return table

    table.add_row(f"{label} err", f"{pid.error:+7.1f}")
    table.add_row("  P", f"{pid.p:+6.2f}")
    table.add_row("  I", f"{pid.i:+6.3f}")
    table.add_row("  D", f"{pid.d:+6.3f}")
    table.add_row("  output", f"{pid.output:+6.2f}")
    table.add_row("  scale", f"{pid.gain_scale:.2f}")
    table.add_row("  integ", f"{pid.integral:+6.1f}")
    table.add_row("  dt", f"{pid.dt:.4f}s")

    return table


def _build_pid_panel(state: DashboardState) -> Panel:
    """Left panel: PID state for yaw and pitch."""
    table = Table(show_header=False, box=None, padding=(0, 0))
    table.add_column("content")

    table.add_row(_pid_table("YAW", state.yaw_pid))
    table.add_row("")
    table.add_row(_pid_table("PITCH", state.pitch_pid))

    return Panel(table, title="PID State", border_style="blue")


def _build_info_panel(state: DashboardState) -> Panel:
    """Right panel: target info + system status."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="bold yellow", width=10)
    table.add_column("value", width=22)

    # Target section
    if state.target_id is not None:
        table.add_row("Target ID", str(state.target_id))
        table.add_row("Class", state.target_class or "?")
        if state.target_position:
            table.add_row("Position", f"({state.target_position[0]:.0f}, {state.target_position[1]:.0f})")
        if state.target_bbox_size:
            table.add_row("BBox", f"{state.target_bbox_size[0]:.0f} x {state.target_bbox_size[1]:.0f}")
        if state.target_confidence is not None:
            table.add_row("Conf", f"{state.target_confidence:.2f}")
    else:
        table.add_row("Target", "[dim]none[/dim]")

    table.add_row("", "")

    # System section
    cam_status = "[green]OK[/green]" if state.camera_connected else "[red]OFF[/red]"
    gimbal_status = "[green]OK[/green]" if state.gimbal_connected else "[red]OFF[/red]"
    table.add_row("Camera", cam_status)
    table.add_row("Gimbal", gimbal_status)
    table.add_row("Frame", f"{state.frame_size[0]}x{state.frame_size[1]}")
    table.add_row("Loop dt", f"{state.loop_dt:.3f}s")
    table.add_row("Detect", str(state.detection_count))
    table.add_row("Yaw cmd", f"{state.yaw_speed:+.2f}")
    table.add_row("Pitch cmd", f"{state.pitch_speed:+.2f}")

    return Panel(table, title="Target / System", border_style="green")


def _build_event_panel(event_log: EventLog) -> Panel:
    """Bottom panel: scrolling event log."""
    entries = event_log.get_recent(12)
    lines = Text()

    for ts, msg in entries:
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        lines.append(f"{time_str} ", style="dim")
        lines.append(f"{msg}\n")

    if not entries:
        lines.append("[dim]No events yet[/dim]")

    return Panel(lines, title="Event Log", border_style="magenta")


def _generate_layout(state: DashboardState, event_log: EventLog) -> Layout:
    """Build the full dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="events", size=16),
    )

    layout["header"].update(
        Panel(Text("LEGOLAS Tracking System", justify="center", style="bold white"), border_style="bright_white")
    )

    layout["body"].split_row(
        Layout(name="pid", ratio=1),
        Layout(name="info", ratio=1),
    )

    layout["pid"].update(_build_pid_panel(state))
    layout["info"].update(_build_info_panel(state))
    layout["events"].update(_build_event_panel(event_log))

    return layout


class Dashboard:
    def __init__(self, state: DashboardState, event_log: EventLog):
        self.state = state
        self.event_log = event_log
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        try:
            with Live(
                _generate_layout(self.state, self.event_log),
                refresh_per_second=4,
                screen=True,
            ) as live:
                while not self._stop_event.is_set():
                    live.update(_generate_layout(self.state, self.event_log))
                    time.sleep(0.25)
        except Exception:
            # Live display failed (no terminal capabilities) -- fall back to periodic stderr logging
            import sys
            while not self._stop_event.is_set():
                s = self.state
                pid_yaw = f"Y err={s.yaw_pid.error:+.0f} out={s.yaw_pid.output:+.2f}" if s.yaw_pid else "Y idle"
                pid_pitch = f"P err={s.pitch_pid.error:+.0f} out={s.pitch_pid.output:+.2f}" if s.pitch_pid else "P idle"
                target = f"ID={s.target_id}" if s.target_id is not None else "none"
                print(
                    f"[LEGOLAS] target={target} det={s.detection_count} {pid_yaw} {pid_pitch} dt={s.loop_dt:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(1.0)
