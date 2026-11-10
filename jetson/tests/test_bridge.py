"""Bridge safety logic against a fake PX4. No MAVSDK connection is ever made here."""

import threading
import time

import pytest

from jetson.bridge import (DroneAdapter, DryRunSink, MavsdkOffboardSink, NativeMavsdkAdapter, make_adapter)
from jetson.controller import HOVER, LOST, TIMEOUT, TRACKING, ControlOutput, VelocityCommand


class FakePX4(DroneAdapter):
    """Behaves like PX4 where it matters: OFFBOARD is refused unless setpoints are streaming."""

    def __init__(self, in_air=False, engage=True):
        self.log = []
        self.velocity = []
        self.in_air = in_air
        self.armed = in_air
        self.flight_mode = "POSCTL" if in_air else "HOLD"
        self.relative_altitude_m = 3.0 if in_air else 0.0
        self.engage = engage
        self._last_sp = None

    def _rec(self, name, *args):
        self.log.append((time.monotonic(), name) + args)

    def names(self):
        return [entry[1] for entry in self.log]

    async def connect(self, url, timeout_s):
        self._rec("connect", url)

    async def wait_ready(self, need_armable, require_global, timeout_s):
        self._rec("wait_ready", need_armable)

    async def arm(self):
        self.armed = True
        self._rec("arm")

    async def takeoff(self, altitude_m):
        self.in_air, self.relative_altitude_m = True, altitude_m
        self._rec("takeoff", altitude_m)

    async def land(self):
        self.flight_mode = "LAND"
        self._rec("land")

    async def return_to_launch(self):
        self.flight_mode = "RETURN_TO_LAUNCH"
        self._rec("rtl")

    async def hold(self):
        self.flight_mode = "HOLD"
        self._rec("hold")

    async def set_velocity_body(self, cmd):
        self._last_sp = time.monotonic()
        self.velocity.append((self._last_sp, cmd))
        self._rec("setpoint")

    async def start_offboard(self):
        if self._last_sp is None or time.monotonic() - self._last_sp > 1.0:
            raise RuntimeError("PX4: offboard rejected, no setpoints")
        if self.engage:
            self.flight_mode = "OFFBOARD"
        self._rec("start_offboard")

    async def stop_offboard(self):
        self.flight_mode = "HOLD"
        self._rec("stop_offboard")


def tracking(fwd=0.5, yaw=10.0):
    return ControlOutput(VelocityCommand(fwd, 0.0, 0.0, yaw), TRACKING, "test")


def make_sink(px4, **kw):
    defaults = dict(rate_hz=20.0, command_timeout_s=0.3, stall_timeout_s=0.8, connect_timeout_s=1.0,
                    ready_timeout_s=1.0, wait_for_pilot_s=1.0, engage_timeout_s=0.5, emit=lambda m: None)
    defaults.update(kw)
    return MavsdkOffboardSink(px4, **defaults)


def feed(sink, out, seconds, period=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not sink.finished:
        sink.submit(out)
        time.sleep(period)


# ---------------------------------------------------------------- start-up sequence
def test_auto_takeoff_streams_setpoints_before_offboard_start():
    px4 = FakePX4(in_air=False)
    sink = make_sink(px4, auto_takeoff_m=3.0)
    sink.start()
    try:
        names = px4.names()
        assert names[:4] == ["connect", "wait_ready", "arm", "takeoff"]
        assert px4.log[1][2] is True                      # waited for armable because we arm
        i_start = names.index("start_offboard")
        assert names[i_start - 1] == "setpoint" and names.count("setpoint") >= 3
        assert sink.offboard_active and px4.flight_mode == "OFFBOARD"
    finally:
        sink.stop()


def test_without_auto_takeoff_it_waits_for_the_pilot_and_never_arms():
    px4 = FakePX4(in_air=False)
    sink = make_sink(px4, wait_for_pilot_s=2.0)
    threading.Timer(0.3, lambda: setattr(px4, "in_air", True)).start()
    sink.start()
    try:
        assert "arm" not in px4.names() and "takeoff" not in px4.names()
        assert "start_offboard" in px4.names()
    finally:
        sink.stop()


def test_pilot_never_takes_off_times_out():
    sink = make_sink(FakePX4(in_air=False), wait_for_pilot_s=0.3)
    with pytest.raises(TimeoutError):
        sink.start()


# ---------------------------------------------------------------- streaming
def test_streams_newest_command_at_a_steady_rate():
    px4 = FakePX4(in_air=True)
    sink = make_sink(px4, rate_hz=20.0)
    sink.start()
    try:
        t0 = len(px4.velocity)
        feed(sink, tracking(0.5, 10.0), 0.5)
        feed(sink, tracking(0.2, -5.0), 0.5)
        sent = px4.velocity[t0:]
        assert 15 <= len(sent) <= 25                      # ~20 Hz for 1 s
        assert sent[-1][1] == VelocityCommand(0.2, 0.0, 0.0, -5.0)
        gaps = [b[0] - a[0] for a, b in zip(sent, sent[1:])]
        assert max(gaps) < 0.15                           # never a long silence
    finally:
        sink.stop()


def test_stale_commands_become_hover_then_stall_lands():
    px4 = FakePX4(in_air=True)
    sink = make_sink(px4, command_timeout_s=0.2, stall_timeout_s=0.6, stall_action="land")
    sink.start()
    sink.submit(tracking(0.8, 20.0))
    time.sleep(1.0)  # perception "hangs"
    try:
        cmds = [c for _, c in px4.velocity]
        assert VelocityCommand(0.8, 0.0, 0.0, 20.0) in cmds
        assert cmds[-1] == HOVER                          # hovered once commands went stale
        assert sink.finished and "stalled" in sink.finish_reason
        assert px4.names()[-2:] == ["stop_offboard", "land"]
    finally:
        sink.stop()


def test_controller_timeout_executes_the_requested_action():
    px4 = FakePX4(in_air=True)
    sink = make_sink(px4)
    sink.start()
    try:
        feed(sink, ControlOutput(HOVER, LOST, "lost"), 0.2)
        sink.submit(ControlOutput(HOVER, TIMEOUT, "target lost for 5.0 s -> rtl", action="rtl"))
        time.sleep(0.3)
        assert sink.finished and px4.names()[-2:] == ["stop_offboard", "rtl"]
        n = len(px4.velocity)
        time.sleep(0.2)
        assert len(px4.velocity) == n                     # nothing sent after finishing
    finally:
        sink.stop()


def test_pilot_override_stops_streaming_and_never_reengages():
    px4 = FakePX4(in_air=True)
    sink = make_sink(px4)
    sink.start()
    feed(sink, tracking(), 0.3)
    px4.flight_mode = "POSCTL"                            # pilot flips the RC mode switch
    feed(sink, tracking(), 0.4)
    n = len(px4.velocity)
    time.sleep(0.2)
    sink.stop()
    assert sink.pilot_override and sink.finished
    assert len(px4.velocity) == n                         # stopped sending
    assert px4.names().count("start_offboard") == 1       # never re-requested offboard
    assert "hold" not in px4.names() and "land" not in px4.names()  # pilot keeps control


def test_offboard_that_never_engages_ends_the_session():
    px4 = FakePX4(in_air=True, engage=False)              # PX4 accepted but mode never changed
    sink = make_sink(px4, engage_timeout_s=0.3)
    sink.start()
    feed(sink, tracking(), 0.6)
    sink.stop()
    assert sink.finished and "never engaged" in sink.finish_reason


def test_normal_stop_uses_on_exit_action():
    px4 = FakePX4(in_air=True)
    sink = make_sink(px4, on_exit="hold")
    sink.start()
    feed(sink, tracking(), 0.2)
    sink.stop()
    assert px4.names()[-2:] == ["stop_offboard", "hold"]


@pytest.mark.parametrize("kw", [{"rate_hz": 2.0}, {"stall_action": "flip"}, {"on_exit": "crash"},
                                {"auto_takeoff_m": 50.0}])
def test_bad_sink_config(kw):
    with pytest.raises(ValueError):
        make_sink(FakePX4(), **kw)


# ---------------------------------------------------------------- dry run + adapters
def test_dry_run_logs_mode_changes_and_throttles_repeats():
    lines = []
    sink = DryRunSink(log_every_s=10.0, emit=lines.append, finish_on_timeout=True)
    for _ in range(50):
        sink.submit(tracking())
    sink.submit(ControlOutput(HOVER, LOST, "lost"))
    assert len(lines) == 2 and "TRACKING" in lines[0] and "LOST" in lines[1]
    sink.submit(ControlOutput(HOVER, TIMEOUT, "gone -> land", action="land"))
    assert sink.finished and "LAND" in lines[-1]
    assert "DRY-RUN" in sink.status()


def test_make_adapter_picks_native_when_available():
    pytest.importorskip("mavsdk.asyncio")
    assert isinstance(make_adapter("auto"), NativeMavsdkAdapter)
    with pytest.raises(ValueError):
        make_adapter("pymavlink")


def test_linkcheck_is_read_only():
    import asyncio

    from jetson.linkcheck import watch

    px4 = FakePX4(in_air=True)
    lines = []
    assert asyncio.run(watch(px4, "fake://", seconds=0.3, connect_timeout=1.0, emit=lines.append)) == 0
    assert px4.names() == ["connect"]                     # nothing but the connection
    assert any("mode=POSCTL" in line for line in lines) and "no commands" in lines[-1]
