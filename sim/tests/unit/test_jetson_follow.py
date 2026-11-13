"""QuackdroneSink (Jetson CommandSink -> quackdrone Drone) on the FakeBackend: command path,
clamping, lost-target / stall / PX4-override endings, and a small closed loop through the real
Jetson FollowPipeline with a synthetic face detector."""
import math
import time

import numpy as np
import pytest

from quackdrone import Drone, DroneConfig, Limits
from quackdrone.backends.fake import FakeBackend
from quackdrone.integrations.jetson_follow import QuackdroneSink, sim_follow_config

from jetson.controller import HOVER, LOST, TIMEOUT, TRACKING, ControlOutput, FaceFollowController, VelocityCommand
from jetson.face import Face
from jetson.pipeline import FollowPipeline
from jetson.sources import Frame

FAST = DroneConfig(settle_time_s=0.1, move_timeout_s=5.0, takeoff_timeout_s=5.0, ready_timeout_s=3.0,
                   cmd_timeout_s=0.3, link_timeout_s=0.5, offboard_engage_timeout_s=1.0)


def wait_for(cond, timeout=3.0, period=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(period)
    return False


def out(fwd=0.0, yaw=0.0, mode=TRACKING, action=None):
    return ControlOutput(VelocityCommand(fwd, 0.0, 0.0, yaw), mode, "test", action=action)


@pytest.fixture
def fake():
    f = FakeBackend.fast()
    yield f
    f.shutdown()


@pytest.fixture
def drone(fake):
    d = Drone("fake://", backend=fake, config=FAST, limits=Limits(fence_radius_m=30.0), log=None)
    yield d
    d.close(land=False)


@pytest.fixture
def sink(drone):
    s = QuackdroneSink(drone, takeoff_alt_m=2.0, stall_timeout_s=1.0, emit=None)
    yield s
    s.stop()


def test_start_takes_off_and_commands_fly(sink, drone, fake):
    sink.start()
    assert drone.in_control and drone.altitude_m == pytest.approx(2.0, abs=0.2)
    assert sink.altitude_m == pytest.approx(2.0, abs=0.2)
    for _ in range(20):
        sink.submit(out(fwd=1.0, yaw=30.0))
        time.sleep(0.02)
    assert fake.state.mode == "OFFBOARD"
    assert sink.sent == 20 and sink.last.sent.forward_m_s == pytest.approx(1.0)
    assert fake.state.yaw > 5.0                      # + yaw rate = clockwise
    assert math.hypot(fake.state.n, fake.state.e) > 0.1


def test_commands_are_clamped_by_quackdrone_limits(sink, drone):
    sink.start()
    sink.submit(out(fwd=5.0, yaw=200.0))
    assert sink.last.requested.forward_m_s == 5.0
    assert sink.last.sent.forward_m_s == pytest.approx(drone.limits.max_speed_xy)
    assert sink.last.sent.yawspeed_deg_s == pytest.approx(drone.limits.max_yaw_rate)


def test_soft_fence_still_applies(fake):
    d = Drone("fake://", backend=fake, config=FAST, limits=Limits(fence_radius_m=1.0), log=None)
    s = QuackdroneSink(d, takeoff_alt_m=2.0, stall_timeout_s=2.0, emit=None)
    try:
        s.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            s.submit(out(fwd=2.0))
            time.sleep(0.02)
        assert math.hypot(fake.state.n, fake.state.e) < 1.2
        assert s.last.sent.forward_m_s < 0.3
    finally:
        s.stop()
        d.close(land=False)


@pytest.mark.parametrize("action,mode", [("land", "LAND"), ("rtl", "RETURN_TO_LAUNCH")])
def test_lost_target_timeout_runs_action_once(sink, drone, fake, action, mode):
    sink.start()
    sink.submit(out(mode=LOST))
    sink.submit(out(mode=TIMEOUT, action=action))
    assert sink.finished and sink.action_taken == action
    assert fake.state.mode == mode and not drone.in_control
    sink.submit(out(fwd=1.0))                       # ignored after the session ended
    assert sink.sent == 1


def test_hold_action_keeps_control_in_position_hold(sink, drone, fake):
    sink.start()
    sink.submit(out(mode=TIMEOUT, action="hold"))
    assert sink.finished and sink.action_taken == "hold"
    assert drone.in_control and fake.state.mode == "OFFBOARD"


def test_stalled_perception_watchdog_then_stall_action(sink, drone, fake):
    sink.start()
    sink.submit(out(fwd=1.0))
    assert wait_for(lambda: any(e.kind == "cmd_timeout" for e in drone.events()), 1.0)  # 0.3 s watchdog
    assert not sink.finished
    assert wait_for(lambda: sink.finished, 2.0)                                           # 1.0 s stall
    assert sink.action_taken == "land" and fake.state.mode == "LAND"


def test_px4_override_ends_session_and_never_reengages(sink, drone, fake):
    sink.start()
    sink.submit(out(fwd=0.5))
    fake.inject_mode("HOLD", "Geofence: exceeding maximum distance to Home")
    assert wait_for(lambda: sink.finished, 2.0)
    assert "HOLD" in sink.finish_reason and sink.action_taken is None
    sink.submit(out(fwd=0.5))
    assert fake.state.mode == "HOLD"


def test_stop_lands_if_still_in_control(sink, fake):
    sink.start()
    sink.submit(out())
    sink.stop()
    assert sink.finished and sink.action_taken == "land" and fake.state.mode == "LAND"


def test_adopts_flying_drone_and_refuses_ground_without_takeoff(drone, fake):
    s = QuackdroneSink(drone, takeoff_alt_m=None, emit=None)
    with pytest.raises(Exception):
        s.start()
    drone.takeoff(2.0)
    drone.land(wait=False)
    fake.inject_mode("HOLD")                         # flying, not in our control
    assert wait_for(lambda: fake.state.in_air)
    s2 = QuackdroneSink(drone, takeoff_alt_m=None, emit=None)
    s2.start()
    assert drone.in_control
    s2.stop()


def test_nan_command_brakes_and_raises(sink, drone):
    sink.start()
    with pytest.raises(ValueError):
        sink.submit(out(fwd=float("nan")))
    assert drone.last_velocity.is_zero()


def test_rejects_bad_settings(drone):
    with pytest.raises(ValueError):
        QuackdroneSink(drone, stall_action="explode")
    with pytest.raises(ValueError):
        QuackdroneSink(drone, stall_timeout_s=0.1)  # shorter than the drone's watchdog


# ----------------------------------------------------------------------------- closed loop
class ProjectingDetector:
    """'Detects' a face by projecting a world point through a pinhole camera on the fake drone."""

    name = "synthetic"

    def __init__(self, fake, person_ne, hfov_deg=68.75, w=640, h=480, face_m=0.16):
        self.fake, self.person, self.hfov, self.w, self.h, self.face_m = fake, person_ne, hfov_deg, w, h, face_m

    def detect(self, image):
        s = self.fake.state
        dn, de = self.person[0] - s.n, self.person[1] - s.e
        dist = math.hypot(dn, de)
        bearing = math.degrees(math.atan2(de, dn)) - s.yaw
        bearing = (bearing + 180) % 360 - 180
        if abs(bearing) > self.hfov / 2 - 2 or dist < 0.5:
            return []
        f = (self.w / 2) / math.tan(math.radians(self.hfov / 2))
        cx = self.w / 2 + f * math.tan(math.radians(bearing))
        fw = f * self.face_m / dist
        return [Face(cx - fw / 2, self.h / 2 - fw / 2, cx + fw / 2, self.h / 2 + fw / 2, 0.9)]

    def info(self):
        return {"backend": self.name}


def test_closed_loop_turns_and_approaches_face(drone, fake):
    """Face 30 deg right and 8 m away: the real pipeline + controller + sink yaw right and close in."""
    sink = QuackdroneSink(drone, takeoff_alt_m=2.0, stall_timeout_s=2.0, emit=None)
    sink.start()
    person = (8 * math.cos(math.radians(30)), 8 * math.sin(math.radians(30)))   # NED, drone faces north
    det = ProjectingDetector(fake, person, face_m=sim_follow_config().face_width_m)
    pipe = FollowPipeline(det, FaceFollowController(sim_follow_config()), sink=sink)
    img = np.zeros((480, 640, 3), np.uint8)
    t0, seq, modes = time.monotonic(), 0, []
    try:
        while time.monotonic() - t0 < 6.0:
            seq += 1
            res = pipe.step(Frame(img, seq, time.monotonic()))
            modes.append(res.control.mode)
            time.sleep(1 / 15)
        s = fake.state
        dist = math.hypot(person[0] - s.n, person[1] - s.e)
        assert TRACKING in modes
        assert s.yaw == pytest.approx(30.0, abs=6.0)
        assert dist < 6.0
        assert dist > sim_follow_config().min_distance_m - 0.3
    finally:
        sink.stop()
    assert sink.action_taken == "land"


def test_closed_loop_lost_face_hovers_then_lands(drone, fake):
    sink = QuackdroneSink(drone, takeoff_alt_m=2.0, stall_timeout_s=2.0, emit=None)
    sink.start()
    det = ProjectingDetector(fake, (5.0, 0.0))
    pipe = FollowPipeline(det, FaceFollowController(sim_follow_config(lost_timeout_s=1.0)), sink=sink)
    img = np.zeros((480, 640, 3), np.uint8)
    seq = 0
    for _ in range(10):
        seq += 1
        pipe.step(Frame(img, seq, time.monotonic()))
        time.sleep(1 / 15)
    det.person = (-5.0, 0.0)                         # person behind the drone: face gone
    t0 = time.monotonic()
    while not sink.finished and time.monotonic() - t0 < 4.0:
        seq += 1
        res = pipe.step(Frame(img, seq, time.monotonic()))
        if res.control.mode == LOST:
            assert res.control.command == HOVER
        time.sleep(1 / 15)
    assert sink.finished and sink.action_taken == "land" and fake.state.mode == "LAND"
    assert 1.0 <= time.monotonic() - t0 < 2.5
