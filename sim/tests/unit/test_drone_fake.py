"""Drone + FakeBackend: the sequencing, watchdog and failsafe logic, end to end, in ~10 s.
The fake reproduces PX4's rules (setpoint before offboard, offboard-loss failsafe, auto-disarm),
not its physics. The same scenarios run against real PX4 SITL in tests/sitl/."""
import math
import threading
import time

import pytest

from quackdrone import ArmingDenied, ControlLost, Drone, DroneConfig, Limits
from quackdrone.backends.fake import FakeBackend

FAST = DroneConfig(settle_time_s=0.1, move_timeout_s=5.0, takeoff_timeout_s=5.0, ready_timeout_s=3.0,
                   cmd_timeout_s=0.3, link_timeout_s=0.5, offboard_engage_timeout_s=1.0)


def wait_for(cond, timeout=3.0, period=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(period)
    return False


@pytest.fixture
def fake():
    f = FakeBackend.fast()
    yield f
    f.shutdown()


@pytest.fixture
def drone(fake):
    d = Drone("fake://", backend=fake, config=FAST, limits=Limits(fence_radius_m=30.0), log=None)
    d.connect()
    yield d
    d.close(land=False)


def test_takeoff_move_turn_land(drone, fake):
    assert drone.takeoff(2.0)
    t = drone.telemetry
    assert t.flight_mode == "OFFBOARD" and t.armed and t.in_air
    assert drone.altitude_m == pytest.approx(2.0, abs=0.2)
    assert drone.move(forward=1.0)                 # facing north
    assert drone.telemetry.north_m == pytest.approx(1.0, abs=0.2)
    assert drone.turn(90)                          # clockwise -> facing east
    assert drone.telemetry.yaw_deg == pytest.approx(90, abs=5)
    assert drone.move(forward=1.0)                 # now east
    assert drone.telemetry.east_m == pytest.approx(1.0, abs=0.2)
    assert drone.move(up=0.5)
    assert drone.altitude_m == pytest.approx(2.5, abs=0.2)
    assert drone.land()
    assert not fake.state.armed and not drone.in_control
    assert [e.kind for e in drone.events() if e.kind in ("takeoff", "land")] == ["takeoff", "land"]


def test_relative_moves_build_on_target_not_measurement(drone):
    drone.takeoff(2.0)
    for _ in range(4):
        drone.move(forward=0.5, wait=False)       # no waiting at all ...
    assert drone._target[0] == pytest.approx(2.0)  # ... still exactly 4 x 0.5 m from the takeoff spot


def test_velocity_stream_keeps_flying(drone):
    drone.takeoff(2.0)
    for _ in range(20):                            # 1 s at 20 Hz
        drone.set_velocity_body(forward_m_s=1.0)
        time.sleep(0.05)
    assert drone.telemetry.north_m > 0.7
    assert not drone.events("cmd_timeout")


def test_velocity_watchdog_brakes_then_holds(drone):
    drone.takeoff(2.0)
    drone.set_velocity_body(forward_m_s=1.0, yawspeed_deg_s=20)   # ...and then the "controller" hangs
    assert wait_for(lambda: drone.events("cmd_timeout"), timeout=1.5)
    assert wait_for(lambda: drone.events("hold"), timeout=3.0)
    n0 = drone.telemetry.north_m
    time.sleep(0.5)
    assert drone.telemetry.north_m == pytest.approx(n0, abs=0.05)  # really stopped
    assert drone.in_control                                        # still ours, just holding


def test_stop_brakes_and_holds(drone):
    drone.takeoff(2.0)
    drone.set_velocity_body(forward_m_s=1.5)
    time.sleep(0.2)
    drone.stop()
    assert drone.telemetry.speed < 0.3
    assert drone._target is not None                 # latched a position hold


def test_px4_override_is_never_fought(drone, fake):
    drone.takeoff(2.0)
    drone.move(forward=3.0)                          # away from home so RTL has somewhere to go
    fake.inject_mode("RETURN_TO_LAUNCH", "Failsafe activated: geofence")
    assert wait_for(lambda: not drone.in_control, timeout=1.0)
    assert drone.override_mode == "RETURN_TO_LAUNCH"
    with pytest.raises(ControlLost):
        drone.set_velocity_body(forward_m_s=1.0)
    with pytest.raises(ControlLost):
        drone.move(forward=1.0)
    ev = drone.events("px4_override")[-1]
    assert ev.data["mode"] == "RETURN_TO_LAUNCH"
    assert drone.events("px4")                      # PX4's status text was surfaced as an event
    time.sleep(0.3)
    assert fake.mode_log[-1] in ("RETURN_TO_LAUNCH", "LAND")       # we never switched it back


def test_override_interrupts_a_blocking_move(drone, fake):
    drone.takeoff(2.0)
    threading.Timer(0.1, fake.inject_mode, args=("HOLD",)).start()
    with pytest.raises(ControlLost):
        drone.goto_ned(25.0, 0.0, drone.telemetry.down_m)


def test_resume_control_after_override(drone, fake):
    drone.takeoff(2.0)
    fake.inject_mode("HOLD")
    assert wait_for(lambda: not drone.in_control, timeout=1.0)
    drone.resume_control()
    assert drone.in_control and drone.move(forward=1.0)


def test_low_battery_lands_before_px4_does(drone, fake):
    drone.takeoff(2.0)
    fake.set_battery(25)
    assert wait_for(lambda: drone.events("low_battery"), timeout=1.0)
    assert drone.in_control                          # warning only
    fake.set_battery(15)
    assert wait_for(lambda: not drone.in_control, timeout=1.0)
    assert wait_for(lambda: not fake.state.armed, timeout=5.0)
    with pytest.raises(ControlLost):
        drone.set_velocity_body(forward_m_s=1.0)


def test_exception_inside_with_block_lands(fake):
    with pytest.raises(RuntimeError):
        with Drone("fake://", backend=fake, config=FAST, log=None) as d:
            d.takeoff(2.0)
            raise RuntimeError("perception crashed")
    assert not fake.state.in_air and not fake.state.armed
    assert d.events("exception")


def test_process_death_triggers_px4_offboard_loss_failsafe(drone, fake):
    drone.takeoff(2.0)
    drone.close(land=False)        # what kill -9 does: setpoints stop, nobody lands
    assert fake.state.mode == "OFFBOARD"
    assert wait_for(lambda: fake.state.mode in ("RETURN_TO_LAUNCH", "LAND"), timeout=2.5)


def test_arming_denied_carries_px4_reason(drone, fake):
    fake.deny_next_arm("Preflight Fail: Accel #0 fail: TIMEOUT!")
    with pytest.raises(ArmingDenied) as ei:
        drone.takeoff(2.0)
    assert "Accel #0" in str(ei.value) and not drone.in_control


def test_soft_fence_clamps_targets(fake):
    d = Drone("fake://", backend=fake, config=FAST, limits=Limits(fence_radius_m=5.0), log=None).connect()
    try:
        d.takeoff(2.0)
        assert d.goto_ned(20.0, 0.0, d.telemetry.down_m)
        assert d.telemetry.north_m == pytest.approx(5.0, abs=0.2)
        assert d.events("limit")
    finally:
        d.close(land=False)


def test_bad_commands(drone):
    with pytest.raises(ControlLost):
        drone.set_velocity_body(forward_m_s=1.0)   # not flying yet
    drone.takeoff(2.0)
    with pytest.raises(ValueError):
        drone.set_velocity_body(forward_m_s=math.nan)
    cmd = drone.set_velocity_body(forward_m_s=50.0)
    assert cmd.forward_m_s == pytest.approx(drone.limits.max_speed_xy)


def test_link_loss_releases_control(drone, fake):
    drone.takeoff(2.0)
    fake.set_link_alive(False)
    assert wait_for(lambda: drone.events("link_lost"), timeout=2.0)
    assert not drone.in_control
    fake.set_link_alive(True)
    assert wait_for(lambda: drone.events("link_ok"), timeout=2.0)


def test_from_dict_and_fake_url():
    d = Drone.from_dict({"connection": "fake://fast", "max_speed_xy": 0.7, "takeoff_alt_m": 1.2,
                         "camera_url": "ignored"}, log=None)
    assert d.limits.max_speed_xy == 0.7 and d.config.takeoff_alt_m == 1.2
    assert isinstance(d.backend, FakeBackend)
    d.backend.shutdown()


def test_link_loss_replaces_resent_velocity_with_zero(drone, fake):
    # MAVSDK re-sends the last setpoint by itself; after a link drop it must not be a velocity
    drone.takeoff(2.0)
    drone.set_velocity_body(forward_m_s=1.0)
    fake.set_link_alive(False)
    assert wait_for(lambda: drone.events("link_lost"), timeout=2.0)
    assert fake._sp == ("vel", 0.0, 0.0, 0.0, 0.0)


def test_exit_lands_when_px4_still_offboard_after_control_lost(fake):
    # e.g. the laptop slept: link_lost -> ControlLost raised -> leaving the block must still land
    with pytest.raises(ControlLost):
        with Drone("fake://", backend=fake, config=FAST, log=None) as d:
            d.takeoff(2.0)
            fake.set_link_alive(False)
            assert wait_for(lambda: d.events("link_lost"), timeout=2.0)
            fake.set_link_alive(True)
            assert wait_for(lambda: d.events("link_ok"), timeout=2.0)
            d.move(forward=1.0)                      # raises ControlLost
    assert not fake.state.in_air and not fake.state.armed


def test_exit_leaves_a_px4_failsafe_alone(fake):
    with Drone("fake://", backend=fake, config=FAST, log=None) as d:
        d.takeoff(2.0)
        fake.inject_mode("HOLD", "Failsafe activated")    # PX4 / the pilot took over
        assert wait_for(lambda: not d.in_control, timeout=1.0)
    assert fake.state.mode == "HOLD" and fake.state.in_air   # we did not land over PX4's decision
