"""Flight behaviour against real PX4 SITL (Gazebo or SIH). ~3 min.

    sim/scripts/start_sim.sh
    sim/.venv/bin/python -m pytest -m sitl -v -s tests/sitl/test_flight.py
"""
import math
import time

import pytest

from quackdrone import Limits
from quackdrone.geometry import wrap_deg

from .conftest import horizontal_dist, stream_velocity

pytestmark = pytest.mark.sitl


def test_takeoff_move_turn_climb_land(pilot):
    assert pilot.takeoff(2.0)
    assert pilot.altitude_m == pytest.approx(2.0, abs=0.25)
    t0 = pilot.telemetry
    assert pilot.move(forward=2.0)
    t1 = pilot.telemetry
    moved = horizontal_dist(t1.position_ned, t0.position_ned)
    assert moved == pytest.approx(2.0, abs=0.3)
    # moved along the heading we were facing
    bearing = math.degrees(math.atan2(t1.east_m - t0.east_m, t1.north_m - t0.north_m))
    assert abs(wrap_deg(bearing - t0.yaw_deg)) < 10
    assert pilot.turn(90)
    assert wrap_deg(pilot.telemetry.yaw_deg - (t0.yaw_deg + 90)) == pytest.approx(0, abs=6)
    assert pilot.move(up=1.0)
    assert pilot.altitude_m == pytest.approx(3.0, abs=0.25)
    assert pilot.land()
    assert not pilot.telemetry.armed


def test_velocity_body_frame_is_yaw_aligned(pilot):
    """The contract the Jetson follower relies on: forward/right rotate with heading, yaw rate
    is clockwise-positive."""
    pilot.takeoff(2.0)
    assert pilot.yaw_to(90)                      # face east
    t0 = pilot.telemetry
    stream_velocity(pilot, 3.0, forward_m_s=1.0)
    pilot.stop()
    t1 = pilot.telemetry
    de, dn = t1.east_m - t0.east_m, t1.north_m - t0.north_m
    print(f"forward 1 m/s x 3 s facing east: dN={dn:+.2f} dE={de:+.2f}")
    assert de > 2.0 and abs(dn) < 0.6
    stream_velocity(pilot, 2.0, right_m_s=1.0)   # facing east, right = south
    pilot.stop()
    t2 = pilot.telemetry
    dn, de = t2.north_m - t1.north_m, t2.east_m - t1.east_m
    print(f"right 1 m/s x 2 s facing east: dN={dn:+.2f} dE={de:+.2f}")
    assert dn < -1.2 and abs(de) < 0.6
    y0 = pilot.telemetry.yaw_deg
    stream_velocity(pilot, 3.0, yawspeed_deg_s=30.0)
    pilot.stop()
    dy = wrap_deg(pilot.telemetry.yaw_deg - y0)
    print(f"yawspeed +30 deg/s x 3 s: heading change {dy:+.0f} deg")
    assert 60 < dy < 110                         # clockwise
    alt0 = pilot.altitude_m
    stream_velocity(pilot, 2.0, down_m_s=-0.5)   # negative down = climb
    pilot.stop()
    assert pilot.altitude_m - alt0 > 0.6


def test_watchdog_stops_a_hung_controller_and_mavsdk_would_not(pilot):
    """MAVSDK re-sends the last setpoint at 20 Hz by itself. Without our watchdog a controller
    that stops sending (hung, crashed thread) leaves the drone flying its last velocity."""
    pilot.takeoff(2.0)
    p0 = pilot.telemetry.position_ned
    pilot.set_velocity_body(forward_m_s=1.0)     # one command, then silence
    time.sleep(4.0)
    d_watchdog = horizontal_dist(pilot.telemetry.position_ned, p0)
    v_watchdog = pilot.telemetry.speed
    print(f"with watchdog: travelled {d_watchdog:.2f} m, speed now {v_watchdog:.2f} m/s")
    assert pilot.events("cmd_timeout") and pilot.events("hold")

    pilot._watchdog.timeout_s = 1e9               # what plain MAVSDK code does
    p0 = pilot.telemetry.position_ned
    pilot.set_velocity_body(forward_m_s=1.0)     # same single command
    time.sleep(4.0)
    d_plain = horizontal_dist(pilot.telemetry.position_ned, p0)
    v_plain = pilot.telemetry.speed
    print(f"without watchdog: travelled {d_plain:.2f} m and still moving at {v_plain:.2f} m/s")
    pilot._watchdog.timeout_s = pilot.config.cmd_timeout_s
    pilot.stop()
    # 0.5 s timeout + PX4 braking ~ 1.3 m; a 4 s fly-away at 1 m/s ~ 3.5 m and still going
    assert d_watchdog < 2.0 and v_watchdog < 0.3
    assert d_plain > 3.0 and v_plain > 0.7


def test_soft_fence_stops_velocity_at_radius(make_pilot):
    pilot = make_pilot(limits=Limits(fence_radius_m=3.0))
    pilot.takeoff(2.0)
    home = pilot.home_ned
    stream_velocity(pilot, 7.0, forward_m_s=1.0)
    pilot.stop()
    r = horizontal_dist(pilot.telemetry.position_ned, home)
    print(f"7 s at 1 m/s with a 3 m fence: ended {r:.2f} m from home")
    assert r < 3.3
    assert pilot.events("limit")


def test_soft_fence_stops_sideways_velocity(make_pilot):
    """jetson --head-lateral flies right/left (blind: forward camera only); the fence must clamp it too."""
    pilot = make_pilot(limits=Limits(fence_radius_m=3.0, max_speed_xy=0.8))
    pilot.takeoff(2.0)
    home = pilot.home_ned
    stream_velocity(pilot, 8.0, right_m_s=2.0)           # asks for 2 m/s sideways: capped to 0.8
    pilot.stop()
    r = horizontal_dist(pilot.telemetry.position_ned, home)
    print(f"8 s at 2 m/s sideways (cap 0.8) with a 3 m fence: ended {r:.2f} m from home")
    assert r < 3.3
    assert pilot.events("limit")
