import math

import pytest

from quackdrone.safety import (BatteryMonitor, CommandWatchdog, Limits, VelocityCommand, clamp_position_target,
                               clamp_velocity_body)

L = Limits(max_speed_xy=2.0, max_speed_z=1.0, max_yaw_rate=60.0, min_alt_m=0.5, max_alt_m=10.0,
           fence_radius_m=30.0)


def clampv(cmd, pos=(0.0, 0.0, -2.0), yaw=0.0, home=(0.0, 0.0), ground=0.0, limits=L):
    return clamp_velocity_body(cmd, limits, yaw_deg=yaw, pos_ned=pos, home_ne=home, ground_d=ground)


def test_inside_limits_unchanged():
    cmd = VelocityCommand(1.0, -0.5, 0.2, 10.0)
    out, notes = clampv(cmd)
    assert out == cmd and notes == []


def test_speed_limits_keep_direction():
    out, notes = clampv(VelocityCommand(3.0, 4.0, -5.0, 200.0))
    assert math.hypot(out.forward_m_s, out.right_m_s) == pytest.approx(2.0)
    assert out.forward_m_s / out.right_m_s == pytest.approx(0.75)
    assert out.down_m_s == -1.0 and out.yawspeed_deg_s == 60.0
    assert set(notes) == {"max_speed_xy", "max_speed_z", "max_yaw_rate"}


def test_nan_rejected():
    with pytest.raises(ValueError):
        clampv(VelocityCommand(float("nan"), 0, 0, 0))
    with pytest.raises(ValueError):
        clampv(VelocityCommand(0, 0, 0, float("inf")))


def test_altitude_band():
    out, notes = clampv(VelocityCommand(0, 0, -0.5, 0), pos=(0, 0, -10.0))  # at max alt, climbing
    assert out.down_m_s == 0 and notes == ["max_alt"]
    out, _ = clampv(VelocityCommand(0, 0, 0.5, 0), pos=(0, 0, -10.0))       # descending is fine
    assert out.down_m_s == 0.5
    out, notes = clampv(VelocityCommand(0, 0, 0.5, 0), pos=(0, 0, -0.4))    # below min alt, descending
    assert out.down_m_s == 0 and notes == ["min_alt"]
    out, notes = clampv(VelocityCommand(0, 0, -1.0, 0), pos=(0, 0, -9.7))   # 0.3 m below the ceiling
    assert out.down_m_s == pytest.approx(-0.3) and notes == ["max_alt"]     # slows down approaching it


def test_fence_slows_the_approach():
    # 29.5 m out heading straight for the fence: outward speed capped at gain x 0.5 m = 0.5 m/s
    out, notes = clampv(VelocityCommand(2.0, 0, 0, 0), pos=(29.5, 0.0, -2.0))
    assert out.forward_m_s == pytest.approx(0.5) and notes == ["fence"]
    # far from the fence nothing changes
    out, notes = clampv(VelocityCommand(2.0, 0, 0, 0), pos=(10.0, 0.0, -2.0))
    assert out.forward_m_s == 2.0 and notes == []


def test_fence_removes_only_outward_component():
    # 31 m north of home facing north: forward is outward, right (east) is tangential
    out, notes = clampv(VelocityCommand(1.0, 1.0, 0, 0), pos=(31.0, 0.0, -2.0))
    assert out.forward_m_s == pytest.approx(0.0, abs=1e-9)
    assert out.right_m_s == pytest.approx(1.0)
    assert notes == ["fence"]
    # flying back towards home is allowed
    out, notes = clampv(VelocityCommand(-1.0, 0, 0, 0), pos=(31.0, 0.0, -2.0))
    assert out.forward_m_s == -1.0 and notes == []
    # facing east at the north edge: forward is tangential, left (negative right) is outward
    out, _ = clampv(VelocityCommand(1.0, -1.0, 0, 0), pos=(31.0, 0.0, -2.0), yaw=90.0)
    assert out.forward_m_s == pytest.approx(1.0) and out.right_m_s == pytest.approx(0.0, abs=1e-9)


def test_fence_disabled_or_no_home():
    out, notes = clampv(VelocityCommand(1, 0, 0, 0), pos=(100, 0, -2), limits=Limits(fence_radius_m=0))
    assert notes == []
    out, notes = clampv(VelocityCommand(1, 0, 0, 0), pos=(100, 0, -2), home=None)
    assert notes == []


def test_clamp_position_target():
    (n, e, d), notes = clamp_position_target(60.0, 0.0, -20.0, L, home_ne=(0.0, 0.0), ground_d=0.0)
    assert (n, e) == pytest.approx((30.0, 0.0)) and d == pytest.approx(-10.0)
    assert set(notes) == {"fence", "alt_band"}
    (n, e, d), notes = clamp_position_target(1.0, 2.0, -0.1, L, home_ne=(0.0, 0.0), ground_d=0.0)
    assert d == pytest.approx(-0.5) and notes == ["alt_band"]
    # ground is not at d=0 (EKF origin drift): band is relative to the takeoff point
    (_, _, d), _ = clamp_position_target(0, 0, -3.0, L, home_ne=(0, 0), ground_d=-2.0)
    assert d == pytest.approx(-3.0)


def test_watchdog_brake_then_hold():
    w = CommandWatchdog(timeout_s=0.5, brake_max_s=2.0, settle_speed=0.2)
    assert w.update(0.0, 1.0) is None            # idle: nothing to do
    w.command(10.0)
    assert w.update(10.3, 1.0) is None           # fresh
    w.command(10.4)
    assert w.update(10.8, 1.0) is None
    assert w.update(10.95, 1.0) == "brake"       # 0.55 s since last command
    assert w.update(11.0, 0.8) is None           # still braking, still fast
    assert w.update(11.2, 0.1) == "hold"         # slow enough
    assert w.update(20.0, 0.0) is None           # holding: done
    w.command(21.0)                              # new commands re-arm it
    assert w.state == w.ACTIVE


def test_watchdog_brake_timeout_and_release():
    w = CommandWatchdog(timeout_s=0.5, brake_max_s=2.0)
    w.command(0.0)
    assert w.update(1.0, 3.0) == "brake"
    assert w.update(2.0, 3.0) is None
    assert w.update(3.1, 3.0) == "hold"          # never slowed down: hold after brake_max_s anyway
    w.command(5.0)
    w.release()
    assert w.update(9.0, 1.0) is None            # released: position control owns the drone


def test_battery_monitor():
    b = BatteryMonitor(warn_pct=30, act_pct=20, hysteresis_pct=3)
    assert b.update(None) is None and b.update(float("nan")) is None
    assert b.update(80) is None
    assert b.update(30) == "warn"
    assert b.update(29) is None                  # edge-triggered
    assert b.update(20) == "act"
    assert b.update(10) is None
    assert b.update(40) is None                  # swapped battery: re-arms
    assert b.update(15) == "act"                 # straight to act (implies warned)
    with pytest.raises(ValueError):
        BatteryMonitor(warn_pct=10, act_pct=20)


# ---- sideways commands (jetson --head-lateral flies right/left; the camera only looks forward)
def test_lateral_only_command_is_speed_capped():
    out, notes = clampv(VelocityCommand(0.0, 3.0, 0, 0))
    assert out.right_m_s == pytest.approx(2.0) and out.forward_m_s == 0.0 and notes == ["max_speed_xy"]
    out, notes = clampv(VelocityCommand(0.0, -0.5, 0, 0))          # jetson max_lateral_m_s: untouched
    assert out.right_m_s == -0.5 and notes == []


def test_fence_clamps_a_purely_sideways_approach():
    # 29.5 m north of home facing EAST: flying LEFT (negative right) is straight at the fence
    out, notes = clampv(VelocityCommand(0.0, -0.5, 0, 0), pos=(29.5, 0.0, -2.0), yaw=90.0)
    assert notes == [] and out.right_m_s == pytest.approx(-0.5)     # 0.5 m left: 0.5 m/s allowed
    out, notes = clampv(VelocityCommand(0.0, -2.0, 0, 0), pos=(29.8, 0.0, -2.0), yaw=90.0)
    assert out.right_m_s == pytest.approx(-0.2) and notes == ["fence"]
    out, notes = clampv(VelocityCommand(0.0, -0.5, 0, 0), pos=(30.5, 0.0, -2.0), yaw=90.0)
    assert out.right_m_s == pytest.approx(0.0, abs=1e-9) and notes == ["fence"]
    # flying right (back toward home) is never limited by the fence
    out, notes = clampv(VelocityCommand(0.0, 0.5, 0, 0), pos=(30.5, 0.0, -2.0), yaw=90.0)
    assert out.right_m_s == pytest.approx(0.5) and notes == []
