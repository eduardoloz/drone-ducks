"""Safety behaviours against real PX4 SITL. Each test prints the PX4 flight-mode sequence it saw,
which is what docs/ARCHITECTURE.md's failsafe matrix records. ~6 min.

    sim/scripts/start_sim.sh            # (SIM=sih is fine too, and faster)
    sim/.venv/bin/python -m pytest -m sitl -v -s tests/sitl/test_failsafes.py

Tests that change PX4 parameters restore them in `finally`; start_sim.sh also starts every run
with fresh parameters, so nothing leaks into the next session.
"""
import signal
import time

import pytest

from quackdrone import ControlLost, DroneConfig, Limits

from .conftest import ModeRecorder, SITL_CONFIG, child_pilot, horizontal_dist, wait_for, wait_line, wait_on_ground

pytestmark = pytest.mark.sitl


def test_pilot_process_killed_mid_flight(observer):
    """kill -9 of our script: setpoints stop -> PX4 offboard-loss failsafe (COM_OF_LOSS_T, then
    COM_OBL_RC_ACT; with no RC in SITL that falls through to NAV_RCL_ACT = Return)."""
    wait_on_ground(observer)
    proc = child_pilot("velocity")
    try:
        lines = wait_line(proc, "READY", timeout=150)
        assert any("READY" in line for line in lines), "child never got airborne"
        time.sleep(2.0)
        assert observer.telemetry.flight_mode == "OFFBOARD"
        rec = ModeRecorder(observer)
        proc.send_signal(signal.SIGKILL)
        alt_max = 0.0
        t_end = time.monotonic() + 90
        while time.monotonic() < t_end:
            rec.poll(0.5)
            t = observer.telemetry
            alt_max = max(alt_max, -t.down_m)
            if not t.armed and not t.in_air:
                break
        print(f"after kill -9: {rec}; max height above EKF origin {alt_max:.1f} m")
        left = next((ts for ts, m in rec.changes if m != "OFFBOARD"), None)
        assert left is not None and left < 8.0, f"PX4 stayed in OFFBOARD: {rec}"
        assert not observer.telemetry.armed, f"not landed after 90 s: {rec}"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_exception_in_script_lands(observer):
    """An exception inside `with Drone(...)` lands the drone before the process exits."""
    wait_on_ground(observer)
    proc = child_pilot("raise")
    try:
        wait_line(proc, "READY", timeout=150)
        rec = ModeRecorder(observer)
        rec.poll(60, until=lambda t: not t.armed and not t.in_air)
        rc = proc.wait(timeout=30)
        print(f"after the exception: {rec}; child exit code {rc}")
        assert rc != 0
        assert rec.first("LAND") is not None, f"never landed: {rec}"
        assert rec.first("RETURN_TO_LAUNCH") is None, "PX4 failsafe fired instead of our landing"
        assert not observer.telemetry.armed
    finally:
        if proc.poll() is None:
            proc.kill()


def test_geofence_breach_px4_takes_over(observer, make_pilot):
    """PX4's hard geofence (GF_MAX_HOR_DIST, GF_ACTION=Hold) wins; quackdrone notices the mode
    change, stops commanding and raises ControlLost."""
    saved = {k: observer.get_param(k) for k in ("GF_MAX_HOR_DIST", "GF_ACTION")}
    print(f"saved {saved}")
    try:
        observer.set_param("GF_MAX_HOR_DIST", 6.0)
        observer.set_param("GF_ACTION", 2)   # 2 = Hold (the PX4 default)
        pilot = make_pilot(limits=Limits(fence_radius_m=0))  # our soft fence off: let PX4's trip
        pilot.takeoff(2.0)
        home = pilot.home_ned
        rec = ModeRecorder(observer)
        with pytest.raises(ControlLost):
            end = time.monotonic() + 20
            while time.monotonic() < end:
                pilot.set_velocity_body(forward_m_s=1.5)
                rec.poll(0.1)
        rec.poll(2.0)
        r = horizontal_dist(pilot.telemetry.position_ned, home)
        px4_msgs = [e.message for e in pilot.events("px4")]
        print(f"geofence: {rec}; stopped {r:.1f} m from home; override={pilot.override_mode}; PX4 said {px4_msgs[-4:]}")
        assert pilot.override_mode in ("HOLD", "RETURN_TO_LAUNCH", "LAND")
        assert r < 9.0
        pilot.land()
    finally:
        for k, v in saved.items():
            observer.set_param(k, v)


def test_low_battery_quackdrone_lands_first(make_pilot):
    """Our layer lands at battery_land_pct, well before PX4's critical threshold (7 %).
    (Sim battery: 100 % at arming, drains 100 % per SIM_BAT_DRAIN = 60 s while armed.)"""
    cfg = DroneConfig(move_timeout_s=30.0, takeoff_timeout_s=45.0, ready_timeout_s=120.0,
                      battery_warn_pct=85.0, battery_land_pct=78.0)
    pilot = make_pilot(config=cfg)
    pilot.takeoff(2.0)
    assert wait_for(lambda: pilot.events("low_battery"), timeout=40)
    assert wait_for(lambda: not pilot.in_control, timeout=40)
    msgs = [e.message for e in pilot.events("low_battery")]
    print(f"low battery events: {msgs}")
    assert any("landing" in m for m in msgs)
    with pytest.raises(ControlLost):
        pilot.set_velocity_body(forward_m_s=0.5)
    assert wait_for(lambda: not pilot.telemetry.armed, timeout=60)


def test_low_battery_px4_backstop(observer, make_pilot):
    """If our layer only warns, PX4's battery failsafe (COM_LOW_BAT_ACT; SITL default 3 = Return
    at critical, land at emergency) takes over. PX4 first holds for COM_FAIL_ACT_T (5 s, time for
    a pilot to react), then acts."""
    saved = {k: observer.get_param(k) for k in ("SIM_BAT_DRAIN", "SIM_BAT_MIN_PCT", "COM_LOW_BAT_ACT")}
    print(f"saved {saved}")
    try:
        cfg = DroneConfig(move_timeout_s=30.0, takeoff_timeout_s=45.0, ready_timeout_s=120.0,
                          low_battery_action="warn")
        pilot = make_pilot(config=cfg)
        pilot.takeoff(2.0)
        pilot.move(forward=4.0)                  # away from home, so Return has somewhere to go
        observer.set_param("SIM_BAT_DRAIN", 20.0)
        observer.set_param("SIM_BAT_MIN_PCT", 3.0)
        rec = ModeRecorder(observer)
        rec.poll(60, until=lambda t: t.flight_mode not in ("OFFBOARD", "UNKNOWN"))
        rec.poll(20, until=lambda t: t.flight_mode in ("RETURN_TO_LAUNCH", "LAND"))
        rec.poll(2)
        print(f"battery failsafe: {rec}; battery {observer.telemetry.battery_pct}%; "
              f"COM_LOW_BAT_ACT={saved['COM_LOW_BAT_ACT']}; override={pilot.override_mode}")
        assert pilot.override_mode is not None and not pilot.in_control
        assert rec.first("RETURN_TO_LAUNCH") is not None or rec.first("LAND") is not None, str(rec)
        assert pilot.events("low_battery")
        pilot.land()
        wait_on_ground(observer)
    finally:
        for k, v in saved.items():
            observer.set_param(k, v)
