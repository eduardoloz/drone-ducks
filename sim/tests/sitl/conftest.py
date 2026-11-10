"""Fixtures for tests that fly PX4 SITL. Start it first:  sim/scripts/start_sim.sh  (or SIM=sih).

Two MAVLink links to the same simulated PX4:
  * pilot     udpin://0.0.0.0:14540  (PX4's onboard/API port: where our code runs on the real drone)
  * observer  udpin://0.0.0.0:14550  (PX4's ground-station port: watches modes, sets params, and
              cleans up, so a test can kill the pilot process and still see what PX4 does)
"""
import math
import os
import subprocess
import sys
import time

import pytest

from quackdrone import Drone, DroneConfig, Limits

PILOT_URL = os.environ.get("QUACK_PILOT_URL", "udpin://0.0.0.0:14540")
OBSERVER_URL = os.environ.get("QUACK_OBSERVER_URL", "udpin://0.0.0.0:14550")
# Generous timeouts: SITL on a busy laptop runs slower than real time.
SITL_CONFIG = DroneConfig(move_timeout_s=30.0, takeoff_timeout_s=45.0, ready_timeout_s=120.0)


def px4_running() -> bool:
    if os.environ.get("QUACK_SITL") == "1":
        return True
    return subprocess.run(["pgrep", "-f", "bin/px4"], capture_output=True).returncode == 0


def pytest_collection_modifyitems(config, items):
    if px4_running():
        return
    skip = pytest.mark.skip(reason="PX4 SITL not running (start it with sim/scripts/start_sim.sh)")
    for item in items:
        if "sitl" in item.keywords:
            item.add_marker(skip)


def wait_for(cond, timeout: float, period: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(period)
    return False


def tag(name):
    return lambda msg: print(f"[{name}] {msg}", flush=True)


def wait_on_ground(drone: Drone, timeout: float = 90.0) -> None:
    """Previous test may still be landing: wait until PX4 reports landed and disarmed."""
    ok = wait_for(lambda: not drone.telemetry.in_air and not drone.telemetry.armed, timeout)
    assert ok, f"drone still flying/armed after {timeout:.0f} s: {drone.telemetry}"


@pytest.fixture
def make_pilot():
    made = []

    def _make(limits: Limits = None, config: DroneConfig = None) -> Drone:
        d = Drone(PILOT_URL, limits=limits, config=config or SITL_CONFIG, log=tag("pilot"))
        d.connect(timeout_s=20)
        made.append(d)
        wait_on_ground(d)
        return d

    yield _make
    for d in made:
        d.close()  # lands if still flying in Offboard


@pytest.fixture
def pilot(make_pilot):
    return make_pilot()


@pytest.fixture(scope="module")
def observer():
    d = Drone(OBSERVER_URL, config=SITL_CONFIG, log=tag("observer"))
    d.connect(timeout_s=20)
    yield d
    t = d.telemetry
    if t.armed and t.in_air:
        d.land(wait=True)
    d.close(land=False)


class ModeRecorder:
    """Records PX4 flight-mode changes seen by a Drone (with timestamps) on a background poll."""

    def __init__(self, drone: Drone):
        self.drone = drone
        self.t0 = time.monotonic()
        self.changes = []  # (seconds since start, mode)

    def poll(self, seconds: float, until=None, period: float = 0.05) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            mode = self.drone.telemetry.flight_mode
            if not self.changes or self.changes[-1][1] != mode:
                self.changes.append((round(time.monotonic() - self.t0, 2), mode))
            if until and until(self.drone.telemetry):
                return
            time.sleep(period)

    def first(self, mode: str):
        return next((t for t, m in self.changes if m == mode), None)

    def __str__(self):
        return " -> ".join(f"{m}@{t:.1f}s" for t, m in self.changes)


def horizontal_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def stream_velocity(drone: Drone, seconds: float, hz: float = 10.0, **cmd) -> None:
    """Send the same velocity command at `hz` for `seconds`, like a perception loop would."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        drone.set_velocity_body(**cmd)
        time.sleep(1.0 / hz)


def child_pilot(behaviour: str) -> subprocess.Popen:
    """Start tests/sitl/child_pilot.py in its own process (so the test can kill it)."""
    script = os.path.join(os.path.dirname(__file__), "child_pilot.py")
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return subprocess.Popen([sys.executable, script, behaviour, PILOT_URL], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)


def wait_line(proc: subprocess.Popen, needle: str, timeout: float) -> list[str]:
    """Read the child's output until a line contains needle (or timeout/exit). Returns the lines."""
    import selectors

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    lines, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not sel.select(timeout=0.5):
            if proc.poll() is not None:
                break
            continue
        line = proc.stdout.readline()
        if not line:
            break
        print("   child>", line.rstrip(), flush=True)
        lines.append(line)
        if needle in line:
            break
    return lines
