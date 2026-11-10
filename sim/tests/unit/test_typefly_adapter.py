"""TypeFly PX4Wrapper on top of Drone + FakeBackend (ported from Eddie's tests/test_px4_wrapper.py).
Skipped unless TypeFly is installed (sim/scripts/setup_typefly.sh)."""
import math
import threading
import types

import pytest

pytest.importorskip("typefly")
from typefly.plan_execution import PlanCancelled, PlanPolicy  # noqa: E402
from typefly.robot_info import RobotInfo  # noqa: E402
from typefly.robot_wrapper import RobotWrapper  # noqa: E402

from quackdrone import Drone, DroneConfig, Limits  # noqa: E402
from quackdrone.backends.fake import FakeBackend  # noqa: E402
from quackdrone.integrations.typefly_px4 import PX4Wrapper, drone_settings  # noqa: E402

FAST = DroneConfig(settle_time_s=0.1, move_timeout_s=5.0, takeoff_timeout_s=5.0, ready_timeout_s=3.0)


class NoCamera:
    image = None
    image_process_result = {"yolo": []}
    yolo_client = None

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def robot():
    RobotWrapper.set_controller_func([lambda m: True, lambda q, r: "x"])
    fake = FakeBackend.fast(start_yaw=90.0)                  # facing east, like the old test
    drone = Drone("fake://", backend=fake, config=FAST, limits=Limits(max_step_m=5.0), log=None)
    r = PX4Wrapper(RobotInfo("t", "px4", {"takeoff_alt": 1.5}), drone=drone, obs=NoCamera())
    r.start()
    yield r
    r.stop()
    fake.shutdown()


def test_start_takes_off(robot):
    assert robot.drone.in_control
    assert robot.altitude() == pytest.approx(1.5, abs=0.2)


def test_relative_moves_compose(robot):
    t = robot.drone.telemetry
    n0, e0 = t.north_m, t.east_m
    robot.move_forward(2.0)            # facing east -> +E
    assert robot.drone._target[:2] == pytest.approx([n0, e0 + 2.0], abs=1e-6)
    robot.rotate_left(90)              # TypeFly: + = left -> east becomes north
    assert robot.drone._target[3] == pytest.approx(0.0, abs=1e-6)
    robot.move_forward(1.0)            # +N
    robot.move_right(1.0)              # facing north, right = east
    assert robot.drone._target[:2] == pytest.approx([n0 + 1.0, e0 + 3.0], abs=1e-6)
    robot.rotate_right(270)            # north -> west (-90)
    assert robot.drone._target[3] == pytest.approx(-90.0, abs=1e-6)
    t = robot.drone.telemetry
    assert (t.north_m, t.east_m) == pytest.approx((n0 + 1.0, e0 + 3.0), abs=0.2)


def test_move_is_clamped(robot):
    e0 = robot.drone.telemetry.east_m
    robot.move_forward(50)
    assert robot.drone._target[1] == pytest.approx(e0 + 5.0, abs=1e-6)


def test_lift_respects_altitude_band(robot):
    robot.lift(100)                    # one step capped at max_step (5 m), alt band max 10 m
    assert robot.altitude() == pytest.approx(1.5 + 5.0, abs=0.25)
    robot.lift(-100)
    robot.lift(-100)
    assert robot.altitude() == pytest.approx(robot.drone.limits.min_alt_m, abs=0.25)


def test_ground_distance_estimate(robot):
    cam_h = robot.altitude() + robot.cam_z_offset
    on_axis = types.SimpleNamespace(x=0.5, y=0.45, w=0.1, h=0.1)   # base exactly on the optical axis
    assert robot.ground_distance(on_axis) == pytest.approx(cam_h / math.tan(math.radians(robot.cam_pitch_deg)),
                                                           rel=0.05)
    near = types.SimpleNamespace(x=0.5, y=0.8, w=0.1, h=0.1)       # lower in the image = closer
    assert robot.ground_distance(near) < robot.ground_distance(on_axis)
    assert robot.ground_distance(types.SimpleNamespace(x=0.5, y=0.95, w=0.1, h=0.1)) is None  # cut off


def test_px4_extra_skills_go_through_policy(robot):
    for name in ("lift",):
        skill = robot.skillset.get_skill(name)
        policy = PlanPolicy(max_total_calls=1000, max_probe_calls=1000, max_delay=0.0)
        robot.set_policy(policy)
        skill(*[0.5 for _ in skill.args])
        robot.clear_policy()
        assert policy.total_calls >= 1, name


def test_cancel_interrupts_a_long_move(robot):
    cancel = threading.Event()
    robot.set_policy(PlanPolicy(cancel_event=cancel))
    robot.drone.config.move_timeout_s = 5.0
    threading.Timer(0.05, cancel.set).start()
    with pytest.raises(PlanCancelled):
        robot.move_forward(5.0)
    robot.clear_policy()
    robot.stop_motion()                # what the controller does after a cancel: hover here


def test_old_config_keys_still_work():
    s = drone_settings({"takeoff_alt": 2.5, "min_alt": 0.8, "move_timeout": 9, "connection": "fake://"})
    d = Drone.from_dict(s, log=None)
    assert d.config.takeoff_alt_m == 2.5 and d.limits.min_alt_m == 0.8 and d.config.move_timeout_s == 9
    d.backend.shutdown()
