import math
import random

import pytest

from jetson.controller import (IDLE, LOST, TIMEOUT, TRACKING, FaceFollowController, FollowConfig,
                               TargetObservation, VelocityCommand, clamp, slew, soft_deadband)
from jetson.tests.helpers import obs_for

DT = 0.05  # 20 Hz control loop


def settle(ctrl, make_obs, t0=0.0, seconds=5.0, altitude=None):
    """Step the controller at 20 Hz with obs = make_obs(t); return the last output."""
    out, t = None, t0
    n = int(round(seconds / DT))
    for _ in range(n + 1):
        out = ctrl.step(make_obs(t), t, altitude_m=altitude)
        t += DT
    return out


# ---------------------------------------------------------------- helpers
def test_soft_deadband_is_continuous_and_signed():
    assert soft_deadband(2.9, 3.0) == 0.0
    assert soft_deadband(-3.0, 3.0) == 0.0
    assert soft_deadband(3.5, 3.0) == pytest.approx(0.5)
    assert soft_deadband(-3.5, 3.0) == pytest.approx(-0.5)
    assert soft_deadband(3.0 + 1e-9, 3.0) == pytest.approx(0.0, abs=1e-6)  # no jump at the edge


def test_clamp_and_slew():
    assert clamp(5, -1, 1) == 1 and clamp(-5, -1, 1) == -1 and clamp(0.3, -1, 1) == 0.3
    assert slew(0.0, 10.0, max_rate=2.0, dt=0.5) == pytest.approx(1.0)
    assert slew(1.0, -10.0, max_rate=2.0, dt=0.5) == pytest.approx(0.0)
    assert slew(0.0, 0.1, max_rate=2.0, dt=0.5) == pytest.approx(0.1)


# ---------------------------------------------------------------- tracking behaviour
def test_centred_face_at_target_distance_gives_zero_command():
    cfg = FollowConfig()
    out = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, cfg.target_distance_m, cfg, t))
    assert out.mode == TRACKING
    assert out.command.is_zero()
    assert out.distance_m == pytest.approx(cfg.target_distance_m, rel=1e-6)


@pytest.mark.parametrize("bearing,sign", [(15.0, +1), (-15.0, -1)])
def test_face_right_turns_right_and_left_turns_left(bearing, sign):
    cfg = FollowConfig()
    out = settle(FaceFollowController(cfg), lambda t: obs_for(bearing, cfg.target_distance_m, cfg, t))
    expected = cfg.k_yaw * (abs(bearing) - cfg.yaw_deadband_deg)
    assert out.command.yawspeed_deg_s == pytest.approx(sign * expected, rel=1e-3)
    assert out.bearing_deg == pytest.approx(bearing, rel=1e-6)


def test_yaw_deadband_ignores_small_offsets():
    cfg = FollowConfig(yaw_deadband_deg=3.0)
    out = settle(FaceFollowController(cfg), lambda t: obs_for(2.5, cfg.target_distance_m, cfg, t))
    assert out.command.yawspeed_deg_s == 0.0


def test_yaw_rate_saturates():
    cfg = FollowConfig(max_yaw_rate_deg_s=20.0)
    out = settle(FaceFollowController(cfg), lambda t: obs_for(29.0, cfg.target_distance_m, cfg, t))
    assert out.command.yawspeed_deg_s == pytest.approx(20.0)


def test_far_face_flies_forward_capped_and_near_face_backs_off_slowly():
    cfg = FollowConfig(max_forward_m_s=1.0, max_backward_m_s=0.4)
    far = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 12.0, cfg, t))
    assert far.command.forward_m_s == pytest.approx(1.0)
    near = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 1.0, cfg, t))
    assert near.command.forward_m_s == pytest.approx(-0.4)


def test_forward_speed_proportional_inside_limits():
    cfg = FollowConfig(k_forward=0.5, distance_deadband_m=0.5, max_forward_m_s=5.0)
    out = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, cfg.target_distance_m + 1.5, cfg, t))
    assert out.command.forward_m_s == pytest.approx(0.5 * (1.5 - 0.5), rel=1e-3)


def test_turn_first_then_approach():
    """An off-centre far face gets less forward speed than a centred one; none at the edge."""
    cfg = FollowConfig(max_forward_m_s=5.0)
    centred = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 8.0, cfg, t))
    off = settle(FaceFollowController(cfg), lambda t: obs_for(15.0, 8.0, cfg, t))
    edge = settle(FaceFollowController(cfg), lambda t: obs_for(cfg.hfov_deg / 2 - 0.01, 8.0, cfg, t))
    assert centred.command.forward_m_s > off.command.forward_m_s > 0
    assert edge.command.forward_m_s == pytest.approx(0.0, abs=1e-3)


def test_never_flies_forward_when_too_close():
    # Large deadband would otherwise allow forward motion inside min_distance.
    cfg = FollowConfig(target_distance_m=3.0, min_distance_m=2.8, distance_deadband_m=0.1)
    out = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 2.0, cfg, t))
    assert out.command.forward_m_s <= 0.0
    assert "too close" in out.reason


def test_lateral_mode_strafes_toward_face():
    cfg = FollowConfig(k_yaw=0.0, k_lateral=0.05)
    out = settle(FaceFollowController(cfg), lambda t: obs_for(10.0, cfg.target_distance_m, cfg, t))
    assert out.command.yawspeed_deg_s == 0.0
    assert out.command.right_m_s > 0.0


# ---------------------------------------------------------------- smoothness
def test_first_step_cannot_jump_and_slew_limits_ramp():
    cfg = FollowConfig(max_accel_m_s2=1.0, max_yaw_accel_deg_s2=60.0)
    ctrl = FaceFollowController(cfg)
    first = ctrl.step(obs_for(25.0, 12.0, cfg, 0.0), 0.0)
    assert first.command.is_zero()  # dt = 0 on the first step
    second = ctrl.step(obs_for(25.0, 12.0, cfg, 0.1), 0.1)
    assert second.command.forward_m_s == pytest.approx(0.1)        # 1 m/s^2 * 0.1 s
    assert second.command.yawspeed_deg_s == pytest.approx(6.0)     # 60 deg/s^2 * 0.1 s


def test_stalled_loop_does_not_unlock_a_big_jump():
    cfg = FollowConfig(max_accel_m_s2=1.0, max_dt_s=0.25)
    ctrl = FaceFollowController(cfg)
    ctrl.step(obs_for(0.0, 12.0, cfg, 0.0), 0.0)
    out = ctrl.step(obs_for(0.0, 12.0, cfg, 3.0), 3.0)  # 3 s gap
    assert out.command.forward_m_s == pytest.approx(0.25)


def test_time_going_backwards_is_harmless():
    cfg = FollowConfig()
    ctrl = FaceFollowController(cfg)
    ctrl.step(obs_for(10.0, 6.0, cfg, 1.0), 1.0)
    out = ctrl.step(obs_for(10.0, 6.0, cfg, 0.9), 0.9)
    assert out.mode == TRACKING and all(math.isfinite(v) for v in out.command.as_tuple())


# ---------------------------------------------------------------- target loss
def test_idle_hovers_until_acquire_timeout():
    cfg = FollowConfig(acquire_timeout_s=10.0, lost_action="rtl")
    ctrl = FaceFollowController(cfg)
    out = ctrl.step(None, 0.0)
    assert out.mode == IDLE and out.command.is_zero() and out.action is None
    out = ctrl.step(None, 9.9)
    assert out.mode == IDLE
    out = ctrl.step(None, 10.0)
    assert out.mode == TIMEOUT and out.action == "rtl"


def test_lost_target_hovers_immediately_then_times_out_and_latches():
    cfg = FollowConfig(lost_timeout_s=5.0, lost_action="land")
    ctrl = FaceFollowController(cfg)
    out = settle(ctrl, lambda t: obs_for(20.0, 10.0, cfg, t), seconds=3.0)
    assert not out.command.is_zero()
    t_lost = 3.0 + DT
    out = ctrl.step(None, t_lost)
    assert out.mode == LOST and out.command.is_zero()      # no slew on loss: hover now
    assert out.lost_for_s == pytest.approx(DT, abs=1e-6)
    out = ctrl.step(None, 3.0 + 4.9)
    assert out.mode == LOST
    out = ctrl.step(None, 3.0 + 5.0)
    assert out.mode == TIMEOUT and out.action == "land" and out.command.is_zero()
    # Face comes back: still TIMEOUT (latched) until reset().
    out = ctrl.step(obs_for(0.0, 4.0, cfg, 9.0), 9.0)
    assert out.mode == TIMEOUT and out.action == "land"
    ctrl.reset()
    out = ctrl.step(obs_for(0.0, 4.0, cfg, 9.1), 9.1)
    assert out.mode == TRACKING


def test_reacquire_before_timeout_resumes_tracking():
    cfg = FollowConfig(lost_timeout_s=5.0)
    ctrl = FaceFollowController(cfg)
    settle(ctrl, lambda t: obs_for(0.0, 4.0, cfg, t), seconds=1.0)
    assert ctrl.step(None, 2.0).mode == LOST
    assert ctrl.step(obs_for(10.0, 4.0, cfg, 3.0), 3.0).mode == TRACKING


def test_stale_observation_counts_as_lost():
    cfg = FollowConfig(max_obs_age_s=0.5)
    ctrl = FaceFollowController(cfg)
    settle(ctrl, lambda t: obs_for(10.0, 6.0, cfg, t), seconds=1.0)
    stale = obs_for(10.0, 6.0, cfg, stamp=1.0)
    out = ctrl.step(stale, 1.0 + 0.6)
    assert out.mode == LOST and out.command.is_zero() and "stale" in out.reason


@pytest.mark.parametrize("bad", [
    TargetObservation(float("nan"), 0.5, 0.1, 0.1, 0.0),
    TargetObservation(0.5, 0.5, float("inf"), 0.1, 0.0),
    TargetObservation(0.5, 0.5, 0.0, 0.1, 0.0),
    TargetObservation(0.5, 0.5, 1.5, 0.1, 0.0),
    TargetObservation(1.2, 0.5, 0.1, 0.1, 0.0),
    TargetObservation(0.5, 0.5, 0.1, 0.1, 5.0),  # stamp in the future
])
def test_invalid_observations_never_raise_and_mean_hover(bad):
    out = FaceFollowController().step(bad, 0.0)
    assert out.mode == IDLE and out.command.is_zero()


# ---------------------------------------------------------------- vertical axis
def test_vertical_disabled_by_default():
    cfg = FollowConfig()
    out = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 4.0, cfg, t, cy=0.95), altitude=4.0)
    assert out.command.down_m_s == 0.0


def test_vertical_follows_image_row_within_altitude_limits():
    cfg = FollowConfig(enable_vertical=True, min_altitude_m=2.0, max_altitude_m=6.0)
    low_face = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 4.0, cfg, t, cy=0.9), altitude=4.0)
    assert low_face.command.down_m_s > 0           # face below centre -> descend
    high_face = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 4.0, cfg, t, cy=0.1), altitude=4.0)
    assert high_face.command.down_m_s < 0          # face above centre -> climb
    floor = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 4.0, cfg, t, cy=0.9), altitude=2.0)
    assert floor.command.down_m_s == 0 and "min altitude" in floor.reason
    ceiling = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 4.0, cfg, t, cy=0.1), altitude=6.0)
    assert ceiling.command.down_m_s == 0 and "max altitude" in ceiling.reason
    blind = settle(FaceFollowController(cfg), lambda t: obs_for(0.0, 4.0, cfg, t, cy=0.9), altitude=None)
    assert blind.command.down_m_s == 0 and "no altitude" in blind.reason


# ---------------------------------------------------------------- config + invariants
@pytest.mark.parametrize("kwargs", [
    {"hfov_deg": 0}, {"hfov_deg": 180}, {"lost_action": "flip"},
    {"min_distance_m": 5.0, "target_distance_m": 4.0}, {"max_forward_m_s": -1},
    {"min_altitude_m": 6.0, "max_altitude_m": 2.0},
])
def test_bad_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FollowConfig(**kwargs)


def test_random_inputs_always_give_finite_bounded_commands():
    rng = random.Random(1234)
    cfg = FollowConfig(enable_vertical=True)
    ctrl = FaceFollowController(cfg)
    t = 0.0
    for _ in range(5000):
        t += rng.uniform(0.0, 0.3)
        r = rng.random()
        if r < 0.1:
            obs = None
        else:
            obs = TargetObservation(rng.uniform(-0.2, 1.2), rng.uniform(-0.2, 1.2), rng.uniform(-0.1, 1.2),
                                    rng.uniform(0.001, 1.0), t - rng.uniform(-0.1, 1.0))
        out = ctrl.step(obs, t, altitude_m=rng.uniform(0.0, 10.0))
        cmd = out.command
        assert all(math.isfinite(v) for v in cmd.as_tuple())
        assert -cfg.max_backward_m_s - 1e-9 <= cmd.forward_m_s <= cfg.max_forward_m_s + 1e-9
        assert abs(cmd.yawspeed_deg_s) <= cfg.max_yaw_rate_deg_s + 1e-9
        assert abs(cmd.down_m_s) <= cfg.max_vertical_m_s + 1e-9
        assert abs(cmd.right_m_s) <= cfg.max_lateral_m_s + 1e-9
        if out.mode != TRACKING:
            assert cmd.is_zero()
        if out.mode == TIMEOUT:
            ctrl.reset()


def test_velocity_command_str_and_tuple():
    cmd = VelocityCommand(1.0, -0.5, 0.25, 12.0)
    assert cmd.as_tuple() == (1.0, -0.5, 0.25, 12.0)
    assert "fwd +1.00" in str(cmd) and "yaw +12.0" in str(cmd)
