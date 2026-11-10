"""Closed-loop checks: controller + a kinematic drone + a (walking) person, no PX4 needed."""

import math

import numpy as np
import pytest

from jetson.controller import LOST, TIMEOUT, FaceFollowController, FollowConfig
from jetson.toysim import OracleDetector, ToyWorld, run_closed_loop


def world_with_person(distance, bearing_deg, **kw):
    b = math.radians(bearing_deg)
    return ToyWorld(person_n=distance * math.cos(b), person_e=distance * math.sin(b), **kw)


def test_projection_matches_geometry():
    w = ToyWorld(person_n=4.0, person_e=0.0, altitude=1.6)
    x1, y1, x2, y2 = w.face_box()
    assert (x1 + x2) / 2 == pytest.approx(320.0) and (y1 + y2) / 2 == pytest.approx(240.0)
    assert x2 - x1 == pytest.approx(640 * 0.16 / (2 * 4.0 * math.tan(math.radians(31.1))))
    w.visible = False
    assert w.face_box() is None and w.observe() is None


def test_converges_to_standoff_and_centres_the_face():
    cfg = FollowConfig(hfov_deg=62.2)
    world = world_with_person(9.0, 20.0)
    hist = run_closed_loop(world, FaceFollowController(cfg), seconds=25)
    assert abs(world.distance_m - cfg.target_distance_m) <= cfg.distance_deadband_m + 0.3
    assert abs(world.bearing_deg) <= cfg.yaw_deadband_deg + 1.0
    assert all(h["seen"] for h in hist[2:])           # never lost the face on the way in


def test_follows_a_walking_person_without_losing_them():
    """P-only control lags a moving target. Once the drone is tail-chasing, forward speed
    k * (d - target - deadband) must equal the walking speed v, so it settles at
    d = target + deadband + v / k  (= 4 + 0.5 + 0.8 / 0.5 = 6.1 m here), not at 4 m.
    A feed-forward term from the track's velocity would remove that lag (a good first task)."""
    cfg = FollowConfig(hfov_deg=62.2)
    v = 0.8
    world = world_with_person(6.0, 0.0, person_ve=v)  # walks sideways at 0.8 m/s
    hist = run_closed_loop(world, FaceFollowController(cfg), seconds=40)
    assert all(h["seen"] for h in hist[2:])
    expected = cfg.target_distance_m + cfg.distance_deadband_m + v / cfg.k_forward
    tail = [h["distance"] for h in hist if h["t"] > 32]
    assert abs(tail[-1] - expected) < 0.5
    assert max(tail) - min(tail) < 0.4                 # settled, not drifting away


def test_perception_latency_still_converges_without_growing_oscillation():
    cfg = FollowConfig(hfov_deg=62.2)
    world = world_with_person(8.0, 25.0)
    hist = run_closed_loop(world, FaceFollowController(cfg), seconds=30, latency_s=0.3)
    tail = [h["bearing"] for h in hist if h["t"] > 22]
    assert max(abs(b) for b in tail) <= cfg.yaw_deadband_deg + 1.5
    assert abs(world.distance_m - cfg.target_distance_m) <= cfg.distance_deadband_m + 0.4


def test_target_disappears_hover_then_land():
    cfg = FollowConfig(hfov_deg=62.2, lost_timeout_s=4.0, lost_action="land")
    world = world_with_person(9.0, 0.0)
    ctrl = FaceFollowController(cfg)
    run_closed_loop(world, ctrl, seconds=3.0)
    assert world.speed_m_s > 0.3                        # flying toward the person
    world.visible = False
    t_lost = world.t
    hist = run_closed_loop(world, ctrl, seconds=6.0)
    lost = [h for h in hist if h["mode"] == LOST]
    assert lost and lost[0]["cmd"].is_zero()
    slowed = [h for h in hist if h["t"] - t_lost > 1.5]
    assert all(h["speed"] < 0.05 for h in slowed)       # PX4-like lag decays to a hover
    timeout = [h for h in hist if h["mode"] == TIMEOUT]
    assert timeout and timeout[0]["action"] == "land"
    assert timeout[0]["t"] - t_lost == pytest.approx(4.0, abs=0.2)


def test_backs_off_when_too_close():
    cfg = FollowConfig(hfov_deg=62.2)
    world = world_with_person(1.6, 0.0, altitude=1.8)
    run_closed_loop(world, FaceFollowController(cfg), seconds=15)
    assert world.distance_m >= cfg.min_distance_m


def test_level_camera_cannot_see_a_close_face_from_altitude():
    """Design input: at 3 m altitude a level camera (VFOV 48.8 deg) loses a 1.6 m-high face closer
    than (3 - 1.6) / tan(24.4 deg) = 3.1 m. Tilting the camera down ~20 deg fixes that."""
    level = world_with_person(2.5, 0.0, altitude=3.0)
    assert level.face_box() is None
    tilted = world_with_person(2.5, 0.0, altitude=3.0, cam_pitch_deg=20.0)
    assert tilted.face_box() is not None


def test_noisy_detections_through_the_full_pipeline():
    """Oracle detector with noise + dropouts -> tracker -> selector -> controller still converges."""
    from jetson.pipeline import FollowPipeline
    from jetson.sources import Frame

    cfg = FollowConfig(hfov_deg=62.2)
    world = world_with_person(8.0, -15.0)
    pipe = FollowPipeline(OracleDetector(world, noise_px=1.0, dropout=0.1, seed=1), FaceFollowController(cfg))
    dt = 1 / 15
    blank = np.zeros((world.height, world.width, 3), np.uint8)
    for i in range(int(25 / dt)):
        res = pipe.step(Frame(blank, i + 1, world.t), now=world.t)
        world.step(res.control.command, dt)
    assert abs(world.distance_m - cfg.target_distance_m) <= cfg.distance_deadband_m + 0.4
    assert abs(world.bearing_deg) <= cfg.yaw_deadband_deg + 1.5


def test_render_draws_a_face_where_projected():
    world = world_with_person(4.0, 0.0)
    img = world.render()
    assert img.shape == (480, 640, 3)
    x1, y1, x2, y2 = (int(v) for v in world.face_box())
    face_patch = img[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
    assert face_patch[2] > 150                        # skin-ish (red channel high) where the face is
