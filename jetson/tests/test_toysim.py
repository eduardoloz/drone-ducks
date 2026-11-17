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


@pytest.mark.parametrize("alt,pitch,target", [(3.0, 20.0, 4.0), (5.0, 30.0, 5.0)])
def test_altitude_aware_distance_holds_the_ground_distance(alt, pitch, target):
    from jetson.controller import FaceFollowController, FollowConfig
    res = {}
    for pa in (False, True):
        w = ToyWorld(altitude=alt, cam_pitch_deg=pitch, person_n=target + 3)
        c = FaceFollowController(FollowConfig(target_distance_m=target, cam_pitch_deg=pitch, distance_deadband_m=0.05))
        h = run_closed_loop(w, c, 30, pass_altitude=pa)
        res[pa] = sum(x["distance"] for x in h[-60:]) / 60
    assert abs(res[True] - target) < 0.1           # holds the commanded ground distance
    assert res[False] < target - 0.15              # slant range: stops too close


def test_assumed_altitude_used_when_no_telemetry():
    from jetson.controller import FaceFollowController, FollowConfig, TargetObservation
    obs = TargetObservation(0.5, 0.5, 0.05, 0.06, stamp=0.0)
    raw = FaceFollowController(FollowConfig()).step(obs, 0.0)
    assumed = FaceFollowController(FollowConfig(assumed_altitude_m=3.0)).step(obs, 0.0)
    tele = FaceFollowController(FollowConfig(assumed_altitude_m=10.0)).step(obs, 0.0, altitude_m=3.0)
    assert raw.distance_m == raw.range_m
    assert assumed.distance_m < raw.distance_m and assumed.distance_m == pytest.approx(tele.distance_m)


# ----------------------------------------------------------------------------- round-5 review: ego compensation
class _StaticTarget:
    """Pipeline driven in the toy world with a fixed yaw command (no controller feedback), so the
    image motion of a static face is caused only by the drone's own rotation."""

    def __init__(self, yaw_cmd, ego, telemetry=False):
        from jetson.controller import VelocityCommand
        from jetson.pipeline import FollowPipeline

        self.world = world_with_person(8.0, 0.0, altitude=1.6)
        ctrl = FaceFollowController(FollowConfig(hfov_deg=self.world.hfov_deg))
        ctrl._cmd = VelocityCommand(yawspeed_deg_s=yaw_cmd)   # what the pipeline's lag model reads
        self.cmd = ctrl._cmd
        self.ctrl, self.telemetry = ctrl, telemetry
        self.pipe = FollowPipeline(OracleDetector(self.world), ctrl, ego_compensation=ego)
        self.errors = []   # |predicted centre - detected centre| per frame, px

    def run(self, frames=16, dt=1 / 15):
        from jetson.sources import Frame

        blank = np.zeros((self.world.height, self.world.width, 3), np.uint8)
        trk = self.pipe.tracker
        orig_update = trk.update

        def spy(faces, stamp):
            if faces and trk.tracks:
                p = trk.tracks[0].predicted_box(stamp)
                self.errors.append(abs((p[0] + p[2]) / 2 - (faces[0].x1 + faces[0].x2) / 2))
            return orig_update(faces, stamp)

        trk.update = spy
        for i in range(frames):
            rate = self.world.yaw_rate if self.telemetry else None
            self.pipe.step(Frame(blank, i + 1, self.world.t), now=self.world.t, yaw_rate_deg_s=rate)
            self.ctrl._cmd = self.cmd          # keep yawing regardless of what the controller wanted
            self.world.step(self.cmd, dt)
        return self


@pytest.mark.parametrize("yaw_cmd", [20.0, -20.0])
def test_ego_compensation_predicts_the_image_shift_of_the_drones_own_turn(yaw_cmd):
    # Sign check in both directions: without compensation a static face 8 m away appears to move
    # by up to ~5 px more per frame than the tracker predicts while the turn builds up (its
    # constant-velocity model only catches up later); with compensation the prediction is right.
    # A wrong sign would double the error instead.
    off = _StaticTarget(yaw_cmd, ego=False).run()
    lag = _StaticTarget(yaw_cmd, ego=True).run()                    # commanded rate through the 0.35 s lag
    tel = _StaticTarget(yaw_cmd, ego=True, telemetry=True).run()    # measured yaw rate passed to step()
    onset = slice(0, 8)
    assert len(off.errors) == len(lag.errors) == len(tel.errors) == 15
    assert np.mean(off.errors[onset]) > 2.5
    assert np.mean(lag.errors[onset]) < 0.15 * np.mean(off.errors[onset])
    assert np.mean(tel.errors[onset]) < 0.15 * np.mean(off.errors[onset])
    assert abs(lag.world.bearing_deg) > 10.0 and lag.world.bearing_deg * yaw_cmd < 0   # it really turned


def test_ego_lag_model_starts_slow_like_the_drone():
    # The first frame after a step in the yaw command: the toy drone (tau 0.35 s) has only reached
    # dt / (tau + dt) of the commanded rate, and so has the pipeline's estimate.
    s = _StaticTarget(30.0, ego=True).run(frames=2)
    a = (1 / 15) / (0.35 + 1 / 15)
    assert s.pipe._yaw_rate_est == pytest.approx(30.0 * a, rel=1e-6)
