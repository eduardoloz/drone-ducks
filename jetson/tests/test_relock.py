"""Confirmed re-lock (jetson.relock), head-yaw proxy (jetson.headpose), the k_head_lateral term
and --camera-preset. No models needed."""

import argparse
import math

import numpy as np
import pytest

from jetson import cli
from jetson.controller import FaceFollowController, FollowConfig, TargetObservation, TRACKING
from jetson.face import Face
from jetson.headpose import head_yaw_deg, synth_landmarks, yaw_proxy
from jetson.pipeline import FollowPipeline
from jetson.relock import ENGAGED, RELOCKING, RelockGate
from jetson.sources import Frame
from jetson.tracker import IouTracker

from .helpers import obs_for


# ----------------------------------------------------------------------------- RelockGate
def test_gate_passes_flicker_but_confirms_after_a_loss():
    g = RelockGate(confirm_hits=3, confirm_score=None, lost_after_s=0.3)
    assert g.update(1, True, 0.7, 0.0) and g.state == ENGAGED
    assert g.update(1, False, 0.7, 0.1)            # one missed frame: still engaged, coast passes
    assert g.update(1, True, 0.7, 0.2)
    assert not g.update(1, False, 0.7, 0.6) and g.state == RELOCKING   # 0.4 s without a detection
    # a single detection (back of the head) does not re-lock, nor do two
    assert not g.update(1, True, 0.5, 0.7)
    assert not g.update(1, False, 0.5, 0.8)
    assert not g.update(1, True, 0.5, 0.9)
    assert not g.update(1, True, 0.5, 1.0)
    assert g.update(1, True, 0.5, 1.1) and g.state == ENGAGED and g.relocks == 1


def test_gate_streak_must_be_on_one_track_and_high_score_fast_path():
    g = RelockGate(confirm_hits=3, confirm_score=0.9, lost_after_s=0.3)
    g.update(1, True, 0.8, 0.0)
    g.update(None, False, 0.0, 1.0)
    assert g.state == RELOCKING
    assert not g.update(2, True, 0.6, 1.1)
    assert not g.update(3, True, 0.6, 1.2)          # different track: streak restarts
    assert not g.update(3, True, 0.6, 1.3)
    assert g.update(4, True, 0.95, 1.4)              # one confident detection is enough
    with pytest.raises(ValueError):
        RelockGate(confirm_hits=0)


def test_gate_with_one_hit_is_round2_behaviour():
    g = RelockGate(confirm_hits=1, confirm_score=None)
    g.update(1, True, 0.6, 0.0)
    g.update(None, False, 0.0, 2.0)
    assert g.update(5, True, 0.5, 2.1)


class ScriptedFaces:
    """Detector replaying a list of per-frame face lists."""

    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def detect(self, image):
        out = self.frames[self.i] if self.i < len(self.frames) else []
        self.i += 1
        return out

    def info(self):
        return {"backend": "scripted"}


def _run(frames, **kw):
    img = np.zeros((480, 640, 3), np.uint8)
    pipe = FollowPipeline(ScriptedFaces(frames), FaceFollowController(FollowConfig(lost_timeout_s=5.0)), **kw)
    return [pipe.step(Frame(img, k, k / 15.0), now=k / 15.0) for k in range(len(frames))]


FACE = Face(300, 200, 340, 250, 0.8)
FAR = Face(318, 230, 328, 242, 0.65)         # 10 px "face" on the back of a head, 7-10 m away


def test_pipeline_single_far_hits_after_a_loss_do_not_restart_tracking():
    frames = [[FACE]] * 10 + [[]] * 30 + [[FAR], [], [], [FAR], [FAR], []] + [[]] * 10
    res = _run(frames)
    assert res[9].control.mode == TRACKING
    after = res[40:]
    assert all(r.control.mode != TRACKING for r in after)          # never resumes
    assert any(r.gated for r in after)                              # a target was there, withheld
    # the lost timer was not restarted by the far hits: lost_for keeps growing
    lf = [r.control.lost_for_s for r in after if r.control.lost_for_s is not None]
    assert lf == sorted(lf)
    # round-2 behaviour for comparison: the far pair re-locks
    old = _run(frames, relock=RelockGate(confirm_hits=1, confirm_score=None))
    assert any(r.control.mode == TRACKING for r in old[40:])


def test_pipeline_relocks_on_a_confirmed_face():
    frames = [[FACE]] * 10 + [[]] * 30 + [[FACE]] * 6
    res = _run(frames)
    modes = [r.control.mode for r in res[40:]]
    assert modes[:3] != [TRACKING] * 3 and modes[-1] == TRACKING


def test_face_only_holdoff_blocks_new_tracks():
    frames = [[FACE]] * 10 + [[]] * 30 + [[FACE]] * 10
    res = _run(frames, relock_holdoff_s=10.0)
    assert all(r.control.mode != TRACKING for r in res[40:])


# ----------------------------------------------------------------------------- head yaw
@pytest.mark.parametrize("yaw", [-50, -20, 0, 15, 40])
def test_head_yaw_roundtrip_and_order_independent(yaw):
    for order in (True, False):
        lm = synth_landmarks(320, 240, 60, yaw, yunet_order=order)
        assert head_yaw_deg(lm) == pytest.approx(yaw, abs=0.5)
    mirrored = [(640 - x, y) for x, y in synth_landmarks(320, 240, 60, yaw)]
    assert head_yaw_deg(mirrored) == pytest.approx(-yaw, abs=0.5)


def test_head_yaw_rejects_bad_input():
    assert head_yaw_deg(None) is None
    assert yaw_proxy([(1, 1), (1, 1), (1, 2)]) is None
    assert head_yaw_deg(synth_landmarks(0, 0, 60, 20), face_width_px=10) is None   # too small
    assert yaw_proxy([(0, 0), (10, 0), (float("nan"), 1)]) is None


def test_pipeline_passes_head_yaw_from_landmarks():
    f = Face(300, 200, 360, 270, 0.9, synth_landmarks(330, 235, 60, 25))
    res = _run([[f]] * 5)
    assert res[-1].observation.head_yaw_deg == pytest.approx(25, abs=1)


# ----------------------------------------------------------------------------- lateral term
def _lateral(cfg, head, n=40):
    c = FaceFollowController(cfg)
    out = None
    for k in range(n):
        o = obs_for(0.0, 4.0, cfg, k / 15)
        o = TargetObservation(o.cx, o.cy, o.w, o.h, o.stamp, head_yaw_deg=head)
        out = c.step(o, k / 15)
    return out


def test_head_lateral_off_by_default():
    assert _lateral(FollowConfig(), 40.0).command.right_m_s == 0.0


def test_head_lateral_sign_deadband_saturation():
    cfg = FollowConfig(k_head_lateral=0.02, max_lateral_m_s=0.5)
    assert _lateral(cfg, 30.0).command.right_m_s == pytest.approx(0.02 * 22, abs=0.02)   # face points right
    assert _lateral(cfg, -30.0).command.right_m_s < 0
    assert _lateral(cfg, 5.0).command.right_m_s == 0.0             # inside the 8 deg deadband
    assert _lateral(cfg, 89.0).command.right_m_s == pytest.approx(0.5)
    assert _lateral(cfg, None).command.right_m_s == 0.0


def test_head_lateral_estimate_decays_when_unknown():
    cfg = FollowConfig(k_head_lateral=0.02)
    c = FaceFollowController(cfg)
    for k in range(30):
        o = obs_for(0.0, 4.0, cfg, k / 15)
        c.step(TargetObservation(o.cx, o.cy, o.w, o.h, o.stamp, head_yaw_deg=40.0), k / 15)
    for k in range(30, 50):          # body-only steering / tiny face: no yaw for > head_yaw_hold_s
        out = c.step(obs_for(0.0, 4.0, cfg, k / 15), k / 15)
    assert out.head_yaw_deg is None and out.command.right_m_s == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------------- presets
def _args(argv):
    ap = argparse.ArgumentParser()
    cli.add_controller_args(ap)
    return ap.parse_args(argv)


def test_camera_presets():
    d = cli.make_config(_args([]))
    assert (d.hfov_deg, d.face_width_m) == (FollowConfig().hfov_deg, FollowConfig().face_width_m)
    s = cli.make_config(_args(["--camera-preset", "sim"]))
    assert s.hfov_deg == pytest.approx(math.degrees(1.2), abs=0.01) and s.face_width_m == 0.13
    o = cli.make_config(_args(["--camera-preset", "sim", "--face-width", "0.15", "--head-lateral", "0.02"]))
    assert o.face_width_m == 0.15 and o.hfov_deg == s.hfov_deg and o.k_head_lateral == 0.02
    w = cli.make_config(_args(["--camera-preset", "webcam"]))
    assert 65 < w.hfov_deg < 75
    kw = cli.relock_kwargs(_args(["--relock-hits", "1", "--relock-score", "0"]))
    assert kw["relock"].confirm_hits == 1 and kw["relock"].confirm_score is None


def test_gate_distance_cap_matches_sitl_run2():
    # sim/docs/runs/follow-sitl-run2.csv: genuine re-lock at 34.4 s estimated 4.04 m, the false
    # back-of-head re-lock at 68.8 s 6.0-6.8 m (in streaks of up to 16 frames). Cap = 4 + 1.5 m.
    g = RelockGate(max_distance_m=5.5)
    g.update(1, True, 0.7, 0.0)
    g.update(None, False, 0.0, 1.0)
    assert not any(g.update(2, True, 0.7, 1.0 + k / 15, distance_m=6.0 + 0.05 * k) for k in range(16))
    assert [g.update(3, True, 0.7, 3.0 + k / 15, distance_m=4.04) for k in range(3)] == [False, False, True]


def test_headeval_lateral_term_keeps_the_walker_in_front():
    from jetson.headeval import run

    off = run(30.0, 0.0, seconds=25.0, heading_noise_deg=0.0)
    on = run(30.0, 0.02, seconds=25.0, heading_noise_deg=0.0)
    assert off["timeout_s"] is not None and on["timeout_s"] is None
    assert on["lock_pct"] > 95 and on["front_pct"] > off["front_pct"] + 50


def test_unconfirmed_relock_candidate_is_not_the_target_so_it_gets_blurred():
    frames = [[FACE]] * 10 + [[]] * 30 + [[FACE]] * 6
    res = _run(frames)
    gated = [r for r in res[40:] if r.gated]
    assert gated and all(r.target is None and r.faces for r in gated)


def test_gate_distance_cap_follows_a_far_target():
    g = RelockGate(max_distance_m=5.5)
    g.update(1, True, 0.7, 0.0, distance_m=8.0)      # following someone 8 m away
    g.update(None, False, 0.0, 0.5)
    assert g.distance_cap() == pytest.approx(9.5)
    assert [g.update(2, True, 0.7, 0.6 + k / 15, distance_m=8.1) for k in range(3)] == [False, False, True]
    g2 = RelockGate(max_distance_m=5.5)
    g2.update(1, True, 0.7, 0.0, distance_m=3.5)      # run 2: lost at ~3.5 m -> cap stays 5.5 m
    g2.update(None, False, 0.0, 0.5)
    assert g2.distance_cap() == 5.5
