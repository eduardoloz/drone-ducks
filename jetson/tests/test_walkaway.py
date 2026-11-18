"""Walk-away end (FollowConfig.end_on_walkaway, jetson/walkaway.py): detector, controller, toy closed
loop and the SITL-log replay."""

import os
import random

import pytest

from jetson.controller import TIMEOUT, TRACKING, FaceFollowController, FollowConfig, TargetObservation
from jetson.walkaway import WalkAwayDetector

DT = 1 / 15


def drive(det, seconds, person_speed, drone_speed=0.0, start_dist=6.0, face=lambda t: False,
          noise=0.0, seed=0, dist_fn=None):
    """Person recedes at person_speed, drone flies toward them at drone_speed. Returns trigger time."""
    rng = random.Random(seed)
    d = start_dist
    t = 0.0
    while t < seconds:
        meas = dist_fn(t, d) if dist_fn else d
        meas += rng.gauss(0.0, noise) if noise else 0.0
        if det.update(t, drone_speed, meas, face_visible=face(t)).triggered:
            return t
        t += DT
        d += (person_speed - drone_speed) * DT
    return None


def test_person_walking_off_while_drone_hovers_fires_after_the_window():
    t = drive(WalkAwayDetector(), 10.0, person_speed=1.2)
    assert t is not None and 3.0 <= t <= 3.6


def test_drone_keeping_pace_still_sees_the_walk_away():
    # Distance grows only 0.2 m/s because the drone follows at 1.0 m/s; the person does 1.2 m/s.
    t = drive(WalkAwayDetector(), 10.0, person_speed=1.2, drone_speed=1.0)
    assert t is not None and t <= 4.0


def test_drone_approaching_a_standing_person_is_not_a_walk_away():
    assert drive(WalkAwayDetector(), 20.0, person_speed=0.0, drone_speed=0.6, start_dist=12.0) is None


@pytest.mark.parametrize("speed", [0.0, 0.4, -0.8])
def test_standing_slow_drift_or_approaching_never_fires(speed):
    assert drive(WalkAwayDetector(), 30.0, person_speed=speed, start_dist=7.0) is None


@pytest.mark.parametrize("seed", range(5))
def test_noisy_distance_of_a_standing_person_never_fires(seed):
    # 0.3 m box noise plus the odd 2 m outlier (SITL body-box distances are about this bad)
    rng = random.Random(100 + seed)

    def bad_box(t, d):
        return d + (2.0 if rng.random() < 0.05 else 0.0)

    assert drive(WalkAwayDetector(), 60.0, person_speed=0.0, start_dist=6.0, noise=0.3, seed=seed,
                 dist_fn=bad_box) is None


def test_face_visible_never_fires_and_mostly_face_blocks_it():
    assert drive(WalkAwayDetector(), 10.0, person_speed=1.2, face=lambda t: True) is None
    # face in every other frame: more than 1/3 of the window -> not "back turned"
    assert drive(WalkAwayDetector(), 10.0, person_speed=1.2, face=lambda t: int(t / DT) % 2 == 0) is None


def test_a_back_of_head_face_streak_does_not_restart_the_clock():
    # 5 face frames (a hand-back confirmed on the back of the head) in the middle of walking off.
    t = drive(WalkAwayDetector(), 10.0, person_speed=1.2, face=lambda t: 2.0 <= t < 2.0 + 5 * DT)
    assert t is not None and t <= 3.6


def test_a_distance_step_when_the_geometry_is_relearned_is_not_motion():
    # Standing person; after 3 face frames the body distance scale steps up by 1.2 m (SITL 34.8 s).
    def step(t, d):
        return d + (1.2 if t >= 2.0 else 0.0)

    face = lambda t: 2.0 - 3 * DT <= t < 2.0
    assert drive(WalkAwayDetector(), 10.0, person_speed=0.0, start_dist=6.0, face=face, dist_fn=step) is None


def test_closer_than_target_plus_margin_does_not_fire():
    det = WalkAwayDetector(min_distance_m=20.0)
    assert drive(det, 10.0, person_speed=1.2) is None
    assert det.last.speed_m_s == pytest.approx(1.2, abs=0.1)


def test_a_gap_longer_than_a_second_restarts_the_window():
    det = WalkAwayDetector()
    t, d = 0.0, 6.0
    while t < 2.5:
        assert not det.update(t, 0.0, d).triggered
        t, d = t + DT, d + 1.2 * DT
    t += 1.5   # nothing for 1.5 s
    fired = None
    while t < 10.0 and fired is None:
        if det.update(t, 0.0, d).triggered:
            fired = t
        t, d = t + DT, d + 1.2 * DT
    assert fired is not None and fired >= 2.5 + 1.5 + 3.0 - 1e-6


def test_detector_validates_its_parameters():
    with pytest.raises(ValueError):
        WalkAwayDetector(seconds=0)
    with pytest.raises(ValueError):
        WalkAwayDetector(speed_m_s=0)


# ----------------------------------------------------------------------------- controller
def _obs(t, dist, cfg, source):
    import math
    # box width for this distance (level camera, centred)
    w = cfg.face_width_m / (2.0 * dist * math.tan(math.radians(cfg.hfov_deg) / 2.0))
    return TargetObservation(0.5, 0.5, w, 1.25 * w, stamp=t, source=source)


def _run_controller(cfg, source, seconds=10.0, speed=1.3):
    c = FaceFollowController(cfg)
    t, d, outs = 0.0, cfg.target_distance_m + 1.5, []
    while t < seconds:
        out = c.step(_obs(t, d, cfg, source), t)
        outs.append(out)
        # the drone flies what the controller says; the person walks away
        d += (speed - out.command.forward_m_s) * DT
        t += DT
    return outs


def test_walkaway_end_is_off_by_default():
    assert FollowConfig().end_on_walkaway is False
    outs = _run_controller(FollowConfig(), "body")
    assert all(o.mode == TRACKING for o in outs)


def test_walkaway_end_times_out_with_the_lost_action_and_latches():
    cfg = FollowConfig(end_on_walkaway=True, lost_action="hold")
    outs = _run_controller(cfg, "body")
    first = next(i for i, o in enumerate(outs) if o.mode == TIMEOUT)
    o = outs[first]
    assert o.action == "hold" and "walked away" in o.reason and o.command.is_zero()
    assert 2.5 <= first * DT <= 5.0
    assert all(x.mode == TIMEOUT and x.action == "hold" for x in outs[first:])   # latched
    # the estimate is exported for logs (StepLog recede_m_s) while tracking
    far = _run_controller(FollowConfig(end_on_walkaway=True, walkaway_margin_m=50.0), "body")
    assert all(x.mode == TRACKING for x in far)
    assert [x.recede_m_s for x in far if x.recede_m_s is not None][-1] == pytest.approx(1.3, abs=0.2)


def test_walkaway_end_never_fires_while_the_face_is_visible():
    outs = _run_controller(FollowConfig(end_on_walkaway=True), "face")
    assert all(o.mode == TRACKING for o in outs)


def test_walkaway_config_validation():
    for bad in ({"walkaway_s": 0.0}, {"walkaway_speed_m_s": -1.0}, {"walkaway_margin_m": -0.5}):
        with pytest.raises(ValueError):
            FollowConfig(**bad)


def test_cli_flags_reach_the_config():
    import argparse

    from jetson import cli
    ap = argparse.ArgumentParser()
    cli.add_controller_args(ap)
    cfg = cli.make_config(ap.parse_args(["--end-on-walkaway", "--walkaway-speed", "1.1",
                                         "--walkaway-seconds", "4", "--walkaway-margin", "2"]))
    assert cfg.end_on_walkaway and cfg.walkaway_speed_m_s == 1.1 and cfg.walkaway_s == 4.0
    assert cfg.walkaway_margin_m == 2.0
    assert cli.make_config(ap.parse_args([])).end_on_walkaway is False


# ----------------------------------------------------------------------------- closed loop (toy world)
def _toy(name, on, seed=0):
    from jetson.lockeval import run
    from jetson.toysim import ToyWorld
    cfg = FollowConfig(hfov_deg=ToyWorld().hfov_deg, end_on_walkaway=on)
    return run(name, use_body=True, seed=seed, config=cfg)


def test_toy_person_who_leaves_ends_the_follow_only_when_enabled():
    off = _toy("leaves", False)
    assert off["timeout"] == 0.0 and off["on_target"] > 0.95      # round-3 behaviour: the chase
    on = _toy("leaves", True)
    assert on["walked_away"] == 1.0 and 4.0 < on["timeout_t"] < 8.0  # turns away at 4 s
    assert on["on_wrong"] == 0.0


@pytest.mark.parametrize("name", ["turn_around", "face_visible"])
def test_toy_standing_with_back_turned_or_facing_never_ends_it(name):
    r = _toy(name, True)
    assert r["timeout"] == 0.0 and r["walked_away"] == 0.0


# ----------------------------------------------------------------------------- SITL log replay
RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "sim", "docs", "runs")


def _log(name):
    p = os.path.join(RUNS, name)
    if not os.path.exists(p):
        pytest.skip("sim run log not in this checkout")
    return p


def test_sitl_replay_turnaway_walker_ends_on_the_fast_walk_away_only():
    from jetson.lockeval import walkaway_replay
    r = walkaway_replay(_log("r3-turnaway-body.csv.gz"))
    assert r["false_triggers"] == 0
    assert r["triggers"] and 74.0 <= r["triggers"][0]["t"] <= 79.0   # walks off at 1.37 m/s from ~74 s
    # the slow 0.39 m/s drift away (28-41 s) and all face-on following stay untouched at defaults
    assert all(tr["t"] > 70.0 for tr in r["triggers"])
    assert all(ep["detected_at"] is not None for ep in r["episodes"])


def test_sitl_replay_alt3_walk_away_is_caught_before_the_body_is_lost():
    from jetson.lockeval import walkaway_replay
    r = walkaway_replay(_log("r3-alt3-body-t5.csv.gz"))
    assert r["target_distance_m"] == 5.0 and r["false_triggers"] == 0
    assert 66.0 <= r["triggers"][0]["t"] <= 71.0     # round 3 chased this walker until LOST at ~81 s
    assert r["triggers"][0]["true_dist"] < 10.0


@pytest.mark.parametrize("name", ["r3-turnaway-face.csv.gz", "r3-alt3-face-t5.csv.gz", "follow-sitl-run2.csv.gz"])
def test_sitl_replay_face_only_runs_never_trigger(name):
    from jetson.lockeval import walkaway_replay
    assert walkaway_replay(_log(name))["triggers"] == []
