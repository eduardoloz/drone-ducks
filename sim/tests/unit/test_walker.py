"""The Python copy of the walker trajectory must match the actor script in quack_people.sdf."""
import math
import os
import re

import pytest

from quackdrone.integrations import walker

SDF = os.path.join(os.path.dirname(__file__), "..", "..", "gazebo", "worlds", "quack_people.sdf")


def test_waypoints_match_world_file():
    text = open(SDF).read()
    actor = text[text.index('<actor name="walker">'):text.index("</actor>")]
    found = [(float(t), *map(float, p.split()))
             for t, p in re.findall(r"<time>([\d.]+)</time><pose>([^<]+)</pose>", actor)]
    got = [v for t, x, y, _z, _r, _p, yaw in found for v in (t, x, y, yaw)]
    assert got == pytest.approx([v for wp in walker.WAYPOINTS for v in wp])


def test_pose_interpolates_and_loops():
    assert walker.walker_pose(0.0)[:2] == pytest.approx((-5.0, -12.0))
    assert walker.walker_pose(7.75)[:2] == pytest.approx((-5.0, -8.9))
    assert walker.walker_pose(walker.PERIOD_S + 7.75)[:2] == pytest.approx((-5.0, -8.9))


def test_facing_legs_are_slow_and_within_17_deg_of_north():
    for (t0, x0, y0, a0), (t1, x1, y1, _a1) in zip(walker.WAYPOINTS, walker.WAYPOINTS[1:]):
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-6:
            continue
        speed = dist / (t1 - t0)
        if any(w0 <= t0 < w1 for w0, w1 in walker.FACING_WINDOWS):
            assert speed == pytest.approx(0.4, abs=0.02)
            assert abs(math.degrees(a0) - 90) <= 17


@pytest.mark.parametrize("script", list(walker.SCRIPTS.values()), ids=lambda s: s.name)
def test_every_script_matches_its_world_file(script):
    path = os.path.join(os.path.dirname(SDF), f"{script.name}.sdf")
    text = open(path).read()
    assert f'<world name="{script.name}">' in text
    actor = text[text.index('<actor name="walker">'):text.index("</actor>")]
    found = [(float(t), *map(float, p.split()))
             for t, p in re.findall(r"<time>([\d.]+)</time><pose>([^<]+)</pose>", actor)]
    got = [v for t, x, y, _z, _r, _p, yaw in found for v in (t, x, y, yaw)]
    assert got == pytest.approx([v for wp in script.waypoints for v in wp])


def test_turnaway_has_a_face_away_stretch_longer_than_the_lost_timeout():
    sc = walker.TURNAWAY
    (_, end1), (start2, _) = sc.facing_windows
    assert start2 - end1 > 5.0 + 5.0          # well past the 5 s lost timeout ...
    assert start2 - end1 < 15.0               # ... but inside jetson LockKeeper.max_body_only_s
    # while turned away the walker walks away from the drone (south), slowly enough to follow
    x0, y0, a0 = walker.walker_pose(30.0, sc)
    x1, y1, _ = walker.walker_pose(34.0, sc)
    assert y1 < y0 and abs(a0 - walker.S) < 1e-6 and (y0 - y1) / 4.0 == pytest.approx(0.4, abs=0.02)
    assert walker.get("turnaway") is sc and walker.get(None) is walker.LEAD


def test_scripts_stay_inside_the_follow_fence():
    for sc in walker.SCRIPTS.values():
        for _t, x, y, _a in sc.waypoints:
            assert (x * x + y * y) ** 0.5 < 25.0 - 5.0   # drone stays ~4 m from the walker


def test_angled_legs_cross_the_line_of_sight_at_25_and_40_deg():
    sc = walker.ANGLED
    assert walker.get("angled") is sc and [lab for *_t, lab in sc.legs] == ["A25", "B40"]
    for (t0, t1, label), want in zip(sc.legs, (25.0, 40.0)):
        x0, y0, yaw0 = walker.walker_pose(t0 + 0.01, sc)
        x1, y1, _ = walker.walker_pose(t1 - 0.01, sc)
        heading = math.degrees(math.atan2(y1 - y0, x1 - x0))
        assert heading == pytest.approx(math.degrees(yaw0), abs=0.5)        # walks where it faces
        assert abs(heading - 90.0) == pytest.approx(want, abs=0.5)           # vs. north (drone ahead)
        assert math.hypot(x1 - x0, y1 - y0) / (t1 - t0) == pytest.approx(0.4, abs=0.02)
    # the drone start is ~4 m straight ahead of the walker's start
    sx, sy = sc.drone_start_enu
    x, y, _ = walker.walker_pose(0.0, sc)
    assert sx == pytest.approx(x) and 3.0 < sy - y < 5.0


def test_angled_path_keeps_the_static_people_out_of_view():
    """The drone ~4 m ahead of the walker, looking at it, sees the static people (x = 4) only
    > 40 deg off its camera axis (half HFOV 34.4 deg)."""
    sc = walker.ANGLED
    for t in [i * 0.5 for i in range(int(sc.final_turn_s * 2))]:
        x, y, yaw = walker.walker_pose(t, sc)
        dx, dy = x + 4 * math.cos(yaw), y + 4 * math.sin(yaw)      # drone in front of the face
        axis = math.atan2(y - dy, x - dx)
        for py in (-2.0, 0.0, 2.0):
            off = math.degrees(math.atan2(py - dy, 4.0 - dx) - axis)
            assert abs((off + 180) % 360 - 180) > 40.0, (t, py)
