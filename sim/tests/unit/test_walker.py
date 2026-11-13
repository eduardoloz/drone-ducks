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
