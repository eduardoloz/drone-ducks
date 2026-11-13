"""The quack_people walker's scripted trajectory, evaluated in Python (ground-truth fallback).

Must match the <actor name="walker"> <trajectory> in sim/gazebo/worlds/quack_people.sdf
(tests/unit/test_walker.py checks that). Gazebo ENU: x = east, y = north; yaw 0 = facing +x.

Scenario ("leading" follow, like a selfie drone flying backwards in front of a person):
  0 - 62.6 s   walk north along an S-curve at 0.4 m/s, facing the drone ahead of them,
               with one look-back (face turned away) at 27.0 - 31.5 s
  62.6 - 78.9  turn round and walk back to the start at 1.4 m/s (face away: the follower
               loses the face, hovers, and lands after its lost timeout)
Loop period 78.9 s. The static people (x = 4, y = -2..2) are east of this lane and out of view.

Linear interpolation; Gazebo interpolates actor waypoints with a spline, so near the corners the
true pose can differ by a few tens of cm. Prefer the pose probe's "walker" pose when available.
"""
from __future__ import annotations

import math

# (time s, x, y, yaw rad)
WAYPOINTS = [
    (0.0, -5.0, -12.0, 1.1071),
    (11.2, -3.0, -8.0, 1.1071),
    (11.7, -3.0, -8.0, 2.1910),
    (27.0, -6.5, -3.0, 2.1910),
    (28.0, -6.5, -3.0, -0.9506),   # look back over the shoulder: face away from the drone
    (30.5, -6.5, -3.0, -0.9506),
    (31.5, -6.5, -3.0, 2.1910),
    (32.0, -6.5, -3.0, 1.0304),
    (46.6, -3.5, 2.0, 1.0304),
    (47.1, -3.5, 2.0, 1.8158),
    (62.6, -5.0, 8.0, 1.8158),
    (63.6, -5.0, 8.0, -1.5708),    # turn round and walk back to the start
    (77.9, -5.0, -12.0, -1.5708),
    (78.9, -5.0, -12.0, 1.1071),
]
PERIOD_S = WAYPOINTS[-1][0]
FACING_WINDOWS = [(0.0, 27.0), (31.5, 62.6)]  # walker's face points along its path towards the drone
FACE_HEIGHT_M = 1.6                           # approx. face height of the walking actor

# Suggested drone start for follow_person.py: ~4 m ahead of the walker's start, facing it.
DRONE_START_ENU = (-4.0, -8.0)


def _lerp_angle(a: float, b: float, f: float) -> float:
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * f


def walker_pose(sim_time_s: float) -> tuple[float, float, float]:
    """(x, y, yaw) of the walker at a Gazebo sim time (the script starts at t = 0 and loops)."""
    t = sim_time_s % PERIOD_S
    for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(WAYPOINTS, WAYPOINTS[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, _lerp_angle(a0, a1, f)
    return WAYPOINTS[-1][1:]


def phase(sim_time_s: float) -> float:
    return sim_time_s % PERIOD_S


def enu_to_ned(x: float, y: float) -> tuple[float, float]:
    """Gazebo ENU (x east, y north) -> PX4 local NED (north, east), assuming the drone spawned at the
    world origin (start_sim.sh does) so PX4's local origin is the world origin."""
    return y, x
