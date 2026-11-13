"""The quack_people walker's scripted trajectory, evaluated in Python (ground-truth fallback).

Must match the <actor name="walker"> <trajectory> in sim/gazebo/worlds/quack_people.sdf
(tests/unit/test_walker.py checks that). Gazebo ENU: x = east, y = north; yaw 0 = facing +x.

Scenario ("leading" follow, like a selfie drone flying backwards in front of a person):
  0 - 60.7 s   walk north at 0.4 m/s, zig-zagging 14-17 deg off north, facing the drone ahead of
               them, with one look-back (face turned away) at 29.1 - 33.6 s
  60.7 - 77.7  turn round and walk back to the start at 1.4 m/s (face away: the follower
               loses the face, hovers, and lands after its lost timeout)
Loop period 77.7 s. Why only ~15 deg: the follower centres the face with yaw and keeps distance
by flying along its camera axis; it has no idea which way the person faces. When the person walks
at an angle to the line of sight they drift sideways past the drone and the facing angle grows
(a kinematic dry run with 35-39 deg legs lost the face after ~20 s for good). The static people (x = 4, y = -2..2) are east of this
lane and out of view.

Linear interpolation; Gazebo interpolates actor waypoints with a spline, so near the corners the
true pose can differ by a few tens of cm. Prefer the pose probe's "walker" pose when available.
"""
from __future__ import annotations

import math

# (time s, x, y, yaw rad)
WAYPOINTS = [
    (0.0, -5.0, -12.0, 1.3258),
    (15.5, -3.5, -6.0, 1.3258),
    (16.0, -3.5, -6.0, 1.8623),
    (29.1, -5.0, -1.0, 1.8623),
    (30.1, -5.0, -1.0, -1.2793),   # look back over the shoulder: face away from the drone
    (32.6, -5.0, -1.0, -1.2793),
    (33.6, -5.0, -1.0, 1.8623),
    (34.1, -5.0, -1.0, 1.2793),
    (47.1, -3.5, 4.0, 1.2793),
    (47.6, -3.5, 4.0, 1.8623),
    (60.7, -5.0, 9.0, 1.8623),
    (61.7, -5.0, 9.0, -1.5708),   # turn round and walk back to the start
    (76.7, -5.0, -12.0, -1.5708),
    (77.7, -5.0, -12.0, 1.3258),
]
PERIOD_S = WAYPOINTS[-1][0]
FACING_WINDOWS = [(0.0, 29.1), (33.6, 60.7)]  # walker's face points along its path towards the drone
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
