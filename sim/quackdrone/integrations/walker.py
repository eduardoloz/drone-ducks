"""The walker actors' scripted trajectories, evaluated in Python (ground-truth fallback).

Each script must match the <actor name="walker"> <trajectory> in the world file of the same name
(tests/unit/test_walker.py checks that). Gazebo ENU: x = east, y = north; yaw 0 = facing +x.
The pose probe does not get actor poses from Gazebo, so follow_metrics uses these scripts at the
probe's sim time as ground truth for the walker.

quack_people ("lead": like a selfie drone flying backwards in front of a person):
  0 - 62.0 s   walk straight north along x = -5 at 0.4 m/s, facing the drone ahead of them,
               with one look-back (face turned away) at 27.5 - 32.0 s
  62.0 - 80.4  turn round and walk back to the start at 1.4 m/s (face away: the follower
               loses the face, hovers, and lands after its lost timeout)

quack_people_turnaway (body-lock test: one turned-away stretch LONGER than the 5 s lost timeout):
  0 - 25.0 s   walk north at 0.4 m/s facing the drone (face lock acquired)
  25.0 - 39.0  turn round, walk 12 s AWAY from the drone (south) at 0.4 m/s, turn back:
               ~14 s with no face. Face-only following times out and lands here; face+body
               keeps the body lock (jetson LockKeeper max_body_only_s = 15 s) and re-acquires
  39.0 - 69.0  walk north again facing the drone
  69.0 - 83.3  turn round and walk back to the start at 1.4 m/s (end of the follow either way)

The static people (x = 4, y = -2..2) are east of the lane and out of view.

Why straight lines: the Jetson follower centres the face (yaw) and keeps distance along its camera
axis, but cannot observe which way the person faces. Any position on the line of sight centres
the face, so nothing pulls the drone back in front of a person who walks at an angle to it: the
facing angle random-walks and grows. Kinematic dry runs (FakeBackend + ideal detector) with legs
14-39 deg off the line of sight lost the face for good after 20-50 s, even with lateral gain.

Linear interpolation; Gazebo interpolates actor waypoints with a spline, so near the corners the
true pose can differ by a few tens of cm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

N, S = 1.5708, -1.5708   # ENU yaw: facing north (+y) / south (-y)


@dataclass(frozen=True)
class Script:
    name: str                                         # = world file name (sim/gazebo/worlds/<name>.sdf)
    waypoints: tuple                                  # (time s, x, y, yaw rad)
    facing_windows: tuple                             # (t0, t1): face points along the path at the drone
    final_turn_s: float                               # walker turns away for good; follow is over after this
    drone_start_enu: tuple = (-5.0, -8.0)             # ~4 m ahead of the walker's start, facing it

    @property
    def period_s(self) -> float:
        return self.waypoints[-1][0]


LEAD = Script(
    name="quack_people",
    waypoints=(
        (0.0, -5.0, -12.0, N),
        (27.5, -5.0, -1.0, N),
        (28.5, -5.0, -1.0, S),     # look back over the shoulder: face away from the drone
        (31.0, -5.0, -1.0, S),
        (32.0, -5.0, -1.0, N),
        (62.0, -5.0, 11.0, N),
        (63.0, -5.0, 11.0, S),     # turn round and walk back to the start
        (79.4, -5.0, -12.0, S),
        (80.4, -5.0, -12.0, N),
    ),
    facing_windows=((0.0, 27.5), (32.0, 62.0)),
    final_turn_s=62.0,
)

TURNAWAY = Script(
    name="quack_people_turnaway",
    waypoints=(
        (0.0, -5.0, -12.0, N),
        (25.0, -5.0, -2.0, N),
        (26.0, -5.0, -2.0, S),     # turn round ...
        (38.0, -5.0, -6.8, S),     # ... and walk away from the drone for 12 s (body only, no face)
        (39.0, -5.0, -6.8, N),     # face the drone again
        (69.0, -5.0, 5.2, N),
        (70.0, -5.0, 5.2, S),      # turn round and walk back to the start
        (82.3, -5.0, -12.0, S),
        (83.3, -5.0, -12.0, N),
    ),
    facing_windows=((0.0, 25.0), (39.0, 69.0)),
    final_turn_s=69.0,
)

SCRIPTS = {s.name: s for s in (LEAD, TURNAWAY)}
ALIASES = {"lead": LEAD.name, "turnaway": TURNAWAY.name}


def get(name: str | None) -> Script:
    """Script by world name or alias ('lead', 'turnaway'); None -> LEAD."""
    if not name:
        return LEAD
    return SCRIPTS[ALIASES.get(name, name)]


# Back-compat names (= the lead walker in quack_people)
WAYPOINTS = list(LEAD.waypoints)
PERIOD_S = LEAD.period_s
FACING_WINDOWS = list(LEAD.facing_windows)
DRONE_START_ENU = LEAD.drone_start_enu
FACE_HEIGHT_M = 1.6                           # approx. face height of the walking actor


def _lerp_angle(a: float, b: float, f: float) -> float:
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * f


def walker_pose(sim_time_s: float, script: Script = LEAD) -> tuple[float, float, float]:
    """(x, y, yaw) of the walker at a Gazebo sim time (the script starts at t = 0 and loops)."""
    wps = script.waypoints
    t = sim_time_s % script.period_s
    for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(wps, wps[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, _lerp_angle(a0, a1, f)
    return wps[-1][1:]


def phase(sim_time_s: float, script: Script = LEAD) -> float:
    return sim_time_s % script.period_s


def enu_to_ned(x: float, y: float) -> tuple[float, float]:
    """Gazebo ENU (x east, y north) -> PX4 local NED (north, east), assuming the drone spawned at the
    world origin (start_sim.sh does) so PX4's local origin is the world origin."""
    return y, x
