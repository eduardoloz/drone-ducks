"""Frame conventions and small pure-math helpers. No MAVSDK, no numpy.

Conventions used everywhere in quackdrone (the same as PX4 and MAVSDK):

* World frame: local NED in metres. north = +x, east = +y, **down = +z** (so 2 m up is d = -2).
  The origin is PX4's EKF origin, which is near (not exactly at) where the drone booted.
* Heading / yaw: degrees, 0 = north, +90 = east (clockwise seen from above), kept in (-180, 180].
* Body velocities (MAVSDK ``VelocityBodyYawspeed``, MAVLink ``MAV_FRAME_BODY_NED``):
  ``forward``, ``right``, ``down`` in m/s. PX4 rotates forward/right by the current **yaw only**
  (roll and pitch are ignored, see mavlink_receiver.cpp), and down is world-vertical.
  ``yawspeed`` is deg/s, positive = clockwise = turning right.
* Gazebo worlds are ENU: x = east, y = north, z = up, yaw 0 = facing east, counter-clockwise.
"""
from __future__ import annotations

import math


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def body_to_ned(forward: float, right: float, yaw_deg: float) -> tuple[float, float]:
    """Rotate a horizontal (forward, right) vector into (north, east) for a given heading."""
    y = math.radians(yaw_deg)
    north = forward * math.cos(y) - right * math.sin(y)
    east = forward * math.sin(y) + right * math.cos(y)
    return north, east


def ned_to_body(north: float, east: float, yaw_deg: float) -> tuple[float, float]:
    """Inverse of body_to_ned: (north, east) -> (forward, right)."""
    y = math.radians(yaw_deg)
    forward = north * math.cos(y) + east * math.sin(y)
    right = -north * math.sin(y) + east * math.cos(y)
    return forward, right


def enu_to_ned(x_east: float, y_north: float, z_up: float) -> tuple[float, float, float]:
    """Gazebo (ENU) position -> PX4 (NED) position."""
    return y_north, x_east, -z_up


def ned_to_enu(north: float, east: float, down: float) -> tuple[float, float, float]:
    """PX4 (NED) position -> Gazebo (ENU) position."""
    return east, north, -down


def enu_yaw_to_heading(yaw_rad: float) -> float:
    """Gazebo yaw (radians, 0 = east, CCW) -> PX4 heading (degrees, 0 = north, CW)."""
    return wrap_deg(90.0 - math.degrees(yaw_rad))


def heading_to_enu_yaw(heading_deg: float) -> float:
    """PX4 heading (degrees) -> Gazebo yaw (radians)."""
    return math.radians(wrap_deg(90.0 - heading_deg))


def bearing_deg(n0: float, e0: float, n1: float, e1: float) -> float:
    """Heading (deg) you would face to look from (n0, e0) towards (n1, e1)."""
    return wrap_deg(math.degrees(math.atan2(e1 - e0, n1 - n0)))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def limit_norm(x: float, y: float, max_norm: float) -> tuple[float, float]:
    """Scale (x, y) down so its length is at most max_norm (direction is kept)."""
    n = math.hypot(x, y)
    if n <= max_norm or n == 0.0:
        return x, y
    k = max_norm / n
    return x * k, y * k


def all_finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
