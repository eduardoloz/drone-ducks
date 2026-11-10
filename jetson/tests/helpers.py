"""Small helpers shared by several test modules."""

import math


def obs_for(bearing_deg, distance_m, cfg, stamp, cy=None):
    """Build a TargetObservation that the controller will read as (bearing, distance)."""
    from jetson.controller import TargetObservation

    t = math.tan(math.radians(cfg.hfov_deg) / 2)
    cx = 0.5 + math.tan(math.radians(bearing_deg)) / (2 * t)
    w = cfg.face_width_m / (2 * distance_m * t)
    return TargetObservation(cx=cx, cy=0.5 if cy is None else cy, w=w, h=w * cfg.aspect, stamp=stamp)
