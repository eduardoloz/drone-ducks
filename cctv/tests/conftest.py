"""Shared test fixtures: a physically plausible camera looking down a highway."""

from __future__ import annotations

import numpy as np
import pytest


def pinhole_ground_homography(height_m=12.0, pitch_deg=12.0, f_px=600.0, cx=320.0, cy=240.0,
                              cam_x=0.0, cam_y=5.5) -> np.ndarray:
    """H mapping image (u, v) -> road (X, Y) for a camera above the road looking along +X.

    Built from a pinhole model (K [r1 r2 t]) so perspective is realistic: points
    near the horizon cover many metres per pixel, points near the bottom few cm.
    """
    p = np.radians(pitch_deg)
    z = np.array([np.cos(p), 0.0, -np.sin(p)])   # optical axis
    x = np.array([0.0, -1.0, 0.0])                # image right
    y = np.cross(z, x)                            # image down
    R = np.vstack([x, y, z])
    C = np.array([cam_x, cam_y, height_m])
    t = -R @ C
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1.0]])
    H_road_to_img = K @ np.c_[R[:, 0], R[:, 1], t]
    H = np.linalg.inv(H_road_to_img)
    return H / H[2, 2]


@pytest.fixture
def H_cam() -> np.ndarray:
    return pinhole_ground_homography()
