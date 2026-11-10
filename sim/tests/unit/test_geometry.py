import math

import pytest

from quackdrone.geometry import (bearing_deg, body_to_ned, enu_to_ned, enu_yaw_to_heading, heading_to_enu_yaw,
                                 limit_norm, ned_to_body, ned_to_enu, wrap_deg)


@pytest.mark.parametrize("a,expected", [(0, 0), (190, -170), (-190, 170), (180, 180), (-180, 180),
                                        (360, 0), (540, 180), (-721, -1)])
def test_wrap_deg(a, expected):
    assert wrap_deg(a) == pytest.approx(expected)


@pytest.mark.parametrize("yaw,fwd,right,exp", [
    (0, 1, 0, (1, 0)),      # facing north: forward = north
    (0, 0, 1, (0, 1)),      # facing north: right = east
    (90, 1, 0, (0, 1)),     # facing east: forward = east
    (90, 0, 1, (-1, 0)),    # facing east: right = south
    (180, 1, 0, (-1, 0)),
    (-90, 1, 0, (0, -1)),   # facing west
])
def test_body_to_ned(yaw, fwd, right, exp):
    n, e = body_to_ned(fwd, right, yaw)
    assert (n, e) == pytest.approx(exp, abs=1e-9)


def test_body_ned_roundtrip():
    for yaw in range(-180, 181, 15):
        f, r = ned_to_body(*body_to_ned(1.3, -0.4, yaw), yaw)
        assert (f, r) == pytest.approx((1.3, -0.4), abs=1e-9)


def test_matches_px4_body_ned_rotation():
    # PX4 mavlink_receiver.cpp, MAV_FRAME_BODY_NED: vx = cos(yaw)*bx - sin(yaw)*by ; vy = sin(yaw)*bx + cos(yaw)*by
    yaw = 37.0
    bx, by = 0.8, 0.3
    y = math.radians(yaw)
    assert body_to_ned(bx, by, yaw) == pytest.approx((math.cos(y) * bx - math.sin(y) * by,
                                                      math.sin(y) * bx + math.cos(y) * by))


def test_enu_ned():
    assert enu_to_ned(1, 2, 3) == (2, 1, -3)
    assert ned_to_enu(*enu_to_ned(1, 2, 3)) == (1, 2, 3)
    # quack_demo chair at Gazebo (-1, 6) -> PX4 N 6, E -1
    assert enu_to_ned(-1, 6, 0)[:2] == (6, -1)


def test_gazebo_yaw_to_heading():
    assert enu_yaw_to_heading(0.0) == pytest.approx(90.0)          # Gazebo +x (east)
    assert enu_yaw_to_heading(math.pi / 2) == pytest.approx(0.0)    # Gazebo +y (north)
    assert heading_to_enu_yaw(enu_yaw_to_heading(0.3)) == pytest.approx(0.3)


def test_bearing_and_limit_norm():
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0)
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(90)
    assert bearing_deg(0, 0, -1, -1) == pytest.approx(-135)
    assert limit_norm(3, 4, 10) == (3, 4)
    assert limit_norm(3, 4, 1) == pytest.approx((0.6, 0.8))
