import math

import pytest

from jetson import geometry as g


def test_face_pixels_matches_hand_calculation():
    # 640 px wide, 62.2 deg HFOV, face 0.16 m at 4 m: 640*0.16 / (2*4*tan(31.1 deg)) = 21.2 px
    px = g.face_pixels(4.0, 62.2, 640)
    assert px == pytest.approx(640 * 0.16 / (2 * 4 * math.tan(math.radians(31.1))))
    assert px == pytest.approx(21.2, abs=0.1)


def test_max_range_is_inverse_of_face_pixels():
    for hfov in (22.2, 62.2, 120.0):
        for width in (320, 640, 1920):
            r = g.max_range_m(20, hfov, width)
            assert g.face_pixels(r, hfov, width) == pytest.approx(20.0)


def test_distance_from_width_roundtrip():
    hfov = 68.75
    for d in (1.0, 4.0, 12.5):
        w_norm = g.face_pixels(d, hfov, 640) / 640
        assert g.distance_from_width(w_norm, hfov) == pytest.approx(d)


def test_bearing_centre_edges_and_sign():
    assert g.bearing_deg(0.5, 60.0) == pytest.approx(0.0)
    assert g.bearing_deg(1.0, 60.0) == pytest.approx(30.0)
    assert g.bearing_deg(0.0, 60.0) == pytest.approx(-30.0)
    assert g.bearing_deg(0.75, 60.0) > 0


def test_fov_conversions():
    # C920: 78 deg diagonal, 16:9 -> ~70.4 deg horizontal
    assert g.hfov_from_dfov(78.0, 16 / 9) == pytest.approx(70.4, abs=0.1)
    # IMX477 (6.287 mm wide) + 6 mm lens -> ~55.3 deg
    assert g.hfov_from_focal(6.287, 6.0) == pytest.approx(55.3, abs=0.1)
    # vertical from horizontal for 4:3 should be smaller
    assert g.vfov_from_hfov(62.2, 4 / 3) == pytest.approx(48.8, abs=0.3)  # RPi v2 spec: 48.8 deg


def test_depression_angle():
    assert g.depression_deg(1.6, 5.0) == pytest.approx(0.0)
    assert g.depression_deg(6.6, 5.0) == pytest.approx(45.0)


@pytest.mark.parametrize("fn,args", [
    (g.face_pixels, (0.0, 60.0, 640)),
    (g.face_pixels, (5.0, 0.0, 640)),
    (g.face_pixels, (5.0, 180.0, 640)),
    (g.max_range_m, (0.0, 60.0, 640)),
    (g.distance_from_width, (0.0, 60.0)),
])
def test_invalid_inputs_raise(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_tables_render():
    t = g.range_table()
    assert t.count("\n") >= 6 and "RPi Camera v2" in t
    assert "px" in g.pixels_table()
    assert g.main([]) == 0
