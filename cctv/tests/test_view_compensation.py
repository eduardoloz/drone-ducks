"""Small PTZ offsets: compensate the calibration instead of rejecting the clip (views.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cctv.geometry import MPH_PER_MPS, Calibration, apply_homography
from cctv.speed import estimate_speed
from cctv.tests.conftest import pinhole_ground_homography
from cctv.tests.test_camera_motion import scene, shift
from cctv.views import match_view


def _pinhole_cal(view_id="v1") -> Calibration:
    """A calibration whose H is the realistic pinhole camera from conftest, with an ROI."""
    H = pinhole_ground_homography()
    Hi = np.linalg.inv(H)
    road = np.array([[x, y] for x in (15, 25, 35, 45) for y in (0.0, 3.66, 7.32)])
    img = apply_homography(Hi, road)
    cal = Calibration.fit("CAM", (640, 480), img, road, view_id=view_id)
    cal.roi_polygon = apply_homography(Hi, [[12, -1], [50, -1], [50, 9], [12, 9]]).tolist()
    return cal


def _track_uv(H, mph=60.0, lane_y=1.8, n=30, fps=15.0, x0=16.0):
    t = np.arange(n) / fps
    x = x0 + mph / MPH_PER_MPS * t
    return t, apply_homography(np.linalg.inv(H), np.c_[x, np.full(n, lane_y)])


def _apply(M, uv):
    M3 = np.vstack([np.asarray(M, float), [0, 0, 1]])
    return (np.c_[uv, np.ones(len(uv))] @ M3.T)[:, :2]


def test_with_image_transform_maps_moved_pixels_to_the_same_road_points():
    cal = _pinhole_cal()
    M = [[1.0, 0.0, 12.0], [0.0, 1.0, -5.0]]
    cal2 = cal.with_image_transform(M)
    ref_px = np.asarray(cal.img_pts)
    np.testing.assert_allclose(cal2.image_to_road(_apply(M, ref_px)), cal.image_to_road(ref_px), atol=1e-6)
    np.testing.assert_allclose(cal2.roi_polygon, _apply(M, np.asarray(cal.roi_polygon)), atol=1e-9)
    assert "compensated" in cal2.notes and cal.notes == ""        # original untouched


def test_speed_through_compensated_calibration_is_unbiased():
    cal = _pinhole_cal()
    M = cv2.getRotationMatrix2D((320, 240), 0.8, 1.015)           # 0.8 deg roll, 1.5% zoom ...
    M[:, 2] += (9.0, -6.0)                                         # ... and a 9/-6 px pan/tilt
    t, uv_ref = _track_uv(cal.H_np)
    uv_cur = _apply(M, uv_ref)                                     # what the nudged camera sees
    good = estimate_speed(t, uv_cur, cal.with_image_transform(M).H_np)
    stale = estimate_speed(t, uv_cur, cal.H_np)
    assert good.accepted and good.speed_mph == pytest.approx(60.0, abs=0.05)
    assert abs(stale.speed_mph - 60.0) > 1.0                       # what rejecting protects against


def _view_dir(tmp_path, ref_img, tag=None):
    d = tmp_path / "CAM"
    d.mkdir(parents=True, exist_ok=True)
    _pinhole_cal().save(d / "v1.json")
    cv2.imwrite(str(d / ("v1.ref.jpg" if not tag else f"v1.ref-{tag}.jpg")), ref_img)
    return tmp_path


def test_match_compensates_a_small_nudge(tmp_path):
    cal_dir = _view_dir(tmp_path, scene(0))
    m = match_view("CAM", shift(scene(0), 15, -4), cal_dir=cal_dir)
    assert m.status == "matched" and m.compensated and m.usable
    assert "compensating" in m.reason and m.to_dict()["compensated"] is True
    M = np.asarray(m.transform)
    np.testing.assert_allclose(M[:, :2], np.eye(2), atol=0.01)
    np.testing.assert_allclose(M[:, 2], [15, -4], atol=0.6)
    # the calibration handed out is the moved one: a ref pixel shifted by (15, -4) maps to the same road point
    p = np.asarray(_pinhole_cal().img_pts)
    np.testing.assert_allclose(m.calibration.image_to_road(p + [15, -4]), _pinhole_cal().image_to_road(p),
                               atol=0.35)


def test_compensation_can_be_switched_off(tmp_path):
    cal_dir = _view_dir(tmp_path, scene(0))
    m = match_view("CAM", shift(scene(0), 15, 0), cal_dir=cal_dir, compensate=False)
    assert m.status == "no_match" and "shift" in m.reason


def test_large_moves_are_still_rejected(tmp_path):
    cal_dir = _view_dir(tmp_path, scene(0))
    assert match_view("CAM", shift(scene(0), 60, 0), cal_dir=cal_dir).status == "no_match"
    zoom = cv2.warpAffine(scene(0), cv2.getRotationMatrix2D((320, 240), 0, 1.08), (640, 480),
                          borderMode=cv2.BORDER_REFLECT)
    m = match_view("CAM", zoom, cal_dir=cal_dir)
    assert m.status == "no_match" and not m.compensated


def test_extra_references_are_never_used_for_compensation(tmp_path):
    # Only a night reference: it need not be aligned with the frame the calibration was clicked on.
    cal_dir = _view_dir(tmp_path, scene(0), tag="night")
    assert match_view("CAM", shift(scene(0), 15, 0), cal_dir=cal_dir).status == "no_match"
    assert match_view("CAM", shift(scene(0), 1, 0), cal_dir=cal_dir).status == "matched"   # plain match still ok


def test_uncalibrated_view_is_not_compensated(tmp_path):
    d = tmp_path / "CAM"
    d.mkdir()
    cv2.imwrite(str(d / "v2.ref.jpg"), scene(0))
    m = match_view("CAM", shift(scene(0), 15, 0), cal_dir=tmp_path)
    assert m.status == "no_match" and not m.compensated
