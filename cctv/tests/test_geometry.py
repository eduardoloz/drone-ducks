import numpy as np
import pytest

from cctv.geometry import (FT, Calibration, apply_homography, fit_homography, local_scale,
                           points_in_polygon, reprojection_errors, road_points_from_dashes)


def test_exact_fit_from_four_points(H_cam):
    road = np.array([[20, 0], [20, 11], [80, 0], [80, 11]], float)
    img = apply_homography(np.linalg.inv(H_cam), road)
    H = fit_homography(img, road)
    assert np.allclose(apply_homography(H, img), road, atol=1e-6)
    # same matrix up to scale
    assert np.allclose(H / H[2, 2], H_cam / H_cam[2, 2], rtol=1e-6, atol=1e-9)


def test_round_trip_image_road_image(H_cam):
    rng = np.random.default_rng(1)
    img = np.c_[rng.uniform(50, 590, 50), rng.uniform(200, 470, 50)]
    road = apply_homography(H_cam, img)
    back = apply_homography(np.linalg.inv(H_cam), road)
    assert np.allclose(back, img, atol=1e-6)


def test_noisy_fit_reprojection_error_is_small(H_cam):
    rng = np.random.default_rng(2)
    road = np.c_[rng.uniform(20, 70, 40), rng.uniform(0, 11, 40)]
    img = apply_homography(np.linalg.inv(H_cam), road) + rng.normal(0, 0.5, (40, 2))
    H = fit_homography(img, road)
    err = reprojection_errors(H, img, road)
    assert np.sqrt(np.mean(err ** 2)) < 0.3   # metres, with 0.5 px click noise


def test_collinear_points_rejected():
    img = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
    road = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    with pytest.raises(ValueError):
        fit_homography(img, road)


def test_needs_four_points():
    with pytest.raises(ValueError):
        fit_homography([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]])


def test_far_pixels_cover_more_road(H_cam):
    near = local_scale(H_cam, (320, 460))[1]
    far = local_scale(H_cam, (320, 150))[1]
    assert far > 5 * near


def test_dash_geometry_is_40ft_cycle():
    x0, y0 = road_points_from_dashes(0, 0, "start")
    x1, _ = road_points_from_dashes(0, 1, "start")
    xe, _ = road_points_from_dashes(0, 0, "end")
    _, y1 = road_points_from_dashes(1, 0, "start")
    assert (x0, y0) == (0.0, 0.0)
    assert x1 == pytest.approx(40 * FT)
    assert xe == pytest.approx(10 * FT)
    assert y1 == pytest.approx(12 * FT)


def test_calibration_save_load_round_trip(tmp_path, H_cam):
    road = np.array([[20, 0], [20, 11], [80, 0], [80, 11], [50, 5.5]], float)
    img = apply_homography(np.linalg.inv(H_cam), road)
    cal = Calibration.fit("TEST", (640, 480), img, road, roi_polygon=[[0, 200], [640, 200], [640, 480], [0, 480]])
    assert cal.rmse_m < 1e-6
    p = tmp_path / "cal.json"
    cal.save(p)
    cal2 = Calibration.load(p)
    assert cal2.image_size == (640, 480)
    assert np.allclose(cal2.image_to_road(img), road, atol=1e-6)
    assert np.allclose(cal2.road_to_image(road), img, atol=1e-6)
    assert cal2.in_roi([[320, 300], [320, 100]]).tolist() == [True, False]


def test_points_in_polygon():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    pts = np.array([[5, 5], [15, 5], [-1, -1], [9.9, 0.1]], float)
    assert points_in_polygon(pts, sq).tolist() == [True, False, False, True]


def test_jacobian_matches_finite_differences(H_cam):
    from cctv.geometry import homography_jacobian
    pts = np.array([[320, 300], [100, 450], [500, 220]], float)
    J = homography_jacobian(H_cam, pts)
    eps = 1e-3
    for k, (u, v) in enumerate(pts):
        du = (apply_homography(H_cam, [[u + eps, v]]) - apply_homography(H_cam, [[u - eps, v]]))[0] / (2 * eps)
        dv = (apply_homography(H_cam, [[u, v + eps]]) - apply_homography(H_cam, [[u, v - eps]]))[0] / (2 * eps)
        assert np.allclose(J[k][:, 0], du, rtol=1e-5)
        assert np.allclose(J[k][:, 1], dv, rtol=1e-5)
