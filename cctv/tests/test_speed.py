import numpy as np
import pytest

from cctv.geometry import MPH_PER_MPS
from cctv.speed import (SpeedConfig, estimate_speed, is_speeding, ols, synthetic_track,
                        theil_sen)


@pytest.mark.parametrize("mph", [25, 55, 65, 85])
def test_noise_free_track_recovered_exactly(H_cam, mph):
    t, img = synthetic_track(H_cam, mph, start_xy=(25, 3), duration_s=2.0)
    est = estimate_speed(t, img, H_cam, scale_uncertainty=0.0)
    assert est.accepted, est.reject_reason
    assert est.speed_mph == pytest.approx(mph, abs=0.05)
    assert est.direction == 1


def test_noisy_track_within_tolerance_and_sigma_is_honest(H_cam):
    errs, sig = [], []
    for seed in range(200):
        t, img = synthetic_track(H_cam, 65, start_xy=(25, 3), duration_s=2.0, noise_px=1.0, seed=seed)
        est = estimate_speed(t, img, H_cam, scale_uncertainty=0.0,
                             config=SpeedConfig(timing_sigma_rel=0.0))
        assert est.accepted, est.reject_reason
        errs.append(est.speed_mph - 65)
        sig.append(est.sigma_mph)
    errs, sig = np.array(errs), np.array(sig)
    assert abs(errs.mean()) < 0.5                 # no meaningful bias
    assert np.abs(errs).max() < 6                 # every estimate within 6 mph
    coverage = np.mean(np.abs(errs) <= 2 * sig)   # ~95% expected for +-2 sigma
    assert coverage > 0.85


def test_outliers_are_rejected_by_robust_fit(H_cam):
    t, img = synthetic_track(H_cam, 60, start_xy=(25, 3), duration_s=2.5, noise_px=0.5,
                             outlier_frac=0.15, outlier_px=60, seed=3)
    est = estimate_speed(t, img, H_cam, scale_uncertainty=0.0)
    assert est.accepted, est.reject_reason
    assert est.speed_mph == pytest.approx(60, abs=2.0)
    assert est.n_inliers < est.n_points           # the jumps were dropped


def test_plain_least_squares_would_be_fooled_by_the_same_outliers(H_cam):
    from cctv.geometry import apply_homography
    t, img = synthetic_track(H_cam, 60, start_xy=(25, 3), duration_s=2.5, noise_px=0.5,
                             outlier_frac=0.15, outlier_px=60, seed=3)
    xy = apply_homography(H_cam, img)
    naive = ols(t, xy[:, 0])[0] * MPH_PER_MPS
    robust = theil_sen(t, xy[:, 0])[0] * MPH_PER_MPS
    assert abs(robust - 60) < abs(naive - 60)


def test_vehicle_moving_towards_minus_x(H_cam):
    t, img = synthetic_track(H_cam, 50, start_xy=(70, 7), heading_deg=180, duration_s=2.0)
    est = estimate_speed(t, img, H_cam)
    assert est.accepted
    assert est.direction == -1
    assert est.speed_mph == pytest.approx(50, abs=0.1)


def test_reject_too_few_points(H_cam):
    t, img = synthetic_track(H_cam, 60, start_xy=(25, 3), duration_s=0.3)  # 5 frames
    est = estimate_speed(t, img, H_cam)
    assert not est.accepted and "too few" in est.reject_reason


def test_reject_stationary_vehicle(H_cam):
    t, img = synthetic_track(H_cam, 0.5, start_xy=(30, 3), duration_s=3.0, noise_px=0.3)
    est = estimate_speed(t, img, H_cam)
    assert not est.accepted and "moved only" in est.reject_reason


def test_reject_motion_across_the_road(H_cam):
    t, img = synthetic_track(H_cam, 20, start_xy=(30, 0), heading_deg=80, duration_s=1.5)
    est = estimate_speed(t, img, H_cam)
    assert not est.accepted and "not moving along the road" in est.reject_reason


def test_reject_implausible_speed(H_cam):
    t, img = synthetic_track(H_cam, 200, start_xy=(20, 3), duration_s=1.0)
    est = estimate_speed(t, img, H_cam)
    assert not est.accepted and "implausible" in est.reject_reason


def test_reject_jumpy_track(H_cam):
    t, img = synthetic_track(H_cam, 60, start_xy=(25, 3), duration_s=2.0, outlier_frac=0.5,
                             outlier_px=80, seed=4)
    est = estimate_speed(t, img, H_cam)
    assert not est.accepted and "jumpy" in est.reject_reason


def test_timestamp_jitter_small_effect(H_cam):
    t, img = synthetic_track(H_cam, 65, start_xy=(25, 3), duration_s=2.0, timestamp_jitter_s=0.005,
                             seed=5)
    est = estimate_speed(t, img, H_cam)
    assert est.accepted
    assert est.speed_mph == pytest.approx(65, abs=1.5)


def test_sigma_includes_calibration_term(H_cam):
    t, img = synthetic_track(H_cam, 60, start_xy=(25, 3), duration_s=2.0)
    est = estimate_speed(t, img, H_cam, scale_uncertainty=0.05)
    assert est.sigma_cal_mph == pytest.approx(3.0, abs=0.05)
    assert est.sigma_mph >= est.sigma_cal_mph


def test_speeding_flag_uses_margin(H_cam):
    t, img = synthetic_track(H_cam, 70, start_xy=(25, 3), duration_s=2.0)
    est = estimate_speed(t, img, H_cam, scale_uncertainty=0.02)   # sigma ~1.4 mph -> margin 3
    assert is_speeding(est, 65)            # 70 - 3 > 65
    assert not is_speeding(est, 67.5)      # 70 - 3 < 67.5
    est.accepted = False
    assert not is_speeding(est, 50)        # rejected tracks are never flagged
