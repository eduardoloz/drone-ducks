"""Speed estimation: a track of image points + timestamps -> speed with uncertainty.

Pipeline for one tracked vehicle:

1. take the bbox bottom-centre (u, v) in every frame, with the frame's PTS time t
2. map (u, v) -> road (X, Y) metres with the camera's homography
3. robust straight-line fit of X(t) and Y(t) (Theil-Sen, then weighted least
   squares on inliers), so occlusions and ID-switch jumps don't drag the slope.
   Residuals are measured in *pixel-equivalent* units: a 1 px wobble far away
   is several metres on the road, so far points get less weight.
4. speed = |(dX/dt, dY/dt)|, with a 1-sigma uncertainty that combines
   - statistical error of the slope (inflated for frame-to-frame correlation)
   - calibration scale error (relative, from the calibration)
   - timing error (relative; tiny when PTS timestamps are used)
5. reject tracks that are too short, too jumpy, implausibly fast, or not moving
   along the road

``is_speeding`` flags a vehicle only if (estimate - margin) > posted limit, where
margin = max(k * sigma, min_margin). This is analytics, not enforcement.

Pure numpy; fully unit tested in tests/test_speed.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cctv.geometry import MPH_PER_MPS, apply_homography, homography_jacobian


@dataclass
class SpeedConfig:
    min_points: int = 8               # detections in the track (after ROI filtering)
    min_duration_s: float = 0.8
    min_distance_m: float = 8.0       # must travel at least this far
    min_pair_dt_s: float = 0.2        # Theil-Sen ignores pairs closer than this in time
    inlier_k: float = 3.0             # residual threshold = max(k * robust_sigma, floor), in px
    inlier_floor_px: float = 2.0
    min_inlier_frac: float = 0.7
    max_heading_dev_deg: float = 25.0  # travel direction vs road axis (X)
    max_speed_mph: float = 120.0      # above this we assume a tracking/calibration error
    max_sigma_stat_rel: float = 0.08  # reject if the fit alone is this uncertain (8%)
    timing_sigma_rel: float = 0.005   # relative timing uncertainty (PTS-based)
    max_points: int = 300             # subsample very long tracks for Theil-Sen


@dataclass
class SpeedEstimate:
    track_id: int
    cls: str | None
    n_points: int
    n_inliers: int
    t_start: float
    t_end: float
    duration_s: float
    distance_m: float
    speed_mps: float
    speed_mph: float
    sigma_mph: float
    sigma_stat_mph: float
    sigma_cal_mph: float
    sigma_time_mph: float
    heading_deg: float     # 0 = +X (along road), 180 = -X
    direction: int         # +1 travelling towards +X, -1 towards -X
    accepted: bool
    reject_reason: str | None = None

    def margin_mph(self, k: float = 2.0, min_margin_mph: float = 3.0) -> float:
        return max(k * self.sigma_mph, min_margin_mph)

    def to_dict(self) -> dict:
        return asdict(self)


def is_speeding(est: SpeedEstimate, limit_mph: float, k: float = 2.0,
                min_margin_mph: float = 3.0) -> bool:
    """Conservative flag: only if even the low end of the estimate exceeds the limit."""
    return bool(est.accepted and est.speed_mph - est.margin_mph(k, min_margin_mph) > limit_mph)


# ------------------------------------------------------------------ robust fits
def theil_sen(t: np.ndarray, x: np.ndarray, min_dt: float = 0.0) -> tuple[float, float]:
    """Median of pairwise slopes (robust to ~29% outliers). Returns (slope, intercept)."""
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    keep = dt > max(min_dt, 1e-9)
    if not np.any(keep):
        keep = dt > 1e-9
    slopes = (x[j] - x[i])[keep] / dt[keep]
    slope = float(np.median(slopes))
    return slope, float(np.median(x - slope * t))


def ols(t: np.ndarray, x: np.ndarray, w: np.ndarray | None = None
        ) -> tuple[float, float, float, np.ndarray]:
    """(Weighted) least squares line. Returns (slope, intercept, slope_std_error, residuals).

    ``w`` are inverse-variance weights (default: all equal). The standard error
    is inflated for lag-1 autocorrelation of the normalised residuals: tracker
    boxes jitter smoothly, so neighbouring frames are not independent.
    """
    n = len(t)
    w = np.ones(n) if w is None else np.asarray(w, float)
    sw = w.sum()
    tm = float((w * t).sum() / sw)
    xm = float((w * x).sum() / sw)
    sxx = float((w * (t - tm) ** 2).sum())
    slope = float((w * (t - tm) * (x - xm)).sum() / sxx)
    icpt = xm - slope * tm
    r = x - (icpt + slope * t)
    if n <= 2:
        return slope, icpt, float("inf"), r
    z = r * np.sqrt(w)                      # residuals in "unit variance" units
    s2 = float((z ** 2).sum() / (n - 2))
    se = np.sqrt(s2 / sxx)
    if n > 4 and z.std() > 0:
        rho = float(np.clip(np.corrcoef(z[:-1], z[1:])[0, 1], 0.0, 0.95))
        se *= np.sqrt((1 + rho) / (1 - rho))
    return slope, icpt, float(se), r


def _robust_sigma(r: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


# ------------------------------------------------------------------- estimator
def estimate_speed(t, img_pts, H, scale_uncertainty: float = 0.03,
                   config: SpeedConfig | None = None, track_id: int = -1,
                   cls: str | None = None) -> SpeedEstimate:
    """Estimate one track's speed. ``t`` seconds, ``img_pts`` (N, 2) bbox bottom-centres."""
    cfg = config or SpeedConfig()
    t = np.asarray(t, dtype=float)
    uv = np.asarray(img_pts, dtype=float).reshape(-1, 2)
    order = np.argsort(t)
    t, uv = t[order], uv[order]
    _, uniq = np.unique(t, return_index=True)
    t, uv = t[uniq], uv[uniq]

    nan = float("nan")

    def reject(reason: str, **kw) -> SpeedEstimate:
        base = dict(track_id=track_id, cls=cls, n_points=len(t), n_inliers=kw.get("n_inliers", 0),
                    t_start=float(t[0]) if len(t) else nan, t_end=float(t[-1]) if len(t) else nan,
                    duration_s=float(t[-1] - t[0]) if len(t) else 0.0,
                    distance_m=kw.get("distance_m", nan), speed_mps=kw.get("speed_mps", nan),
                    speed_mph=kw.get("speed_mph", nan), sigma_mph=kw.get("sigma_mph", nan),
                    sigma_stat_mph=nan, sigma_cal_mph=nan, sigma_time_mph=nan,
                    heading_deg=kw.get("heading_deg", nan), direction=kw.get("direction", 0),
                    accepted=False, reject_reason=reason)
        return SpeedEstimate(**base)

    if len(t) < cfg.min_points:
        return reject(f"too few points ({len(t)} < {cfg.min_points})")
    if t[-1] - t[0] < cfg.min_duration_s:
        return reject(f"too short ({t[-1] - t[0]:.2f}s < {cfg.min_duration_s}s)")

    if len(t) > cfg.max_points:  # keep evenly spaced subset for the O(n^2) step
        idx = np.linspace(0, len(t) - 1, cfg.max_points).round().astype(int)
        t, uv = t[idx], uv[idx]

    try:
        xy = apply_homography(H, uv)
    except ValueError:
        return reject("point above horizon (bad calibration or ROI)")
    tt = t - t[0]

    # Metres-per-pixel at each point (perspective): rows of the Jacobian of H.
    J = homography_jacobian(H, uv)                  # (N, 2, 2)
    mx = np.hypot(J[:, 0, 0], J[:, 0, 1])           # metres of X per px of image noise
    my = np.hypot(J[:, 1, 0], J[:, 1, 1])

    # 1) robust initial fit (Theil-Sen), 2) inliers judged in pixel units,
    # 3) weighted LS on inliers, repeated a few times so the noise scale is
    #    estimated from inliers only (MAD alone breaks down near 50% outliers).
    wx_all, wy_all = 1.0 / mx ** 2, 1.0 / my ** 2
    sx, ix = theil_sen(tt, xy[:, 0], cfg.min_pair_dt_s)
    sy, iy = theil_sen(tt, xy[:, 1], cfg.min_pair_dt_s)
    inl = np.ones(len(tt), dtype=bool)
    for _ in range(4):
        zx = (xy[:, 0] - (ix + sx * tt)) / mx
        zy = (xy[:, 1] - (iy + sy * tt)) / my
        thr_x = max(cfg.inlier_k * _robust_sigma(zx[inl]), cfg.inlier_floor_px)
        thr_y = max(cfg.inlier_k * _robust_sigma(zy[inl]), cfg.inlier_floor_px)
        new_inl = (np.abs(zx - np.median(zx[inl])) <= thr_x) & (np.abs(zy - np.median(zy[inl])) <= thr_y)
        if new_inl.sum() < 3:
            break
        converged = np.array_equal(new_inl, inl)
        inl = new_inl
        sx, ix, _, _ = ols(tt[inl], xy[inl, 0], wx_all[inl])
        sy, iy, _, _ = ols(tt[inl], xy[inl, 1], wy_all[inl])
        if converged:
            break
    n_in = int(inl.sum())
    if n_in < cfg.min_points or n_in / len(tt) < cfg.min_inlier_frac:
        return reject(f"jumpy track: {n_in}/{len(tt)} inliers", n_inliers=n_in)
    ti = tt[inl]
    if ti[-1] - ti[0] < cfg.min_duration_s:
        return reject("inliers span too short a time", n_inliers=n_in)
    vx, _, se_x, _ = ols(ti, xy[inl, 0], wx_all[inl])
    vy, _, se_y, _ = ols(ti, xy[inl, 1], wy_all[inl])
    v = float(np.hypot(vx, vy))
    heading = float(np.degrees(np.arctan2(vy, vx)))
    direction = 1 if vx >= 0 else -1
    dist = v * float(ti[-1] - ti[0])
    common = dict(n_inliers=n_in, distance_m=dist, speed_mps=v, speed_mph=v * MPH_PER_MPS,
                  heading_deg=heading, direction=direction)

    if dist < cfg.min_distance_m:
        return reject(f"moved only {dist:.1f} m", **common)
    dev = min(abs(heading), 180 - abs(heading))
    if dev > cfg.max_heading_dev_deg:
        return reject(f"not moving along the road ({dev:.0f} deg off axis)", **common)
    if v * MPH_PER_MPS > cfg.max_speed_mph:
        return reject(f"implausible speed {v * MPH_PER_MPS:.0f} mph", **common)

    sigma_stat = float(np.sqrt((vx * se_x) ** 2 + (vy * se_y) ** 2) / v) if v > 0 else float("inf")
    if sigma_stat > cfg.max_sigma_stat_rel * v:
        return reject(f"too noisy (fit sigma {sigma_stat / v:.0%} of speed)", **common)
    sigma_cal = v * scale_uncertainty
    sigma_time = v * cfg.timing_sigma_rel
    sigma = float(np.sqrt(sigma_stat ** 2 + sigma_cal ** 2 + sigma_time ** 2))
    return SpeedEstimate(
        track_id=track_id, cls=cls, n_points=len(tt), n_inliers=n_in,
        t_start=float(t[0]), t_end=float(t[-1]), duration_s=float(t[-1] - t[0]),
        distance_m=dist, speed_mps=v, speed_mph=v * MPH_PER_MPS, sigma_mph=sigma * MPH_PER_MPS,
        sigma_stat_mph=sigma_stat * MPH_PER_MPS, sigma_cal_mph=sigma_cal * MPH_PER_MPS,
        sigma_time_mph=sigma_time * MPH_PER_MPS, heading_deg=heading, direction=direction,
        accepted=True, reject_reason=None)


# ------------------------------------------------------------ synthetic tracks
def synthetic_track(H: np.ndarray, speed_mph: float, start_xy=(0.0, 1.8), duration_s: float = 2.0,
                    fps: float = 15.0, heading_deg: float = 0.0, noise_px: float = 0.0,
                    timestamp_jitter_s: float = 0.0, outlier_frac: float = 0.0,
                    outlier_px: float = 40.0, seed: int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """Make a constant-speed track in *road* space and project it into the image.

    Returns ``(t, img_pts)`` like a real track (t from frame timestamps, which
    get ``timestamp_jitter_s`` of error; image points get ``noise_px`` of
    Gaussian noise and an ``outlier_frac`` of big jumps, e.g. occlusions).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps)) + 1
    t_true = np.arange(n) / fps
    v = speed_mph / MPH_PER_MPS
    h = np.radians(heading_deg)
    road = np.c_[start_xy[0] + v * np.cos(h) * t_true, start_xy[1] + v * np.sin(h) * t_true]
    img = apply_homography(np.linalg.inv(H), road)
    img = img + rng.normal(0, noise_px, img.shape) if noise_px else img
    if outlier_frac > 0:
        k = int(round(outlier_frac * n))
        idx = rng.choice(n, size=k, replace=False)
        img[idx] += rng.normal(0, outlier_px, (k, 2))
    t_obs = t_true + (rng.normal(0, timestamp_jitter_s, n) if timestamp_jitter_s else 0.0)
    return t_obs, img
