"""Detect that a (PTZ) camera moved since it was calibrated.

NYSDOT LIE cameras are mostly pan-tilt-zoom domes that operators re-aim at
incidents. A homography calibrated for one view is wrong for any other view, so
before trusting speeds we compare the current frame with the calibration's
reference frame.

Method: ORB keypoints on both frames (with on-screen-display text masked out,
because the clock/caption overlay never moves even when the camera does), match,
fit a similarity transform with RANSAC, and look at the shift / zoom / rotation
and at how many matches survive. A camera pointed somewhere else entirely gives
few consistent matches; a slightly nudged one gives a clear shift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Default overlay boxes as fractions of (W, H): caption top-left, clock top-right.
DEFAULT_OSD_BOXES = [(0.0, 0.0, 0.30, 0.30), (0.62, 0.0, 1.0, 0.16)]


@dataclass
class MotionResult:
    moved: bool
    reason: str
    shift_px: float          # median keypoint displacement (pixels, full-res)
    scale: float             # zoom factor (1.0 = unchanged)
    rotation_deg: float
    inlier_ratio: float      # RANSAC inliers / good matches
    n_matches: int
    transform: list[list[float]] | None = None  # 2x3 similarity, reference pixel -> current pixel

    def to_dict(self) -> dict:
        return asdict(self)


def osd_mask(shape: tuple[int, int], boxes=DEFAULT_OSD_BOXES) -> np.ndarray:
    """255 where features are allowed, 0 on overlay boxes (fractions of W/H)."""
    h, w = shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[int(y1 * h): int(y2 * h), int(x1 * w): int(x2 * w)] = 0
    return mask


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class CameraMotionDetector:
    """Compare frames against a reference view.

    Thresholds are for ~640x480 video: a 4 px shift at the bottom of the frame is
    already ~0.3-1 m on the road, i.e. a few percent of speed.
    """

    def __init__(self, reference: np.ndarray, osd_boxes=DEFAULT_OSD_BOXES,
                 max_shift_px: float = 4.0, max_scale_change: float = 0.02,
                 max_rotation_deg: float = 1.0, min_inlier_ratio: float = 0.25,
                 min_matches: int = 12, n_features: int = 1500):
        self.ref = _gray(reference)
        self.mask = osd_mask(self.ref.shape, osd_boxes)
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref, self.mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.max_shift_px = max_shift_px
        self.max_scale_change = max_scale_change
        self.max_rotation_deg = max_rotation_deg
        self.min_inlier_ratio = min_inlier_ratio
        self.min_matches = min_matches

    def check(self, frame: np.ndarray) -> MotionResult:
        cur = _gray(frame)
        if cur.shape != self.ref.shape:
            return MotionResult(True, "resolution changed", float("nan"), float("nan"),
                                float("nan"), 0.0, 0)
        if self.des_ref is None or len(self.kp_ref) < self.min_matches:
            return MotionResult(True, "reference frame has too few features", float("nan"),
                                float("nan"), float("nan"), 0.0, 0)
        kp, des = self.orb.detectAndCompute(cur, self.mask)
        if des is None or len(kp) < self.min_matches:
            return MotionResult(True, "current frame has too few features (dark/blank?)",
                                float("nan"), float("nan"), float("nan"), 0.0, 0)
        # Lowe ratio test on 2-NN matches.
        good = []
        for pair in self.matcher.knnMatch(self.des_ref, des, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < self.min_matches:
            return MotionResult(True, f"only {len(good)} feature matches: view changed",
                                float("nan"), float("nan"), float("nan"), 0.0, len(good))
        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is None:
            return MotionResult(True, "no consistent transform", float("nan"), float("nan"),
                                float("nan"), 0.0, len(good))
        inl = inl.ravel().astype(bool)
        ratio = float(inl.mean())
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        shift = float(np.median(np.linalg.norm(dst[inl] - src[inl], axis=1))) if inl.any() else float("nan")
        reasons = []
        if ratio < self.min_inlier_ratio:
            reasons.append(f"inlier ratio {ratio:.2f} < {self.min_inlier_ratio}")
        if shift > self.max_shift_px:
            reasons.append(f"shift {shift:.1f}px > {self.max_shift_px}px")
        if abs(scale - 1) > self.max_scale_change:
            reasons.append(f"zoom x{scale:.3f}")
        if abs(rot) > self.max_rotation_deg:
            reasons.append(f"rotation {rot:.2f} deg")
        return MotionResult(bool(reasons), "; ".join(reasons) or "static", shift, scale, rot,
                            ratio, len(good), [[round(float(x), 6) for x in row] for row in M])
