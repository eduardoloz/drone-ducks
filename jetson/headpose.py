"""Which way is the face pointing? A yaw proxy from the 5 face landmarks (YuNet / SCRFD).

Not a head-pose model, and not identity: one number per detected face from where the nose tip sits
between the eyes (and the mouth corners), in image pixels.

    eyes sorted by image x: eL (image-left), eR (image-right); nose tip n
    r_eyes  = ((n.x - eL.x) - (eR.x - n.x)) / (eR.x - eL.x)        0 frontal, + nose toward image-right
    r_mouth = same with the mouth corners (if present and sane); r = mean of the two

Simple head model: eyes at +-a, nose tip d in front of the eye plane. Turning the head by theta
puts the eyes at +-a cos(theta) and the nose at d sin(theta), so r = (d / a) tan(theta) and
theta = atan(r / k) with k = d / a ~ 1 for adult faces (nose tip ~30 mm in front of the eyes,
eyes ~63 mm apart). k is a calibration constant, not a measurement; the sign is what control needs.

Sign: + = the face points toward the camera's RIGHT (the person is looking at a spot to the right
of the drone), so to get back in front of them the drone should move right (right_m_s > 0).

Sorting the points by x makes this independent of the detector's landmark order and of mirroring.
Small faces have pixel-noisy landmarks: below `min_face_px` we return None (unknown).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]

NOSE_DEPTH_RATIO = 1.0   # k = nose depth / half eye distance
MIN_FACE_PX = 20.0       # narrower faces: landmarks too noisy for a yaw estimate


def _ratio(left: Point, right: Point, nose: Point) -> Optional[float]:
    span = right[0] - left[0]
    if span <= 1e-6:
        return None
    return ((nose[0] - left[0]) - (right[0] - nose[0])) / span


def yaw_proxy(landmarks: Optional[Sequence[Point]]) -> Optional[float]:
    """Dimensionless r (see module doc), clipped to [-3, 3]; None if landmarks are unusable."""
    if landmarks is None or len(landmarks) < 3:
        return None
    pts = [(float(x), float(y)) for x, y in landmarks]
    if not all(math.isfinite(v) for p in pts for v in p):
        return None
    eyes = sorted(pts[:2])
    nose = pts[2]
    rs = []
    r = _ratio(eyes[0], eyes[1], nose)
    if r is not None:
        rs.append(r)
    if len(pts) >= 5:
        mouth = sorted(pts[3:5])
        # mouth corners must be below the eyes (image y grows down) to be trusted
        if min(mouth[0][1], mouth[1][1]) > max(eyes[0][1], eyes[1][1]):
            rm = _ratio(mouth[0], mouth[1], nose)
            if rm is not None:
                rs.append(rm)
    if not rs:
        return None
    return max(-3.0, min(3.0, sum(rs) / len(rs)))


def head_yaw_deg(landmarks: Optional[Sequence[Point]], face_width_px: Optional[float] = None,
                 k: float = NOSE_DEPTH_RATIO, min_face_px: float = MIN_FACE_PX) -> Optional[float]:
    """Yaw proxy in degrees (+ = face points to the camera's right); None if unknown."""
    if face_width_px is not None and face_width_px < min_face_px:
        return None
    r = yaw_proxy(landmarks)
    if r is None:
        return None
    return math.degrees(math.atan2(r, k))


def synth_landmarks(cx: float, cy: float, w: float, yaw_deg: float, k: float = NOSE_DEPTH_RATIO,
                    yunet_order: bool = True) -> Tuple[Point, ...]:
    """5 landmarks of a face of box width w centred at (cx, cy) turned by yaw_deg (+ = toward the
    camera's right), from the same head model. For the toy world and tests."""
    th = math.radians(yaw_deg)
    a = 0.2 * w                     # half eye distance ~ 0.2 face-box widths
    c, s = math.cos(th), math.sin(th)
    eye_l, eye_r = (cx - a * c, cy - 0.12 * w), (cx + a * c, cy - 0.12 * w)
    nose = (cx + k * a * s, cy + 0.05 * w)
    m = a                           # same lateral span as the eyes, so both ratios agree
    mouth_l, mouth_r = (cx - m * c, cy + 0.25 * w), (cx + m * c, cy + 0.25 * w)
    # YuNet order: right eye, left eye (the subject's), nose, right mouth, left mouth -> image left first
    return (eye_l, eye_r, nose, mouth_l, mouth_r) if yunet_order else (eye_r, eye_l, nose, mouth_r, mouth_l)
