import cv2
import numpy as np

from cctv.camera_motion import CameraMotionDetector


def scene(seed=0, w=640, h=480):
    """A textured synthetic 'road scene' with plenty of corners for ORB."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 90, np.uint8)
    for _ in range(400):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(4, 25))
        cv2.rectangle(img, (x, y), (x + s, y + s // 2), int(rng.integers(0, 255)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def with_osd(img):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (180, 120), 0, -1)
    cv2.putText(out, "Camera 041", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.putText(out, "16:43:34", (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return out


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_same_view_with_noise_is_static():
    ref = scene()
    noisy = np.clip(ref.astype(int) + np.random.default_rng(1).normal(0, 3, ref.shape), 0, 255).astype(np.uint8)
    r = CameraMotionDetector(ref).check(noisy)
    assert not r.moved, r.reason


def test_small_pan_is_detected_even_with_static_overlay():
    ref = scene()
    moved = with_osd(shift(ref, 12, 0))   # picture moves, overlay text does not
    r = CameraMotionDetector(with_osd(ref)).check(moved)
    assert r.moved
    assert 9 < r.shift_px < 15


def test_zoom_is_detected():
    ref = scene()
    M = cv2.getRotationMatrix2D((320, 240), 0, 1.06)
    zoomed = cv2.warpAffine(ref, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
    r = CameraMotionDetector(ref).check(zoomed)
    assert r.moved and "zoom" in r.reason


def test_completely_different_view_is_detected():
    r = CameraMotionDetector(scene(0)).check(scene(99))
    assert r.moved


def test_resolution_change_is_detected():
    r = CameraMotionDetector(scene()).check(cv2.resize(scene(), (352, 240)))
    assert r.moved and "resolution" in r.reason
