"""Pinhole-camera maths for "how many pixels will a face be, and how far away is it?".

All angles are in degrees at the API, all lengths in metres. These formulas assume a rectilinear
(normal, non-fisheye) lens. Ultra-wide "160 degree" lenses are fisheye-like: near the centre they
behave roughly like a rectilinear lens with the same focal length, but towards the edges faces
shrink, so treat numbers for them as optimistic.

Key formula (face of width F metres at distance d, horizontal FOV theta, image width W pixels):

    pixels_on_face = W * F / (2 * d * tan(theta / 2))

W is the width of the image *the detector actually sees*. If a 1920-px camera frame is resized to
640 px before inference, use 640. Cropping/tiling is how you get the range back.

Run `python -m jetson.geometry` to print the range tables used in docs/WHAT_WE_NEED.md.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Iterable, NamedTuple, Optional

FACE_WIDTH_M = 0.16    # typical adult face width as boxed by detectors (calibrate per detector!)
PERSON_HEIGHT_M = 1.7  # for the person-detection stage of a two-stage pipeline


def _tan_half(fov_deg: float) -> float:
    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"field of view must be in (0, 180) degrees, got {fov_deg}")
    return math.tan(math.radians(fov_deg) / 2.0)


def pixels_on_object(size_m: float, distance_m: float, fov_deg: float, image_px: float) -> float:
    """Pixels spanned by an object of `size_m` at `distance_m` along an axis with `fov_deg`."""
    if distance_m <= 0:
        raise ValueError("distance must be positive")
    return image_px * size_m / (2.0 * distance_m * _tan_half(fov_deg))


def face_pixels(distance_m: float, hfov_deg: float, image_width_px: float,
                face_width_m: float = FACE_WIDTH_M) -> float:
    """Width in pixels of a face at `distance_m` (the formula in the module docstring)."""
    return pixels_on_object(face_width_m, distance_m, hfov_deg, image_width_px)


def max_range_m(min_pixels: float, hfov_deg: float, image_width_px: float,
                size_m: float = FACE_WIDTH_M) -> float:
    """Farthest distance at which an object still spans `min_pixels` (inverse of the above)."""
    if min_pixels <= 0:
        raise ValueError("min_pixels must be positive")
    return image_width_px * size_m / (2.0 * min_pixels * _tan_half(hfov_deg))


def distance_from_width(width_norm: float, hfov_deg: float,
                        face_width_m: float = FACE_WIDTH_M) -> float:
    """Estimate distance (m) from a face box width given as a fraction of image width (0..1]."""
    if width_norm <= 0:
        raise ValueError("width must be positive")
    return face_width_m / (2.0 * width_norm * _tan_half(hfov_deg))


def bearing_deg(x_norm: float, fov_deg: float) -> float:
    """Angle (deg) of an image point from the optical axis along one axis.

    x_norm is 0 at the left (or top) edge, 0.5 at the centre and 1 at the right (or bottom) edge.
    Positive means right of centre (or below centre for the vertical axis).
    """
    return math.degrees(math.atan((2.0 * x_norm - 1.0) * _tan_half(fov_deg)))


def vfov_from_hfov(hfov_deg: float, aspect: float) -> float:
    """Vertical FOV for a rectilinear lens, aspect = width / height (e.g. 4/3, 16/9)."""
    return math.degrees(2.0 * math.atan(_tan_half(hfov_deg) / aspect))


def hfov_from_dfov(dfov_deg: float, aspect: float) -> float:
    """Horizontal FOV from a diagonal FOV (what webcam spec sheets usually quote)."""
    diag = math.hypot(aspect, 1.0)
    return math.degrees(2.0 * math.atan(_tan_half(dfov_deg) * aspect / diag))


def hfov_from_focal(sensor_width_mm: float, focal_mm: float) -> float:
    """Horizontal FOV from sensor width and lens focal length."""
    return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal_mm)))


def depression_deg(altitude_m: float, horizontal_m: float, target_height_m: float = 1.6) -> float:
    """Look-down angle from the drone to a face: 0 = level, 90 = straight down."""
    return math.degrees(math.atan2(altitude_m - target_height_m, horizontal_m))


def slant_from_depth(depth_m: float, x_norm: float, y_norm: float, hfov_deg: float, vfov_deg: float) -> float:
    """Straight-line camera-to-face range from the *depth* a box width gives.

    distance_from_width() measures depth along the optical axis (pinhole: width_px = f * W / depth).
    A face off the image centre is farther than that along its ray by sqrt(1 + x^2 + y^2), with x, y
    its normalised image-plane coordinates (tan of its angles off the axis)."""
    x = (2.0 * x_norm - 1.0) * _tan_half(hfov_deg)
    y = (2.0 * y_norm - 1.0) * _tan_half(vfov_deg)
    return depth_m * math.sqrt(1.0 + x * x + y * y)


def horizontal_distance(depth_m: float, x_norm: float, y_norm: float, hfov_deg: float, vfov_deg: float,
                        altitude_m: Optional[float], cam_pitch_deg: float = 0.0,
                        target_height_m: float = 1.6) -> float:
    """Ground (horizontal) distance from the drone to a face, from its box-width depth estimate.

    altitude_m None (unknown): returns depth_m unchanged (the pre-round-4 behaviour).
    Otherwise uses the height difference dh = altitude - face height:
        slant = slant_from_depth(...);  horizontal = sqrt(slant^2 - dh^2)
    Altitude (PX4 relative altitude, i.e. height above the take-off point; flat ground assumed) is
    preferred over the camera ray because the ray's look-down angle moves with the body's pitch
    when the drone accelerates (a few degrees at 1 m/s), whereas dh barely moves. Only if the two
    disagree physically (slant <= |dh|: the box says closer than the height difference, e.g. a
    child, a person on stairs, or a bad box) do we fall back to the ray: rotate it by the camera
    pitch (+ = tilted down) and take its horizontal component.
    """
    if altitude_m is None or not math.isfinite(altitude_m):
        return depth_m
    slant = slant_from_depth(depth_m, x_norm, y_norm, hfov_deg, vfov_deg)
    dh = altitude_m - target_height_m
    if slant > abs(dh) * 1.02:
        return math.sqrt(slant * slant - dh * dh)
    x = (2.0 * x_norm - 1.0) * _tan_half(hfov_deg)
    y = (2.0 * y_norm - 1.0) * _tan_half(vfov_deg)
    p = math.radians(cam_pitch_deg)
    fwd = math.cos(p) - y * math.sin(p)      # camera ray (x right, y down, 1 forward) -> body frame
    return depth_m * math.hypot(max(fwd, 0.0), x)


class CameraPreset(NamedTuple):
    name: str
    hfov_deg: float
    native_width: int
    note: str


# HFOVs: see docs/WHAT_WE_NEED.md (camera section) for sources. IMX477 sensor width 6.287 mm.
CAMERAS: Dict[str, CameraPreset] = {
    "rpi-v2": CameraPreset("RPi Camera v2 (IMX219, stock lens)", 62.2, 3280, "rectilinear"),
    "rpi-hq-6mm": CameraPreset("RPi HQ (IMX477) + 6 mm CS lens", round(hfov_from_focal(6.287, 6.0), 1),
                               4056, "rectilinear"),
    "rpi-hq-16mm": CameraPreset("RPi HQ (IMX477) + 16 mm C lens", round(hfov_from_focal(6.287, 16.0), 1),
                                4056, "rectilinear, narrow"),
    "c920": CameraPreset("Logitech C920 USB (78 deg diagonal, 16:9)", round(hfov_from_dfov(78.0, 16 / 9), 1),
                         1920, "rectilinear"),
    "imx219-wide": CameraPreset("IMX219 'wide' module (~120 deg H)", 120.0, 3280,
                                "fisheye-like: optimistic at the edges"),
    "sim": CameraPreset("Gazebo quack_cam (1.2 rad HFOV)", round(math.degrees(1.2), 2), 640, "simulated"),
}


def range_table(cameras: Iterable[str] = ("rpi-v2", "rpi-hq-6mm", "rpi-hq-16mm", "c920", "imx219-wide"),
                widths: Iterable[int] = (320, 640, 1280, 1920),
                min_px: Iterable[int] = (10, 20), face_width_m: float = FACE_WIDTH_M) -> str:
    """Markdown table of max face-detection range per camera / inference width / min face size."""
    widths, min_px = list(widths), list(min_px)
    head = "| Camera (HFOV) | " + " | ".join(
        f"{w} px @ {p} px face" for w in widths for p in min_px) + " |"
    sep = "|---|" + "---:|" * (len(widths) * len(min_px))
    rows = [head, sep]
    for key in cameras:
        cam = CAMERAS[key]
        cells = [f"{max_range_m(p, cam.hfov_deg, w, face_width_m):.1f} m" for w in widths for p in min_px]
        rows.append(f"| {cam.name} ({cam.hfov_deg:g} deg) | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def pixels_table(cameras: Iterable[str] = ("rpi-v2", "c920", "imx219-wide"), width: int = 640,
                 distances: Iterable[float] = (2, 3, 5, 8, 12, 20),
                 face_width_m: float = FACE_WIDTH_M) -> str:
    distances = list(distances)
    rows = ["| Camera @ %d px wide | " % width + " | ".join(f"{d:g} m" for d in distances) + " |",
            "|---|" + "---:|" * len(distances)]
    for key in cameras:
        cam = CAMERAS[key]
        rows.append(f"| {cam.name} | " + " | ".join(
            f"{face_pixels(d, cam.hfov_deg, width, face_width_m):.0f} px" for d in distances) + " |")
    return "\n".join(rows)


def person_range_m(min_px: float, hfov_deg: float, image_width_px: int, aspect: float = 4 / 3,
                   height_m: float = PERSON_HEIGHT_M) -> float:
    """Max range for a person detector that needs `min_px` of body height (vertical axis)."""
    vfov = vfov_from_hfov(hfov_deg, aspect)
    return max_range_m(min_px, vfov, image_width_px / aspect, size_m=height_m)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Print face-detection range tables (pinhole model).")
    ap.add_argument("--face-width", type=float, default=FACE_WIDTH_M, help="face width in metres")
    args = ap.parse_args(argv)
    print(f"Max face range (m) = W * {args.face_width} / (2 * min_px * tan(HFOV/2))\n")
    print(range_table(face_width_m=args.face_width))
    print("\nPixels on a face at 640 px inference width\n")
    print(pixels_table(face_width_m=args.face_width))
    print("\nTwo-stage idea: a person detector needing ~40 px of body height (4:3 frame)\n")
    for key in ("rpi-v2", "c920", "imx219-wide"):
        cam = CAMERAS[key]
        print(f"- {cam.name}: {person_range_m(40, cam.hfov_deg, 640):.0f} m at 640 px, "
              f"{person_range_m(40, cam.hfov_deg, 1280):.0f} m at 1280 px")
    print("\nLook-down angle to a face at 1.6 m height:")
    for alt in (3, 5, 8):
        print(f"- altitude {alt} m: " + ", ".join(
            f"{h} m away -> {depression_deg(alt, h):.0f} deg" for h in (3, 5, 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
