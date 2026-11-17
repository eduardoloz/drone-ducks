"""Command-line pieces shared by demo, follow and bench (argument groups, builders, small helpers)."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Optional

from . import geometry
from .controller import LOST_ACTIONS, FollowConfig
from .detectors import BACKENDS, create_detector
from .detectors.base import FaceDetector


def add_detector_args(ap: argparse.ArgumentParser, extra_backends=()) -> None:
    g = ap.add_argument_group("detector")
    g.add_argument("--backend", default="yunet", choices=sorted(BACKENDS) + list(extra_backends),
                   help="face detector backend (default: yunet)")
    g.add_argument("--model", default=None,
                   help="model name or path (onnx: yunet|scrfd|path; ultralytics: .pt/.onnx/.engine)")
    g.add_argument("--engine", default=None, help="TensorRT engine path (backend tensorrt)")
    g.add_argument("--device", default=None, help="cpu | cuda | cuda_fp16 (yunet) | tensorrt (onnx)")
    g.add_argument("--input-width", type=int, default=640,
                   help="resize frames to this width before detection (speed vs range!)")
    g.add_argument("--score", type=float, default=None, help="detection score threshold")
    g.add_argument("--person", default="none", choices=("none", "yolox", "hog", "tensorrt"),
                   help="also detect bodies and keep the lock when the face turns away "
                        "(yolox: Apache-2.0 YOLOX-nano, needs `jetson.fetch models`; hog: OpenCV, slow; "
                        "tensorrt: engine from scripts/export_trt.py, Jetson only)")
    g.add_argument("--person-engine", default="opencv", choices=("opencv", "onnxruntime"),
                   help="yolox runtime (onnxruntime + --device cuda|tensorrt on a Jetson)")
    g.add_argument("--person-every", type=int, default=1, metavar="N",
                   help="run the person detector on every Nth frame (saves time on a Nano)")


def make_person_detector(args):
    """The --person detector, or None. Body tracking only: nothing identifies anyone."""
    name = getattr(args, "person", "none")
    if name in (None, "none"):
        return None
    from .detectors.person import create_person_detector

    if name == "yolox":
        dev = args.device if args.device in ("cpu", "cuda", "tensorrt") else "cpu"
        return create_person_detector("yolox", engine=args.person_engine, device=dev)
    return create_person_detector(name)


def make_detector(args) -> FaceDetector:
    kw = {"input_width": args.input_width}
    if args.score is not None:
        kw["score_threshold"] = args.score
    if args.backend == "onnx":
        kw["model"] = args.model or "yunet"
        kw["device"] = args.device or "cpu"
    elif args.backend == "yunet":
        if args.model:
            kw["model"] = args.model
        kw["device"] = args.device or "cpu"
    elif args.backend == "tensorrt":
        kw = {"engine": args.engine or args.model or ""}
        if args.score is not None:
            kw["score_threshold"] = args.score
    elif args.backend == "ultralytics":
        kw["model"] = args.model or ""
        kw["device"] = args.device or ""
    elif args.backend == "haar":
        kw.pop("score_threshold", None)
    return create_detector(args.backend, **kw)


# --camera-preset: (horizontal FOV deg, face width m as boxed by YuNet, note). --hfov / --face-width
# override. Face widths: 0.16 m is the adult default; the Gazebo walking actor boxes at 0.13 m
# (sim agent, 3 frames at 2.3-3.9 m ground truth). Re-measure face width on each real camera.
CAMERA_PRESETS = {
    "imx219": (geometry.CAMERAS["rpi-v2"].hfov_deg, geometry.FACE_WIDTH_M, "RPi Camera v2 / IMX219 stock lens"),
    "imx219-wide": (geometry.CAMERAS["imx219-wide"].hfov_deg, geometry.FACE_WIDTH_M,
                    "IMX219 ~120 deg module (fisheye-ish: distance is optimistic off-centre)"),
    "webcam": (geometry.CAMERAS["c920"].hfov_deg, geometry.FACE_WIDTH_M, "Logitech C920-class USB webcam"),
    "sim": (geometry.CAMERAS["sim"].hfov_deg, 0.13, "Gazebo quack_cam + walking actor"),
}
DEFAULT_CAMERA_PRESET = "imx219"   # == the FollowConfig defaults (62.2 deg, 0.16 m)


def camera_params(preset: Optional[str] = None, hfov: Optional[float] = None,
                  face_width: Optional[float] = None):
    """(hfov_deg, face_width_m): the preset's values, overridden by explicit ones."""
    p_hfov, p_face, _ = CAMERA_PRESETS[preset or DEFAULT_CAMERA_PRESET]
    return (p_hfov if hfov is None else hfov, p_face if face_width is None else face_width)


def add_controller_args(ap: argparse.ArgumentParser) -> None:
    d = FollowConfig()
    g = ap.add_argument_group("controller")
    g.add_argument("--camera-preset", default=None, choices=sorted(CAMERA_PRESETS),
                   help="HFOV + face width for a camera: " + "; ".join(
                       f"{k}: {v[0]:g} deg, {v[1]:g} m" for k, v in sorted(CAMERA_PRESETS.items()))
                   + f" (default {DEFAULT_CAMERA_PRESET})")
    g.add_argument("--hfov", type=float, default=None,
                   help=f"camera horizontal FOV in degrees, overrides the preset (default {d.hfov_deg:g})")
    g.add_argument("--face-width", type=float, default=None,
                   help=f"face width (m) for the distance estimate, overrides the preset (default {d.face_width_m:g})")
    g.add_argument("--target-distance", type=float, default=d.target_distance_m, help="standoff distance (m)")
    g.add_argument("--min-distance", type=float, default=d.min_distance_m, help="never fly closer (m)")
    g.add_argument("--max-forward", type=float, default=d.max_forward_m_s, help="max forward speed (m/s)")
    g.add_argument("--max-yaw-rate", type=float, default=d.max_yaw_rate_deg_s, help="max yaw rate (deg/s)")
    g.add_argument("--lost-timeout", type=float, default=d.lost_timeout_s, help="hover this long when lost (s)")
    g.add_argument("--lost-action", default=d.lost_action, choices=LOST_ACTIONS, help="then do this")
    g.add_argument("--vertical", action="store_true", help="enable altitude control (needs altitude telemetry)")
    g.add_argument("--head-lateral", type=float, default=d.k_head_lateral, metavar="K",
                   help="fly sideways to stay in front of the face: m/s per deg of head yaw "
                        "(0 = off; 0.02 tested in the toy world)")
    g.add_argument("--max-lateral", type=float, default=d.max_lateral_m_s, help="max sideways speed (m/s)")
    g.add_argument("--relock-hits", type=int, default=3,
                   help="after a loss, resume only after this many consecutive detections (1 = round-2 behaviour)")
    g.add_argument("--relock-score", type=float, default=0.9,
                   help="...or one detection scoring at least this (<= 0 disables)")
    g.add_argument("--relock-max-distance", type=float, default=None, metavar="M",
                   help="...and only on a face estimated within M metres (default: target distance + 1.5)")
    g.add_argument("--relock-holdoff", type=float, default=0.0,
                   help="face-only: after a loss adopt no new face track for this long (s)")
    g.add_argument("--end-on-walkaway", action="store_true",
                   help="body lock: end the follow (TIMEOUT -> --lost-action) when the person walks away "
                        "with no face visible, instead of chasing them (off by default; see README)")
    g.add_argument("--walkaway-seconds", type=float, default=d.walkaway_s, metavar="S",
                   help=f"...receding this long with no face (default {d.walkaway_s:g})")
    g.add_argument("--walkaway-speed", type=float, default=d.walkaway_speed_m_s, metavar="M_S",
                   help=f"...at least this fast over the ground (default {d.walkaway_speed_m_s:g} m/s)")
    g.add_argument("--walkaway-margin", type=float, default=d.walkaway_margin_m, metavar="M",
                   help=f"...and farther than target distance + M (default {d.walkaway_margin_m:g})")


def make_config(args) -> FollowConfig:
    hfov, face_w = camera_params(getattr(args, "camera_preset", None), args.hfov, args.face_width)
    d = FollowConfig()
    return FollowConfig(hfov_deg=hfov, face_width_m=face_w, target_distance_m=args.target_distance,
                        min_distance_m=args.min_distance, max_forward_m_s=args.max_forward,
                        max_yaw_rate_deg_s=args.max_yaw_rate, lost_timeout_s=args.lost_timeout,
                        lost_action=args.lost_action, enable_vertical=args.vertical,
                        k_head_lateral=getattr(args, "head_lateral", d.k_head_lateral),
                        max_lateral_m_s=getattr(args, "max_lateral", d.max_lateral_m_s),
                        end_on_walkaway=getattr(args, "end_on_walkaway", False),
                        walkaway_s=getattr(args, "walkaway_seconds", d.walkaway_s),
                        walkaway_speed_m_s=getattr(args, "walkaway_speed", d.walkaway_speed_m_s),
                        walkaway_margin_m=getattr(args, "walkaway_margin", d.walkaway_margin_m))


def relock_kwargs(args) -> dict:
    """FollowPipeline kwargs (relock=, relock_holdoff_s=) from the --relock-* flags."""
    from .relock import RelockGate

    score = getattr(args, "relock_score", 0.9)
    maxd = getattr(args, "relock_max_distance", None)
    if maxd is None:
        maxd = getattr(args, "target_distance", 4.0) + 1.5
    gate = RelockGate(confirm_hits=getattr(args, "relock_hits", 3),
                      confirm_score=score if score and score > 0 else None,
                      max_distance_m=maxd if maxd > 0 else None)
    return {"relock": gate, "relock_holdoff_s": getattr(args, "relock_holdoff", 0.0)}


class FpsMeter:
    """Exponential moving average of a loop's rate."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha, self.fps, self._t = alpha, 0.0, None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._t is not None:
            inst = 1.0 / max(1e-6, now - self._t)
            self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._t = now
        return self.fps


class StepLog:
    """Optional CSV log, one row per processed frame, for tuning plots."""

    FIELDS = ["t", "seq", "mode", "action", "faces", "target_id", "bearing_deg", "distance_m", "fwd", "right",
              "down", "yaw", "detect_ms", "total_ms", "frame_age_ms", "source", "bodies", "person_ms",
              "recede_m_s"]

    def __init__(self, path: Optional[str]) -> None:
        self._f = self._w = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._f = open(path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
            self._w.writeheader()

    def write(self, t: float, res) -> None:
        if self._w is None:
            return
        c, o = res.control.command, res.control
        self._w.writerow({
            "t": f"{t:.3f}", "seq": res.frame.seq, "mode": o.mode, "action": o.action or "",
            "faces": len(res.faces), "target_id": res.target.track_id if res.target else "",
            "bearing_deg": "" if o.bearing_deg is None else f"{o.bearing_deg:.2f}",
            "distance_m": "" if o.distance_m is None else f"{o.distance_m:.2f}",
            "fwd": f"{c.forward_m_s:.3f}", "right": f"{c.right_m_s:.3f}", "down": f"{c.down_m_s:.3f}",
            "yaw": f"{c.yawspeed_deg_s:.2f}", "detect_ms": f"{res.timings_ms['detect']:.1f}",
            "total_ms": f"{res.timings_ms['total']:.1f}", "frame_age_ms": f"{res.timings_ms['frame_age']:.1f}",
            "source": getattr(res, "target_source", ""), "bodies": len(getattr(res, "bodies", [])),
            "person_ms": f"{res.timings_ms['person']:.1f}" if "person" in res.timings_ms else "",
            "recede_m_s": "" if getattr(o, "recede_m_s", None) is None else f"{o.recede_m_s:.2f}"})

    def close(self) -> None:
        if self._f:
            self._f.close()


def percentile(values, q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(len(s) - 1, int(k) + 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
