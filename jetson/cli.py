"""Command-line pieces shared by demo, follow and bench (argument groups, builders, small helpers)."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Optional

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


def add_controller_args(ap: argparse.ArgumentParser) -> None:
    d = FollowConfig()
    g = ap.add_argument_group("controller")
    g.add_argument("--hfov", type=float, default=d.hfov_deg,
                   help="camera horizontal FOV in degrees (sim camera: 68.75, RPi v2: 62.2)")
    g.add_argument("--face-width", type=float, default=d.face_width_m, help="face width (m) for distance estimate")
    g.add_argument("--target-distance", type=float, default=d.target_distance_m, help="standoff distance (m)")
    g.add_argument("--min-distance", type=float, default=d.min_distance_m, help="never fly closer (m)")
    g.add_argument("--max-forward", type=float, default=d.max_forward_m_s, help="max forward speed (m/s)")
    g.add_argument("--max-yaw-rate", type=float, default=d.max_yaw_rate_deg_s, help="max yaw rate (deg/s)")
    g.add_argument("--lost-timeout", type=float, default=d.lost_timeout_s, help="hover this long when lost (s)")
    g.add_argument("--lost-action", default=d.lost_action, choices=LOST_ACTIONS, help="then do this")
    g.add_argument("--vertical", action="store_true", help="enable altitude control (needs altitude telemetry)")


def make_config(args) -> FollowConfig:
    return FollowConfig(hfov_deg=args.hfov, face_width_m=args.face_width, target_distance_m=args.target_distance,
                        min_distance_m=args.min_distance, max_forward_m_s=args.max_forward,
                        max_yaw_rate_deg_s=args.max_yaw_rate, lost_timeout_s=args.lost_timeout,
                        lost_action=args.lost_action, enable_vertical=args.vertical)


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
              "down", "yaw", "detect_ms", "total_ms", "frame_age_ms"]

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
            "total_ms": f"{res.timings_ms['total']:.1f}", "frame_age_ms": f"{res.timings_ms['frame_age']:.1f}"})

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
