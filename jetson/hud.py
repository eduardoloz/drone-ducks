"""Draw detections, the followed target and the controller's command on a frame.

Privacy: blur_others=True blurs every face that is not the followed target. Use it for anything
you save: we only keep imagery of consenting team members, never of bystanders.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from .controller import IDLE, LOST, TIMEOUT, TRACKING, FollowConfig
from .face import iou

MODE_COLOURS = {TRACKING: (60, 200, 60), LOST: (0, 200, 255), IDLE: (180, 180, 180), TIMEOUT: (40, 40, 230)}


def _cv2():
    import cv2
    return cv2


def blur_faces(img: np.ndarray, boxes, k: int = 31) -> None:
    """Blur boxes in place (expanded a little so hairlines/ears are covered too)."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        mx, my = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
        a, b = int(max(0, x1 - mx)), int(max(0, y1 - my))
        c, d = int(min(w, x2 + mx)), int(min(h, y2 + my))
        if c - a > 2 and d - b > 2:
            img[b:d, a:c] = cv2.GaussianBlur(img[b:d, a:c], (k, k), 0)


def _bar(img, x, y, label, value, limit, unit, width=120):
    cv2 = _cv2()
    cv2.putText(img, f"{label} {value:+.2f}{unit}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    cx = x + 170 + width // 2
    cv2.rectangle(img, (x + 170, y - 9), (x + 170 + width, y + 1), (90, 90, 90), 1)
    frac = 0.0 if limit <= 0 else max(-1.0, min(1.0, value / limit))
    cv2.rectangle(img, (cx, y - 8), (cx + int(frac * width / 2), y), (0, 220, 255), -1)


def draw(result, cfg: FollowConfig, fps: float = 0.0, backend: str = "", status: str = "",
         blur_others: bool = False) -> np.ndarray:
    """Return an annotated copy of result.frame.image."""
    cv2 = _cv2()
    img = result.frame.image.copy()
    h, w = img.shape[:2]
    target = result.target

    if blur_others:  # every face that isn't the followed person gets blurred
        boxes = [f.box for f in result.faces] + [tr.box for tr in result.tracks]
        if target is not None:
            boxes = [b for b in boxes if iou(b, target.box) < 0.3]
        blur_faces(img, boxes)

    # Image centre and the yaw deadband (faces inside the band don't cause a turn).
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
    t = math.tan(math.radians(cfg.hfov_deg) / 2)
    dx = int(math.tan(math.radians(cfg.yaw_deadband_deg)) / t * w / 2)
    for xx in (w // 2 - dx, w // 2 + dx):
        cv2.line(img, (xx, int(h * 0.42)), (xx, int(h * 0.58)), (200, 200, 200), 1)

    for f in result.faces:
        cv2.rectangle(img, (int(f.x1), int(f.y1)), (int(f.x2), int(f.y2)), (140, 140, 140), 1)
    for tr in result.tracks:
        if not tr.confirmed:
            continue
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        is_target = target is not None and tr.track_id == target.track_id
        colour = (60, 255, 60) if is_target else (255, 180, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3 if is_target else 1)
        cv2.putText(img, f"#{tr.track_id} {tr.score:.2f}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)
        if is_target:
            cv2.line(img, (w // 2, h // 2), (int(tr.cx), int(tr.cy)), colour, 1)

    out = result.control
    panel_h = 118
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (min(w, 330), panel_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    cv2.putText(img, out.mode + (f" -> {out.action.upper()}" if out.action else ""), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, MODE_COLOURS.get(out.mode, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.putText(img, out.reason[:46], (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    c = out.command
    _bar(img, 8, 58, "fwd  ", c.forward_m_s, cfg.max_forward_m_s, "m/s")
    _bar(img, 8, 74, "right", c.right_m_s, cfg.max_lateral_m_s, "m/s")
    _bar(img, 8, 90, "down ", c.down_m_s, cfg.max_vertical_m_s, "m/s")
    _bar(img, 8, 106, "yaw  ", c.yawspeed_deg_s, cfg.max_yaw_rate_deg_s, "d/s")

    tm = result.timings_ms
    footer = (f"{fps:4.1f} FPS  det {tm.get('detect', 0):.0f} ms  age {tm.get('frame_age', 0):.0f} ms  "
              f"{backend}  {status}")
    cv2.rectangle(img, (0, h - 20), (w, h), (0, 0, 0), -1)
    cv2.putText(img, footer[:110], (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class VideoSink:
    """Write annotated frames to a video file (mp4v; plays in VLC/QuickTime)."""

    def __init__(self, path: str, fps: float = 15.0) -> None:
        self.path, self.fps = path, fps
        self._vw = None

    def write(self, img: np.ndarray) -> None:
        cv2 = _cv2()
        if self._vw is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            h, w = img.shape[:2]
            self._vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self._vw.isOpened():
                raise RuntimeError(f"could not open video writer for {self.path}")
        self._vw.write(img)

    def close(self) -> Optional[str]:
        if self._vw is not None:
            self._vw.release()
            self._vw = None
            return self.path
        return None
