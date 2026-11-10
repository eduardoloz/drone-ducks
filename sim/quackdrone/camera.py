"""Camera sources: one interface for the sim camera, a webcam, a phone, a video file, or the
Jetson's camera. Perception code reads frames from a CameraSource and never cares which.

    from quackdrone.camera import open_camera
    cam = open_camera("http://127.0.0.1:5601")      # Gazebo drone camera (raw, numpy only)
    cam = open_camera("http://127.0.0.1:5601/video") # same camera as MJPEG, through OpenCV
    cam = open_camera(0)                             # laptop webcam
    cam = open_camera("http://10.18.237.58:8080/video")  # phone "IP Webcam" app (Stephen's scripts)
    cam = open_camera("nvarguscamerasrc ! ... ! appsink")  # Jetson CSI camera (GStreamer string)
    frame = cam.read()         # Frame or None; frame.rgb is HxWx3 uint8 RGB, frame.bgr() for OpenCV

The OpenCV source keeps only the newest frame in a background thread (the pattern from
BTMOB.py), so a slow detector always works on a fresh image instead of a growing backlog.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray        # H x W x 3, uint8, RGB order
    stamp: float           # time.time() when the frame was captured (bridge) or received
    seq: int               # increasing frame counter (gaps = frames you did not see)

    def bgr(self) -> np.ndarray:
        """OpenCV channel order (a view, no copy)."""
        return self.rgb[:, :, ::-1]

    @property
    def age_s(self) -> float:
        return time.time() - self.stamp


class CameraSource(ABC):
    @abstractmethod
    def read(self, timeout_s: float = 1.0) -> Optional[Frame]:
        """Newest frame, or None if none arrived within timeout_s."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BridgeCamera(CameraSource):
    """Pulls raw RGB frames from sim/gazebo/gz_camera_bridge.py (GET /frame). Needs only numpy."""

    def __init__(self, base_url: str = "http://127.0.0.1:5601", timeout_s: float = 1.0):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/frame"):
            self.base_url = self.base_url[: -len("/frame")]
        self.timeout_s = timeout_s

    def read(self, timeout_s: Optional[float] = None) -> Optional[Frame]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/frame", timeout=timeout_s or self.timeout_s) as r:
                w, h = int(r.headers["X-Width"]), int(r.headers["X-Height"])
                seq = int(r.headers.get("X-Seq", 0))
                stamp = float(r.headers.get("X-Stamp", 0)) or time.time() - int(r.headers.get("X-Age-Ms", 0)) / 1e3
                data = r.read()
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return None
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return Frame(rgb, stamp, seq)

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            return {"ok": False}


class OpenCVCamera(CameraSource):
    """cv2.VideoCapture with a newest-frame-only reader thread."""

    def __init__(self, source: Union[int, str], open_timeout_s: float = 10.0):
        import cv2  # optional dependency: pip install opencv-python

        self.source = source
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera source {source!r}")
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._spin, name="camera-reader", daemon=True)
        self._thread.start()
        deadline = time.time() + open_timeout_s
        with self._cond:
            while self._frame is None and self._running and time.time() < deadline:
                self._cond.wait(0.1)

    def _spin(self) -> None:
        while self._running:
            ok, bgr = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._seq += 1
                self._frame = Frame(np.ascontiguousarray(bgr[:, :, ::-1]), time.time(), self._seq)
                self._cond.notify_all()

    def read(self, timeout_s: float = 1.0) -> Optional[Frame]:
        deadline = time.time() + timeout_s
        with self._cond:
            while self._frame is None and time.time() < deadline:
                self._cond.wait(0.05)
            return self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class PatternCamera(CameraSource):
    """Synthetic frames (a bright square moving across a grey image) for tests and fake:// demos."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 15.0):
        self.width, self.height, self.fps = width, height, fps
        self._t0 = time.time()

    def read(self, timeout_s: float = 1.0) -> Optional[Frame]:
        t = time.time() - self._t0
        seq = int(t * self.fps)
        img = np.full((self.height, self.width, 3), 90, np.uint8)
        x = int((seq * 8) % (self.width - 60))
        img[200:260, x:x + 60] = (240, 200, 40)
        return Frame(img, time.time(), seq)


def is_bridge_url(source: str) -> bool:
    """True for the raw bridge endpoint: http://host:port, http://host:port/ or .../frame."""
    u = urllib.parse.urlparse(source)
    return u.scheme in ("http", "https") and u.path in ("", "/", "/frame")


def open_camera(source: Union[int, str, CameraSource]) -> CameraSource:
    """Build a CameraSource from an int (webcam index), a URL, a file path or a GStreamer string."""
    if isinstance(source, CameraSource):
        return source
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return OpenCVCamera(int(source))
    if source in ("fake://", "test://"):
        return PatternCamera()
    if is_bridge_url(source):
        return BridgeCamera(source)
    return OpenCVCamera(source)
