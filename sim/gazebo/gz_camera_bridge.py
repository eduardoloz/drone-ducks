#!/usr/bin/env python3
"""gz_camera_bridge.py: serve the simulated drone's Gazebo camera over local HTTP.

Originally written for ProjectQuack's TypeFly/PX4 branch; moved here and extended with JPEG and
MJPEG endpoints so OpenCV code (Stephen's scripts, the Jetson face follower) can read the sim
camera exactly like a phone "IP Webcam" stream.

Why a separate process: the Gazebo Python bindings (gz.transport13, gz.msgs10) only exist for the
Python Gazebo was built against: the pixi env's python3 on macOS, the system python3 on Ubuntu
(apt python3-gz-transport13). Our venv has torch/OpenCV/MAVSDK instead, so this bridge runs with
Gazebo's Python and everything else talks HTTP to it. Needs only the stdlib, numpy and gz.

Endpoints (default http://127.0.0.1:5601), all return the NEWEST frame:
  GET /frame        raw RGB bytes (H*W*3), headers X-Width X-Height X-Seq X-Stamp X-Age-Ms
  GET /frame.jpg    JPEG snapshot                         (needs a JPEG encoder, see below)
  GET /video        MJPEG stream (multipart/x-mixed-replace); alias /stream.mjpg
                    -> cv2.VideoCapture("http://127.0.0.1:5601/video") or open it in a browser
  GET /frame.bmp    BMP snapshot (always available, no encoder needed)
  GET /health       JSON {ok, topic, frames, width, height, age_ms, fps, jpeg}

JPEG encoder: Pillow or OpenCV if importable, otherwise libturbojpeg through ctypes (ships inside
the PX4 macOS pixi env and as libturbojpeg0 on Ubuntu). Without one, /frame.jpg and /video
return 501 and everything else still works.

Usage:
  sim/scripts/run_camera_bridge.sh                    # picks Gazebo's python for you
  python3 gz_camera_bridge.py [--topic T] [--port 5601] [--width 640] [--quality 80]
With no --topic, the first camera image topic (not depth) is used.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import json
import os
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# gz.msgs.PixelFormatType values we handle (gz-msgs image.proto / pixel_format_type.proto).
L_INT8, RGB_INT8, RGBA_INT8, BGRA_INT8, BGR_INT8 = 1, 3, 4, 5, 8
_CHANNELS = {RGB_INT8: 3, BGR_INT8: 3, RGBA_INT8: 4, BGRA_INT8: 4, L_INT8: 1}
BOUNDARY = b"quackframe"


# ----------------------------------------------------------------------------- pure helpers
def to_rgb(width: int, height: int, pixel_format: int, step: int, data: bytes, out_width: int = 0) -> np.ndarray:
    """Convert a gz.msgs.Image payload to a contiguous HxWx3 uint8 RGB array, downscaled by an
    integer factor (cheap striding) so its width is about out_width (0 = keep size)."""
    ch = _CHANNELS.get(pixel_format)
    if ch is None:
        raise ValueError(f"unsupported pixel format {pixel_format}")
    step = step or width * ch
    buf = np.frombuffer(data, dtype=np.uint8).reshape(height, step)[:, : width * ch].reshape(height, width, ch)
    if pixel_format in (BGR_INT8, BGRA_INT8):
        buf = buf[:, :, 2::-1]
    elif ch == 4:
        buf = buf[:, :, :3]
    elif ch == 1:
        buf = np.repeat(buf, 3, axis=2)
    k = max(1, width // out_width) if out_width else 1
    return np.ascontiguousarray(buf[::k, ::k])


def to_bmp(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    row = w * 3
    pad = (4 - row % 4) % 4
    bgr = rgb[::-1, :, ::-1]  # BMP rows are bottom-up, BGR
    if pad:
        bgr = np.concatenate([bgr.reshape(h, row), np.zeros((h, pad), np.uint8)], axis=1)
    pixels = np.ascontiguousarray(bgr).tobytes()
    header = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    return header + info + pixels


def mjpeg_part(jpeg: bytes) -> bytes:
    """One part of a multipart/x-mixed-replace MJPEG stream (format FFmpeg's mpjpeg demuxer and
    browsers accept)."""
    return (b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")


class _TurboJpeg:
    """Minimal libturbojpeg binding (legacy tjCompress2 API: present in libjpeg-turbo 1.4 .. 3.x)."""

    TJPF_RGB, TJSAMP_420, TJFLAG_FASTDCT = 0, 2, 2048

    def __init__(self, path: str):
        lib = ctypes.CDLL(path)
        lib.tjInitCompress.restype = ctypes.c_void_p
        lib.tjCompress2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                                    ctypes.POINTER(ctypes.c_ulong), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.tjCompress2.restype = ctypes.c_int
        lib.tjFree.argtypes = [ctypes.c_void_p]
        lib.tjDestroy.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self._handle = lib.tjInitCompress()
        if not self._handle:
            raise OSError("tjInitCompress failed")
        self.path = path

    def encode(self, rgb: np.ndarray, quality: int) -> bytes:
        h, w, _ = rgb.shape
        src = np.ascontiguousarray(rgb)
        out = ctypes.POINTER(ctypes.c_ubyte)()
        size = ctypes.c_ulong(0)
        rc = self._lib.tjCompress2(self._handle, src.ctypes.data_as(ctypes.c_void_p), w, w * 3, h,
                                   self.TJPF_RGB, ctypes.byref(out), ctypes.byref(size), self.TJSAMP_420,
                                   int(quality), self.TJFLAG_FASTDCT)
        if rc != 0:
            raise RuntimeError("tjCompress2 failed")
        try:
            return ctypes.string_at(out, size.value)
        finally:
            self._lib.tjFree(out)


def _turbojpeg_candidates() -> list[str]:
    names = []
    if os.environ.get("QUACK_TURBOJPEG"):
        names.append(os.environ["QUACK_TURBOJPEG"])
    for prefix in filter(None, {os.environ.get("CONDA_PREFIX"), sys.prefix}):
        names += glob.glob(os.path.join(prefix, "lib", "libturbojpeg.*"))
    found = ctypes.util.find_library("turbojpeg")
    if found:
        names.append(found)
    names += glob.glob("/usr/lib/*-linux-gnu/libturbojpeg.so*") + glob.glob("/opt/homebrew/lib/libturbojpeg.*") \
        + glob.glob("/usr/local/lib/libturbojpeg.*")
    return names


class JpegEncoder:
    """Picks the first available JPEG encoder: Pillow, OpenCV, then libturbojpeg via ctypes."""

    def __init__(self, quality: int = 80):
        self.quality = quality
        self.name = None
        self._encode = None
        self._lock = threading.Lock()
        try:
            from io import BytesIO

            from PIL import Image

            def pil(rgb):
                b = BytesIO()
                Image.fromarray(rgb, "RGB").save(b, "JPEG", quality=self.quality)
                return b.getvalue()

            self._encode, self.name = pil, "pillow"
            return
        except ImportError:
            pass
        try:
            import cv2

            def ocv(rgb):
                ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise RuntimeError("cv2.imencode failed")
                return buf.tobytes()

            self._encode, self.name = ocv, "opencv"
            return
        except ImportError:
            pass
        for path in _turbojpeg_candidates():
            try:
                tj = _TurboJpeg(path)
            except OSError:
                continue
            self._encode, self.name = (lambda rgb: tj.encode(rgb, self.quality)), f"turbojpeg ({os.path.basename(path)})"
            return

    @property
    def available(self) -> bool:
        return self._encode is not None

    def encode(self, rgb: np.ndarray) -> bytes:
        if self._encode is None:
            raise RuntimeError("no JPEG encoder available")
        with self._lock:
            return self._encode(rgb)


class Latest:
    """Newest frame, plus a lazily encoded JPEG of it (encoded at most once per frame)."""

    def __init__(self, encoder: JpegEncoder):
        self.cond = threading.Condition()
        self.rgb = None
        self.stamp = 0.0
        self.seq = 0
        self.encoder = encoder
        self._jpeg = (0, b"")
        self._arrivals: deque[float] = deque(maxlen=30)

    def put(self, rgb: np.ndarray) -> None:
        with self.cond:
            self.rgb, self.stamp, self.seq = rgb, time.time(), self.seq + 1
            self._arrivals.append(self.stamp)
            self.cond.notify_all()

    def get(self):
        with self.cond:
            return self.rgb, self.stamp, self.seq

    def fps(self) -> float:
        with self.cond:
            a = list(self._arrivals)
        return (len(a) - 1) / (a[-1] - a[0]) if len(a) > 1 and a[-1] > a[0] else 0.0

    def jpeg(self, after_seq: int = -1, timeout_s: float = 2.0):
        """(jpeg_bytes, seq) for the newest frame newer than after_seq, or (None, after_seq)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > after_seq and self.rgb is not None, timeout_s):
                return None, after_seq
            rgb, seq = self.rgb, self.seq
            if self._jpeg[0] == seq:
                return self._jpeg[1], seq
        data = self.encoder.encode(rgb)
        with self.cond:
            if seq >= self._jpeg[0]:
                self._jpeg = (seq, data)
        return data, seq


def make_handler(latest: Latest, topic: str, max_stream_fps: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            if not latest.encoder.available:
                return self._send(501, b"no JPEG encoder on the bridge's python; use /frame or /frame.bmp",
                                  "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            seq, min_dt, last = -1, 1.0 / max_stream_fps if max_stream_fps > 0 else 0.0, 0.0
            try:
                while True:
                    jpeg, seq = latest.jpeg(after_seq=seq, timeout_s=2.0)
                    if jpeg is None:
                        continue
                    wait = min_dt - (time.time() - last)
                    if wait > 0:
                        time.sleep(wait)
                    self.wfile.write(mjpeg_part(jpeg))
                    self.wfile.flush()
                    last = time.time()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            rgb, stamp, seq = latest.get()
            age_ms = int((time.time() - stamp) * 1000) if stamp else -1
            if path == "/health":
                body = {"ok": rgb is not None and 0 <= age_ms < 2000, "topic": topic, "frames": seq,
                        "width": None if rgb is None else rgb.shape[1],
                        "height": None if rgb is None else rgb.shape[0], "age_ms": age_ms,
                        "fps": round(latest.fps(), 1), "jpeg": latest.encoder.name}
                return self._send(200, json.dumps(body).encode(), "application/json")
            if path in ("/video", "/stream.mjpg", "/stream.mjpeg"):
                return self._stream()
            if path in ("/", ""):
                return self._send(200, b"quack camera bridge: /frame /frame.jpg /video /frame.bmp /health",
                                  "text/plain")
            if rgb is None:
                return self._send(503, b"no frame yet", "text/plain")
            hdr = {"X-Width": rgb.shape[1], "X-Height": rgb.shape[0], "X-Seq": seq, "X-Stamp": f"{stamp:.3f}",
                   "X-Age-Ms": age_ms}
            if path == "/frame":
                return self._send(200, rgb.tobytes(), "application/octet-stream", hdr)
            if path == "/frame.bmp":
                return self._send(200, to_bmp(rgb), "image/bmp", hdr)
            if path == "/frame.jpg":
                if not latest.encoder.available:
                    return self._send(501, b"no JPEG encoder; use /frame.bmp", "text/plain")
                jpeg, _ = latest.jpeg(after_seq=seq - 1, timeout_s=0.5)
                return self._send(200, jpeg or b"", "image/jpeg", hdr)
            self._send(404, b"try /frame, /frame.jpg, /video, /frame.bmp or /health", "text/plain")

    return Handler


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True          # streaming clients must not keep the process alive
    allow_reuse_address = True


# ----------------------------------------------------------------------------- gz glue
def find_camera_topic(node, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            topics = list(node.topic_list())
        except Exception:
            topics = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True).stdout.split()
        cams = [t for t in topics if t.endswith("/image") and "/sensor/" in t and "depth" not in t]
        if cams:
            return sorted(cams)[0]
        time.sleep(1.0)
    raise SystemExit("[bridge] no camera image topic found. Is the sim running with a camera drone "
                     "(x500_quack)?")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default=None)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let other machines (a Jetson) connect")
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--width", type=int, default=640, help="approximate output width (integer downscale)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality for /frame.jpg and /video")
    ap.add_argument("--max-stream-fps", type=float, default=30.0)
    ap.add_argument("--wait", type=float, default=120.0, help="seconds to wait for a camera topic")
    args = ap.parse_args()

    try:
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node
    except ImportError as e:  # pragma: no cover - environment specific
        sys.exit(f"[bridge] Gazebo Python bindings not importable ({e}).\n"
                 "  macOS: run with PX4-Autopilot/Tools/setup/macos/.pixi/envs/default/bin/python3\n"
                 "  Ubuntu: sudo apt install python3-gz-transport13 python3-gz-msgs10, use /usr/bin/python3\n"
                 "  (sim/scripts/run_camera_bridge.sh picks the right one)")

    encoder = JpegEncoder(args.quality)
    latest = Latest(encoder)
    node = Node()
    topic = args.topic or find_camera_topic(node, args.wait)

    def on_image(msg):
        try:
            rgb = to_rgb(msg.width, msg.height, msg.pixel_format_type, msg.step, msg.data, args.width)
        except Exception as e:
            print(f"[bridge] frame dropped: {e}", flush=True)
            return
        latest.put(rgb)

    if not node.subscribe(Image, topic, on_image):
        sys.exit(f"[bridge] subscribe failed for {topic}")
    print(f"[bridge] subscribed to {topic}", flush=True)
    print(f"[bridge] JPEG encoder: {encoder.name or 'none (/frame.jpg and /video disabled)'}", flush=True)
    server = QuietServer((args.host, args.port), make_handler(latest, topic, args.max_stream_fps))
    print(f"[bridge] serving http://{args.host}:{args.port}/  (/frame /frame.jpg /video /frame.bmp /health)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
