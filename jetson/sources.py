"""Camera sources behind one interface + a reader thread that only ever keeps the newest frame.

Every source implements FrameSource.read() -> (ok, bgr_image, capture_stamp). `open_source(spec)`
turns a command-line string into the right source:

  spec                                   source
  -------------------------------------  ---------------------------------------------------------
  0, 1, ...                              USB webcam by index (OpenCV)
  /dev/video0                            V4L2 device (Linux, OpenCV)
  csi://0  csi://0?width=1640&height=1232&fps=30&out=1280x960&flip=0
                                         Jetson CSI camera via GStreamer nvarguscamerasrc (OpenCV
                                         must be built with GStreamer: JetPack's is, pip's is NOT)
  jetson-utils:csi://0                   CSI/V4L2/RTSP through jetson_utils.videoSource (no OpenCV
                                         GStreamer needed; comes with jetson-inference)
  gst:<pipeline> (or any "a ! b" string) custom GStreamer pipeline ending in appsink
  sim  /  sim://127.0.0.1:5601           Gazebo camera bridge (sim/ track): GET /frame, raw RGB
  http://phone:8080/video                MJPEG stream (Android "IP Webcam" app, like BTMOB.py)
  http://phone:8080/shot.jpg             single-image snapshot URL, polled
  rtsp://...                             RTSP via OpenCV/FFmpeg
  opencv:<anything>                      force plain cv2.VideoCapture(<anything>)
  path/to/video.mp4                      video file (frame by frame; --realtime paces it like a camera)
  path/to/photo.jpg                      still image, repeated

Why LatestFrameReader: cameras and streams buffer frames. If detection takes 80 ms and the
camera produces a frame every 33 ms, reading frames in order means acting on ever older images,
and a drone acting on a 2-second-old picture of a face is dangerous. The reader thread reads as
fast as the camera delivers and overwrites a single slot; the main loop always takes the newest
frame and can wait for one it hasn't seen yet (get(newer_than=seq)). Same idea as BTMOB.py.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

try:  # OpenCV is optional for the HTTP raw-RGB path and for the reader itself
    import cv2
except ImportError:  # pragma: no cover - exercised only on machines without OpenCV
    cv2 = None

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
SIM_DEFAULT_URL = "http://127.0.0.1:5601/frame"
ReadResult = Tuple[bool, Optional[np.ndarray], float]


class SourceError(RuntimeError):
    """A camera could not be opened or read; the message says how to fix it."""


def _need_cv2(what: str) -> None:
    if cv2 is None:
        raise SourceError(f"{what} needs OpenCV: pip install 'opencv-python<4.12' (or use JetPack's)")


@dataclass
class Frame:
    image: np.ndarray   # BGR uint8, H x W x 3
    seq: int            # 1, 2, 3, ... per reader (gaps = frames the reader dropped for you)
    stamp: float        # time.monotonic() at capture, best estimate
    source: str = ""

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class FrameSource:
    """Blocking camera. Subclasses implement open/read/close."""

    name = "source"
    live = True          # live cameras: stale frames should be dropped
    exhausted = False    # finite sources (files) set this when they run out

    def open(self) -> None:
        pass

    def read(self) -> ReadResult:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


# ------------------------------------------------------------------------------------ OpenCV
class OpenCVSource(FrameSource):
    """Anything cv2.VideoCapture can open: webcam index, /dev/videoN, file, RTSP, GStreamer."""

    def __init__(self, target: Union[int, str], api: Optional[int] = None, width: Optional[int] = None,
                 height: Optional[int] = None, fps: Optional[float] = None, mjpeg: bool = False,
                 live: bool = True, name: Optional[str] = None) -> None:
        self.target, self.api = target, api
        self.width, self.height, self.fps, self.mjpeg = width, height, fps, mjpeg
        self.live = live
        self.name = name or str(target)
        self.cap = None

    def open(self) -> None:
        _need_cv2(self.name)
        self.cap = cv2.VideoCapture(self.target) if self.api is None else cv2.VideoCapture(self.target,
                                                                                          self.api)
        if not self.cap.isOpened():
            raise SourceError(f"could not open {self.name!r}. {self._hint()}")
        if self.mjpeg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # many backends ignore it; the reader thread doesn't

    def _hint(self) -> str:
        if isinstance(self.target, int) or str(self.target).startswith("/dev/video"):
            return ("Is the camera plugged in and not used by another program? On Linux check "
                    "`v4l2-ctl --list-devices`; on macOS allow camera access for your terminal.")
        return "Check the path/URL."

    def read(self) -> ReadResult:
        if self.cap is None:
            raise SourceError(f"{self.name} is not open")
        ok, img = self.cap.read()
        stamp = time.monotonic()
        if not ok or img is None:
            if not self.live:
                self.exhausted = True
            return False, None, stamp
        return True, img, stamp

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoFileSource(OpenCVSource):
    """A video file. realtime=True paces reads at the file's FPS so it behaves like a camera."""

    def __init__(self, path: str, realtime: bool = False, loop: bool = False) -> None:
        if not os.path.exists(path):
            raise SourceError(f"video file not found: {path}")
        super().__init__(path, live=realtime, name=os.path.basename(path))
        self.realtime, self.loop = realtime, loop
        self._period = 0.0
        self._next_t = 0.0

    def open(self) -> None:
        super().open()
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._period = 1.0 / fps if (self.realtime and 1.0 <= fps <= 240.0) else 0.0
        self._next_t = time.monotonic()

    def read(self) -> ReadResult:
        if self._period:
            delay = self._next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_t = max(self._next_t + self._period, time.monotonic() - self._period)
        ok, img = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        stamp = time.monotonic()
        if not ok or img is None:
            self.exhausted = True
            return False, None, stamp
        return True, img, stamp


class ImageSource(FrameSource):
    """A still image served repeatedly (count frames, or forever), optionally at a fixed rate."""

    live = False

    def __init__(self, path_or_image: Union[str, np.ndarray], fps: float = 0.0,
                 count: Optional[int] = None, name: Optional[str] = None) -> None:
        self._path = path_or_image if isinstance(path_or_image, str) else None
        self._img = None if self._path else np.ascontiguousarray(path_or_image)
        self.name = name or (os.path.basename(self._path) if self._path else "image")
        self.fps, self.count = fps, count
        self._served = 0
        self._next_t = 0.0

    def open(self) -> None:
        if self._img is None:
            _need_cv2("reading image files")
            img = cv2.imread(self._path)
            if img is None:
                raise SourceError(f"could not read image {self._path!r}")
            self._img = img
        self._served, self._next_t = 0, time.monotonic()

    def read(self) -> ReadResult:
        if self.count is not None and self._served >= self.count:
            self.exhausted = True
            return False, None, time.monotonic()
        if self.fps > 0:
            delay = self._next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_t += 1.0 / self.fps
        self._served += 1
        return True, self._img.copy(), time.monotonic()


# ------------------------------------------------------------------------------------ GStreamer / CSI
def csi_pipeline(sensor_id: int = 0, width: int = 1640, height: int = 1232, fps: int = 30,
                 out_width: Optional[int] = None, out_height: Optional[int] = None,
                 flip: int = 0) -> str:
    """GStreamer pipeline for a Jetson CSI camera (IMX219/IMX477) into an OpenCV appsink.

    width/height/fps must be a sensor mode the driver lists (nvarguscamerasrc prints them at
    start-up). IMX219 on Jetson: 3264x2464@21, 3264x1848@28, 1920x1080@30, 1640x1232@30,
    1280x720@60. Pick a full-field-of-view (binned) mode such as 1640x1232: some modes are crops,
    which silently narrows the HFOV and breaks the distance maths.
    nvvidconv resizes on the VIC hardware for free, so ask for out_width/out_height there.
    appsink drop=true max-buffers=1 keeps GStreamer from queueing stale frames.
    """
    ow, oh = out_width or width, out_height or height
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){fps}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width=(int){ow}, height=(int){oh}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def opencv_has_gstreamer() -> bool:
    if cv2 is None:
        return False
    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer" in line:
            return "YES" in line.upper()
    return False


class GStreamerSource(OpenCVSource):
    def __init__(self, pipeline: str, name: str = "gstreamer") -> None:
        super().__init__(pipeline, api=None, live=True, name=name)

    def open(self) -> None:
        _need_cv2("GStreamer capture")
        if not opencv_has_gstreamer():
            raise SourceError(
                "this OpenCV was built without GStreamer (pip wheels are). On a Jetson use JetPack's "
                "OpenCV (python3 -c 'import cv2; print(cv2.__file__)' should be under /usr/lib), e.g. "
                "a venv made with --system-site-packages, or use the jetson-utils:csi://0 source.")
        self.api = cv2.CAP_GSTREAMER
        super().open()


class JetsonUtilsSource(FrameSource):
    """Capture through jetson_utils.videoSource (csi://0, /dev/video0, rtsp://...). UNVERIFIED on HW."""

    def __init__(self, uri: str = "csi://0", width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.uri, self.width, self.height, self.fps = uri, width, height, fps
        self.name = f"jetson-utils:{uri}"
        self._src = None
        self._ju = None

    def open(self) -> None:
        try:
            import jetson_utils  # noqa: WPS433 - optional, Jetson only
        except ImportError as e:
            raise SourceError("jetson_utils not importable: build/install jetson-inference "
                              "(https://github.com/dusty-nv/jetson-inference) or use its container") from e
        self._ju = jetson_utils
        argv = [f"--input-width={self.width}", f"--input-height={self.height}", f"--input-rate={self.fps}"]
        self._src = jetson_utils.videoSource(self.uri, argv=argv)

    def read(self) -> ReadResult:
        img = self._src.Capture(format="rgb8", timeout=1000)
        stamp = time.monotonic()
        if img is None:
            return False, None, stamp
        self._ju.cudaDeviceSynchronize()
        rgb = self._ju.cudaToNumpy(img)
        return True, np.ascontiguousarray(rgb[:, :, ::-1]), stamp

    def close(self) -> None:
        if self._src is not None:
            self._src.Close()
            self._src = None


# ------------------------------------------------------------------------------------ HTTP
def _decode_image(data: bytes) -> Optional[np.ndarray]:
    _need_cv2("decoding JPEG/PNG")
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class HttpSnapshotSource(FrameSource):
    """Poll a URL that returns one image per GET.

    Understands the sim/ Gazebo camera bridge (GET /frame -> raw RGB bytes with X-Width, X-Height,
    X-Seq, X-Age-Ms headers) and ordinary JPEG/PNG/BMP snapshot URLs (IP Webcam's /shot.jpg).
    For the bridge, the capture stamp is corrected by X-Age-Ms and repeated X-Seq values are
    skipped, so we never feed the detector the same simulated frame twice.
    """

    def __init__(self, url: str, timeout: float = 2.0, max_fps: float = 30.0) -> None:
        self.url, self.timeout = url, timeout
        self.min_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.name = url
        self._last_req = 0.0
        self._last_seq: Optional[str] = None

    def read(self) -> ReadResult:
        deadline = time.monotonic() + self.timeout
        while True:
            wait = self._last_req + self.min_period - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.monotonic()
            with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                headers, data = r.headers, r.read()
            now = time.monotonic()
            if headers.get("X-Width") and headers.get("X-Height"):
                seq = headers.get("X-Seq")
                if seq is not None and seq == self._last_seq:
                    if now > deadline:
                        return False, None, now  # bridge alive but no new frame
                    continue
                self._last_seq = seq
                w, h = int(headers["X-Width"]), int(headers["X-Height"])
                if len(data) != w * h * 3:
                    raise SourceError(f"bridge sent {len(data)} bytes for {w}x{h} RGB")
                bgr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)[:, :, ::-1]
                age_ms = float(headers.get("X-Age-Ms", "0") or 0)
                stamp = now - age_ms / 1000.0 if age_ms > 0 else now
                return True, np.ascontiguousarray(bgr), stamp
            img = _decode_image(data)
            return (img is not None), img, now


class MjpegStreamSource(FrameSource):
    """Read a multipart MJPEG stream over HTTP (IP Webcam's /video, many IP cameras).

    Frames are cut at JPEG start/end markers (FFD8 ... FFD9). If several complete frames are
    already buffered we decode only the newest, the same latest-frame idea as the reader thread.
    """

    def __init__(self, url: str, timeout: float = 5.0, chunk: int = 65536) -> None:
        self.url, self.timeout, self.chunk = url, timeout, chunk
        self.name = url
        self._resp = None
        self._buf = b""

    def open(self) -> None:
        try:
            self._resp = urllib.request.urlopen(self.url, timeout=self.timeout)
        except OSError as e:
            raise SourceError(f"could not open MJPEG stream {self.url}: {e}. Phone and computer on "
                              "the same Wi-Fi? Is the IP Webcam app's server started?") from e
        self._buf = b""

    def read(self) -> ReadResult:
        if self._resp is None:
            raise SourceError("stream not open")
        while True:
            last_end = self._buf.rfind(b"\xff\xd9")
            if last_end != -1:
                start = self._buf.rfind(b"\xff\xd8", 0, last_end)
                if start != -1:
                    jpg = self._buf[start:last_end + 2]
                    self._buf = self._buf[last_end + 2:]
                    img = _decode_image(jpg)
                    return (img is not None), img, time.monotonic()
            data = self._resp.read1(self.chunk) if hasattr(self._resp, "read1") else self._resp.read(4096)
            if not data:
                return False, None, time.monotonic()
            self._buf += data
            if len(self._buf) > 8 * 1024 * 1024:  # never grow without bound on a corrupt stream
                self._buf = self._buf[-1024 * 1024:]

    def close(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            finally:
                self._resp = None


# ------------------------------------------------------------------------------------ factory
def _parse_query(q: str) -> dict:
    return {k: v[-1] for k, v in urllib.parse.parse_qs(q).items()}


def open_source(spec: str, width: Optional[int] = None, height: Optional[int] = None,
                fps: Optional[float] = None, realtime: bool = False, loop: bool = False) -> FrameSource:
    """Build (not open) the FrameSource for a command-line spec. See the module docstring."""
    s = str(spec).strip()
    if s.startswith("opencv:"):
        return OpenCVSource(s[len("opencv:"):], width=width, height=height, fps=fps)
    if s == "sim" or s.startswith("sim://"):
        hostport = s[len("sim://"):] if s.startswith("sim://") else ""
        return HttpSnapshotSource(f"http://{hostport}/frame" if hostport else SIM_DEFAULT_URL)
    if s.isdigit():
        return OpenCVSource(int(s), width=width, height=height, fps=fps, name=f"usb:{s}")
    if s.startswith("/dev/video"):
        api = cv2.CAP_V4L2 if (cv2 is not None and sys.platform.startswith("linux")) else None
        return OpenCVSource(s, api=api, width=width, height=height, fps=fps, mjpeg=True)
    if s.startswith("jetson-utils:"):
        return JetsonUtilsSource(s[len("jetson-utils:"):], width or 1280, height or 720, int(fps or 30))
    if s.startswith("csi:"):
        u = urllib.parse.urlparse(s if "://" in s else s.replace("csi:", "csi://", 1))
        q = _parse_query(u.query)
        sensor = int(u.netloc or u.path.strip("/") or 0)
        out_w = out_h = None
        if "out" in q:
            out_w, out_h = (int(v) for v in q["out"].lower().split("x"))
        pipe = csi_pipeline(sensor, int(q.get("width", width or 1640)), int(q.get("height", height or 1232)),
                            int(q.get("fps", fps or 30)), out_w, out_h, int(q.get("flip", 0)))
        return GStreamerSource(pipe, name=f"csi:{sensor}")
    if s.startswith("gst:") or " ! " in s:
        return GStreamerSource(s[len("gst:"):] if s.startswith("gst:") else s)
    if s.startswith(("http://", "https://")):
        path = urllib.parse.urlparse(s).path.lower()
        if path.endswith(("/frame", "/frame.bmp")) or path.endswith(IMAGE_EXTS) or "shot" in path \
                or "snapshot" in path:
            return HttpSnapshotSource(s)
        return MjpegStreamSource(s)
    if s.startswith(("rtsp://", "rtmp://", "udp://")):
        return OpenCVSource(s, name=s)
    ext = os.path.splitext(s)[1].lower()
    if ext in IMAGE_EXTS:
        return ImageSource(s, fps=fps or (30.0 if realtime else 0.0))
    if ext in VIDEO_EXTS or os.path.isfile(s):
        return VideoFileSource(s, realtime=realtime, loop=loop)
    raise SourceError(f"don't know how to open {spec!r}: expected a webcam index, /dev/videoN, csi://N, "
                      "sim, http(s)://..., rtsp://..., or an existing image/video file")


# ------------------------------------------------------------------------------------ reader thread
@dataclass
class ReaderStats:
    frames: int = 0           # frames read from the source
    consumed: int = 0         # frames handed to get()
    dropped: int = 0          # frames overwritten before anyone took them (that's the point)
    read_errors: int = 0
    reconnects: int = 0
    last_error: str = ""
    fps: float = 0.0          # source frame rate (EMA)
    _last_t: float = field(default=0.0, repr=False)


class LatestFrameReader:
    """Background thread that keeps only the newest frame of a FrameSource.

        reader = LatestFrameReader(open_source("csi://0")).start()
        last = 0
        while True:
            frame = reader.get(timeout=1.0, newer_than=last)   # never the same frame twice
            if frame is None: ...                              # camera stalled or ended
            last = frame.seq
    """

    def __init__(self, source: FrameSource, reconnect: bool = True, reconnect_delay_s: float = 1.0,
                 max_failures: int = 50) -> None:
        self.source = source
        self.reconnect = reconnect
        self.reconnect_delay_s = reconnect_delay_s
        self.max_failures = max_failures
        self.stats = ReaderStats()
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._taken = True
        self._seq = 0
        self._eof = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "LatestFrameReader":
        self.source.open()  # open here so a bad path/URL fails loudly in the caller
        self._thread = threading.Thread(target=self._run, name=f"reader:{self.source.name}", daemon=True)
        self._thread.start()
        return self

    @property
    def eof(self) -> bool:
        return self._eof

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                ok, img, stamp = self.source.read()
            except Exception as e:  # network hiccup, unplugged camera...
                ok, img, stamp = False, None, time.monotonic()
                self.stats.last_error = f"{type(e).__name__}: {e}"
            if not ok or img is None:
                self.stats.read_errors += 1
                if self.source.exhausted:
                    break
                failures += 1
                if self.reconnect and failures >= self.max_failures:
                    failures = 0
                    self._reopen()
                else:
                    self._stop.wait(0.01)
                continue
            failures = 0
            now = time.monotonic()
            with self._cond:
                if not self._taken:
                    self.stats.dropped += 1
                self._seq += 1
                self._frame = Frame(img, self._seq, stamp, self.source.name)
                self._taken = False
                self.stats.frames += 1
                if self.stats._last_t:
                    inst = 1.0 / max(1e-6, now - self.stats._last_t)
                    self.stats.fps = inst if self.stats.fps == 0 else 0.9 * self.stats.fps + 0.1 * inst
                self.stats._last_t = now
                self._cond.notify_all()
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def _reopen(self) -> None:
        log.warning("camera %s: too many failed reads (%s), reconnecting", self.source.name,
                    self.stats.last_error or "no frame")
        try:
            self.source.close()
        except Exception:
            pass
        self._stop.wait(self.reconnect_delay_s)
        try:
            self.source.open()
            self.stats.reconnects += 1
        except Exception as e:
            self.stats.last_error = f"reconnect failed: {e}"

    def get(self, timeout: float = 1.0, newer_than: Optional[int] = None) -> Optional[Frame]:
        """Newest frame, waiting up to `timeout` s for one with seq > newer_than. None on timeout/EOF."""
        deadline = time.monotonic() + timeout

        def fresh() -> bool:
            return self._frame is not None and (newer_than is None or self._frame.seq > newer_than)

        with self._cond:
            while not fresh():
                remaining = deadline - time.monotonic()
                if self._eof or remaining <= 0:
                    return None
                self._cond.wait(remaining)
            self._taken = True
            self.stats.consumed += 1
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.source.close()
        except Exception:
            pass

    def __enter__(self) -> "LatestFrameReader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
