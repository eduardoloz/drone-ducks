import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from jetson import sources as S


# ---------------------------------------------------------------- fakes / fixtures
class FakeCamera(S.FrameSource):
    """Produces numbered frames at `fps`; frame k is filled with value k % 256."""

    def __init__(self, fps=200.0, n=None, fail_first=0):
        self.fps, self.n, self.fail_first = fps, n, fail_first
        self.name = "fake"
        self.k = 0
        self.opens = 0
        self.live = n is None

    def open(self):
        self.opens += 1

    def read(self):
        time.sleep(1.0 / self.fps)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise IOError("camera hiccup")
        if self.n is not None and self.k >= self.n:
            self.exhausted = True
            return False, None, time.monotonic()
        self.k += 1
        return True, np.full((4, 4, 3), self.k % 256, np.uint8), time.monotonic()


def _jpeg(color_bgr, size=(48, 64)):
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:] = color_bgr
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def http_server():
    """Local server that mimics the sim camera bridge, a snapshot URL and an MJPEG stream."""
    state = {"seq": 0, "repeat_seq": False}
    red_rgb = np.zeros((6, 8, 3), np.uint8)
    red_rgb[:, :, 0] = 255  # pure red in RGB

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/frame"):
                if not state["repeat_seq"]:
                    state["seq"] += 1
                body = red_rgb.tobytes()
                self.send_response(200)
                for k, v in (("X-Width", 8), ("X-Height", 6), ("X-Seq", state["seq"]), ("X-Age-Ms", 40),
                             ("Content-Length", len(body))):
                    self.send_header(k, str(v))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/shot.jpg"):
                body = _jpeg((0, 255, 0))
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/video"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    for i in range(30):
                        jpg = _jpeg((i * 8 % 256, 0, 0))
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()
    srv.server_close()


# ---------------------------------------------------------------- open_source parsing
@pytest.mark.parametrize("spec,cls", [
    ("0", S.OpenCVSource),
    ("/dev/video2", S.OpenCVSource),
    ("csi://0", S.GStreamerSource),
    ("csi://1?width=1280&height=720&fps=60&out=640x360&flip=2", S.GStreamerSource),
    ("csi:0", S.GStreamerSource),
    ("gst:videotestsrc ! videoconvert ! appsink", S.GStreamerSource),
    ("videotestsrc ! appsink", S.GStreamerSource),
    ("jetson-utils:csi://0", S.JetsonUtilsSource),
    ("sim", S.HttpSnapshotSource),
    ("sim://10.0.0.5:5601", S.HttpSnapshotSource),
    ("http://10.18.237.58:8080/video", S.MjpegStreamSource),
    ("http://10.18.237.58:8080/shot.jpg", S.HttpSnapshotSource),
    ("http://127.0.0.1:5601/frame", S.HttpSnapshotSource),
    ("rtsp://cam/stream", S.OpenCVSource),
    ("tcp://127.0.0.1:5000", S.MjpegStreamSource),
    ("opencv:http://cam/video", S.OpenCVSource),
    ("photo.jpg", S.ImageSource),
])
def test_open_source_picks_the_right_class(spec, cls):
    assert type(S.open_source(spec)) is cls


def test_open_source_details():
    assert S.open_source("sim").url == "http://127.0.0.1:5601/frame"
    assert S.open_source("sim://10.0.0.5:5601").url == "http://10.0.0.5:5601/frame"
    src = S.open_source("csi://1?width=1280&height=720&fps=60&out=640x360&flip=2")
    p = src.target
    assert "sensor-id=1" in p and "width=(int)1280" in p and "framerate=(fraction)60/1" in p
    assert "width=(int)640, height=(int)360" in p and "flip-method=2" in p
    assert p.rstrip().endswith("appsink drop=true max-buffers=1 sync=false")
    with pytest.raises(S.SourceError):
        S.open_source("no/such/file/or/scheme")
    with pytest.raises(S.SourceError):
        S.open_source("missing_video.mp4")


def test_csi_pipeline_defaults_use_full_fov_mode():
    p = S.csi_pipeline()
    assert "nvarguscamerasrc sensor-id=0" in p and "width=(int)1640, height=(int)1232" in p
    assert "format=(string)BGR" in p


def test_gstreamer_source_explains_missing_gstreamer():
    if S.opencv_has_gstreamer():
        pytest.skip("this OpenCV has GStreamer")
    with pytest.raises(S.SourceError, match="without GStreamer"):
        S.GStreamerSource(S.csi_pipeline()).open()


# ---------------------------------------------------------------- HTTP sources
def test_sim_bridge_raw_rgb_becomes_bgr_with_age_corrected_stamp(http_server):
    base, _ = http_server
    src = S.HttpSnapshotSource(base + "/frame", max_fps=0)
    before = time.monotonic()
    ok, img, stamp = src.read()
    assert ok and img.shape == (6, 8, 3)
    assert tuple(img[0, 0]) == (0, 0, 255)            # red, now in BGR order
    assert stamp < time.monotonic() - 0.03 and stamp > before - 0.1  # 40 ms X-Age-Ms applied


def test_sim_bridge_repeated_seq_is_not_a_new_frame(http_server):
    base, state = http_server
    src = S.HttpSnapshotSource(base + "/frame", timeout=0.3, max_fps=100)
    assert src.read()[0]
    state["repeat_seq"] = True
    t0 = time.monotonic()
    ok, img, _ = src.read()
    assert not ok and img is None and time.monotonic() - t0 >= 0.25


def test_jpeg_snapshot(http_server):
    base, _ = http_server
    ok, img, _ = S.open_source(base + "/shot.jpg").read()
    assert ok and img.shape == (48, 64, 3) and img[0, 0, 1] > 200


def test_mjpeg_stream_decodes_frames(http_server):
    base, _ = http_server
    src = S.open_source(base + "/video")
    src.open()
    try:
        frames = [src.read() for _ in range(5)]
    finally:
        src.close()
    assert all(ok for ok, _, _ in frames)
    blues = [int(img[0, 0, 0]) for _, img, _ in frames]
    assert blues == sorted(blues)                       # newer frames come later


def test_raw_mjpeg_over_tcp():
    """What a `gst-launch ... nvjpegenc ! multipartmux ! tcpserversink` pipeline would serve."""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        try:
            for i in range(10):
                conn.sendall(b"--ThisRandomString\r\nContent-Type: image/jpeg\r\n\r\n" + _jpeg((0, 0, i * 20)))
                time.sleep(0.01)
        except OSError:
            pass
        finally:
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    src = S.open_source(f"tcp://127.0.0.1:{port}")
    src.open()
    try:
        ok, img, _ = src.read()
        assert ok and img.shape == (48, 64, 3)
    finally:
        src.close()
        srv.close()
    assert "nvjpegenc" in S.csi_mjpeg_server_cmd() and "tcpserversink" in S.csi_mjpeg_server_cmd()


def test_mjpeg_bad_url_has_helpful_error():
    with pytest.raises(S.SourceError, match="IP Webcam"):
        S.MjpegStreamSource("http://127.0.0.1:9/video", timeout=0.5).open()


# ---------------------------------------------------------------- file sources
def test_image_source_repeats_then_stops(media_image):
    src = S.ImageSource(media_image, count=3)
    src.open()
    oks = [src.read()[0] for _ in range(4)]
    assert oks == [True, True, True, False] and src.exhausted


def test_video_file_reads_every_frame_then_is_exhausted(tmp_path):
    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 48))
    for i in range(10):
        vw.write(np.full((48, 64, 3), i * 20, np.uint8))
    vw.release()
    src = S.open_source(path)
    assert isinstance(src, S.VideoFileSource) and not src.live
    src.open()
    n = 0
    while src.read()[0]:
        n += 1
    assert n == 10 and src.exhausted
    rt = S.VideoFileSource(path, realtime=True)        # paced at 20 fps like a camera
    rt.open()
    t0 = time.monotonic()
    for _ in range(5):
        assert rt.read()[0]
    assert time.monotonic() - t0 >= 0.15


# ---------------------------------------------------------------- LatestFrameReader
def test_reader_hands_out_newest_frames_and_drops_stale_ones():
    with S.LatestFrameReader(FakeCamera(fps=300)) as reader:
        last, seqs = 0, []
        for _ in range(8):
            f = reader.get(timeout=1.0, newer_than=last)
            assert f is not None and f.seq > last
            seqs.append(f.seq)
            last = f.seq
            time.sleep(0.03)  # slow consumer, ~10 frames arrive meanwhile
        assert reader.stats.dropped > 0
        assert all(b - a > 1 for a, b in zip(seqs[1:], seqs[2:]))  # we skipped stale frames
        f = reader.get(timeout=0.5)
        assert f.image[0, 0, 0] == f.seq % 256  # frame content matches its number


def test_reader_waits_for_a_new_frame_and_times_out():
    with S.LatestFrameReader(FakeCamera(fps=5)) as reader:
        f = reader.get(timeout=1.0)
        assert f is not None
        t0 = time.monotonic()
        assert reader.get(timeout=0.05, newer_than=f.seq) is None
        assert time.monotonic() - t0 < 0.2


def test_reader_reports_eof_for_finite_sources():
    with S.LatestFrameReader(FakeCamera(fps=500, n=5)) as reader:
        last = 0
        got = 0
        while True:
            f = reader.get(timeout=1.0, newer_than=last)
            if f is None:
                break
            last, got = f.seq, got + 1
        assert reader.eof and 1 <= got <= 5


def test_reader_survives_errors_and_reconnects():
    cam = FakeCamera(fps=500, fail_first=12)
    reader = S.LatestFrameReader(cam, max_failures=5, reconnect_delay_s=0.01).start()
    try:
        assert reader.get(timeout=2.0) is not None
        assert reader.stats.read_errors >= 12 and reader.stats.reconnects >= 2 and cam.opens >= 3
        assert "camera hiccup" in reader.stats.last_error
    finally:
        reader.stop()


def test_reader_open_failure_is_raised_to_caller(tmp_path):
    bogus = tmp_path / "x.jpg"
    bogus.write_bytes(b"not an image")
    with pytest.raises(S.SourceError):
        S.LatestFrameReader(S.ImageSource(str(bogus))).start()
