import math
import os
import threading
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "live_start_index;-1"

import cv2
import numpy as np
from ultralytics import YOLO

STREAM_URL = "https://video.dot.state.mn.us/public/C607.stream/playlist.m3u8"
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
SHOW_WINDOW = True


class RealTimeHLSStream:
    """Reads an HLS stream at its native 1x frame rate without burst fast-forwarding."""

    def __init__(self, url: str):
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.stream_generation = 0  # Increments on reconnect so tracker can reset
        self.fps = 15.0
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _connect(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        detected_fps = cap.get(cv2.CAP_PROP_FPS)
        if detected_fps and not math.isnan(detected_fps) and 1.0 < detected_fps <= 120.0:
            self.fps = detected_fps
        return cap

    def _run(self):
        cap = self._connect()
        frame_interval = 1.0 / self.fps
        next_frame_time = time.perf_counter()

        while not self.stopped:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Stream interrupted or buffering. Reconnecting in 2 seconds...")
                cap.release()
                time.sleep(2)
                cap = self._connect()
                with self.lock:
                    self.stream_generation += 1
                frame_interval = 1.0 / self.fps
                next_frame_time = time.perf_counter()
                continue

            with self.lock:
                self.frame = frame
                self.frame_id += 1

            next_frame_time += frame_interval
            now = time.perf_counter()
            sleep_time = next_frame_time - now

            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1.0:
                next_frame_time = now

        cap.release()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, 0, self.stream_generation
            return self.frame.copy(), self.frame_id, self.stream_generation

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)


class AsyncYOLOTracker:
    """Initializes and runs YOLO + ByteTrack within a dedicated worker thread."""

    def __init__(self, model_path: str, stream: RealTimeHLSStream):
        self.model_path = model_path
        self.stream = stream
        self.lock = threading.Lock()
        self.tracks = []
        self.status = "Loading YOLO model..."
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            model = YOLO(self.model_path)
            # Warm up detection weights only (do not use .track() on a blank image)
            model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            with self.lock:
                self.status = "ByteTrack Active (0 objects)"
        except Exception as exc:
            err_msg = f"YOLO Init Error: {exc}"
            print(err_msg)
            with self.lock:
                self.status = err_msg
            return

        last_id = 0
        last_generation = 0

        while not self.stopped:
            frame, frame_id, generation = self.stream.get_frame()
            if frame is None or frame_id == last_id:
                time.sleep(0.01)
                continue

            # Reset ByteTrack state if the HLS stream dropped and reconnected
            if generation != last_generation:
                if hasattr(model, "predictor") and model.predictor is not None:
                    for tracker in getattr(model.predictor, "trackers", []):
                        tracker.reset()
                last_generation = generation

            last_id = frame_id
            try:
                results = model.track(
                    frame,
                    tracker="bytetrack.yaml",
                    persist=True,
                    classes=TRAFFIC_CLASSES,
                    conf=0.20,
                    imgsz=640,
                    verbose=False,
                )

                parsed = []
                counts = {}
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    xyxys = boxes.xyxy.cpu().tolist()
                    cls_ids = boxes.cls.cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    # boxes.id can be None if no tracks are confirmed yet
                    track_ids = (
                        boxes.id.int().cpu().tolist()
                        if boxes.id is not None
                        else [None] * len(xyxys)
                    )

                    for xyxy, cls_id, conf, track_id in zip(
                        xyxys, cls_ids, confs, track_ids
                    ):
                        label = model.names[int(cls_id)]
                        counts[label] = counts.get(label, 0) + 1
                        parsed.append((*map(int, xyxy), track_id, label, conf))

                    print(f"[Live Frame {frame_id}] Tracked: {counts}")

                with self.lock:
                    self.tracks = parsed
                    self.status = f"ByteTrack Active ({len(parsed)} objects)"
            except Exception as exc:
                print(f"Tracking error on frame {frame_id}: {exc}")
                with self.lock:
                    self.status = f"Tracking Error: {exc}"
                time.sleep(0.1)

    def get_state(self):
        with self.lock:
            return list(self.tracks), self.status

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)


stream = RealTimeHLSStream(STREAM_URL)
detector = AsyncYOLOTracker("yolo26n.pt", stream)

try:
    while True:
        frame, _, _ = stream.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        if SHOW_WINDOW:
            tracks, status = detector.get_state()

            # Draw bounding boxes with persistent Track IDs
            for x1, y1, x2, y2, track_id, label, conf in tracks:
                id_prefix = f"ID {track_id} | " if track_id is not None else ""
                caption = f"{id_prefix}{label} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    caption,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            # Draw top-left HUD status banner
            cv2.rectangle(frame, (8, 8), (380, 38), (0, 0, 0), -1)
            cv2.putText(
                frame,
                status,
                (15, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("MnDOT C607 Live Stream (ByteTrack)", frame)
            wait_ms = max(1, int(1000 / stream.fps))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.05)
finally:
    detector.stop()
    stream.stop()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()