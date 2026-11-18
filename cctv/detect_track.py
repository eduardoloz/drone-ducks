"""Vehicle detection + tracking over a recorded clip (YOLO + ByteTrack).

This is Stephen's ``ByteTrackFinal.py`` idea (``model.track(..., persist=True,
tracker="bytetrack.yaml")``) adapted for offline CCTV clips:

- frames come from a file with their real timestamps (``ingest.iter_frames``)
- only vehicle classes are kept (COCO: car, motorcycle, bus, truck)
- results are saved as JSON lines, so the slow part runs once and the speed
  maths can be re-run instantly while you tune it

Needs ``requirements-detect.txt``. On the 2019 Intel MacBook, yolo11n at
imgsz 640 runs at roughly 10-20 frames/s on CPU (see README for measured numbers).

    python -m cctv.detect_track --clip cctv/data/clips/R10_041_....ts --out tracks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cctv.ingest import iter_frames
from cctv.registry import DATA_DIR

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids
MODELS_DIR = DATA_DIR / "models"
DEFAULT_MODEL = "yolo11n.pt"


def cpu_threads(requested: int | None = None) -> int | None:
    """Torch CPU threads to use: ``requested``, else $CCTV_THREADS, else $OMP_NUM_THREADS, else None (library default).

    Needed because ultralytics' ``select_device("cpu")`` calls ``torch.set_num_threads(min(8, cpus - 1))``
    on the first predict, silently overriding OMP_NUM_THREADS (measured 2026-11-17: 613% CPU with
    OMP_NUM_THREADS=2). On a shared laptop that matters.
    """
    for v in (requested, os.environ.get("CCTV_THREADS"), os.environ.get("OMP_NUM_THREADS")):
        if v not in (None, ""):
            n = int(v)
            if n > 0:
                return n
    return None


def apply_thread_cap(n: int | None) -> None:
    """Cap both torch's and OpenCV's CPU thread pools to ``n`` (None: leave library defaults).

    Torch alone was not enough: with torch at 2 threads the tracker still sampled at ~250% CPU
    (2026-11-17), because OpenCV (decode, letterbox resize) runs its own pool sized to all cores.
    """
    if not n:
        return
    try:
        import cv2

        cv2.setNumThreads(n)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


@dataclass
class TrackPoint:
    frame: int
    t: float          # seconds from first frame (PTS-based)
    track_id: int
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


def load_model(name: str = DEFAULT_MODEL):
    """Load YOLO weights from data/models/, downloading once if missing.

    When the weights are already on disk we set YOLO_OFFLINE=1 so ultralytics
    makes no network calls (no update checks, no usage analytics).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    if path.exists():
        os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO

    if path.exists():
        return YOLO(str(path))
    cwd = os.getcwd()
    try:  # ultralytics downloads known weights into the working directory
        os.chdir(MODELS_DIR)
        model = YOLO(name)
    finally:
        os.chdir(cwd)
    return model


def run_tracker(clip: Path | str, model_name: str = DEFAULT_MODEL, imgsz: int = 640,
                conf: float = 0.25, tracker: str = "bytetrack.yaml", max_frames: int | None = None,
                device: str = "cpu", progress_every: int = 100,
                threads: int | None = None) -> tuple[list[TrackPoint], dict]:
    model = load_model(model_name)
    n_threads = cpu_threads(threads)
    apply_thread_cap(n_threads if device == "cpu" else None)
    points: list[TrackPoint] = []
    times = []
    t_start = time.time()
    n = 0
    for idx, t, frame in iter_frames(clip, max_frames):
        t0 = time.perf_counter()
        res = model.track(source=frame, persist=True, tracker=tracker, imgsz=imgsz, conf=conf,
                          classes=list(VEHICLE_CLASSES), verbose=False, device=device)
        times.append(time.perf_counter() - t0)
        n += 1
        if n == 1 and n_threads and device == "cpu":
            apply_thread_cap(n_threads)        # undo ultralytics' reset (see cpu_threads)
        boxes = res[0].boxes
        if boxes is not None and boxes.id is not None:
            for xyxy, tid, c, cf in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(),
                                        boxes.cls.int().tolist(), boxes.conf.tolist()):
                points.append(TrackPoint(idx, round(t, 4), int(tid), VEHICLE_CLASSES.get(c, str(c)),
                                         round(cf, 3), *(round(v, 2) for v in xyxy)))
        if progress_every and n % progress_every == 0:
            print(f"  frame {n}: {np.mean(times[-progress_every:]) * 1000:.0f} ms/frame", flush=True)
    stats = {"frames": n, "wall_s": round(time.time() - t_start, 1),
             "ms_per_frame_mean": round(1000 * float(np.mean(times)), 1) if times else None,
             "ms_per_frame_p90": round(1000 * float(np.percentile(times, 90)), 1) if times else None,
             "model": model_name, "imgsz": imgsz, "conf": conf, "tracker": tracker, "threads": n_threads,
             "n_points": len(points), "n_tracks": len({p.track_id for p in points})}
    return points, stats


def save_jsonl(points: list[TrackPoint], path: Path | str, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        if meta is not None:
            fh.write(json.dumps({"meta": meta}) + "\n")
        for p in points:
            fh.write(json.dumps(asdict(p)) + "\n")


def load_jsonl(path: Path | str) -> tuple[list[TrackPoint], dict]:
    """Read a tracks JSONL (optionally gzipped, ``.gz``) written by ``save_jsonl``."""
    import gzip

    points, meta = [], {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if "meta" in d:
                meta = d["meta"]
            else:
                points.append(TrackPoint(**d))
    return points, meta


def group_tracks(points: list[TrackPoint]) -> dict[int, dict]:
    """track_id -> {t, uv (bottom-centres), boxes, cls (majority vote), conf}."""
    by_id: dict[int, list[TrackPoint]] = defaultdict(list)
    for p in points:
        by_id[p.track_id].append(p)
    out = {}
    for tid, pts in by_id.items():
        pts.sort(key=lambda p: p.t)
        out[tid] = {
            "t": np.array([p.t for p in pts]),
            "frames": np.array([p.frame for p in pts]),
            "uv": np.array([p.bottom_center for p in pts]),
            "boxes": np.array([[p.x1, p.y1, p.x2, p.y2] for p in pts]),
            "cls": Counter(p.cls for p in pts).most_common(1)[0][0],
            "conf": float(np.mean([p.conf for p in pts])),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="JSONL output (default: data/runs/<clip>.tracks.jsonl)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--threads", type=int, help="torch CPU threads (default: $CCTV_THREADS or $OMP_NUM_THREADS)")
    args = ap.parse_args(argv)
    out = args.out or DATA_DIR / "runs" / f"{args.clip.stem}.{Path(args.model).stem}.tracks.jsonl"
    points, stats = run_tracker(args.clip, args.model, args.imgsz, args.conf, max_frames=args.max_frames,
                                threads=args.threads)
    save_jsonl(points, out, meta={"clip": str(args.clip), **stats})
    print(json.dumps(stats, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
