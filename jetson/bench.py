"""Benchmark face-detector latency per backend and input width.

    python -m jetson.bench                          # yunet, haar, onnx:yunet, onnx:scrfd at 320/640/1280
    python -m jetson.bench --backends yunet --widths 640 --frames 200
    python -m jetson.bench --save                   # also write bench_results/<machine>_<date>.md

Numbers are per detect() call on a real photo resized to each width (pre-processing and
post-processing included, drawing excluded). They are labelled with the machine they ran on: a
laptop number is NOT a Jetson number. The machine's load average is recorded before and after,
because other programs sharing the CPU make results noisy.

Backend spec syntax: name[:model][@device], e.g. onnx:scrfd, onnx:yunet@tensorrt, yunet@cuda.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

from .cli import percentile
from .detectors import BackendUnavailable, create_detector

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "media", "nasa_STS064-S-002.jpg")
RESULTS_DIR = os.path.join(HERE, "bench_results")


def machine_label() -> str:
    """'Jetson Orin Nano ...' on a Jetson, otherwise e.g. 'Intel Mac CPU, not Jetson'."""
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().strip("\x00\n ")
        if "jetson" in model.lower():
            return model
    except OSError:
        pass
    if sys.platform == "darwin":
        arch = platform.machine()
        return ("Intel Mac CPU" if arch == "x86_64" else "Apple Silicon Mac CPU") + ", not Jetson"
    return f"{platform.system()} {platform.machine()} CPU, not Jetson"


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], stdout=subprocess.PIPE,
                                  universal_newlines=True, timeout=5).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def load_avg() -> str:
    try:
        return " / ".join(f"{x:.2f}" for x in os.getloadavg())
    except OSError:
        return "n/a"


def parse_spec(spec: str):
    """'onnx:scrfd@tensorrt' -> ('onnx', 'scrfd', 'tensorrt')."""
    device = None
    if "@" in spec:
        spec, device = spec.split("@", 1)
    name, model = (spec.split(":", 1) + [None])[:2]
    return name, model, device


def make(spec: str, width: int):
    name, model, device = parse_spec(spec)
    kw: Dict[str, object] = {"input_width": width}
    if model:
        kw["model"] = model
    if device:
        kw["device"] = device
    return create_detector(name, **kw)


def bench_one(spec: str, image, width: int, frames: int, warmup: int) -> Optional[dict]:
    import cv2

    h, w = image.shape[:2]
    frame = cv2.resize(image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    try:
        det = make(spec, width)
    except (BackendUnavailable, ValueError) as e:  # missing library/model, or a typo in the name
        return {"spec": spec, "width": width, "error": str(e)}
    for _ in range(warmup):
        det.detect(frame)
    times, counts = [], []
    for _ in range(frames):
        t0 = time.perf_counter()
        faces = det.detect(frame)
        times.append((time.perf_counter() - t0) * 1e3)
        counts.append(len(faces))
    info = det.info()
    det.close()
    mean = statistics.mean(times)
    return {"spec": spec, "width": width, "frame": f"{frame.shape[1]}x{frame.shape[0]}",
            "input": info.get("input", f"{info.get('input_width', width)} wide"), "mean": mean,
            "p50": percentile(times, 0.5), "p90": percentile(times, 0.9), "fps": 1000.0 / mean,
            "faces": int(statistics.median(counts))}


def table(rows: List[dict]) -> str:
    out = ["| backend | frame | network input | mean ms | p50 ms | p90 ms | FPS | faces |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['spec']} | {r['width']} | unavailable: {r['error'][:60]} | | | | | |")
        else:
            out.append(f"| {r['spec']} | {r['frame']} | {r['input']} | {r['mean']:.1f} | {r['p50']:.1f} | "
                       f"{r['p90']:.1f} | {r['fps']:.0f} | {r['faces']} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="yunet,haar,onnx:yunet,onnx:scrfd")
    ap.add_argument("--widths", default="320,640,1280")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="photo with faces (python -m jetson.fetch media)")
    ap.add_argument("--threads", type=int, default=0, help="cv2.setNumThreads (0 = OpenCV default)")
    ap.add_argument("--save", action="store_true", help="write a markdown report to jetson/bench_results/")
    args = ap.parse_args(argv)

    import cv2
    image = cv2.imread(args.image)
    if image is None:
        print(f"cannot read {args.image}; run `python -m jetson.fetch media` first")
        return 2
    if args.threads:
        cv2.setNumThreads(args.threads)
    label = machine_label()
    load_before = load_avg()
    header = [f"Machine: **{label}** ({cpu_name()}, {os.cpu_count()} logical CPUs)",
              f"Python {platform.python_version()}, OpenCV {cv2.__version__} ({cv2.getNumThreads()} threads)"]
    try:
        import onnxruntime as ort
        header[-1] += f", onnxruntime {ort.__version__}"
    except ImportError:
        pass
    header.append(f"Image: {os.path.basename(args.image)}; {args.frames} timed runs after {args.warmup} warm-up")
    header.append(f"Load average (1/5/15 min) before: {load_before}")
    print("\n".join(header) + "\n")

    rows = []
    for spec in [s.strip() for s in args.backends.split(",") if s.strip()]:
        for width in [int(w) for w in args.widths.split(",")]:
            r = bench_one(spec, image, width, args.frames, args.warmup)
            rows.append(r)
            if "error" in r:
                print(f"{spec:14s} {width:5d}  unavailable: {r['error'][:90]}")
            else:
                print(f"{spec:14s} {r['frame']:>10s}  mean {r['mean']:7.1f} ms  p90 {r['p90']:7.1f} ms  "
                      f"{r['fps']:6.1f} FPS  faces {r['faces']}")
    load_after = load_avg()
    print(f"\nLoad average after: {load_after}")
    report = "\n".join(header) + f"\nLoad average after: {load_after}\n\n" + table(rows) + "\n"
    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        slug = "".join(ch if ch.isalnum() else "_" for ch in label.split(",")[0].lower()).strip("_")
        suffix = f"_{args.threads}threads" if args.threads else ""
        path = os.path.join(RESULTS_DIR, f"{slug}_{dt.date.today().isoformat()}{suffix}.md")
        with open(path, "w") as f:
            f.write(f"# Face detector benchmark: {label}\n\n_Generated by `python -m jetson.bench` on "
                    f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}. Laptop numbers are not Jetson numbers._\n\n"
                    + report)
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
