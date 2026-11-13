#!/usr/bin/env python3
"""Build TensorRT engines from our ONNX models, ON THE JETSON (engines are tied to GPU + TensorRT).

    python jetson/scripts/export_trt.py --dry-run          # anywhere: check ONNX files, print commands
    python jetson/scripts/export_trt.py yunet yolox        # on the Jetson: build FP16 engines
    python jetson/scripts/export_trt.py yolox --bench      # ...and time each engine with trtexec

Engines land next to the ONNX files in jetson/models/ with the names the tensorrt backends load
by default (yunet_640_fp16.engine, yolox_nano_416_fp16.engine, ...).

--dry-run never needs TensorRT or a GPU. It checks that each ONNX file exists, reads its input
shape (with onnxruntime or onnx if installed), confirms the shape we will build for is static or
pinned with --shapes, and prints the exact trtexec command. That is everything that can be checked
without a Jetson; whether TensorRT accepts every layer is only known when it runs there.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models")
TRTEXEC_PATHS = ("/usr/src/tensorrt/bin/trtexec", "/usr/local/tensorrt/bin/trtexec")

# name: (onnx file, engine stem, input tensor name, pinned (C, H, W) or None if the ONNX is static)
SPECS: Dict[str, Tuple[str, str, str, Optional[Tuple[int, int, int]]]] = {
    "yunet": ("face_detection_yunet_2023mar.onnx", "yunet_640", "input", None),
    "scrfd": ("scrfd_500m_bnkps.onnx", "scrfd_640x480", "input.1", (3, 480, 640)),
    "yolox": ("yolox_nano.onnx", "yolox_nano_416", "images", None),
}


def find_trtexec(explicit: Optional[str] = None) -> Optional[str]:
    for p in ([explicit] if explicit else []) + list(TRTEXEC_PATHS):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("trtexec")


def trt_major(default: int = 10) -> int:
    """TensorRT major version from the Python bindings (JetPack 6 = 10, JetPack 5 = 8.5, JetPack 4 = 8.2)."""
    try:
        import tensorrt  # noqa: F401

        return int(tensorrt.__version__.split(".")[0])
    except Exception:
        return default


def onnx_input_shape(path: str) -> Optional[Tuple[str, List[object]]]:
    """(input name, shape) via onnxruntime or onnx, or None if neither is installed."""
    try:
        import onnxruntime as ort

        i = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).get_inputs()[0]
        return i.name, list(i.shape)
    except ImportError:
        pass
    try:
        import onnx

        m = onnx.load(path)
        inits = {x.name for x in m.graph.initializer}
        i = [x for x in m.graph.input if x.name not in inits][0]
        dims = [d.dim_value if d.dim_value > 0 else (d.dim_param or "?") for d in i.type.tensor_type.shape.dim]
        return i.name, dims
    except ImportError:
        return None


def build_command(trtexec: str, onnx_path: str, engine_path: str, input_name: str,
                  pinned: Optional[Tuple[int, int, int]], precision: str = "fp16",
                  workspace_mb: int = 1024, major: int = 10) -> List[str]:
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if precision == "fp16":
        cmd.append("--fp16")
    if pinned:
        c, h, w = pinned
        cmd.append(f"--shapes={input_name}:1x{c}x{h}x{w}")
    # --workspace was removed in TensorRT 10; --memPoolSize exists since 8.4. JetPack 4.6 has 8.2.
    cmd.append(f"--memPoolSize=workspace:{workspace_mb}M" if major >= 9 else f"--workspace={workspace_mb}")
    return cmd


def check_shape(name: str, shape: Optional[Tuple[str, List[object]]],
                pinned: Optional[Tuple[int, int, int]]) -> List[str]:
    """Problems with building this ONNX as a fixed-shape engine (empty list = fine)."""
    if shape is None:
        return []
    _, dims = shape
    if len(dims) != 4:
        return [f"{name}: expected a 4-D NCHW input, got {dims}"]
    dynamic = [d for d in dims[1:] if not isinstance(d, int) or d <= 0]
    if dynamic and not pinned:
        return [f"{name}: dynamic input {dims} but no pinned shape; add one to SPECS"]
    if pinned and not dynamic and tuple(dims[1:]) != tuple(pinned):
        return [f"{name}: static input {dims} disagrees with pinned {pinned}"]
    return []


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="*", default=None, help=f"subset of {sorted(SPECS)} (default: all present)")
    ap.add_argument("--dry-run", action="store_true", help="check and print, don't run trtexec")
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--workspace-mb", type=int, default=1024)
    ap.add_argument("--trtexec", default=None, help="path to trtexec (default: JetPack location)")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--bench", action="store_true", help="time each engine with trtexec --loadEngine")
    args = ap.parse_args(argv)

    names = args.models or list(SPECS)
    bad = [n for n in names if n not in SPECS]
    if bad:
        print(f"unknown model(s) {bad}; choose from {sorted(SPECS)}")
        return 2
    trtexec = find_trtexec(args.trtexec)
    if not trtexec and not args.dry_run:
        print("trtexec not found. On a Jetson it is /usr/src/tensorrt/bin/trtexec (package tensorrt / "
              "libnvinfer-bin). Elsewhere use --dry-run.")
        return 2
    major = trt_major()
    problems, built = [], 0
    for n in names:
        onnx_file, stem, input_name, pinned = SPECS[n]
        onnx_path = os.path.join(args.models_dir, onnx_file)
        engine_path = os.path.join(args.models_dir, f"{stem}_{args.precision}.engine")
        if not os.path.isfile(onnx_path):
            msg = f"{n}: {onnx_path} missing (python -m jetson.fetch models{' --all' if n == 'scrfd' else ''})"
            if args.models:
                problems.append(msg)
            print("skip " + msg)
            continue
        shape = onnx_input_shape(onnx_path)
        if shape is not None and shape[0] != input_name:
            problems.append(f"{n}: input is named {shape[0]!r}, SPECS says {input_name!r}")
        problems += check_shape(n, shape, pinned)
        cmd = build_command(trtexec or TRTEXEC_PATHS[0], onnx_path, engine_path, input_name, pinned,
                            args.precision, args.workspace_mb, major)
        print(f"{n}: input {shape[1] if shape else '(not inspected: no onnxruntime/onnx)'}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            problems.append(f"{n}: trtexec failed ({r.returncode})")
            continue
        built += 1
        if args.bench:
            out = subprocess.run([trtexec, f"--loadEngine={engine_path}", "--iterations=200", "--warmUp=500"],
                                 capture_output=True, text=True).stdout
            m = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms", out)
            print(f"  {n}: GPU compute mean {m.group(1) if m else '?'} ms")
    for p in problems:
        print("PROBLEM: " + p)
    if not args.dry_run:
        print(f"built {built} engine(s) in {args.models_dir}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
