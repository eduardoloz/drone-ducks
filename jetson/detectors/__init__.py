"""Pluggable face detectors. All share FaceDetector.detect(frame_bgr) -> list[Face].

    from jetson.detectors import create_detector
    det = create_detector("yunet", input_width=640)
    faces = det.detect(frame)

Backends (heavy libraries are imported only when you create one):

  name              runs on                  needs                                    status
  ----------------  -----------------------  ---------------------------------------  -------------------
  yunet (default)   any CPU; CUDA if OpenCV  OpenCV >= 4.8 + YuNet ONNX (~230 KB)       tested on Mac
                    was built with it
  haar              any CPU                  OpenCV                                   tested on Mac
  onnx              CPU / CUDA / TensorRT    onnxruntime(-gpu) + YuNet or SCRFD ONNX  tested on Mac (CPU)
  tensorrt          Jetson GPU               tensorrt + pycuda + engine built there   untested (no HW)
  jetson_inference  Jetson GPU               jetson-inference (container or build)    untested (no HW)
  ultralytics       CPU / GPU / TensorRT     ultralytics (+torch) + yolov8n-face      optional

`python -m jetson.doctor` lists which of these work on the current machine.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

from .base import BackendUnavailable, FaceDetector

BACKENDS: Dict[str, Tuple[str, str]] = {
    "yunet": ("jetson.detectors.yunet", "YuNetDetector"),
    "haar": ("jetson.detectors.haar", "HaarDetector"),
    "onnx": ("jetson.detectors.onnxrt", "OnnxFaceDetector"),
    "tensorrt": ("jetson.detectors.trt", "TensorRTFaceDetector"),
    "jetson_inference": ("jetson.detectors.jetson_infer", "JetsonInferenceDetector"),
    "ultralytics": ("jetson.detectors.yolo_face", "UltralyticsFaceDetector"),
}


__all__ = ["BACKENDS", "BackendUnavailable", "FaceDetector", "create_detector", "probe_backends"]


def create_detector(name: str, **kwargs) -> FaceDetector:
    """Instantiate a backend by name. Raises BackendUnavailable with an install hint if it can't run."""
    if name not in BACKENDS:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(BACKENDS)}")
    module, cls = BACKENDS[name]
    return getattr(importlib.import_module(module), cls)(**kwargs)


def probe_backends() -> List[Tuple[str, bool, str]]:
    """(name, works, detail) for every backend, constructing each with default settings."""
    out = []
    for name in BACKENDS:
        try:
            det = create_detector(name)
            out.append((name, True, ", ".join(f"{k}={v}" for k, v in det.info().items())))
            det.close()
        except BackendUnavailable as e:
            out.append((name, False, str(e)))
        except Exception as e:  # a broken install should not crash the doctor
            out.append((name, False, f"{type(e).__name__}: {e}"))
    return out
