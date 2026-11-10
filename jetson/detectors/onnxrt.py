"""YuNet or SCRFD through ONNX Runtime: CPU on a laptop, CUDA or TensorRT on a Jetson.

On a Jetson this is the easiest way to get TensorRT speed: install NVIDIA's onnxruntime-gpu
wheel for your JetPack and pass device="tensorrt". ONNX Runtime builds a TensorRT engine on the
first run (slow: a minute or more) and caches it in jetson/models/trt_cache/, so later starts are fast.

    det = OnnxFaceDetector("yunet")                  # opencv_zoo YuNet, MIT licence, fixed 640x640
    det = OnnxFaceDetector("scrfd", device="cuda")   # InsightFace SCRFD-500M (non-commercial weights)

The model family is detected from the ONNX outputs, so any YuNet/SCRFD export works via a path.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..face import Face
from . import decode
from .base import MODELS_DIR, BackendUnavailable, FaceDetector, model_path, require_file

MODELS = {
    "yunet": "face_detection_yunet_2023mar.onnx",
    "scrfd": "scrfd_500m_bnkps.onnx",
}
DEVICES = ("cpu", "cuda", "tensorrt", "coreml")


def providers_for(device: str, available: List[str], fp16: bool = True,
                  cache_dir: Optional[str] = None) -> list:
    """ONNX Runtime execution providers for a device name, best first, CPU always last."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}")
    wanted = {"cpu": [], "cuda": ["CUDAExecutionProvider"], "coreml": ["CoreMLExecutionProvider"],
              "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"]}[device]
    missing = [p for p in wanted if p not in available]
    if missing:
        raise BackendUnavailable(
            f"device={device!r} needs {missing} but this onnxruntime only has {available}. On a Jetson "
            "install NVIDIA's onnxruntime-gpu wheel for your JetPack (see docs/BRINGUP_CHECKLIST.md).")
    out = []
    for p in wanted:
        if p == "TensorrtExecutionProvider":
            cache = cache_dir or os.path.join(MODELS_DIR, "trt_cache")
            os.makedirs(cache, exist_ok=True)
            out.append((p, {"trt_fp16_enable": bool(fp16), "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": cache}))
        else:
            out.append(p)
    out.append("CPUExecutionProvider")
    return out


class OnnxFaceDetector(FaceDetector):
    name = "onnx"

    def __init__(self, model: str = "yunet", device: str = "cpu", input_width: int = 640,
                 score_threshold: float = 0.6, nms_threshold: Optional[float] = None, top_k: int = 50,
                 fp16: bool = True, threads: int = 0) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise BackendUnavailable("onnxruntime not installed: pip install onnxruntime (laptop) or "
                                     "NVIDIA's onnxruntime-gpu wheel (Jetson)") from e
        self._ort = ort
        path = model_path(MODELS[model]) if model in MODELS else model
        self.model = require_file(path, f"ONNX model {model!r}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # SCRFD's ONNX declares 640x640 output shapes; other sizes warn
        if threads:
            opts.intra_op_num_threads = threads
        self.device = device
        self.session = ort.InferenceSession(self.model, sess_options=opts,
                                            providers=providers_for(device, ort.get_available_providers(), fp16))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.family = "yunet" if any(n.startswith("cls_") for n in self.output_names) else "scrfd"
        self.score_threshold, self.top_k = score_threshold, top_k
        self.nms_threshold = nms_threshold if nms_threshold is not None else (0.3 if self.family == "yunet"
                                                                               else 0.4)
        h, w = inp.shape[2], inp.shape[3]
        self.fixed_size: Optional[Tuple[int, int]] = (w, h) if isinstance(w, int) and isinstance(h, int) else None
        self.input_width = decode.round_up(input_width) if not self.fixed_size else self.fixed_size[0]
        self._size: Optional[Tuple[int, int]] = self.fixed_size

    def _input_size(self, frame: np.ndarray) -> Tuple[int, int]:
        if self.fixed_size:
            return self.fixed_size
        if self._size is None:  # dynamic model: keep the frame's aspect ratio, multiples of 32
            h, w = frame.shape[:2]
            self._size = (self.input_width, decode.round_up(self.input_width * h / float(w)))
        return self._size

    def detect(self, frame: np.ndarray) -> List[Face]:
        iw, ih = self._input_size(frame)
        canvas, scale = decode.letterbox(frame, (iw, ih))
        if self.family == "yunet":
            outs = self.session.run(self.output_names, {self.input_name: decode.yunet_blob(canvas)})
            named = dict(zip(self.output_names, outs))
            return decode.decode_yunet(named, iw, ih, self.score_threshold, scale, self.nms_threshold, self.top_k)
        outs = self.session.run(self.output_names, {self.input_name: decode.scrfd_blob(canvas)})
        return decode.decode_scrfd(outs, iw, ih, self.score_threshold, scale, self.nms_threshold, self.top_k)

    def info(self) -> Dict[str, object]:
        size = self._size or self.fixed_size
        return {"backend": f"{self.name}:{self.family}", "model": os.path.basename(self.model),
                "input": f"{size[0]}x{size[1]}" if size else f"{self.input_width}x?", "device": self.device,
                "providers": ",".join(p if isinstance(p, str) else p[0] for p in self.session.get_providers()),
                "onnxruntime": self._ort.__version__}
