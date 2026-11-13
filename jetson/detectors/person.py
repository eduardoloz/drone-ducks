"""Person (whole-body) detectors, so the follower keeps its lock when the face turns away.

A person walking away from the drone shows the back of their head: no face detector will find
it. A body detector still sees the person, and jetson/lock.py ties that body to the face we were
following. Detection only: these models say "a person is here", never *who* it is.

    from jetson.detectors.person import create_person_detector
    det = create_person_detector("yolox")          # YOLOX-nano, Apache-2.0, 3.7 MB ONNX
    people = det.detect(frame)                      # list[Face] (box + score; landmarks None)

Boxes are returned as jetson.face.Face objects (a box and a score) so the existing IouTracker,
nms() and HUD helpers work on them unchanged.

Backends (licences checked 2026-11-12, see docs/WHAT_WE_NEED.md section "Person detector"):

  name    model                              licence                      runs on
  ------  ---------------------------------  ---------------------------  ---------------------------------
  yolox   YOLOX-nano 416x416 (Megvii), COCO  Apache-2.0 code AND weights  OpenCV DNN or onnxruntime (CPU /
          "person" class                                                  CUDA / TensorRT EP); TensorRT
                                                                          engine via scripts/export_trt.py
  hog     OpenCV's built-in HOG+SVM people   OpenCV: Apache-2.0           any CPU, no download; slow and
          detector (Dalal-Triggs)                                         upright-full-body only. CI fallback

Considered and NOT used by default: Ultralytics YOLOv8n/YOLO11n (AGPL-3.0: any program that uses
them over a network or ships them must be AGPL too, or buy a licence), NVIDIA TAO PeopleNet via
jetson-inference (NVIDIA model EULA, fine for research, not redistributable), MobileNet-SSD
Caffe (MIT, but 23 MB and older/less accurate than YOLOX-nano).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..face import Face, nms
from .base import BackendUnavailable, model_path, require_file

YOLOX_NANO = "yolox_nano.onnx"
PERSON_CLASS = 0  # COCO class 0 = person


class PersonDetector:
    """detect(frame_bgr) -> list[Face] of whole-person boxes in the frame's pixel coordinates."""

    name = "base"

    def detect(self, frame: np.ndarray) -> List[Face]:
        raise NotImplementedError

    def info(self) -> Dict[str, object]:
        return {"backend": self.name}

    def close(self) -> None:
        pass

    def __enter__(self) -> "PersonDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ----------------------------------------------------------------------------- YOLOX helpers
def letterbox(frame: np.ndarray, size: Tuple[int, int] = (416, 416)) -> Tuple[np.ndarray, float]:
    """YOLOX preprocessing: resize keeping aspect, pad bottom/right with 114, HWC uint8 -> CHW float32.

    Returns (blob 1x3xHxW, ratio); a box in blob pixels maps back to the frame by dividing by ratio.
    BGR order and 0-255 range, no mean/std: that is what the Megvii ONNX exports (>= 0.1.1) expect.
    """
    import cv2

    ih, iw = size
    h, w = frame.shape[:2]
    r = min(ih / float(h), iw / float(w))
    nh, nw = max(1, int(h * r)), max(1, int(w * r))
    padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), r


def yolox_grids(size: Tuple[int, int] = (416, 416), strides=(8, 16, 32)) -> Tuple[np.ndarray, np.ndarray]:
    """Per-anchor (x, y) grid cell and stride, in the order the YOLOX head flattens its outputs."""
    grids, expanded = [], []
    for s in strides:
        hs, ws = size[0] // s, size[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        expanded.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids, 0).astype(np.float32), np.concatenate(expanded, 0).astype(np.float32)


def decode_yolox(raw: np.ndarray, ratio: float, size: Tuple[int, int] = (416, 416),
                 score_threshold: float = 0.4, nms_threshold: float = 0.45, class_id: int = PERSON_CLASS,
                 frame_shape: Optional[Tuple[int, int]] = None, top_k: int = 50) -> List[Face]:
    """Raw YOLOX output (1, N, 5 + classes) -> person boxes in frame pixels.

    Each row is (dx, dy, log w, log h, objectness, class probs...) relative to its grid cell; the
    exported head has already applied the sigmoids. score = objectness * P(class).
    """
    out = np.asarray(raw, dtype=np.float32).reshape(-1, raw.shape[-1])
    grids, strides = yolox_grids(size)
    if out.shape[0] != grids.shape[0]:
        raise ValueError(f"YOLOX output has {out.shape[0]} rows, expected {grids.shape[0]} for input {size}")
    scores = out[:, 4] * out[:, 5 + class_id]
    keep = scores >= score_threshold
    if not np.any(keep):
        return []
    o, g, s, sc = out[keep], grids[keep], strides[keep], scores[keep]
    cxcy = (o[:, :2] + g) * s / ratio
    wh = np.exp(np.clip(o[:, 2:4], -10, 10)) * s / ratio
    boxes = []
    for (cx, cy), (bw, bh), score in zip(cxcy, wh, sc):
        f = Face(float(cx - bw / 2), float(cy - bh / 2), float(cx + bw / 2), float(cy + bh / 2), float(score))
        if frame_shape is not None:
            f = f.clipped(frame_shape[1], frame_shape[0])
        if f.w > 1 and f.h > 1:
            boxes.append(f)
    return nms(boxes, nms_threshold)[:top_k]


class YoloxPersonDetector(PersonDetector):
    """YOLOX-nano (Apache-2.0) through OpenCV DNN (default) or ONNX Runtime.

    engine="opencv"      : cv2.dnn, CPU (or CUDA if your OpenCV was built with it: device="cuda")
    engine="onnxruntime" : device cpu | cuda | tensorrt (onnxruntime-gpu on the Jetson)
    """

    name = "yolox"

    def __init__(self, model: Optional[str] = None, engine: str = "opencv", device: str = "cpu",
                 score_threshold: float = 0.4, nms_threshold: float = 0.45, input_size: int = 416) -> None:
        self.path = require_file(model or model_path(YOLOX_NANO), "YOLOX-nano person model")
        self.engine, self.device = engine, device
        self.score_threshold, self.nms_threshold = score_threshold, nms_threshold
        self.size = (input_size, input_size)
        if engine == "opencv":
            try:
                import cv2
            except ImportError as e:
                raise BackendUnavailable("OpenCV is not installed: pip install opencv-python-headless") from e
            self.net = cv2.dnn.readNetFromONNX(self.path)
            if device == "cuda":
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self._run = self._run_cv
        elif engine == "onnxruntime":
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise BackendUnavailable("onnxruntime is not installed (Jetson: NVIDIA's onnxruntime-gpu wheel)") from e
            from .onnxrt import providers_for

            self.session = ort.InferenceSession(self.path,
                                                providers=providers_for(device, ort.get_available_providers()))
            self.input_name = self.session.get_inputs()[0].name
            self._run = self._run_ort
        else:
            raise ValueError("engine must be 'opencv' or 'onnxruntime'")

    def _run_cv(self, blob: np.ndarray) -> np.ndarray:
        self.net.setInput(blob)
        return self.net.forward()

    def _run_ort(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob})[0]

    def detect(self, frame: np.ndarray) -> List[Face]:
        blob, r = letterbox(frame, self.size)
        return decode_yolox(self._run(blob), r, self.size, self.score_threshold, self.nms_threshold,
                            frame_shape=frame.shape[:2])

    def info(self) -> Dict[str, object]:
        return {"backend": self.name, "model": os.path.basename(self.path), "engine": self.engine,
                "device": self.device, "input": f"{self.size[1]}x{self.size[0]}", "licence": "Apache-2.0"}


class HogPersonDetector(PersonDetector):
    """OpenCV's built-in HOG people detector. No download, but slow and only finds upright people."""

    name = "hog"

    def __init__(self, input_width: int = 480, score_threshold: float = 0.3, nms_threshold: float = 0.45) -> None:
        try:
            import cv2
        except ImportError as e:
            raise BackendUnavailable("OpenCV is not installed: pip install opencv-python-headless") from e
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.input_width, self.score_threshold, self.nms_threshold = input_width, score_threshold, nms_threshold

    def detect(self, frame: np.ndarray) -> List[Face]:
        from .base import downscale_to_width

        small, scale = downscale_to_width(frame, self.input_width)
        rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        out = []
        for (x, y, w, h), wt in zip(rects, np.ravel(weights) if len(rects) else []):
            if float(wt) >= self.score_threshold:
                out.append(Face.from_xywh(x, y, w, h, min(1.0, float(wt))).scaled(1.0 / scale))
        return nms(out, self.nms_threshold)

    def info(self) -> Dict[str, object]:
        return {"backend": self.name, "model": "OpenCV default people SVM", "device": "cpu",
                "licence": "Apache-2.0 (OpenCV)"}


PERSON_BACKENDS = {"yolox": YoloxPersonDetector, "hog": HogPersonDetector}


def create_person_detector(name: str, **kwargs) -> PersonDetector:
    if name not in PERSON_BACKENDS:
        raise ValueError(f"unknown person detector {name!r}; choose from {sorted(PERSON_BACKENDS)}")
    return PERSON_BACKENDS[name](**kwargs)
