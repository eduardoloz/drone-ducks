"""YOLO face models (e.g. yolov8n-face) through Ultralytics, the API Stephen's scripts already use.

Ultralytics loads .pt (PyTorch), .onnx and .engine (TensorRT) files the same way, so on a Jetson
you export once and point this at the engine:

    yolo export model=jetson/models/yolov8n-face.pt format=engine half=True imgsz=640   # on the Jetson

Licence note: Ultralytics and the community yolov8-face weights are AGPL-3.0. Fine for our
research use; anything distributed or offered as a service must publish its source.
Needs: pip install ultralytics (pulls in torch; on an Intel Mac pin torch==2.2.2 and numpy<2).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ..face import Face
from .base import BackendUnavailable, FaceDetector, model_path, require_file

DEFAULT_MODEL = "yolov8n-face.pt"


class UltralyticsFaceDetector(FaceDetector):
    name = "ultralytics"

    def __init__(self, model: str = "", input_width: int = 640, score_threshold: float = 0.5,
                 device: str = "") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise BackendUnavailable("ultralytics not installed: pip install ultralytics") from e
        self.model_file = require_file(model or model_path(DEFAULT_MODEL), "YOLO face model")
        self.model = YOLO(self.model_file, task="pose" if "pose" in self.model_file else None)
        self.imgsz, self.conf, self.device = int(input_width), float(score_threshold), device or None

    def detect(self, frame: np.ndarray) -> List[Face]:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy() if getattr(r, "keypoints", None) is not None else None
        faces = []
        for i, (b, s) in enumerate(zip(boxes, scores)):
            lm = tuple((float(x), float(y)) for x, y in kps[i][:5]) if kps is not None and len(kps[i]) else None
            faces.append(Face(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), lm))
        return faces

    def info(self) -> Dict[str, object]:
        return {"backend": self.name, "model": os.path.basename(self.model_file), "input_width": self.imgsz,
                "device": self.device or "auto"}
