"""jetson-inference detectNet "facenet" (TensorRT, NVIDIA). Jetson only. UNVERIFIED before hardware.

The quickest way to see GPU face detection on a Jetson: run dusty-nv's jetson-inference container
(or build the repo), and this backend wraps its detectNet. It imports the *new-style* modules
jetson_inference / jetson_utils, not the legacy `jetson.inference`, because our own package is
also called `jetson` and would shadow NVIDIA's legacy `jetson` package.

    det = JetsonInferenceDetector(network="facenet", score_threshold=0.5)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..face import Face
from .base import BackendUnavailable, FaceDetector


class JetsonInferenceDetector(FaceDetector):
    name = "jetson_inference"

    def __init__(self, network: str = "facenet", score_threshold: float = 0.5, input_width: int = 0) -> None:
        try:
            import jetson_inference
            import jetson_utils
        except ImportError as e:
            raise BackendUnavailable(
                "jetson_inference not importable. Use the container "
                "(git clone --recursive https://github.com/dusty-nv/jetson-inference && "
                "cd jetson-inference && docker/run.sh) or build it from source.") from e
        self._ji, self._ju = jetson_inference, jetson_utils
        self.network = network
        self.net = jetson_inference.detectNet(network, threshold=score_threshold)

    def detect(self, frame: np.ndarray) -> List[Face]:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        cuda_img = self._ju.cudaFromNumpy(rgb)
        dets = self.net.Detect(cuda_img, overlay="none")
        return [Face(float(d.Left), float(d.Top), float(d.Right), float(d.Bottom), float(d.Confidence))
                for d in dets]

    def info(self) -> Dict[str, object]:
        return {"backend": self.name, "model": self.network, "device": "gpu (TensorRT)"}
