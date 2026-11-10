"""YuNet face detector through OpenCV's cv2.FaceDetectorYN (the default backend).

Why default: tiny (~230 KB ONNX), MIT licence, only needs OpenCV, decent accuracy, gives 5
landmarks, and runs on any CPU (laptop, Jetson Nano, Orin). On a Jetson with a CUDA build of
OpenCV, backend="cuda" moves it to the GPU.

Model: face_detection_yunet_2023mar.onnx from opencv_zoo, which needs OpenCV >= 4.8.
(The older 2022mar model works with OpenCV 4.5.4 - 4.7; FaceDetectorYN does not exist before 4.5.4,
so JetPack 4.6's bundled OpenCV 4.1.1 cannot run YuNet at all.)

opencv_zoo notes the model finds faces from about 10x10 to 300x300 pixels, so `input_width`
matters both ways: too small and distant faces drop under 10 px, too large and a face right in
front of the camera exceeds 300 px.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..face import Face
from .base import BackendUnavailable, FaceDetector, downscale_to_width, model_path, require_file

DEFAULT_MODEL = "face_detection_yunet_2023mar.onnx"


def _version_tuple(v: str):
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


class YuNetDetector(FaceDetector):
    name = "yunet"

    def __init__(self, model: Optional[str] = None, input_width: int = 640, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, top_k: int = 50, device: str = "cpu") -> None:
        try:
            import cv2
        except ImportError as e:
            raise BackendUnavailable("OpenCV not installed: pip install 'opencv-python<4.12'") from e
        self._cv2 = cv2
        if not hasattr(cv2, "FaceDetectorYN"):
            raise BackendUnavailable(f"OpenCV {cv2.__version__} has no FaceDetectorYN (needs >= 4.5.4; the "
                                     "2023mar model needs >= 4.8)")
        self.model = require_file(model or model_path(DEFAULT_MODEL), "YuNet model")
        if "2023" in self.model and _version_tuple(cv2.__version__) < (4, 8, 0):
            raise BackendUnavailable(f"{DEFAULT_MODEL} needs OpenCV >= 4.8 (have {cv2.__version__}); "
                                     "upgrade OpenCV or use face_detection_yunet_2022mar.onnx")
        backend_id, target_id = cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU
        if device in ("cuda", "cuda_fp16"):
            n = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
            if n <= 0:
                raise BackendUnavailable("device='cuda' but this OpenCV has no CUDA support (pip wheels "
                                         "never do). Use device='cpu' or a CUDA build of OpenCV.")
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA_FP16 if device == "cuda_fp16" else cv2.dnn.DNN_TARGET_CUDA
        elif device != "cpu":
            raise ValueError("device must be 'cpu', 'cuda' or 'cuda_fp16'")
        self.input_width = int(input_width)
        self.score_threshold, self.nms_threshold, self.top_k = score_threshold, nms_threshold, top_k
        self.device = device
        self._size = (320, 320)
        self._det = cv2.FaceDetectorYN.create(self.model, "", self._size, score_threshold, nms_threshold,
                                              top_k, backend_id, target_id)

    def detect(self, frame: np.ndarray) -> List[Face]:
        img, scale = downscale_to_width(frame, self.input_width)
        h, w = img.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self._det.detect(img)
        if rows is None:
            return []
        inv = 1.0 / scale
        faces = []
        for r in rows:
            lm = [(float(r[4 + 2 * i]), float(r[5 + 2 * i])) for i in range(5)]
            f = Face.from_xywh(r[0], r[1], r[2], r[3], float(r[14]), lm)
            faces.append(f.scaled(inv) if scale != 1.0 else f)
        return faces

    def info(self) -> Dict[str, object]:
        return {"backend": self.name, "model": DEFAULT_MODEL, "input_width": self.input_width,
                "device": self.device, "opencv": self._cv2.__version__}
