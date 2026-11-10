"""The interface every face detector implements, plus shared helpers."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from ..face import Face

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class BackendUnavailable(RuntimeError):
    """An optional dependency or model file is missing. The message says how to get it."""


class FaceDetector:
    """detect(frame_bgr) -> list[Face] in the frame's own pixel coordinates.

    Subclasses import their heavy dependencies inside __init__ (never at module import time), so
    `import jetson.detectors` works on any machine and a missing library becomes a clear
    BackendUnavailable error instead of an ImportError at start-up.
    """

    name = "base"

    def detect(self, frame: np.ndarray) -> List[Face]:
        raise NotImplementedError

    def info(self) -> Dict[str, object]:
        """Short description for logs, the HUD and benchmark tables."""
        return {"backend": self.name}

    def close(self) -> None:
        pass

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def require_file(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise BackendUnavailable(f"{what} not found at {path}. Run: python -m jetson.fetch models")
    return path


def downscale_to_width(frame: np.ndarray, width: int) -> Tuple[np.ndarray, float]:
    """Resize so the image is at most `width` px wide; returns (image, scale) with scale <= 1.

    Detections on the returned image are mapped back with face.scaled(1 / scale).
    We never upscale: it costs time and adds no information.
    """
    import cv2

    h, w = frame.shape[:2]
    if width <= 0 or w <= width:
        return frame, 1.0
    scale = width / float(w)
    return cv2.resize(frame, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA), scale
