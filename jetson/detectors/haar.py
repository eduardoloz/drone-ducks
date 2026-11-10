"""OpenCV Haar cascade: the 2001-era baseline. Use it to *compare*, not to fly.

It is fast on a CPU and needs nothing but OpenCV (even JetPack 4.6's OpenCV 4.1.1 has it), but
it only finds fairly frontal, well-lit faces, has no meaningful confidence score (we report 1.0)
and produces more false positives than any CNN detector here.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import numpy as np

from ..face import Face
from .base import BackendUnavailable, FaceDetector, downscale_to_width

CASCADE = "haarcascade_frontalface_default.xml"
SEARCH_DIRS = ("/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades",
               "/usr/local/share/opencv4/haarcascades")


def find_cascade(name: str = CASCADE) -> Optional[str]:
    import cv2

    dirs = []
    if hasattr(cv2, "data"):
        dirs.append(cv2.data.haarcascades)  # pip wheels ship the XML files here
    dirs.extend(SEARCH_DIRS)                # apt/JetPack OpenCV puts them here
    for d in dirs:
        hits = glob.glob(os.path.join(d, name))
        if hits:
            return hits[0]
    return None


class HaarDetector(FaceDetector):
    name = "haar"

    def __init__(self, cascade: Optional[str] = None, input_width: int = 640, scale_factor: float = 1.1,
                 min_neighbors: int = 5, min_size: int = 20, score_threshold: float = 0.0) -> None:
        try:
            import cv2
        except ImportError as e:
            raise BackendUnavailable("OpenCV not installed") from e
        self._cv2 = cv2
        path = cascade or find_cascade()
        if not path:
            raise BackendUnavailable(f"{CASCADE} not found (looked in cv2.data and {SEARCH_DIRS})")
        self._clf = cv2.CascadeClassifier(path)
        if self._clf.empty():
            raise BackendUnavailable(f"could not load cascade {path}")
        self.input_width, self.scale_factor = int(input_width), scale_factor
        self.min_neighbors, self.min_size = min_neighbors, min_size

    def detect(self, frame: np.ndarray) -> List[Face]:
        cv2 = self._cv2
        img, scale = downscale_to_width(frame, self.input_width)
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        rects = self._clf.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                                           minSize=(self.min_size, self.min_size))
        inv = 1.0 / scale
        return [Face.from_xywh(x, y, w, h, 1.0).scaled(inv) for (x, y, w, h) in rects]

    def info(self) -> Dict[str, object]:
        return {"backend": self.name, "model": CASCADE, "input_width": self.input_width,
                "device": "cpu", "opencv": self._cv2.__version__}
