"""Face: one detected face, plus the small box helpers every other module shares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels, x right, y down


@dataclass(frozen=True)
class Face:
    """One face in pixel coordinates of the frame it was detected in.

    landmarks: optional 5 (x, y) points (eyes, nose tip, mouth corners). The order is
    detector-specific; we only draw them, so it doesn't matter for control.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    landmarks: Optional[Tuple[Tuple[float, float], ...]] = None

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def box(self) -> Box:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float, score: float,
                  landmarks: Optional[Sequence[Tuple[float, float]]] = None) -> "Face":
        lm = tuple((float(px), float(py)) for px, py in landmarks) if landmarks is not None else None
        return cls(float(x), float(y), float(x + w), float(y + h), float(score), lm)

    def scaled(self, sx: float, sy: Optional[float] = None) -> "Face":
        """Return the face with coordinates multiplied by (sx, sy).

        Detectors run on a resized copy of the frame; this maps boxes back to the original.
        """
        sy = sx if sy is None else sy
        lm = None
        if self.landmarks is not None:
            lm = tuple((px * sx, py * sy) for px, py in self.landmarks)
        return Face(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy, self.score, lm)

    def clipped(self, width: int, height: int) -> "Face":
        """Clip the box to the image (landmarks are left alone)."""
        return Face(min(max(self.x1, 0.0), width), min(max(self.y1, 0.0), height),
                    min(max(self.x2, 0.0), width), min(max(self.y2, 0.0), height),
                    self.score, self.landmarks)


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes, 0..1."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def nms(faces: Sequence[Face], iou_threshold: float = 0.4) -> list:
    """Greedy non-maximum suppression: keep the best-scoring face of each overlapping group."""
    kept = []
    for f in sorted(faces, key=lambda f: f.score, reverse=True):
        if all(iou(f.box, k.box) <= iou_threshold for k in kept):
            kept.append(f)
    return kept
