"""Pure-numpy pre/post-processing for YuNet and SCRFD raw network outputs.

OpenCV's FaceDetectorYN does all of this internally. We need our own copy for runtimes that only
give us raw tensors: ONNX Runtime (CPU / CUDA / TensorRT execution provider) and native TensorRT
engines. Keeping it pure numpy makes it testable without any of those runtimes.

YuNet (opencv_zoo, MIT):     input BGR 0..255 float, NCHW, no normalisation.
  outputs per stride s in (8, 16, 32), one prior per cell: cls, obj (N x 1), bbox (N x 4), kps (N x 10)
  score = sqrt(clip(cls) * clip(obj));  cx = (col + b0) * s;  w = exp(b2) * s
  (mirrors OpenCV 4.8+ modules/objdetect/src/face_detect.cpp)

SCRFD (InsightFace):         input RGB, (x - 127.5) / 128, NCHW.
  outputs per stride: score (N x 1), bbox distances (N x 4), kps offsets (N x 10), with 2 anchors per
  cell for 3-level models (strides 8, 16, 32) or 1 for 5-level models (8 ... 128).
  box = centre -/+ distance * stride  (mirrors insightface/python-package/insightface/model_zoo/scrfd.py)

SCRFD outputs are grouped by *shape*, not by position, so it doesn't matter whether ONNX Runtime
or TensorRT returns them in a different order.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..face import Face

YUNET_STRIDES = (8, 16, 32)


def letterbox(frame: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Scale `frame` to fit inside size=(w, h) keeping aspect, pad bottom/right with zeros.

    Returns (canvas, scale). Detections in canvas pixels map back to the frame via / scale.
    """
    import cv2

    tw, th = size
    h, w = frame.shape[:2]
    scale = min(tw / float(w), th / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=frame.dtype)
    canvas[:nh, :nw] = resized
    return canvas, scale


def round_up(x: float, multiple: int = 32) -> int:
    return int(math.ceil(x / float(multiple)) * multiple)


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Vectorised greedy NMS over N x 4 (x1, y1, x2, y2) boxes; returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep


def _to_faces(boxes: np.ndarray, scores: np.ndarray, kps: np.ndarray, scale: float,
              iou_threshold: float, top_k: int) -> List[Face]:
    keep = nms_indices(boxes, scores, iou_threshold)[:top_k]
    inv = 1.0 / scale
    faces = []
    for i in keep:
        lm = tuple((float(kps[i, 2 * n]) * inv, float(kps[i, 2 * n + 1]) * inv) for n in range(5))
        b = boxes[i] * inv
        faces.append(Face(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(scores[i]), lm))
    return faces


# ------------------------------------------------------------------------------------ YuNet
def yunet_blob(canvas_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(canvas_bgr.transpose(2, 0, 1)[None].astype(np.float32))


def decode_yunet(outputs: Dict[str, np.ndarray], input_w: int, input_h: int, score_threshold: float,
                 scale: float = 1.0, iou_threshold: float = 0.3, top_k: int = 50) -> List[Face]:
    """outputs: name -> array for cls_8 ... kps_32 (a leading batch dim of 1 is fine)."""
    all_boxes, all_scores, all_kps = [], [], []
    for s in YUNET_STRIDES:
        cols, rows = input_w // s, input_h // s
        cls = np.clip(outputs[f"cls_{s}"].reshape(-1), 0.0, 1.0)
        obj = np.clip(outputs[f"obj_{s}"].reshape(-1), 0.0, 1.0)
        bbox = outputs[f"bbox_{s}"].reshape(-1, 4)
        kps = outputs[f"kps_{s}"].reshape(-1, 10)
        if cls.size != rows * cols:
            raise ValueError(f"YuNet stride {s}: {cls.size} cells, expected {rows}x{cols}")
        score = np.sqrt(cls * obj)
        idx = np.where(score >= score_threshold)[0]
        if idx.size == 0:
            continue
        r, c = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
        b = bbox[idx]
        cx, cy = (c + b[:, 0]) * s, (r + b[:, 1]) * s
        w, h = np.exp(b[:, 2]) * s, np.exp(b[:, 3]) * s
        all_boxes.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))
        all_scores.append(score[idx])
        k = kps[idx].copy()
        k[:, 0::2] = (k[:, 0::2] + c[:, None]) * s
        k[:, 1::2] = (k[:, 1::2] + r[:, None]) * s
        all_kps.append(k)
    if not all_boxes:
        return []
    return _to_faces(np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_kps),
                     scale, iou_threshold, top_k)


# ------------------------------------------------------------------------------------ SCRFD
def scrfd_blob(canvas_bgr: np.ndarray) -> np.ndarray:
    rgb = canvas_bgr[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None])


def group_scrfd_outputs(outputs: Sequence[np.ndarray], input_w: int, input_h: int):
    """Sort raw SCRFD outputs into [(stride, anchors, scores, bbox, kps_or_None), ...] by shape."""
    arrs = [np.asarray(o) for o in outputs]
    arrs = [a[0] if a.ndim == 3 else a for a in arrs]  # drop batch dim if present
    by_dim: Dict[int, List[np.ndarray]] = {1: [], 4: [], 10: []}
    for a in arrs:
        if a.ndim != 2 or a.shape[1] not in by_dim:
            raise ValueError(f"unexpected SCRFD output shape {a.shape}")
        by_dim[a.shape[1]].append(a)
    levels = len(by_dim[1])
    if levels not in (3, 5) or len(by_dim[4]) != levels or len(by_dim[10]) not in (0, levels):
        raise ValueError(f"unexpected SCRFD outputs: {[a.shape for a in arrs]}")
    strides, anchors = ((8, 16, 32), 2) if levels == 3 else ((8, 16, 32, 64, 128), 1)
    groups = []
    for key in by_dim:
        by_dim[key].sort(key=lambda a: -a.shape[0])  # most rows = finest stride
    for i, s in enumerate(strides):
        expected = (input_h // s) * (input_w // s) * anchors
        if by_dim[1][i].shape[0] != expected:
            raise ValueError(f"SCRFD stride {s}: {by_dim[1][i].shape[0]} rows, expected {expected} "
                             f"for a {input_w}x{input_h} input")
        groups.append((s, anchors, by_dim[1][i], by_dim[4][i], by_dim[10][i] if by_dim[10] else None))
    return groups


def decode_scrfd(outputs: Sequence[np.ndarray], input_w: int, input_h: int, score_threshold: float,
                 scale: float = 1.0, iou_threshold: float = 0.4, top_k: int = 50) -> List[Face]:
    all_boxes, all_scores, all_kps = [], [], []
    for s, anchors, scores, bbox, kps in group_scrfd_outputs(outputs, input_w, input_h):
        rows, cols = input_h // s, input_w // s
        ys, xs = np.mgrid[:rows, :cols]
        centres = (np.stack([xs, ys], axis=-1).reshape(-1, 2) * s).astype(np.float32)
        if anchors > 1:
            centres = np.repeat(centres, anchors, axis=0)
        sc = scores.reshape(-1)
        idx = np.where(sc >= score_threshold)[0]
        if idx.size == 0:
            continue
        p, d = centres[idx], bbox[idx] * s
        all_boxes.append(np.stack([p[:, 0] - d[:, 0], p[:, 1] - d[:, 1], p[:, 0] + d[:, 2], p[:, 1] + d[:, 3]],
                                  axis=1))
        all_scores.append(sc[idx])
        if kps is not None:
            k = kps[idx] * s
            k[:, 0::2] += p[:, 0:1]
            k[:, 1::2] += p[:, 1:2]
        else:
            k = np.zeros((idx.size, 10), np.float32)
        all_kps.append(k)
    if not all_boxes:
        return []
    return _to_faces(np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_kps),
                     scale, iou_threshold, top_k)
