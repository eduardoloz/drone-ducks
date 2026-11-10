import os
import random
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from jetson.detectors import BACKENDS, BackendUnavailable, create_detector, probe_backends  # noqa: E402
from jetson.detectors import decode  # noqa: E402
from jetson.detectors.onnxrt import providers_for  # noqa: E402
from jetson.face import iou  # noqa: E402
from jetson.tests.conftest import MODELS_DIR  # noqa: E402

SCRFD = os.path.join(MODELS_DIR, "scrfd_500m_bnkps.onnx")


def best_iou(face, others):
    return max((iou(face.box, o.box) for o in others), default=0.0)


# ---------------------------------------------------------------- registry
def test_unknown_backend():
    with pytest.raises(ValueError):
        create_detector("magic")


def test_probe_lists_every_backend_without_crashing():
    res = probe_backends()
    assert [r[0] for r in res] == list(BACKENDS)
    for name, ok, detail in res:
        assert isinstance(ok, bool) and detail


@pytest.mark.parametrize("backend,module", [
    ("onnx", "onnxruntime"), ("tensorrt", "tensorrt"), ("jetson_inference", "jetson_inference"),
    ("ultralytics", "ultralytics"),
])
def test_missing_library_is_a_clear_error(monkeypatch, backend, module):
    monkeypatch.setitem(sys.modules, module, None)  # makes `import module` raise ImportError
    kwargs = {"engine": "x.engine"} if backend == "tensorrt" else {}
    with pytest.raises(BackendUnavailable) as e:
        create_detector(backend, **kwargs)
    assert "install" in str(e.value).lower() or "jetson" in str(e.value).lower()


def test_missing_model_says_how_to_fetch(tmp_path):
    with pytest.raises(BackendUnavailable, match="jetson.fetch"):
        create_detector("yunet", model=str(tmp_path / "nope.onnx"))


def test_yunet_cuda_request_without_cuda_opencv(yunet_model):
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        pytest.skip("this OpenCV has CUDA")
    with pytest.raises(BackendUnavailable, match="CUDA"):
        create_detector("yunet", device="cuda")


def test_providers_for():
    assert providers_for("cpu", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    with pytest.raises(BackendUnavailable, match="onnxruntime-gpu"):
        providers_for("tensorrt", ["CPUExecutionProvider"])
    with pytest.raises(ValueError):
        providers_for("tpu", [])


def test_providers_for_tensorrt_options(tmp_path):
    prov = providers_for("tensorrt", ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                                      "CPUExecutionProvider"], fp16=True, cache_dir=str(tmp_path / "c"))
    assert prov[0][0] == "TensorrtExecutionProvider" and prov[0][1]["trt_fp16_enable"] is True
    assert prov[1:] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert os.path.isdir(tmp_path / "c")


# ---------------------------------------------------------------- pure decoding
def test_letterbox_keeps_aspect_and_pads_bottom_right():
    frame = np.full((480, 640, 3), 200, np.uint8)
    canvas, scale = decode.letterbox(frame, (320, 320))
    assert canvas.shape == (320, 320, 3) and scale == pytest.approx(0.5)
    assert canvas[:240].min() == 200 and canvas[240:].max() == 0


def test_nms_indices():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    keep = decode.nms_indices(boxes, np.array([0.8, 0.9, 0.7]), 0.4)
    assert keep == [1, 2]
    assert decode.nms_indices(np.zeros((0, 4)), np.zeros(0), 0.4) == []


def synthetic_yunet(w, h, stride, row, col, b=(0.5, 0.5, np.log(4.0), np.log(4.0))):
    outs = {}
    for s in decode.YUNET_STRIDES:
        n = (w // s) * (h // s)
        outs[f"cls_{s}"] = np.zeros((1, n, 1), np.float32)
        outs[f"obj_{s}"] = np.zeros((1, n, 1), np.float32)
        outs[f"bbox_{s}"] = np.zeros((1, n, 4), np.float32)
        outs[f"kps_{s}"] = np.zeros((1, n, 10), np.float32)
    idx = row * (w // stride) + col
    outs[f"cls_{stride}"][0, idx, 0] = 0.81
    outs[f"obj_{stride}"][0, idx, 0] = 1.0
    outs[f"bbox_{stride}"][0, idx] = b
    outs[f"kps_{stride}"][0, idx] = [0.5, 0.5] * 5
    return outs


def test_decode_yunet_synthetic_cell():
    outs = synthetic_yunet(640, 640, 16, row=10, col=20)
    faces = decode.decode_yunet(outs, 640, 640, score_threshold=0.5)
    assert len(faces) == 1
    f = faces[0]
    # centre = (col + 0.5) * 16, (row + 0.5) * 16; size = exp(log 4) * 16 = 64
    assert (f.cx, f.cy, f.w, f.h) == pytest.approx((328.0, 168.0, 64.0, 64.0))
    assert f.score == pytest.approx(0.9)  # sqrt(0.81 * 1.0)
    assert f.landmarks[0] == pytest.approx((328.0, 168.0))
    half = decode.decode_yunet(outs, 640, 640, score_threshold=0.5, scale=0.5)[0]
    assert half.cx == pytest.approx(656.0)  # mapped back through the letterbox scale
    assert decode.decode_yunet(outs, 640, 640, score_threshold=0.95) == []


def synthetic_scrfd(w, h, stride, cell, anchor, dist=(2.0, 3.0, 4.0, 5.0)):
    outs = []
    for kind in (1, 4, 10):
        for s in (8, 16, 32):
            outs.append(np.zeros(((w // s) * (h // s) * 2, kind), np.float32))
    row, col = cell
    i = (row * (w // stride) + col) * 2 + anchor
    level = (8, 16, 32).index(stride)
    outs[level][i, 0] = 0.9
    outs[3 + level][i] = dist
    outs[6 + level][i] = [1.0, -1.0] * 5
    return outs


def test_decode_scrfd_synthetic_and_order_independent():
    outs = synthetic_scrfd(640, 480, 8, cell=(5, 7), anchor=1)
    faces = decode.decode_scrfd(outs, 640, 480, score_threshold=0.5)
    assert len(faces) == 1
    f = faces[0]
    cx, cy = 7 * 8, 5 * 8  # anchor centre in pixels
    assert f.box == pytest.approx((cx - 16, cy - 24, cx + 32, cy + 40))
    assert f.landmarks[0] == pytest.approx((cx + 8, cy - 8))
    shuffled = outs[:]
    random.Random(3).shuffle(shuffled)
    assert decode.decode_scrfd(shuffled, 640, 480, 0.5)[0].box == pytest.approx(f.box)
    batched = [o[None] for o in outs]  # a leading batch dim must also work
    assert decode.decode_scrfd(batched, 640, 480, 0.5)[0].box == pytest.approx(f.box)


def test_group_scrfd_rejects_wrong_input_size():
    outs = synthetic_scrfd(640, 480, 8, cell=(0, 0), anchor=0)
    with pytest.raises(ValueError, match="expected"):
        decode.group_scrfd_outputs(outs, 640, 640)
    with pytest.raises(ValueError):
        decode.group_scrfd_outputs([np.zeros((10, 3))], 640, 480)


# ---------------------------------------------------------------- real models on a real photo
def test_yunet_finds_the_crew_and_maps_boxes_back(media_image, yunet_model):
    img = cv2.imread(media_image)
    small = create_detector("yunet", input_width=640).detect(img)
    big = create_detector("yunet", input_width=1280).detect(img)
    assert len(small) >= 3
    h, w = img.shape[:2]
    for f in small:
        assert 0 <= f.cx <= w and 0 <= f.cy <= h and f.w > 20 and 0.6 <= f.score <= 1.0
        assert len(f.landmarks) == 5
        assert best_iou(f, big) > 0.7  # same faces in original-image coordinates


def test_onnxruntime_yunet_matches_opencv(media_image, yunet_model):
    pytest.importorskip("onnxruntime")
    img = cv2.imread(media_image)
    ref = create_detector("yunet", input_width=640).detect(img)
    ort_faces = create_detector("onnx", model="yunet").detect(img)
    assert len(ort_faces) == len(ref)
    assert all(best_iou(f, ref) > 0.95 for f in ort_faces)


def test_scrfd_agrees_with_yunet(media_image, yunet_model):
    pytest.importorskip("onnxruntime")
    if not os.path.exists(SCRFD):
        pytest.skip("SCRFD model missing: python -m jetson.fetch models --all")
    img = cv2.imread(media_image)
    ref = create_detector("yunet", input_width=640).detect(img)
    det = create_detector("onnx", model="scrfd", input_width=640, score_threshold=0.5)
    faces = det.detect(img)
    assert det.info()["backend"] == "onnx:scrfd"
    assert len(faces) >= len(ref) - 1
    assert sum(best_iou(f, ref) > 0.4 for f in faces) >= len(ref) - 1


def test_haar_baseline_runs(media_image):
    img = cv2.imread(media_image)
    faces = create_detector("haar").detect(img)
    assert isinstance(faces, list) and all(f.score == 1.0 for f in faces)


@pytest.mark.parametrize("backend,kwargs", [("yunet", {}), ("haar", {}), ("onnx", {"model": "yunet"})])
def test_blank_frame_has_no_faces(backend, kwargs, yunet_model):
    if backend == "onnx":
        pytest.importorskip("onnxruntime")
    assert create_detector(backend, **kwargs).detect(np.zeros((480, 640, 3), np.uint8)) == []
