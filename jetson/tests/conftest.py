"""Shared test setup: make `import jetson` work however pytest is launched.

Run from the repo root:   python -m pytest jetson/tests -q
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

JETSON_DIR = os.path.join(REPO_ROOT, "jetson")
MODELS_DIR = os.path.join(JETSON_DIR, "models")
MEDIA_DIR = os.path.join(JETSON_DIR, "media")


@pytest.fixture
def media_image():
    """Path to a downloaded public-domain test photo with faces, or skip."""
    import glob

    paths = sorted(glob.glob(os.path.join(MEDIA_DIR, "nasa_*.jpg")))
    if not paths:
        pytest.skip("no test media: run `python -m jetson.fetch media`")
    return paths[0]


@pytest.fixture
def yunet_model():
    path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(path):
        pytest.skip("YuNet model missing: run `python -m jetson.fetch models`")
    return path
