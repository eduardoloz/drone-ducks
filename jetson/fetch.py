"""Download model weights and public-domain test media into gitignored folders.

    python -m jetson.fetch models          # YuNet (default detector, MIT, ~230 KB)
    python -m jetson.fetch models --all    # + YuNet 2022mar, SCRFD-500M, yolov8n-face
    python -m jetson.fetch media           # NASA photos + a synthetic test video + toy face sprite

Everything lands in jetson/models/ and jetson/media/, both gitignored. Never commit them.
Downloads are checked against pinned SHA-256 hashes when we have one.

Test media are NASA crew portraits. NASA imagery is generally not subject to US copyright, but
they show real, identifiable people: use them for local testing only, never in promotional
material, and never commit them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
MEDIA_DIR = os.path.join(HERE, "media")

MODELS = {
    "yunet": dict(
        file="face_detection_yunet_2023mar.onnx", default=True, licence="MIT (opencv_zoo)",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "yunet-2022": dict(
        file="face_detection_yunet_2022mar.onnx", licence="MIT (opencv_zoo); for OpenCV 4.5.4 - 4.7",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/7e062e54cf5410c09b795ff71b4a255e58498c79/"
            "models/face_detection_yunet/face_detection_yunet_2022mar.onnx",
        sha256="50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7"),  # pinned 2026-11-10
    "scrfd": dict(
        file="scrfd_500m_bnkps.onnx", zip_member="det_500m.onnx",
        licence="InsightFace model zoo: NON-COMMERCIAL RESEARCH ONLY",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a"),
    "yolov8n-face": dict(
        file="yolov8n-face.pt", licence="AGPL-3.0 lineage (Ultralytics); GPL-3.0 repo",
        url="https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
        sha256="d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca"),  # pinned 2026-11-10
}

MEDIA = {
    "nasa_STS064-S-002.jpg": "https://images-assets.nasa.gov/image/STS064-S-002/STS064-S-002~medium.jpg",
    "nasa_jsc2019e022584_alt.jpg":
        "https://images-assets.nasa.gov/image/jsc2019e022584_alt/jsc2019e022584_alt~medium.jpg",
    "nasa_jsc2025e041955.jpg": "https://images-assets.nasa.gov/image/jsc2025e041955/jsc2025e041955~medium.jpg",
}


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "projectquack-jetson-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".part-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o644)
    shutil.move(tmp, path)


def fetch_model(name: str, force: bool = False) -> Optional[str]:
    spec = MODELS[name]
    path = os.path.join(MODELS_DIR, spec["file"])
    if os.path.exists(path) and not force:
        print(f"  {name:13s} ok      {path}")
        return path
    print(f"  {name:13s} fetching {spec['url'][:70]}...")
    data = download(spec["url"])
    if spec.get("zip_member"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(spec["zip_member"])
    digest = sha256_of(data)
    if spec["sha256"] and digest != spec["sha256"]:
        raise SystemExit(f"  {name}: SHA-256 mismatch ({digest}); refusing to save. The upstream file changed?")
    _write_atomic(path, data)
    note = "" if spec["sha256"] else f"  (no pinned hash; sha256={digest})"
    print(f"  {name:13s} saved   {path}  [{spec['licence']}]{note}")
    return path


def fetch_media(force: bool = False) -> None:
    for fname, url in MEDIA.items():
        path = os.path.join(MEDIA_DIR, fname)
        if os.path.exists(path) and not force:
            print(f"  {fname} ok")
            continue
        _write_atomic(path, download(url))
        print(f"  {fname} saved")
    make_derived_media(force)


def make_derived_media(force: bool = False) -> None:
    """Build the synthetic pan/zoom test video and the toy-world face sprite from a NASA photo."""
    import cv2

    src = os.path.join(MEDIA_DIR, "nasa_STS064-S-002.jpg")
    img = cv2.imread(src)
    if img is None:
        print("  (skipping derived media: source photo missing)")
        return
    video = os.path.join(MEDIA_DIR, "crew_pan.mp4")
    if force or not os.path.exists(video):
        write_pan_video(img, video)
        print(f"  {os.path.basename(video)} generated (synthetic camera motion over the crew photo)")
    sprite = os.path.join(MEDIA_DIR, "face_sprite.jpg")
    if force or not os.path.exists(sprite):
        try:
            from .detectors import create_detector
            from .toysim import make_sprite

            faces = create_detector("yunet", input_width=1280).detect(img)
            if faces:
                best = max(faces, key=lambda f: f.w)
                cv2.imwrite(sprite, make_sprite(img, best))
                print(f"  {os.path.basename(sprite)} generated (for --source toy with a real detector)")
        except Exception as e:  # YuNet model missing etc.
            print(f"  (no face sprite: {e})")


def write_pan_video(img, path: str, size=(640, 480), fps: int = 15, seconds: float = 24.0) -> None:
    """A virtual camera flying over a still photo: zoom out (faces shrink), pan, zoom back in,
    then pan right off the crew (faces leave the frame) and back. Deterministic; no real video needed."""
    import cv2
    import numpy as np

    H, W = img.shape[:2]
    out_w, out_h = size
    n = int(fps * seconds)
    keys = [  # (time fraction, centre x frac, centre y frac, crop width as fraction of image width)
        (0.00, 0.50, 0.42, 0.45), (0.25, 0.50, 0.45, 1.00), (0.45, 0.30, 0.42, 0.55),
        (0.62, 0.70, 0.42, 0.55), (0.78, 1.25, 0.42, 0.50), (0.90, 1.25, 0.42, 0.50), (1.00, 0.50, 0.42, 0.45),
    ]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    canvas = cv2.copyMakeBorder(img, 0, 0, W, W, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    for i in range(n):
        u = i / max(1, n - 1)
        for (t0, *a), (t1, *b) in zip(keys, keys[1:]):
            if t0 <= u <= t1:
                k = (u - t0) / (t1 - t0)
                k = k * k * (3 - 2 * k)  # smoothstep
                cx, cy, cw = [x + (y - x) * k for x, y in zip(a, b)]
                break
        crop_w = cw * W
        crop_h = crop_w * out_h / out_w
        x0 = int(W + cx * W - crop_w / 2)
        y0 = int(np.clip(cy * H - crop_h / 2, 0, max(0, H - crop_h)))
        crop = canvas[y0:y0 + int(crop_h), max(0, x0):max(0, x0) + int(crop_w)]
        vw.write(cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA))
    vw.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["models", "media", "all"])
    ap.add_argument("--all", action="store_true", help="also optional models (SCRFD, YOLO-face, YuNet 2022)")
    ap.add_argument("--only", nargs="*", choices=sorted(MODELS), help="fetch just these models")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)
    if args.what in ("models", "all"):
        names = args.only or [n for n, s in MODELS.items() if s.get("default") or args.all]
        print(f"models -> {MODELS_DIR}")
        for n in names:
            fetch_model(n, args.force)
    if args.what in ("media", "all"):
        print(f"media -> {MEDIA_DIR}")
        fetch_media(args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
