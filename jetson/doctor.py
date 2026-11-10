"""Check this machine for everything the jetson/ package can use. Run it first on bring-up day.

    python -m jetson.doctor

Prints OK / WARN / FAIL lines for: Python version, Jetson model + L4T release, power mode,
numpy / OpenCV (+ CUDA, GStreamer, FaceDetectorYN), ONNX Runtime providers, TensorRT, pycuda,
jetson-inference, MAVSDK, models and media, cameras, serial ports for the Pixhawk link, and
finally which detector backends actually construct. Exit code 0 unless something required fails.
"""

from __future__ import annotations

import glob
import grp
import importlib
import os
import platform
import subprocess
import sys
from typing import List, Tuple

from . import MIN_PYTHON

HERE = os.path.dirname(os.path.abspath(__file__))
Result = Tuple[str, str, str]  # (status, item, detail)


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                              timeout=5).stdout.strip()
    except Exception as e:
        return f"({e.__class__.__name__})"


def _version(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return ""


def check_python() -> List[Result]:
    ok = sys.version_info[:2] >= MIN_PYTHON
    return [("OK" if ok else "FAIL", "python", f"{platform.python_version()} ({sys.executable})"
             + ("" if ok else f" -- need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}: on JetPack 4 use "
                              "`sudo apt install python3.8-venv` or the Ultralytics JetPack 4 container"))]


def check_platform() -> List[Result]:
    out = []
    model = ""
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().strip("\x00\n ")
    except OSError:
        pass
    if "jetson" in model.lower():
        out.append(("OK", "jetson", model))
        try:
            with open("/etc/nv_tegra_release") as f:
                out.append(("OK", "l4t", f.readline().strip()))
        except OSError:
            out.append(("WARN", "l4t", "/etc/nv_tegra_release missing"))
        pm = _run(["nvpmodel", "-q"])
        out.append(("OK" if "MAXN" in pm.upper() else "WARN", "power mode",
                    pm.replace("\n", " | ") + ("" if "MAXN" in pm.upper() else "  (sudo nvpmodel -m 0 for max)")))
    else:
        out.append(("OK", "platform", f"{platform.system()} {platform.machine()} -- not a Jetson (fine for dev)"))
    return out


def check_libs() -> List[Result]:
    out = []
    np_v = _version("numpy")
    out.append(("OK" if np_v else "FAIL", "numpy", np_v or "missing: pip install 'numpy<2'"))
    try:
        import cv2
        info = cv2.getBuildInformation()
        gst = any("GStreamer" in line and "YES" in line.upper() for line in info.splitlines())
        cuda_n = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
        yn = hasattr(cv2, "FaceDetectorYN")
        out.append(("OK" if yn else "WARN", "opencv", f"{cv2.__version__} at {os.path.dirname(cv2.__file__)}; "
                    f"FaceDetectorYN={'yes' if yn else 'NO (need >= 4.5.4; 4.8 for YuNet 2023mar)'}; "
                    f"GStreamer={'yes' if gst else 'no (CSI via csi:// needs it)'}; CUDA devices={cuda_n}"))
    except ImportError:
        out.append(("FAIL", "opencv", "missing: pip install 'opencv-python<4.12' (or use JetPack's)"))
    try:
        import onnxruntime as ort
        out.append(("OK", "onnxruntime", f"{ort.__version__}; providers={','.join(ort.get_available_providers())}"))
    except ImportError:
        out.append(("WARN", "onnxruntime", "not installed (optional: onnx backend)"))
    for mod, item, hint in (("tensorrt", "tensorrt", "JetPack only"), ("pycuda", "pycuda", "pip install pycuda"),
                            ("jetson_inference", "jetson-inference", "container or source build"),
                            ("ultralytics", "ultralytics", "optional")):
        v = _version(mod)
        out.append(("OK" if v else "WARN", item, v or f"not available ({hint})"))
    native = _version("mavsdk.asyncio")
    classic = _version("mavsdk_grpc")
    if native or classic:
        out.append(("OK", "mavsdk", f"native={'yes' if native else 'no'} classic={'yes' if classic else 'no'}"))
    else:
        out.append(("WARN", "mavsdk", "not installed: pip install 'mavsdk>=4,<5' (original Nano: mavsdk==4.0.6)"))
    return out


def check_files() -> List[Result]:
    out = []
    models = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "models", "*"))
                    if os.path.isfile(p))
    has_yunet = "face_detection_yunet_2023mar.onnx" in models
    out.append(("OK" if has_yunet else "WARN", "models",
                ", ".join(models) if models else "none: python -m jetson.fetch models"))
    media = glob.glob(os.path.join(HERE, "media", "*"))
    out.append(("OK" if media else "WARN", "media", f"{len(media)} files" if media
                else "none: python -m jetson.fetch media"))
    return out


def check_devices() -> List[Result]:
    out = []
    if not sys.platform.startswith("linux"):
        return [("OK", "devices", "skipped (not Linux)")]
    vids = sorted(glob.glob("/dev/video*"))
    out.append(("OK" if vids else "WARN", "cameras", " ".join(vids) or "no /dev/video* (CSI cams show up as "
                                                                        "/dev/video0 once the driver loads)"))
    if os.path.exists("/proc/device-tree/model"):
        argus = _run(["systemctl", "is-active", "nvargus-daemon"])
        out.append(("OK" if argus == "active" else "WARN", "nvargus-daemon", argus))
    serials = sorted(glob.glob("/dev/ttyTHS*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    out.append(("OK" if serials else "WARN", "serial", " ".join(serials) or "none found"))
    out.append(("OK" if "dialout" in groups else "WARN", "dialout group",
                "yes" if "dialout" in groups else "no: sudo usermod -aG dialout $USER (then log out/in)"))
    getty = _run(["systemctl", "is-enabled", "nvgetty"])
    if getty == "enabled":
        out.append(("WARN", "nvgetty", "enabled: it may hold the UART (original Nano): "
                                       "sudo systemctl disable --now nvgetty"))
    return out


def check_backends() -> List[Result]:
    from .detectors import probe_backends
    out = []
    for name, ok, detail in probe_backends():
        out.append(("OK" if ok else "WARN", f"backend {name}", detail[:150]))
    return out


def main(argv=None) -> int:
    sections = [("Python", check_python), ("Platform", check_platform), ("Libraries", check_libs),
                ("Files", check_files), ("Devices", check_devices), ("Detector backends", check_backends)]
    failed = False
    for title, fn in sections:
        print(f"\n== {title}")
        for status, item, detail in fn():
            failed |= status == "FAIL"
            print(f"  [{status:4s}] {item:18s} {detail}")
    print("\nResult:", "FAIL (fix the FAIL lines first)" if failed else "OK (WARN lines are optional extras)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
