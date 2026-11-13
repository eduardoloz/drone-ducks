#!/usr/bin/env bash
# Set up the jetson/ face-follow stack on a Jetson Orin Nano (Super) running JetPack 6.2.x.
# UNVERIFIED: written before we had the hardware. Every step is idempotent and prints what it does;
# run it, read the output, and fix the checklist (docs/BRINGUP_CHECKLIST.md) where reality differs.
#
#   bash jetson/scripts/setup_jetson_jp6.sh                 # venv + deps + models + tests
#   bash jetson/scripts/setup_jetson_jp6.sh --with-ort      # + NVIDIA onnxruntime-gpu (TensorRT EP)
#   bash jetson/scripts/setup_jetson_jp6.sh --engines       # + build TensorRT engines (minutes)
#
# It never flashes, reboots, changes the power mode or touches the UART/getty config: those are
# deliberate manual steps in the checklist.
set -euo pipefail

VENV="${QUACK_VENV:-$HOME/quack-venv}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WITH_ORT=0
ENGINES=0
ORT_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"

for arg in "$@"; do
  case "$arg" in
    --with-ort) WITH_ORT=1 ;;
    --engines) ENGINES=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

say "1/7 Is this a JetPack 6 Jetson?"
if [[ -r /etc/nv_tegra_release ]]; then
  head -1 /etc/nv_tegra_release
  grep -q "R36" /etc/nv_tegra_release || warn "not L4T R36 (JetPack 6). JetPack 4 Nano: see WHAT_WE_NEED.md section 2."
else
  warn "/etc/nv_tegra_release not found: not a Jetson. Continuing (useful for a dry run only)."
fi
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "system python3 = $PYV (JetPack 6 ships 3.10)"
[[ "$PYV" == "3.10" ]] || warn "expected Python 3.10"

say "2/7 System packages (needs sudo; already-installed ones are skipped)"
NEED=()
dpkg -s python3-venv >/dev/null 2>&1 || NEED+=(python3-venv)
dpkg -s python3-pip >/dev/null 2>&1 || NEED+=(python3-pip)
# trtexec lives in libnvinfer-bin (part of JetPack's nvidia-tensorrt).
[[ -x /usr/src/tensorrt/bin/trtexec ]] || NEED+=(libnvinfer-bin)
# JetPack's OpenCV (4.8, built WITH GStreamer, needed for csi://). Ubuntu's python3-opencv is 4.5: too old.
python3 -c 'import cv2, sys; v = tuple(int(x) for x in cv2.__version__.split(".")[:2]); sys.exit(v < (4, 8))' \
  2>/dev/null || NEED+=(nvidia-opencv)
if ((${#NEED[@]})); then
  echo "installing: ${NEED[*]}"
  sudo apt-get update
  sudo apt-get install -y "${NEED[@]}"
else
  echo "all present"
fi
python3 - <<'EOF'
import cv2
info = cv2.getBuildInformation()
gst = [l.strip() for l in info.splitlines() if "GStreamer" in l]
print("system OpenCV", cv2.__version__, "|", gst[0] if gst else "GStreamer: not listed")
EOF

say "3/7 Virtualenv at $VENV (with system site-packages so JetPack's cv2 and tensorrt stay visible)"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$REPO/jetson/requirements-jetson-jp6.txt"
if python -m pip show opencv-python opencv-python-headless >/dev/null 2>&1; then
  warn "a pip OpenCV is installed in the venv and will shadow JetPack's GStreamer build: pip uninstall it"
fi

say "4/7 Optional GPU runtimes"
if ((WITH_ORT)); then
  python -m pip install onnxruntime-gpu --index-url "$ORT_INDEX" || warn "onnxruntime-gpu install failed (is $ORT_INDEX up?)"
fi
python - <<'EOF'
for mod in ("tensorrt", "onnxruntime", "pycuda"):
    try:
        m = __import__(mod)
        extra = m.get_available_providers() if mod == "onnxruntime" else ""
        print(f"  {mod:12s} {getattr(m, '__version__', '?')} {extra}")
    except Exception as e:
        print(f"  {mod:12s} not available ({type(e).__name__})")
EOF

say "5/7 Models (YuNet face + YOLOX-nano person; pinned hashes)"
cd "$REPO"
python -m jetson.fetch models
python -m jetson.fetch media

say "6/7 TensorRT engines"
python jetson/scripts/export_trt.py --dry-run
if ((ENGINES)); then
  python jetson/scripts/export_trt.py yunet yolox --bench
else
  echo "(skipped; rerun with --engines to build them, takes a few minutes)"
fi

say "7/7 Self-checks"
python -m jetson.doctor || true
python -m pytest jetson/tests -q
groups | grep -qw dialout || warn "add yourself to dialout for the Pixhawk UART: sudo usermod -aG dialout \$USER"
echo
echo "Done. Next: docs/BRINGUP_CHECKLIST.md section 'Camera' (csi://0) and the props-off bench test."
echo "Activate later with: source $VENV/bin/activate"
