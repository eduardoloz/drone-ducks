#!/usr/bin/env bash
# TypeFly vision service: YOLO gateway on :50049 (HTTP) + gRPC worker on :50050.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_venv
[ -d "$TYPEFLY_DIR" ] || die "TypeFly not set up: sim/scripts/setup_typefly.sh"
cd "$TYPEFLY_DIR"
echo "YOLO model: ${TYPEFLY_YOLO_MODEL:-yolo26s.pt (default)}"
exec "$SIM_PYTHON" -m typefly.serving
