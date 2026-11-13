#!/usr/bin/env bash
# Serve Gazebo ground truth (model poses + sim time) on http://127.0.0.1:5602 (see sim/gazebo/gz_pose_probe.py).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ -x "$GZ_PYTHON" ] || die "Gazebo's python not found at $GZ_PYTHON (set GZ_PYTHON)"
exec "$GZ_PYTHON" -u "$SIM_DIR/gazebo/gz_pose_probe.py" "$@"
