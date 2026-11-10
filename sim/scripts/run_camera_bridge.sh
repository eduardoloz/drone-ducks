#!/usr/bin/env bash
# Serve the Gazebo drone camera on http://127.0.0.1:5601 (see sim/gazebo/gz_camera_bridge.py).
# Extra args go to the bridge, e.g. --host 0.0.0.0 to let a Jetson on the LAN read it.
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ -x "$GZ_PYTHON" ] || die "Gazebo's python not found at $GZ_PYTHON (set GZ_PYTHON)"
exec "$GZ_PYTHON" -u "$SIM_DIR/gazebo/gz_camera_bridge.py" "$@"
