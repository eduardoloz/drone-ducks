#!/usr/bin/env bash
# Stop everything start_sim.sh / start_all.sh started. Safe to run any time.
# Gazebo's server sometimes ignores SIGTERM, and a leftover server makes the next PX4 attach to a
# stale world ("Accel #0 fail: TIMEOUT"); a force-killed PX4 leaves /tmp/px4_lock-* and
# /tmp/px4-sock-* behind and the next start hangs at "param import". This script handles both.
source "$(dirname "$0")/common.sh"

# Only processes started by these scripts (PX4 with our rootfs, gz with our worlds). ALL=1 also
# stops any other PX4 / Gazebo on this machine (e.g. one started with PX4's own `make px4_sitl gz_x500`).
PX4_PAT="bin/px4 -w $RUN_DIR"
GZ_PAT="gz sim.*$SIM_DIR/gazebo"
if [ -n "${ALL:-}" ]; then PX4_PAT="bin/px4"; GZ_PAT="gz sim"; fi

for s in $QUACK_SESSIONS; do tmux kill-session -t "$s" 2>/dev/null || true; done
pkill -f "gz_camera_bridge.py" 2>/dev/null || true
pkill -f "$PX4_PAT" 2>/dev/null || true
pkill -f "$GZ_PAT" 2>/dev/null || true
pkill -f "gz sim -g" 2>/dev/null || true
sleep 2
pkill -9 -f "$PX4_PAT" 2>/dev/null || true
pkill -9 -f "$GZ_PAT" 2>/dev/null || true
if ! pgrep -f "bin/px4" >/dev/null; then rm -f /tmp/px4_lock-* /tmp/px4-sock-*; fi
echo "stopped"
