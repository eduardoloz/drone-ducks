#!/usr/bin/env bash
# Start the simulator in tmux and wait until it is ready to fly:
#   quack-sim   PX4 SITL (+ Gazebo when SIM=gz)          attach: tmux attach -t quack-sim  (pxh> shell)
#   quack-cam   camera bridge -> http://127.0.0.1:5601    (SIM=gz only; CAMERA=0 to skip)
#
#   sim/scripts/start_sim.sh                       # Gazebo, quack_demo world, headless
#   WORLD=quack_people sim/scripts/start_sim.sh    # people for face-detection tests
#   HEADLESS=0 sim/scripts/start_sim.sh            # also open the Gazebo GUI
#   SIM=sih sim/scripts/start_sim.sh               # no Gazebo at all (control/failsafe work)
#
# Then fly:  sim/scripts/py.sh sim/examples/01_takeoff_square_land.py
# Stop:      sim/scripts/stop_sim.sh   (always use it: it also clears PX4's stale lock files)
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_px4
command -v tmux >/dev/null || die "tmux not found (macOS: brew install tmux, Ubuntu: sudo apt install tmux)"
S="$SIM_DIR/scripts"
SIM="${SIM:-gz}"

"$S/stop_sim.sh" >/dev/null
rm -f "$LOG_DIR/sim.log" "$LOG_DIR/cam.log"

echo "[1/2] PX4 SITL ($SIM${WORLD:+, world $WORLD})"
tmux new-session -d -s quack-sim "bash '$S/run_sim.sh' 2>&1 | tee '$LOG_DIR/sim.log'"
if ! wait_for 180 "PX4 started" grep -q "Startup script returned successfully" "$LOG_DIR/sim.log"; then
  tail -20 "$LOG_DIR/sim.log"; exit 1
fi

if [ "$SIM" = "gz" ] && [ "${CAMERA:-1}" = "1" ]; then
  echo "[2/2] camera bridge"
  tmux new-session -d -s quack-cam "bash '$S/run_camera_bridge.sh' ${BRIDGE_ARGS:-} 2>&1 | tee '$LOG_DIR/cam.log'"
  wait_for 60 "camera frames on http://127.0.0.1:5601/health" curl -sf -m 2 http://127.0.0.1:5601/health || true
else
  echo "[2/2] camera bridge skipped"
fi
echo "Ready. MAVLink for your code: udpin://0.0.0.0:14540 (ground station / QGC: 14550)."
echo "tmux sessions: $(tmux ls 2>/dev/null | grep -o '^quack-[a-z]*' | tr '\n' ' ')  logs: $LOG_DIR"
