#!/usr/bin/env bash
# The whole TypeFly demo stack, each piece in its own tmux session:
#   quack-sim     PX4 SITL + Gazebo        quack-cam     camera bridge (:5601)
#   quack-vision  YOLO service (:50049)    quack-webui   TypeFly web UI (:50000)  [NO_WEBUI=1 skips it]
# Stop with sim/scripts/stop_sim.sh. Logs in sim/.run/logs/.
set -euo pipefail
source "$(dirname "$0")/common.sh"
S="$SIM_DIR/scripts"
"$S/start_sim.sh"
rm -f "$LOG_DIR/vision.log" "$LOG_DIR/webui.log"
echo "[3/4] vision service (${TYPEFLY_YOLO_MODEL:-yolo26s.pt})"
tmux new-session -d -s quack-vision "bash '$S/run_vision.sh' 2>&1 | tee '$LOG_DIR/vision.log'"
wait_for 240 "vision gateway on :50049" curl -sf -m 2 http://127.0.0.1:50049/health
if [ -z "${NO_WEBUI:-}" ]; then
  echo "[4/4] TypeFly web UI -> http://127.0.0.1:50000 (the drone takes off first)"
  tmux new-session -d -s quack-webui "bash '$S/run_webui.sh' 2>&1 | tee '$LOG_DIR/webui.log'"
  wait_for 180 "web UI" curl -sf -m 2 http://127.0.0.1:50000/health
else
  echo "[4/4] web UI skipped (NO_WEBUI=1)"
fi
echo "Up: $(tmux ls 2>/dev/null | grep -o '^quack-[a-z]*' | tr '\n' ' ')"
