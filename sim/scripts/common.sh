# Sourced by sim/scripts/*.sh. Finds PX4, the right Pythons and our run directory on macOS/Linux.
# Override any of these by exporting them first, e.g. PX4_DIR=~/src/PX4-Autopilot.

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$SIM_DIR/.run}"          # PX4 rootfs (params, flight logs), tmux logs
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# --- PX4 checkout (built with `make px4_sitl`)
if [ -z "${PX4_DIR:-}" ]; then
  for d in "$HOME/PX4-Autopilot" "$HOME/ProjectQuack/setup/PX4-Autopilot" "$SIM_DIR/.deps/PX4-Autopilot"; do
    if [ -x "$d/build/px4_sitl_default/bin/px4" ]; then PX4_DIR="$(cd "$d" && pwd)"; break; fi
  done
fi
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_BUILD="${PX4_BUILD:-$PX4_DIR/build/px4_sitl_default}"
export PX4_DIR PX4_BUILD

# --- Python that has the Gazebo bindings (camera bridge only)
if [ -z "${GZ_PYTHON:-}" ]; then
  if [ "$(uname)" = "Darwin" ]; then
    GZ_PYTHON="$PX4_DIR/Tools/setup/macos/.pixi/envs/default/bin/python3"
  else
    GZ_PYTHON=/usr/bin/python3
  fi
fi
export GZ_PYTHON

# --- our venv (quackdrone, pytest, OpenCV; TypeFly + torch after setup_typefly.sh)
SIM_PYTHON="${SIM_PYTHON:-$SIM_DIR/.venv/bin/python}"
export SIM_PYTHON
TYPEFLY_DIR="${TYPEFLY_DIR:-$SIM_DIR/.deps/TypeFly}"
export TYPEFLY_DIR

# PX4's own `make px4_sitl gz_*` targets run with GZ_IP=127.0.0.1; every gz client must match or
# it will not see the topics.
export GZ_IP="${GZ_IP:-127.0.0.1}"
# On macOS Gazebo comes from PX4's pixi environment, not Homebrew.
if [ "$(uname)" = "Darwin" ] && [ -d "$PX4_DIR/Tools/setup/macos/.pixi/envs/default/bin" ]; then
  export PATH="$PX4_DIR/Tools/setup/macos/.pixi/envs/default/bin:$PATH"
fi
# Intel Macs have no usable GPU for torch: nano YOLO and a longer request timeout.
if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ]; then
  export TYPEFLY_YOLO_MODEL="${TYPEFLY_YOLO_MODEL:-yolo26n.pt}"
  export TYPEFLY_YOLO_TIMEOUT="${TYPEFLY_YOLO_TIMEOUT:-8}"
fi

die() { echo "ERROR: $*" >&2; exit 1; }

need_px4() {
  [ -x "$PX4_BUILD/bin/px4" ] || die "PX4 SITL is not built at $PX4_BUILD.
  Build it once:   cd \$PX4_DIR && make px4_sitl      (see sim/docs/SETUP.md)
  Or point to it:  export PX4_DIR=/path/to/PX4-Autopilot"
}

need_venv() {
  [ -x "$SIM_PYTHON" ] || die "no venv at $SIM_DIR/.venv. Run sim/scripts/setup.sh first."
}

# wait_for <seconds> <description> <command...>: retry a command once per second
wait_for() {
  local t=$1 what=$2; shift 2
  for _ in $(seq 1 "$t"); do
    if "$@" >/dev/null 2>&1; then echo "  ok: $what"; return 0; fi
    sleep 1
  done
  echo "  TIMEOUT waiting for $what (logs: $LOG_DIR)"; return 1
}

# load_params <file>: "NAME VALUE  # comment" lines -> PX4_PARAM_NAME=VALUE (PX4's rcS applies them)
load_params() {
  [ -f "$1" ] || return 0
  local name value _
  while read -r name value _; do
    case "$name" in ''|'#'*) continue ;; esac
    export "PX4_PARAM_${name}=${value}"
  done < "$1"
}

# Our tmux sessions. stop_sim.sh kills exactly these.
QUACK_SESSIONS="quack-webui quack-vision quack-poses quack-cam quack-sim"
