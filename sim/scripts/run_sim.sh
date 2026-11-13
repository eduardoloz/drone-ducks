#!/usr/bin/env bash
# Run PX4 SITL in the foreground (start_sim.sh runs this inside the tmux session `quack-sim`).
#
#   SIM=gz  (default)  Gazebo Harmonic + our x500_quack (x500 + forward camera) in WORLD
#   SIM=sih            PX4's built-in physics, no Gazebo, no camera: light, runs anywhere PX4 builds
#
#   WORLD=quack_demo (default) | quack_people | any PX4 world (default, walls, baylands, ...)
#   MODEL=x500_quack (default) | x500 | x500_depth | ...    POSE="x,y,z,roll,pitch,yaw" (Gazebo ENU)
#   HEADLESS=1 (default) | 0 to also open the Gazebo GUI
#   CAM_PITCH_DEG=20   pitch quack_cam further down (default 8 deg, set in gazebo/models/quack_cam);
#                      use ~20 for 3 m follow altitude. Writes an override model to .run/models/
#   KEEP_PARAMS=1      keep PX4 parameters from the last run (default: every start is a fresh PX4,
#                      with sim/px4/sitl.params applied on top)
#
# Gazebo runs in PX4's "standalone" mode: this script starts `gz sim` with our world, spawns the
# drone, then PX4 attaches to it (PX4_GZ_STANDALONE=1 + PX4_GZ_MODEL_NAME). Nothing is written into
# the PX4 checkout, and PX4's state (params, .ulg flight logs) lives in sim/.run/px4-<sim>/.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_px4

SIM="${SIM:-gz}"
ROOTFS="$RUN_DIR/px4-$SIM"
mkdir -p "$ROOTFS/fs"
if [ -z "${KEEP_PARAMS:-}" ]; then
  # params + dataman (stored missions/geofences): a test that forgets to restore a param cannot
  # leak into the next run
  rm -f "$ROOTFS"/fs/parameters*.bson "$ROOTFS"/fs/dataman
fi
load_params "$SIM_DIR/px4/sitl.params"
ulimit -S -n 2048 2>/dev/null || true

if [ "$SIM" = "sih" ]; then
  export PX4_SIM_MODEL=sihsim_quadx
  echo "PX4 SITL: SIH quadx (no Gazebo) rootfs=$ROOTFS"
  exec "$PX4_BUILD/bin/px4" -w "$ROOTFS" "$PX4_BUILD/etc"
fi
[ "$SIM" = "gz" ] || die "unknown SIM=$SIM (gz or sih)"

# ---------------------------------------------------------------- Gazebo (standalone)
[ -f "$PX4_BUILD/gz_env.sh" ] || die "$PX4_BUILD/gz_env.sh missing: rebuild PX4 (make px4_sitl)"
set +u   # PX4's gz_env.sh and the pixi activation scripts read variables that may be unset
# shellcheck disable=SC1091
source "$PX4_BUILD/gz_env.sh"   # PX4 model/plugin paths, server config; on macOS loads the pixi env
set -u
export GZ_SIM_RESOURCE_PATH="$SIM_DIR/gazebo/models:$SIM_DIR/gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
rm -rf "$RUN_DIR/models"
if [ -n "${CAM_PITCH_DEG:-}" ]; then
  # Same quack_cam, sensor pitched CAM_PITCH_DEG down; found first on the resource path, so
  # x500_quack (and the model name x500_quack_0 the tools expect) is unchanged.
  PITCH_RAD="$(awk -v d="$CAM_PITCH_DEG" 'BEGIN { printf "%.4f", d * 3.14159265 / 180 }')"
  mkdir -p "$RUN_DIR/models"
  cp -R "$SIM_DIR/gazebo/models/quack_cam" "$RUN_DIR/models/"
  sed -i.bak "s|<pose>0 0 0 0 0.14 0</pose>|<pose>0 0 0 0 $PITCH_RAD 0</pose>|" "$RUN_DIR/models/quack_cam/model.sdf"
  rm -f "$RUN_DIR/models/quack_cam/model.sdf.bak"
  grep -q "0 0 0 0 $PITCH_RAD 0" "$RUN_DIR/models/quack_cam/model.sdf" || die "CAM_PITCH_DEG: camera pose not found in quack_cam"
  export GZ_SIM_RESOURCE_PATH="$RUN_DIR/models:$GZ_SIM_RESOURCE_PATH"
  echo "camera pitch: $CAM_PITCH_DEG deg down ($PITCH_RAD rad)"
fi

WORLD="${WORLD:-quack_demo}"
MODEL="${MODEL:-x500_quack}"
NAME="${MODEL}_0"
WORLD_FILE="$SIM_DIR/gazebo/worlds/$WORLD.sdf"
[ -f "$WORLD_FILE" ] || WORLD_FILE="$PX4_GZ_WORLDS/$WORLD.sdf"
[ -f "$WORLD_FILE" ] || die "world '$WORLD' not found in sim/gazebo/worlds or $PX4_GZ_WORLDS"
# The <world name> inside the file is what topics use; quack worlds name it like the file.
WORLD_NAME="$(sed -n 's/.*<world name="\([^"]*\)".*/\1/p' "$WORLD_FILE" | head -1)"
WORLD_NAME="${WORLD_NAME:-$WORLD}"

if gz topic -l 2>/dev/null | grep -q "^/world/.*/clock"; then
  die "a Gazebo world is already running (another sim?). Stop it first: sim/scripts/stop_sim.sh"
fi

GZ_PID=""
cleanup() {
  trap - EXIT INT TERM HUP
  [ -n "$GZ_PID" ] && kill "$GZ_PID" 2>/dev/null || true
  sleep 1
  [ -n "$GZ_PID" ] && kill -9 "$GZ_PID" 2>/dev/null || true   # gz server sometimes ignores SIGTERM
}
trap cleanup EXIT INT TERM HUP

echo "Gazebo: world=$WORLD_FILE headless=${HEADLESS:-1}"
gz sim --verbose="${GZ_VERBOSE:-1}" -r -s "$WORLD_FILE" &
GZ_PID=$!

for i in $(seq 1 90); do
  if gz service -i --service "/world/$WORLD_NAME/scene/info" 2>&1 | grep -q "Service providers"; then break; fi
  kill -0 "$GZ_PID" 2>/dev/null || die "gz sim exited while loading $WORLD_FILE (Fuel models offline? try scripts/fetch_models.sh)"
  [ "$i" = 90 ] && die "timed out waiting for the Gazebo world"
  sleep 1
done
echo "Gazebo world '$WORLD_NAME' is up"
if [ "${HEADLESS:-1}" = "0" ]; then
  gz sim -g >/dev/null 2>&1 &
fi

IFS=, read -r px py pz proll ppitch pyaw <<< "${POSE:-0,0,0,0,0,0}"
gz service -s "/world/$WORLD_NAME/create" --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
  --timeout 5000 --req "name: \"$NAME\", allow_renaming: false, sdf: '<sdf version=\"1.9\"><include><uri>model://$MODEL</uri><pose>${px:-0} ${py:-0} ${pz:-0} ${proll:-0} ${ppitch:-0} ${pyaw:-0}</pose></include></sdf>'" \
  >/dev/null || die "could not spawn $MODEL"
sleep 1

# ---------------------------------------------------------------- PX4
export PX4_GZ_STANDALONE=1 PX4_GZ_WORLD="$WORLD_NAME" PX4_GZ_MODEL_NAME="$NAME"
# x500_quack has no airframe file of its own: it flies with the x500 (mono cam) airframe, 4010.
export PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4010}"
echo "PX4 SITL: model=$NAME world=$WORLD_NAME autostart=$PX4_SYS_AUTOSTART rootfs=$ROOTFS"
"$PX4_BUILD/bin/px4" -w "$ROOTFS" "$PX4_BUILD/etc"
