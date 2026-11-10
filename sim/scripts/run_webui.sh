#!/usr/bin/env bash
# TypeFly web UI driving the PX4 drone (http://127.0.0.1:50000). Takes off when it starts.
# Needs OPENAI_API_KEY in sim/.deps/TypeFly/.env: every instruction is a paid LLM call.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_venv
[ -d "$TYPEFLY_DIR" ] || die "TypeFly not set up: sim/scripts/setup_typefly.sh"
cd "$TYPEFLY_DIR"
export TYPEFLY_ROBOT_INFO="${TYPEFLY_ROBOT_INFO:-$TYPEFLY_DIR/typefly/config/robot_info.px4.json}"
exec "$SIM_PYTHON" -m typefly.webui
