#!/usr/bin/env bash
# Run Python from the sim venv:  sim/scripts/py.sh sim/examples/01_takeoff_square_land.py
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_venv
exec "$SIM_PYTHON" "$@"
