#!/usr/bin/env bash
# Pre-download the Gazebo Fuel models our worlds use into ~/.gz/fuel (one-time, ~150 MB), so the
# first sim start does not stall while Gazebo downloads them.
set -euo pipefail
source "$(dirname "$0")/common.sh"
command -v gz >/dev/null || die "gz not on PATH (macOS: is PX4's pixi env installed? Linux: install gz-harmonic)"
grep -ho 'https://fuel.gazebosim.org/[^<]*' "$SIM_DIR"/gazebo/worlds/*.sdf "$SIM_DIR"/gazebo/models/*/model.sdf 2>/dev/null \
  | sort -u | while read -r url; do
  echo "-> $url"
  gz fuel download -v 1 -u "$url" || echo "   (failed; Gazebo retries when the world loads)"
done
