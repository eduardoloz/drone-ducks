#!/usr/bin/env bash
# One command to capture daylight footage from the LIE cameras we work on. Recording only:
# no detector, no pipeline, nothing left running afterwards.
#
#   cctv/scripts/record_daylight.sh                   # R10_041 R10_007 R10_008, 2 rounds of 60 s clips
#   cctv/scripts/record_daylight.sh -n 3 R10_008      # 3 rounds from one camera
#   cctv/scripts/record_daylight.sh --force R10_041   # record even if it is not daylight (night tests)
#
# Run from the repo root (drone-ducks/). Clips go to cctv/data/clips/ (git-ignored, ~1.8 MB per
# 60 s at 640x480) with a .json sidecar holding the recording start time.
# Etiquette: one stream at a time, 60 s clips, 30 s between clips, 5 min between rounds, at most
# 12 clips per run. This is NOT continuous ingest (that waits for the 511NY developer request).
set -euo pipefail
PY=${PY:-cctv/.venv/bin/python}
N=2
FORCE=0
SECONDS_PER_CLIP=60
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N=${2:?-n needs a number}; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
CAMS=("$@")
[ ${#CAMS[@]} -gt 0 ] || CAMS=(R10_041 R10_007 R10_008)
total=$(( N * ${#CAMS[@]} ))
if [ "$total" -gt 12 ]; then echo "that is $total clips; keep it to 12 or fewer per run" >&2; exit 2; fi
[ -x "$PY" ] || { echo "no python at $PY (see cctv/README.md quickstart, or set PY=...)" >&2; exit 2; }

if command -v pmset >/dev/null; then
  batt=$(pmset -g batt | grep -Eo '[0-9]+%' | tr -d % | head -1)
  if pmset -g batt | grep -q "Battery Power" && [ "${batt:-100}" -lt 20 ]; then
    echo "on battery at ${batt}%: plug in first" >&2; exit 1
  fi
fi
KEEPAWAKE=()
command -v caffeinate >/dev/null && KEEPAWAKE=(caffeinate -i)

clips=()
for r in $(seq 1 "$N"); do
  for cam in "${CAMS[@]}"; do
    if ! "$PY" -m cctv.daylight --camera "$cam" --check; then
      if [ "$FORCE" -eq 0 ]; then
        echo "not daylight at $cam: stopping (use --force to record anyway, e.g. for night tests)" >&2
        break 2
      fi
      echo "(--force: recording $cam outside daylight)"
    fi
    line=$(${KEEPAWAKE[@]+"${KEEPAWAKE[@]}"} "$PY" -m cctv.ingest --camera "$cam" --seconds "$SECONDS_PER_CLIP" | tail -1)
    echo "$line"
    c=$(echo "$line" | grep -Eo 'clip=[^ ]+' | cut -d= -f2 || true)
    [ -n "$c" ] && clips+=("$cam $c")
    sleep 30
  done
  [ "$r" -lt "$N" ] && { echo "round $r done; next round in 5 min"; sleep 270; }
done

echo
echo "recorded ${#clips[@]} clip(s):"
for x in "${clips[@]+"${clips[@]}"}"; do set -- $x; echo "  $1  $2  $(du -h "$2" | cut -f1)"; done
[ ${#clips[@]} -gt 0 ] || exit 1
cat <<MSG

next (one at a time, ~3 min CPU per clip with YOLO11n):
  $PY -m cctv.calibrate match --camera <CAM> --clip <clip>     # which PTZ view is it? calibrated?
  OMP_NUM_THREADS=2 nice -n 10 $PY -m cctv.pipeline --camera <CAM> --clip <clip>
MSG
