#!/usr/bin/env bash
# Daylight ground-truth session: short clips from one camera on an NYC DOT link, then compare.
#
#   cctv/scripts/groundtruth_session.sh R10_008 3     # camera, number of 60 s clips (5 min apart)
#
# Run from the repo root, in daylight, on AC power. Short clips only (one stream, <= 60 s each):
# this is NOT continuous ingest, which waits for the 511NY developer request.
# Steps: battery check -> record clips -> if the camera has no calibrated view matching them, stop and
# print the calibration commands -> else run the pipeline on each clip, fetch link rows, print the table.
set -euo pipefail
CAM=${1:?camera id, e.g. R10_008 or R11_140}
N=${2:-3}
PY=${PY:-cctv/.venv/bin/python}

if command -v pmset >/dev/null; then
  batt=$(pmset -g batt | grep -Eo '[0-9]+%' | tr -d % | head -1)
  if pmset -g batt | grep -q "Battery Power" && [ "${batt:-100}" -lt 30 ]; then
    echo "on battery at ${batt}%: plug in first"; exit 1
  fi
fi
hour=$(TZ=America/New_York date +%H)
if [ "$hour" -lt 8 ] || [ "$hour" -ge 16 ]; then
  echo "warning: it is ${hour}h in New York; November daylight is roughly 07:30-16:30. Continue anyway in 5 s (Ctrl-C to stop)"
  sleep 5
fi

clips=()
for i in $(seq 1 "$N"); do
  out=$(caffeinate -i "$PY" -m cctv.ingest --camera "$CAM" --seconds 60 | tee /dev/stderr | grep -Eo 'clip=[^ ]+' | tail -1 | cut -d= -f2 || true)
  [ -n "$out" ] && clips+=("$out")
  [ "$i" -lt "$N" ] && sleep 240
done
[ ${#clips[@]} -gt 0 ] || { echo "no clips recorded"; exit 1; }

match=$("$PY" -m cctv.calibrate match --camera "$CAM" --clip "${clips[0]}" 2>&1 || true)
echo "$match"
if ! echo "$match" | grep -q "^MATCHED" || echo "$match" | grep -q "not calibrated"; then
  cat <<MSG

No calibrated view of $CAM matches these clips. Calibrate this view (daylight, lane dashes visible):
  $PY -m cctv.calibrate add-view --camera $CAM --view v1 --clip ${clips[0]} --label "<what the camera shows>"
  # click dash corners on http://127.0.0.1:8010/calibrate, save as cctv/data/calibration/$CAM/v1.points.json
  # add "direction_labels": {"+": "EB", "-": "WB"} (which way +X travel goes) so the link comparison can run
  $PY -m cctv.calibrate fit --camera $CAM --view v1
then re-run the pipeline on the clips:
$(for c in "${clips[@]}"; do echo "  caffeinate -i $PY -m cctv.pipeline --camera $CAM --clip $c"; done)
  $PY -m cctv.groundtruth.nyc_links compare --fetch-missing
MSG
  exit 0
fi

for c in "${clips[@]}"; do caffeinate -i "$PY" -m cctv.pipeline --camera "$CAM" --clip "$c" | tail -5; done
"$PY" -m cctv.groundtruth.nyc_links compare --fetch-missing --json cctv/data/groundtruth/cache/compare_latest.json
