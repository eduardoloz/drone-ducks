# Validation drive: checking our speeds against a car with GPS

Goal: drive past a calibrated camera at a steady, known speed while we record the
camera, then compare our estimate for that car with the car's GPS speed. Ten
passes give a bias (do we read high or low?) and a spread (how far off is one
estimate?) for that camera. Tooling: `python -m cctv.groundtruth.gps_drive`.

## Safety and legal rules (non-negotiable)

- **The driver only drives.** A *passenger* runs the GPS logger and notes pass
  times. Nobody touches a phone at the wheel; in New York that's illegal anyway
  (VTL §1225-c/d).
- **Legal speed, normal traffic.** Drive at whatever is legal and safe, typically
  cruise control at or below the limit in the right or middle lane. We don't need
  any particular speed. A steady speed matters more than a round number. Never
  speed, weave or slow down for the camera.
- **Daylight, dry road, light-to-moderate traffic.** Calibrations are daytime and
  we can't measure night accuracy yet. Free-flowing traffic keeps the car steady.
- **No stopping on the shoulder or at the camera.** Passes are just driving
  through. Turn around at the next exits.
- **Privacy.** We identify *our own* car in the video by time, lane and class. We
  never read plates and never record anyone else's identity. Don't film other
  drivers from inside the car.
- Two people minimum; the driver has a licence and insurance for that car.

## Before you go

1. **Pick a calibrated camera and preset.** `python -m cctv.calibrate views --camera R10_041`
   must list a calibrated view. On the day, check that the camera still shows that
   view: record a 10 s clip and run `python -m cctv.calibrate match --camera <CAM> --clip <clip>`.
   If it reports `NO_MATCH`, the operator moved the camera, so choose another one.
2. **Know where the calibrated stretch is.** Open the camera on the site (`:8010`).
   The magenta outline on the reference frame is the measured stretch. Note which
   lanes it covers. The car has to drive through it, in a covered lane.
3. **GPS logger on the passenger's phone.**
   - Android: *GPSLogger* (open source). Log every 1 s, GPX or CSV, and turn on
     "log speed".
   - iOS: any logger that exports GPX with speed, e.g. *GPS Tracks* or *myTracks*.
     Check that `<speed>` appears in the export.
   - Set logging to **1 Hz or faster** and **keep the screen on**. Some phones throttle GPS
     in the background.
   - Phone **clock on automatic network time** (it's what aligns GPS with video).
4. **Plan 10 passes**: at least 5 per direction if both carriageways are calibrated,
   otherwise 10 in the calibrated direction. Use 2–3 different lanes and 2 speeds
   (e.g. 45 and 55 mph, both legal) so a scale error can be told apart from an offset.
5. **Laptop** (stays in the car, run by the passenger or by a third person at home):
   `cctv/.venv` installed, on AC or a full battery.

## During the drive

For each pass:

1. The passenger starts a 60 s recording about 30 s before the camera (or the person at
   home runs it on request):
   `caffeinate -i python -m cctv.ingest --camera <CAM> --seconds 60`.
   Video reaches us 5–20 s late, so start early.
2. The driver holds a steady speed on cruise control from about 500 m before the
   camera to 200 m after it, and stays in one lane.
3. The passenger notes the pass in a text file: `pass 3, 13:42, middle lane, 55 cruise, EB`.
   The lane and direction are what let us find the car in the video.
4. One stream at a time, clips of 60 s or less. This is **not** continuous ingest.

## After the drive

```bash
# 1. track + speeds for every clip (YOLO on CPU, ~2 min per clip on the Intel Mac)
for c in cctv/data/clips/<CAM>_*.ts; do caffeinate -i python -m cctv.pipeline --camera <CAM> --clip "$c"; done

# 2. when did the car pass, how fast, which way? (put the log in cctv/data/groundtruth/drives/, which is gitignored)
python -m cctv.groundtruth.gps_drive passes --gps cctv/data/groundtruth/drives/2026-11-21.gpx --camera <CAM>

# 3. which tracks could be our car? (+-45 s window around each pass)
python -m cctv.groundtruth.gps_drive candidates --gps cctv/data/groundtruth/drives/2026-11-21.gpx --camera <CAM>
```

4. **Find the car by eye.** For each pass, open the run's replay on the site and pick the
   track in the right lane and direction, of the right class, at about the right
   time. The offset between video and GPS is roughly constant during one drive (it is the
   stream latency), so once two passes are matched, the others line up. If two tracks
   are equally plausible, **skip the pass**. Don't guess.
5. Write `matches.json` next to the log:

   ```json
   [{"pass": 0, "run_id": 12, "track_id": 57, "note": "middle lane, silver hatchback = ours"},
    {"pass": 1, "run_id": 13, "track_id": 8}]
   ```

6. Report:

   ```bash
   python -m cctv.groundtruth.gps_drive report --gps cctv/data/groundtruth/drives/2026-11-21.gpx \
       --camera <CAM> --matches cctv/data/groundtruth/drives/2026-11-21.matches.json \
       --json cctv/data/groundtruth/drives/2026-11-21.report.json
   ```

   It prints a per-pass table (GPS speed, our estimate ± σ, error, clock offset) and
   a summary: bias with its standard error, SD, RMSE, the share of passes within
   2σ, and the median clock offset. That offset measures the stream's latency
   as a by-product.

## How to read the result

- **GPS speed is the truth here.** Phone Doppler speed is good to about ±0.5 mph at
  highway speed when the sky is open. The car's speedometer usually reads 1–3 mph
  high, so don't use it. `passes` shows the GPS speed's spread over ±3 s; if it
  is over 1 mph, the car wasn't steady and that pass is weak.
- **Bias** tells us whether to correct the calibration scale. With 10 passes, a
  bias is meaningful when it's bigger than about 2× its standard error. Compare
  it with the error budget (`python -m cctv.error_budget --camera <CAM>`): the
  paint-tolerance term alone allows ±1.5 mph.
- **Spread**: our σ claims about 1.5–2 mph. If far fewer than ~95% of passes land within
  2σ, our σ is too optimistic, so say so in the README.
- Put the table in the PR and in `cctv/README.md` ("Results so far"), with the
  date, the camera, the preset, the number of passes and the conditions. One camera
  doesn't validate every camera.

## GPS log formats the tool reads

| Format | Needed | Optional |
|---|---|---|
| GPX 1.0/1.1 | `<trkpt lat lon>` with `<time>` (UTC "Z" or with an offset) | `<speed>` in m/s (top level or in `<extensions>`), `<hdop>` |
| CSV with header | `time`/`timestamp` (ISO 8601 with zone, or unix s/ms), `lat`/`latitude`, `lon`/`lng`/`longitude` | `speed` (m/s), `speed_mph`, `speed_kmh`, `accuracy` (m) |

Timestamps without a time zone are rejected, because a silent 5-hour error would
match the wrong cars. Without a speed column, speed comes from positions. That works,
but it's noisier, and the output says `positions` instead of `doppler`.
