# cctv/: traffic speed analytics on Long Island Expressway cameras

Estimate how fast traffic moves past public NYSDOT cameras on the LIE (I-495) and show
it on a website: a camera map, speed distributions, and anonymous per-vehicle speed
estimates with ± uncertainty. This is **traffic analytics, not enforcement**: no plates,
no faces, no identities (see [docs/ARCHITECTURE.md §9](docs/ARCHITECTURE.md#9-ethics-privacy-and-legal-required-reading)).

![Tracked vehicles on camera R10_041 with speed estimates](docs/img/r10_041_tracks.jpg)

*Camera R10_041 (LIE at Exit 57), 2026-11-10 16:40. Yellow boxes are vehicles inside the
calibrated stretch, labelled with their speed estimate ± 1σ in mph; grey boxes are
outside it. Video: NYSDOT / 511NY.*

## What's here

| Path | What it does |
|---|---|
| `docs/SOURCES.md` | Where the feeds come from (opencctv.org → 511NY/NYSDOT), terms, the measured fps and timing of 19 cameras, how many cameras can do speeds |
| `docs/ARCHITECTURE.md` | Website architecture: diagram, data model, ports, scaling, cost, error budget, validation plan, ethics, milestones |
| `docs/VALIDATION_DRIVE.md` | How to drive past a camera with GPS (safely, legally) and measure our speed error |
| `data/cameras_lie.json` | **The camera manifest**: 129 cameras from the Queens-Midtown Tunnel to Riverhead |
| `data/feed_measurements.json` | Measured feed characteristics (resolution, fps, timestamp jitter, repeated frames) |
| `data/calibration/R10_041/` | One camera, two PTZ presets: `v1` calibrated (points, homography, reference frame), `v2` known but not calibrated (night reference) |
| `data/groundtruth/nyc_lie_links.json` | Snapshot of NYC DOT's live LIE speed links and the NYSDOT cameras on them |
| `registry.py` | Load the manifest |
| `geometry.py` | Pixel ↔ road-metre homography, calibration files |
| `speed.py` | Track → speed ± σ, rejects bad tracks, conservative over-limit flag |
| `camera_motion.py` | Detects that a pan-tilt-zoom camera was moved since calibration |
| `views.py` | Several calibrations per camera (one per PTZ preset); picks the preset a clip shows, and compensates a preset that came back a few pixels off |
| `daylight.py` | Sun elevation at a camera (no network): sunrise/sunset, day / twilight / night label stored with every run |
| `ingest.py` | Records short HLS clips; reads exact frame timestamps |
| `detect_track.py` | YOLO + ByteTrack over a clip (Stephen's `ByteTrackFinal.py` idea, offline) |
| `pipeline.py` | Clip → tracks → speeds → SQLite |
| `web/` | FastAPI site on port 8010: map, speed estimates, stats, calibration tool, About |
| `calibrate.py` | Calibration helper CLI: fit, draw a check grid, build an empty-road frame |
| `error_budget.py` | Monte Carlo of every error source for a calibrated camera |
| `recon/` | `build_manifest.py` (offline) and `measure_feed.py` (records 20 s per camera) |
| `groundtruth/nyc_links.py` | Compares our speeds with NYC DOT's published link speeds (open data `i4gi-tjb9`) |
| `groundtruth/gps_drive.py` | Lines a phone GPS log up with our tracks and reports the speed error |
| `scripts/record_daylight.sh` | **One command to record daylight footage** (R10_041, R10_007, R10_008 by default); recording only |
| `scripts/groundtruth_session.sh` | Daylight session: a few 60 s clips from one camera, then pipeline and link comparison |
| `tests/` | 103 pytest tests (no network, no YOLO needed), incl. a regression test on real R10_041 tracks (`tests/data/`, boxes only) |

## Quickstart

Commands run from the **repo root** (`drone-ducks/`). They were tested on the 2019 Intel
MacBook with Python 3.12.

```bash
# 1. Make a venv inside cctv/ and install the light dependencies (no torch)
uv venv --python /usr/local/bin/python3.12 cctv/.venv
uv pip install --python cctv/.venv/bin/python -r cctv/requirements.txt
source cctv/.venv/bin/activate

# 2. Run the tests
python -m pytest cctv            # 103 passed

# 3. Start the site (it reads cctv/data/db/cctv.sqlite, which is empty on a fresh clone)
python -m cctv.web.app           # http://127.0.0.1:8010
```

To produce your own speed data you need the detection stack (torch, ultralytics; about
1 GB):

```bash
uv pip install --python cctv/.venv/bin/python -r cctv/requirements-detect.txt

# record 60 s from a camera (one stream, short clip: be a polite viewer)
python -m cctv.ingest --camera R10_041 --seconds 60

# or: capture daylight footage from the three cameras we work on, in one command
# (2 rounds x 3 cameras x 60 s, ~11 MB, ~15 min; refuses to run after dark unless --force)
cctv/scripts/record_daylight.sh
python -m cctv.daylight --camera R10_008        # sunrise / sunset / is it daylight there now?

# detect + track + speeds -> SQLite (YOLO11n on CPU: ~2 min per minute of video here)
python -m cctv.pipeline --camera R10_041 --clip cctv/data/clips/R10_041_<timestamp>.ts

python -m cctv.web.app           # refresh the page
python -m cctv.web.app --publish-delay-min 15   # how a public deployment should run (per-vehicle data delayed)
```

The pipeline first works out which view the camera is showing. LIE cameras are
pan-tilt-zoom, and **camera 041 was re-aimed within an hour of being calibrated** on
2026-11-10. So each camera can hold several calibrations, one per preset ("view"). The
pipeline compares the clip with every view's reference frame and uses the calibration
of the one that matches. If the matching view isn't calibrated, or no view matches,
tracks are stored for counts only, without speeds, and the site says so.

```bash
python -m cctv.calibrate views --camera R10_041                       # v1 calibrated, v2 not
python -m cctv.calibrate match --camera R10_041 --clip cctv/data/clips/<clip>.ts
```

## Ground truth (round 2: tools built, real comparison still to run)

```bash
python -m cctv.groundtruth.nyc_links links      # NYC DOT LIE links + the NYSDOT cameras on them
python -m cctv.groundtruth.nyc_links fetch --start 2026-11-13T12:00 --end 2026-11-13T13:00
python -m cctv.groundtruth.nyc_links quality    # is a link plausible? (some aren't)
python -m cctv.groundtruth.nyc_links compare    # our runs vs link speeds, same direction only

# daylight, on AC power: 3 short clips from a camera on a link, then compare (stops to ask for a calibration if needed)
cctv/scripts/groundtruth_session.sh R10_008 3
```

The validation drive is described in [docs/VALIDATION_DRIVE.md](docs/VALIDATION_DRIVE.md)
(`python -m cctv.groundtruth.gps_drive passes | candidates | report`).

## Calibrating a camera

Speeds need a map from pixels to metres on the road. We use the lane markings: dashed
lines are 10 ft long with 30 ft gaps, and lanes are 12 ft wide.

Each preset (view) of a camera lives in `cctv/data/calibration/<CAM>/` as `<VIEW>.ref.jpg`,
`<VIEW>.points.json` and `<VIEW>.json`. Name views `v1`, `v2`, ... and describe them with `--label`.

```bash
python -m cctv.calibrate add-view --camera <CAM> --view v1 --clip cctv/data/clips/<CAM>_<ts>.ts --label "looking east, both carriageways"
# click dash corners in the browser: http://127.0.0.1:8010/calibrate (or write the JSON by hand)
#   -> save as cctv/data/calibration/<CAM>/v1.points.json (copy R10_041/v1.points.json for the format)
#   -> optional "direction_labels": {"+": "EB", "-": "WB"} (compass direction of +X travel; needed for link comparison)
python -m cctv.calibrate fit --camera <CAM> --view v1          # writes <CAM>/v1.json, prints errors
python -m cctv.calibrate show --camera <CAM> --view v1 --frame cctv/data/calibration/<CAM>/v1.ref.jpg --out check.jpg
python -m cctv.error_budget --camera <CAM> --view v1             # how good can speeds be here?
```

The round-1 flat layout (`<CAM>.points.json` → `<CAM>.json`, `<CAM>.ref.jpg`, no `--view`)
still works as the view `default`. `python -m cctv.calibrate migrate --camera <CAM> --view v1`
moves it into a folder.

![Calibration check grid on R10_041](docs/img/r10_041_calibration.jpg)

*Check grid: cyan lines every 12 ft across the road, yellow every 10 ft along it. They
should sit on the paint. Red rings are the clicked points; the magenta outline is the
stretch where speeds are measured.*

Gotcha: with only two parallel lane lines, the calibration cannot tell whether the road
is "sheared". Add two or more `transverse_lines` (points along concrete pavement joints,
which cross the road at right angles), as in `R10_041.points.json`.

## Results so far (all measured on 2026-11-10)

- **Sources:** the 129 LIE cameras are 108 NYSDOT HLS video streams (about 15 fps; 640×480
  in Nassau/Suffolk, 352×240 in Queens) and 21 NYC DOT still images (refreshed every
  ~2 s, so no speeds).
- **Feed timing:** frame timestamps on the Long Island encoders step 66.7 ms with 0.4 ms
  jitter. The east-end cameras drop and repeat frames, and burned-in clocks are minutes
  off.
- **Tracking:** YOLO11n + ByteTrack at 640 ran at 116 ms/frame on this CPU (shared). On
  R10_041's 60 s clip, 66 tracks touched the calibrated stretch, none fragmented, and 5
  had a gap of 3 or more frames.
- **Speeds on R10_041:**
  - 62 vehicles in 60 s; 54 got a speed.
  - Median 57.2 mph, p85 62.0, mean σ 1.7 mph (posted limit 55).
  - Every accepted track ran within 2° of the lane direction.
- **Error budget (60 mph):**
  - Box jitter, occlusion and timestamps each add under 0.6 mph.
  - Paint tolerance (±2.5%) adds ±1.5 mph and dominates.
  - An undetected 8 px camera nudge would cost −3 mph.
  - Realistic mix: −0.8 ± 1.5 mph.

**Round 2 (2026-11-12/13; it was night in New York, so no new footage):**
- Per-preset calibrations work on the real R10_041 frames. The 16:39 clip and three
  stills from 16:45–16:48 match v1. The 17:43 night clip and a 17:46 still match v2,
  which is known but not calibrated.
- NYC DOT link speeds: the client, the camera-link matching and the comparison table are
  built. A night-time quality check found the 84th–108th St links implausible (20.8 mph
  median at 3 am EB; WB mostly status −101). The Little Neck Pkwy WB link looks sane
  (54.7 mph) and has Region 10 cameras R10_005–R10_008 on it.

**Round 3 (2026-11-17, footage at sunset; clips are git-ignored in `cctv/data/clips/`):**
- Recorded (60 s each, 640×480, ~1.8 MB each): R10_041 at 16:26 and 16:31, R10_008 at 16:27
  and 16:29, R10_007 at 16:28 (sun +1° to −0.5°, clear light, heavy rush-hour traffic);
  R10_008 at 16:47 (sun −3°, twilight); R10_008 at 18:51 (night, only 35 s because the
  laptop slept). R10_041 at 18:52 failed (stream I/O error).
- **No speeds from this footage, so no estimate-vs-link table yet.** R10_041 sat at a preset
  that matches neither v1 nor v2 (a wide view looking along the westbound carriageway).
  R10_007 and R10_008 have no calibration yet (mentee task). The pipeline ran end to end
  on R10_008 and stored counts only.
- NYC DOT link 4362251 (WB Little Neck Pkwy → CVE, R10_007/008): at 16:40 the newest row in
  the open-data API was from 14:27, so the API was running about 2 h behind. The rows for the
  clip window still need fetching:
  `python -m cctv.groundtruth.nyc_links fetch --start 2026-11-17T16:00 --end 2026-11-17T17:00 --link 4362251`.
- Detector at sunset vs twilight on R10_008 (yolo11n, the first 30 s of each clip, measured):

  | Clip | Sun | Boxes/frame | Tracks | Tracks ≥ 15 frames | Median track length |
  |---|---|---|---|---|---|
  | 16:27 | +0.4° | 16.0 | 253 | 119 | 10 frames |
  | 16:47 | −3.0° | 15.5 | 256 | 103 | 8.5 frames |

  Eye count, by one person, of vehicles below the dense far queue on 2 frames per clip:
  about 30 of 52 had a tracked box at sunset (~58%) and 28 of 56 at twilight (~50%). Misses
  are mostly near-field vehicles on the left (eastbound) carriageway, which faces the low
  sun, and small far vehicles. Recall on this camera is much lower than on R10_041 v1.
  A bigger model or imgsz is worth testing before speeds here mean much.
  The night clip has not been run through the detector yet (the laptop was on battery).
- ultralytics resets torch to `min(8, cpus − 1)` threads on its first CPU predict, which
  overrides `OMP_NUM_THREADS` (measured: 613% CPU with `OMP_NUM_THREADS=2`). The tracker now
  puts the cap back (`--threads`, `$CCTV_THREADS` or `$OMP_NUM_THREADS`).
- The R10_041 reference clip of 2026-11-10 (54 of 62 vehicles with speeds) was itself
  recorded at sun −0.9°. The site labels it "twilight", so twilight alone doesn't mean
  the run is bad.

### Night

Each run stores the sun elevation at the camera and a label: day (≥ 0°), twilight, or night
(< −6°), from `daylight.py`. The site shows a note on twilight and night runs. YOLO11n is
trained on daylight photos. At night the LIE cameras show headlights and tail-light streaks
(see `data/calibration/R10_041/v2.ref-night.jpg`), so expect undercounts and weaker tracks.
That hasn't been measured yet: a 35 s night clip of R10_008 from 18:51 on 2026-11-17 is
waiting to be run (`python -m cctv.pipeline --camera R10_008 --clip cctv/data/clips/R10_008_20261117T235157Z.ts --max-frames 450`).

**Not verified yet:**
- no ground truth: no Queens/Little Neck camera is calibrated, so no estimate-vs-link
  table exists yet, and no drive has been done (tools and protocol are ready);
- the far half of each track reads ~4% slower than the near half (braking or a bias? to be
  investigated);
- night-time accuracy (clip recorded, not processed);
- small-offset compensation on a real nudged clip (synthetic tests only);
- the 55 mph limit in Nassau/Suffolk comes from secondary sources.

## Good first tasks

1. **Calibrate a second camera.**
   - Pick one marked `measured_good` in the manifest, e.g. R10_019 or R10_044.
   - Record a daytime clip, build the background frame, and click dash corners on
     `/calibrate`.
   - Fit, check the grid, run the error budget, and open a PR with the `.points.json`,
     `.json` and `.ref.jpg`.
2. **Re-measure feeds in daylight.**
   - Run `python -m cctv.recon.measure_feed --sample 20` around noon, then
     `python -m cctv.recon.build_manifest`.
   - R10_355 was measured after dark and needs a redo. Which cameras changed resolution?
3. **Compare with NYC DOT speeds** (the harness exists: `groundtruth/nyc_links.py`).
   - Calibrate R10_007 or R10_008 (on the Little Neck Pkwy WB link) in daylight, with
     `direction_labels`, using `cctv/scripts/groundtruth_session.sh R10_008 3`.
   - Run `nyc_links compare` and paste the table, with its caveats, into this README.
4. **Speed up tracking.**
   - Export YOLO11n to OpenVINO (`yolo export format=openvino`) and compare ms/frame and
     detections against PyTorch on the same clip.
   - Report the numbers in this README.

## Rules of the road for this folder

- One stream at a time, clips of 60 s or less. Never commit video, frames, weights or
  databases; `cctv/.gitignore` covers them. The only images committed are calibration
  reference frames and the two docs images.
- No licence-plate reading, face detection or vehicle re-identification. Not now, not "just
  to test".
- Show uncertainty with every speed, and say "flagged over limit", never "speeder".
- Before anything runs continuously, the 511NY developer access request must be filed
  (see docs/SOURCES.md).
