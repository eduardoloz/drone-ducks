# jetson/: face detection + face-follow for a Jetson on the drone

Camera -> face detector -> tracker -> controller -> PX4 (MAVSDK Offboard velocity setpoints).
Everything runs on a laptop today; the same code runs on a Jetson later. **Nothing is sent to a
drone unless you pass `--fly`.**

- What to buy, Nano vs Orin, range maths, detector choice, safety, legal: [`docs/WHAT_WE_NEED.md`](docs/WHAT_WE_NEED.md)
- How it fits together + latency budget + sim/PX4 interfaces: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- The day the hardware arrives: [`docs/BRINGUP_CHECKLIST.md`](docs/BRINGUP_CHECKLIST.md)

## Quickstart (laptop, from the repo root)

```bash
uv venv --python 3.12 jetson/.venv
uv pip install --python jetson/.venv/bin/python -r jetson/requirements.txt
source jetson/.venv/bin/activate
python -m jetson.fetch models          # YuNet face (MIT) + YOLOX-nano person (Apache-2.0) -> jetson/models/ (gitignored)
python -m jetson.fetch media           # NASA public-domain photos + test video -> jetson/media/ (gitignored)
python -m pytest jetson/tests -q       # 268 tests, ~45 s on a loaded Intel Mac (16 skip without fetched models/media)

python -m jetson.demo --source toy --out jetson/out/toy.mp4          # closed loop in a toy world
python -m jetson.demo --source jetson/media/crew_pan.mp4 --out jetson/out/demo.mp4
python -m jetson.demo --source 0 --show --person yolox             # + body path: keeps the lock when you turn round
python -m jetson.lockeval              # lock retention, face-only vs face+body (toy world); --end-on-walkaway, --replay-sitl
python -m jetson.demo --source 0 --show                              # webcam
python -m jetson.demo --source http://PHONE_IP:8080/video --show     # phone IP-cam (like BTMOB.py)
python -m jetson.bench                 # detector speed on THIS machine (labelled, with load average)
python -m jetson.doctor                # what works on this machine
python -m jetson.geometry              # how far away can we see a face?
```

SITL (after the sim/ track has PX4 SITL + the camera bridge on :5601 running):
`python -m jetson.follow --source sim --camera-preset sim` (dry run), then add
`--fly --yes --auto-takeoff 3 --on-exit land`. Details in `docs/ARCHITECTURE.md` section 5.

## Layout

| File | What |
|---|---|
| `sources.py` | camera inputs (file, USB, `csi://0`, `sim`, MJPEG, `tcp://`) + `LatestFrameReader` (newest frame only) |
| `detectors/` | `yunet` (default), `haar`, `onnx` (YuNet/SCRFD via ONNX Runtime: CPU/CUDA/TensorRT), `tensorrt`, `jetson_inference`, `ultralytics` |
| `tracker.py` | IoU tracker (ByteTrack-style low-score pass) + sticky target selection |
| `detectors/person.py`, `lock.py` | optional body path: person detector (YOLOX-nano, HOG, TensorRT) + face<->body `LockKeeper` |
| `lockeval.py` | lock-retention scenarios (toy world), recorded-clip check, SITL-log replays (hand-back, walk-away) |
| `walkaway.py` | optional walk-away end: person walks off with their back turned -> end the follow |
| `relock.py` | confirmed re-lock after a loss (N hits on one track, distance cap) |
| `headpose.py`, `headeval.py` | head-yaw proxy from the 5 face landmarks; closed-loop check of the sideways "stay in front" term |
| `controller.py` | pure face-follow controller (deadbands, gains, limits, slew, lost-target -> hover -> land) |
| `bridge.py` | dry-run sink and MAVSDK Offboard bridge (20 Hz, pilot override, stall/timeout safety) |
| `pipeline.py`, `hud.py` | glue + on-screen display (blurs bystander faces in recordings) |
| `toysim.py` | kinematic drone + walking person for closed-loop tests |
| `scripts/` | `export_trt.py` (ONNX -> TensorRT, `--dry-run` anywhere), `check_compat.py` (Python 3.8 checks + multi-interpreter test run), `setup_jetson_jp6.sh` |
| `docker/Dockerfile.jp6` | JetPack 6 container recipe (unverified: no Jetson yet) |
| `demo.py`, `follow.py`, `bench.py`, `fetch.py`, `doctor.py`, `linkcheck.py`, `geometry.py` | command-line tools |

## Python versions

Minimum **Python 3.8**. The full suite passes on 3.8.20, 3.10.0 (with JetPack 6's OpenCV 4.8 and numpy 1.x) and
3.12.8: `python jetson/scripts/check_compat.py --run-tests jetson/.venv38/bin/python ...`. Orin / JetPack 6: system Python 3.10. Original
Nano / JetPack 4 ships 3.6, which is too old for MAVSDK and modern OpenCV; use Ubuntu's `python3.8`
(`requirements-jetson-jp4.txt`) or the Ultralytics JetPack 4 container.

## Keeping the lock when the face turns away

Face-only following loses the person as soon as they turn round or walk away. `--person yolox`
adds a whole-body detector; the follower then ties the target's face to their body, steers by the
body while the face is hidden and switches back when the face reappears inside that body. It never
starts following a body that hasn't shown its face, never switches to a bystander's face, and gives
up after 15 s without a face. Toy-world result: 100 % lock retention vs 16-19 % face-only, and the
bystander-steal case goes from 73 % wrong-person to 0 %. In a dense group (crew photo) it fails safe
(hovers) but does not keep the lock. Details and caveats: `docs/ARCHITECTURE.md` section 2a.
Detection and tracking only: nothing recognises who anyone is, and bystander blur still applies.

## Camera presets, re-lock, staying in front

- `--camera-preset sim|imx219|imx219-wide|webcam` sets HFOV and face width together (sim: 68.75 deg,
  0.13 m; default imx219: 62.2 deg, 0.16 m). `--hfov` / `--face-width` override it. Measure face width per camera.
- After the face is lost, tracking resumes only on a **confirmed** re-lock: 3 consecutive detections
  on one track within target distance + 1.5 m (`--relock-*` flags). This blocks the back-of-head re-lock seen in SITL.
- `--head-lateral 0.02` (off by default) flies sideways to stay in front of the person, based on which way
  their face points. Toy closed loop: 100 % lock on 0-45 deg walker legs vs 8-39 % without it.
  `python -m jetson.headeval`. Details: `docs/ARCHITECTURE.md` 2b/2c.

## Ending the follow when the person walks away (`--end-on-walkaway`, off by default)

With the body lock the drone keeps following someone who turned their back on it, so a person who
walks off gets chased until their body is lost (SITL round 3: out to 11-13 m). `--end-on-walkaway`
ends the follow instead, through the same path as a lost target: TIMEOUT, hover, then
`--lost-action` (land by default; `hold` hands the drone back in Hold mode).

"Walking away" means, over the last `--walkaway-seconds` (3 s):
- the drone is steering by the **body** now and in at least 2/3 of those frames (a face that has not
  passed the hand-back confirmation, e.g. the back of the head, counts as not seen),
- the person moves away **over the ground** at `--walkaway-speed` (0.8 m/s) or faster. This is the
  box distance *plus* how far the drone flew toward them, so it still works while the drone keeps up,
- and they are farther than target distance + `--walkaway-margin` (1 m).

It only acts with `--person yolox` (face-only mode ends on the lost timeout first). The CSV log
(`--csv`) gets a `recede_m_s` column with the speed estimate.

| Check | Result |
|---|---|
| Toy world, person turns and walks off at 1.3 m/s (`lockeval --scenario leaves --end-on-walkaway`) | ended 1.9 s after the turn, 5/5 seeds (off: chased 100 % of the run) |
| Toy world, stands with the back turned twice for 6 s (`turn_around`) / face always visible | never ends, 5/5 seeds |
| SITL replay, `r3-alt3-body-t5` (3 m, 5 m target) | ends at 69.2 s, 2.3 s after the walker set off at 1.37 m/s (true distance 8.2 m); round 3 chased it until LOST at ~81 s |
| SITL replay, `r3-turnaway-body` | ends at 76.9 s, 2.8 s into the 1.37 m/s walk-away (7.8 m); the earlier slow 0.39 m/s drift away (28-41 s) does not end it at 0.8 m/s (it does at 0.6) |
| SITL replay, the three face-only logs | never triggers |

The replays (`python -m jetson.lockeval --replay-sitl sim/docs/runs/*.csv.gz`) are open loop: they feed
each logged frame's detections through the current lock and the walk-away check, and compare with
the walker's true facing and speed, but the logged drone did not actually stop. Not flown in SITL yet.
Why these thresholds: `jetson/walkaway.py` docstring.

## Recommended settings at 3 m altitude

From the round-3 SITL runs (body lock at 3 m / 5 m target: 83 s followed, 1 lost event, vs 27.6 s and
6 lost events face-only):

```bash
python -m jetson.follow --source sim --camera-preset sim --person yolox --target-distance 5 \
    --fly --yes --auto-takeoff 3 --on-exit land --csv jetson/out/sitl.csv
# optional: --end-on-walkaway   (end the session when the person walks off)
```

- `--target-distance 5` with `--person yolox` (body tracking on). The re-lock distance cap follows it
  (5 + 1.5 = 6.5 m).
- Pitch the camera down ~20 deg at 3 m (the sim world does; on the drone it is the mount angle) so
  the face stays near the image centre.
- Distance is horizontal (ground) distance when the altitude is known: `--fly` reads PX4's relative
  altitude, so nothing to set. Without telemetry (dry runs) the raw range is used, which stops ~0.2-0.4 m
  short at 3 m; `FollowConfig(assumed_altitude_m=3.0)` fixes that in code.
- Ego-rotation compensation (tracker boxes shifted by the drone's own yaw) is on by default.

## Rules

Never commit images, videos, model weights or venvs (the `.gitignore` here covers them). Test media
are NASA public-domain photos of real people: local testing only. Only film consenting team members.

## Good first tasks

1. **Calibrate distance:** stand at 3 m and 5 m in front of a webcam, adjust `--face-width` until the
   HUD's `dist` is right. Add the number to `docs/WHAT_WE_NEED.md`.
2. **Feed-forward:** P-only control trails a walking person (`test_follows_a_walking_person...`
   explains why). Add a yaw/forward feed-forward from the tracker's velocity and make that test hit 4 m.
3. **Crop re-detection for range:** after a face is found at 640 px, run the detector on a full-res crop
   around it. Measure the extra range with `--source toy --toy-width 1280`.
4. **Plot a run:** write a small script that plots `--csv` logs (bearing, distance, commands vs time).
