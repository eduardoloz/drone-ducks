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
python -m pytest jetson/tests -q       # 219 tests, ~25 s (15 skip without fetched models/media)

python -m jetson.demo --source toy --out jetson/out/toy.mp4          # closed loop in a toy world
python -m jetson.demo --source jetson/media/crew_pan.mp4 --out jetson/out/demo.mp4
python -m jetson.demo --source 0 --show --person yolox             # + body path: keeps the lock when you turn round
python -m jetson.lockeval              # lock retention, face-only vs face+body (toy world)
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
| `lockeval.py` | lock-retention scenarios (toy world) and recorded-clip check |
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
