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
python -m jetson.fetch models          # YuNet face model (MIT, 230 KB) -> jetson/models/ (gitignored)
python -m jetson.fetch media           # NASA public-domain photos + test video -> jetson/media/ (gitignored)
python -m pytest jetson/tests -q       # 151 tests, ~20 s

python -m jetson.demo --source toy --out jetson/out/toy.mp4          # closed loop in a toy world
python -m jetson.demo --source jetson/media/crew_pan.mp4 --out jetson/out/demo.mp4
python -m jetson.demo --source 0 --show                              # webcam
python -m jetson.demo --source http://PHONE_IP:8080/video --show     # phone IP-cam (like BTMOB.py)
python -m jetson.bench                 # detector speed on THIS machine (labelled, with load average)
python -m jetson.doctor                # what works on this machine
python -m jetson.geometry              # how far away can we see a face?
```

SITL (after the sim/ track has PX4 SITL + the camera bridge on :5601 running):
`python -m jetson.follow --source sim --hfov 68.75` (dry run), then add
`--fly --yes --auto-takeoff 3 --on-exit land`. Details in `docs/ARCHITECTURE.md` section 5.

## Layout

| File | What |
|---|---|
| `sources.py` | camera inputs (file, USB, `csi://0`, `sim`, MJPEG, `tcp://`) + `LatestFrameReader` (newest frame only) |
| `detectors/` | `yunet` (default), `haar`, `onnx` (YuNet/SCRFD via ONNX Runtime: CPU/CUDA/TensorRT), `tensorrt`, `jetson_inference`, `ultralytics` |
| `tracker.py` | IoU tracker (ByteTrack-style low-score pass) + sticky target selection |
| `controller.py` | pure face-follow controller (deadbands, gains, limits, slew, lost-target -> hover -> land) |
| `bridge.py` | dry-run sink and MAVSDK Offboard bridge (20 Hz, pilot override, stall/timeout safety) |
| `pipeline.py`, `hud.py` | glue + on-screen display (blurs bystander faces in recordings) |
| `toysim.py` | kinematic drone + walking person for closed-loop tests |
| `demo.py`, `follow.py`, `bench.py`, `fetch.py`, `doctor.py`, `linkcheck.py`, `geometry.py` | command-line tools |

## Python versions

Minimum **Python 3.8** (tests pass on 3.8.20 and 3.12.8). Orin / JetPack 6: system Python 3.10. Original
Nano / JetPack 4 ships 3.6, which is too old for MAVSDK and modern OpenCV; use Ubuntu's `python3.8`
(`requirements-jetson-jp4.txt`) or the Ultralytics JetPack 4 container.

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
