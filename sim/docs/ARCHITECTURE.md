# sim/ architecture

_2026-11-10. Verified on Eddie's Intel MacBook Pro (macOS 15, PX4 main @ v1.18.0-beta1-1178, Gazebo 8.10,
MAVSDK-Python native 4.0.6). Not yet tried on Apple Silicon, Linux or Windows._

## Components and data flow

```
 your code / TypeFly skills / Jetson follower
        |  Drone API (sync)                         CameraSource.read()  <- camera.py
        v                                                  ^
 quackdrone.Drone ---- supervisor thread (20 Hz):          |  HTTP :5601 (/frame raw, /video MJPEG)
   limits, watchdog, override + battery + link checks      |
        |  Backend                                  gz_camera_bridge.py (Gazebo's python)
        v                                                  ^ gz-transport
 MavsdkBackend (native mavsdk, sync API) --MAVLink UDP 14540--> PX4 SITL <--gz--> Gazebo (x500_quack, world)
                                         (QGC / test observer: 14550)
 FakeBackend (fake://): pure-Python stand-in for unit tests
```

* **PX4 owns all control loops.** quackdrone streams Offboard setpoints: position+yaw (`goto_ned`, `move`,
  `turn`) or body velocity+yaw rate (`set_velocity_body`).
* `scripts/start_sim.sh` runs Gazebo in PX4's *standalone* mode: we start `gz sim` with our world,
  spawn `x500_quack_0`, and PX4 attaches (`PX4_GZ_STANDALONE=1`, `PX4_GZ_MODEL_NAME`). PX4 state
  lives in `sim/.run/px4-gz/` (`-w` rootfs), so **nothing is written into the PX4 checkout** and every
  start uses fresh params plus `px4/sitl.params`.
* TypeFly: pinned upstream commit `ef1f5cc` + `typefly/patches/0001-quack-px4-platform.patch` (4 small
  core edits + a 3-line shim). All PX4 code lives in `quackdrone/integrations/typefly_px4.py`.

## Process / port map

| Process (tmux session) | Port | Notes |
|---|---|---|
| PX4 SITL + gz server (`quack-sim`) | UDP 14540 (API/offboard), 14550 (GCS/QGC) | PX4 binds localhost only (`MAV_x_BROADCAST=1` for LAN) |
| camera bridge (`quack-cam`) | HTTP 5601 | `--host 0.0.0.0` to let a Jetson on the LAN read it |
| TypeFly YOLO (`quack-vision`) | 50049 HTTP, 50050 gRPC | |
| TypeFly web UI (`quack-webui`) | 50000 | paid LLM calls |

## Interfaces for the Jetson track (code against these)

**Camera** (sim: `scripts/start_sim.sh`; real drone: replace the URL with the onboard camera)

| Endpoint | Format |
|---|---|
| `http://127.0.0.1:5601/video` | MJPEG (`multipart/x-mixed-replace`), works with `cv2.VideoCapture(url)` exactly like the phone IP-cam URL in `BTMOB.py` |
| `http://127.0.0.1:5601/frame.jpg` | one JPEG (newest frame) |
| `http://127.0.0.1:5601/frame` | raw RGB, H×W×3 uint8; headers `X-Width`, `X-Height`, `X-Seq`, `X-Stamp` (unix s), `X-Age-Ms` |
| `http://127.0.0.1:5601/health` | JSON `{ok, frames, width, height, age_ms, fps, jpeg}` |

Frames: 640×480, ~13–15 fps measured, RGB (OpenCV gives BGR as usual). Camera: forward-facing, HFOV 1.2 rad
(68.75°), pitched 8° down, 0.24 m above the body origin. `quackdrone.camera.open_camera(src)` accepts any of
these plus a webcam index, a file, RTSP or a GStreamer pipeline string (Jetson CSI), so the follower can take
the source as a single string.

**Setpoints** (MAVLink endpoint: `udpin://0.0.0.0:14540` in SITL; `serial:///dev/ttyTHS1:921600` or similar on the drone)

| Item | Value |
|---|---|
| Call | `Drone.set_velocity_body(forward_m_s, right_m_s, down_m_s, yawspeed_deg_s)` = MAVSDK `VelocityBodyYawspeed` (MAV_FRAME_BODY_NED) |
| Frame | forward/right rotate with **yaw only** (roll/pitch ignored by PX4); down is world vertical (**negative = up**); yaw rate **+ = clockwise** (turn right). Verified live: facing east, forward 1 m/s × 3 s → +3.00 m east |
| Units | m/s, deg/s |
| Rate | send at 10–30 Hz (≥ 5 Hz). MAVSDK itself re-sends the last setpoint at 20 Hz |
| Timeout | no new call for **0.5 s** (`DroneConfig.cmd_timeout_s`) → quackdrone brakes (zero velocity), then holds position. PX4's own offboard-loss failsafe fires only if the whole process dies (COM_OF_LOSS_T = 1 s) |
| Default clamps | 2 m/s horizontal, 1 m/s vertical, 60 °/s yaw, altitude band 0.5–10 m above takeoff, 30 m soft fence (`Limits`) |
| Errors | `ValueError` on NaN/inf; `ControlLost` after PX4 failsafe / pilot takeover / link loss |
| Lifecycle | `connect()` → `takeoff(alt)` → velocity loop → `land()`; `with Drone(...)` lands on exit/exception |

If the Jetson bridge talks to MAVSDK directly instead of using quackdrone, it must add its own command
watchdog: see the live result below (one command and then silence = the drone keeps going).

## Failsafe matrix (observed live in Gazebo SITL, PX4 defaults + SITL rcS defaults)

| Situation | Who reacts | Observed |
|---|---|---|
| Perception loop hangs (process alive) | quackdrone watchdog | 1 m/s command then silence: stopped after 0.54 m. **Without the watchdog: 3.24 m in 4 s and still moving at 1.04 m/s** |
| Script killed (`kill -9`) | PX4 offboard loss | OFFBOARD → RETURN_TO_LAUNCH after 1.2 s → landed, disarmed in 8.3 s (near home) |
| Exception inside `with Drone()` | quackdrone | OFFBOARD → LAND in 1.2 s, no RTL |
| Laptop slept mid-flight (accidental) | PX4 offboard loss | Failsafe → Return → land; led to two fixes (zero the re-sent setpoint on link loss; land on exit if PX4 is still in OFFBOARD) |
| Geofence `GF_MAX_HOR_DIST=6` | PX4 | "Geofence: exceeding maximum distance to Home" → HOLD at 6.8 m; quackdrone stopped commanding, raised `ControlLost` |
| Soft fence 3 m (ours) | quackdrone | velocity slowed near the edge; ended 3.22 m from home |
| Battery below `battery_land_pct` | quackdrone | warn, then land (tested with raised thresholds) |
| Battery critical (ours only warns) | PX4 (COM_LOW_BAT_ACT=3) | OFFBOARD → HOLD (COM_FAIL_ACT_T 5 s) → LAND (battery already at emergency) |
| PX4 leaves Offboard for any reason | quackdrone | never fights it: stops commanding, zeroes the re-sent setpoint |

Real-hardware notes: RTL climbs to `RTL_RETURN_ALT` (60 m default), dangerous indoors; in SITL, offboard
loss → RTL because there is no RC (COM_OBL_RC_ACT=Position falls through to NAV_RCL_ACT=Return).

## Faces in simulation (`quack_people` world, `examples/04_face_check.py`)

Drone at 1.5 m, frontal view. YuNet (OpenCV Zoo) detected every person at 1.5–6 m (score 0.83–0.95,
face 12–59 px wide). OpenCV Haar detected the two MakeHuman models (CC0) at ≤ 4 m and never the low-poly
FemaleVisitor (CC-BY). So sim faces work for pipeline/controller testing, but do not use them to judge
detector accuracy. The walking actor is in the world but was not evaluated.

## Verification log (what ran live, 2026-11-10)

| Run | Result |
|---|---|
| `pytest tests/unit` (incl. TypeFly adapter on the fake) | 57 passed |
| `start_sim.sh` standalone Gazebo, quack_demo and quack_people | up in ~90 s, camera 13–15 fps, TurboJPEG MJPEG |
| `examples/01_takeoff_square_land.py` | PASS, back at start within 2 cm |
| `tests/sitl/test_flight.py` (4) | all pass (watchdog/fence after one fix each) |
| `tests/sitl/test_failsafes.py` (5) | all pass (battery backstop after fixing the expectation) |
| `setup_typefly.sh` | pinned clone + patch, 111 upstream TypeFly tests pass |
| `typefly/e2e_check.py --no-llm` (find the chair) | PASS on a fresh sim: 2.8 m from chair, 2° facing error, 16 s |
| `examples/04_face_check.py` | table above |

## TODO / not verified

* LLM-planned run of the TypeFly demo (real OpenAI call, costs money); web UI frames by eye.
* SIH mode (`SIM=sih`) written but never started; Apple Silicon / Ubuntu / Windows bring-up; Docker
  (not evaluated: Docker Desktop is installed here but its daemon was not started).
* Follow controller closed loop on the walking actor; real camera / Jetson latency.
* Hardware params file (QGC) and real Pixhawk bring-up; Jetson UART URL is an example, untested.
* `mavsdk>=4` wheels need macOS 15+ (Linux aarch64 4.1.0 needs glibc 2.28, i.e. JetPack 5+; JetPack 4 → `mavsdk==4.0.6`).
