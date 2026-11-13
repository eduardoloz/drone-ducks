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
| ground-truth pose probe (`quack-poses`) | HTTP 5602 | `/poses`: Gazebo model poses + sim time (no actor poses) |
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
detector accuracy. The walking actor: see "Follow-person" below.

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
| `examples/follow_person.py` / `tests/sitl/test_follow_person.py` (2026-11-13) | 2 runs, 73-74 s followed, see "Follow-person" |
| `pytest tests/unit` (2026-11-13) | 77 passed |
| Round 3 regression (2026-11-13, after round 2's `start_sim.sh` change) | `test_failsafes.py` 5/5; `test_flight.py` 3/4, then 4/4 on re-run: the failure was a 3 s MAVLink heartbeat gap (`link_lost`) 12 s after takeoff with load average ~8 (other apps at 280 % + 220 % CPU), not a code regression |
| Round 3 follow runs + `test_body_lock_follows_through_long_turn_away` | see "Round 3" below; body-lock SITL test passed once |
| `test_follow_walker_then_land_when_lost`, before the skip-loops fix | fresh world: **failed** (no face in 30 s), then passed (75 s) on the same world |
| same, after the fix, twice in a row from a fresh `start_sim.sh` | **2/2 passed**: 72.8 s / 74.6 s followed, 100 % within 10 deg, 6 / 4 lost events; then the full `tests/sitl` suite on the same world: 10 passed, 1 skipped (body test needs the turnaway world) |
| `pytest tests/unit` (round 3) | 85 passed |

## Follow-person: Jetson pipeline + quackdrone

_2026-11-13, round 2. Live in Gazebo SITL on Eddie's Intel Mac (same stack as above)._

```
 gz camera bridge :5601 /frame --> jetson LatestFrameReader --> YuNet (score 0.5) --> IouTracker/TargetSelector
     --> FaceFollowController (sim_follow_config: hfov 68.75, face 0.13 m, target 4 m)
     --> QuackdroneSink.submit() --> Drone.set_velocity_body()  [Limits clamp + 0.5 s watchdog] --> PX4 Offboard
 gz pose probe :5602 (drone pose, sim time) + walker script  --> ground truth columns in the CSV log
```

* Glue: `quackdrone/integrations/jetson_follow.py`. `QuackdroneSink` implements the Jetson
  `CommandSink` (start / submit / stop / finished / altitude_m / status) on a `Drone`; `jetson/` is
  imported unchanged. No second MAVSDK connection, no second watchdog.
* Example: `examples/follow_person.py` (take off, fly to the walker's lane, wait for its loop to
  restart, follow, land). SITL test: `tests/sitl/test_follow_person.py` (runs the example).
* Ground truth: `gazebo/gz_pose_probe.py` (started by `start_sim.sh`) serves model poses + sim time.
  Gazebo does **not** publish the scripted actor's pose, so the walker's position comes from
  `quackdrone/integrations/walker.py` evaluated at sim time (a unit test keeps it in sync with the
  SDF). Checked against the camera from the ground: the walker crossed the drone at the predicted time.

### Safety behaviour

| Situation | What happens | Tested |
|---|---|---|
| Face lost | controller sends hover (zero velocity) at once | unit (closed loop), SITL |
| Lost for `lost_timeout_s` (5 s) | `--lost-action`: land (default), rtl, or hold; latched, never resumes | unit, SITL (every run ends this way) |
| Frames stop / detector hangs | quackdrone watchdog brakes after 0.5 s and holds; sink lands after 3 s without a submit | unit |
| Soft fence (`--fence`, 25 m) / altitude band / 1.5 m/s cap | velocity clamped by quackdrone `Limits` | unit |
| PX4 leaves Offboard (geofence, battery, pilot, link) | quackdrone stops commanding; session ends; never re-engages | unit (fake) |
| Ctrl-C / exception / script ends | sink `stop()` lands; `with Drone` lands as a backstop | unit |
| Sideways command (`--head-lateral`, jetson `k_head_lateral`) | same clamps as forward: `max_speed_xy` caps \|forward, right\|, the soft fence removes the outward part of a sideways velocity, PX4 GF_* geofence on top | unit (`test_safety.py`, fake-backend sink), SITL (`test_soft_fence_stops_sideways_velocity`) |

**Sideways flight is blind.** The drone has one forward-looking camera; when `--head-lateral` moves it
right/left to stay in front of a person, nothing sees what it moves into (walls, trees, other people).
The jetson controller caps the sideways speed at `max_lateral_m_s` (0.5 m/s) and it is off by
default; quackdrone's fence and speed cap only bound *where* and *how fast*, not *into what*. Before
using it outdoors on hardware: open space only, a safety pilot on the RC switch, and a small
`--max-lateral`.

### The walker (`quack_people`)

Round 1's walker was sunk half into the ground (actor `z = 0`; the `walk.dae` origin is at the hips),
so from 2 m only the top of the head showed and YuNet scored it below 0.6. Now `z = 1.0`, and the path
is a "leading" scenario: straight north along x = -5 at 0.4 m/s facing the drone for 62 s, with one
look-back (face away) at 27.5-32 s, then a 1.4 m/s walk back (face away) so every run ends with the
lost-target landing. Loop 80.4 s.

Why straight: the face controller centres the face with yaw and keeps distance along the camera
axis, but cannot see which way the person faces. Every point on the line of sight centres the face,
so nothing pulls the drone back in front of someone who walks at an angle. In kinematic dry runs
(FakeBackend + ideal detector) with legs 14-39 deg off the line of sight, the facing angle crept
up until the face was lost for good after 20-50 s, even with `k_lateral` 0.04-0.08.

YuNet on the walker (2 m altitude, frontal): score 0.66-0.75 at 2.3-3.5 m, 0.37 at 3.9 m, 0.23 at
2 m (face low in the frame). Hence `--score 0.5` in the example. Measured box width implies a face
of 0.11-0.14 m, hence `face_width_m = 0.13` for the sim.

### Results (2 live runs, 2026-11-13, alt 2 m, target distance 4 m)

| | Run 2 (`docs/runs/follow-sitl-run2.*`) | Run 3 (the SITL test) |
|---|---|---|
| Followed (first lock to lost-timeout landing) | 72.8 s | 73.9 s |
| Frames with a face (TRACKING) while following | 86 % | 84 % |
| True bearing within 5 deg / 10 deg | 100 % / 100 % (mean abs 0.7 deg) | 100 % / 100 % (mean abs 0.4 deg) |
| True distance while TRACKING: mean (median) | 2.99 m (2.90), min 2.24 | 2.99 m (2.86), min 2.16 |
| Estimate minus truth, distance: mean, MAE | -0.16 m, 0.39 m | -0.12 m, 0.39 m |
| Estimate minus truth, bearing | +1.3 deg bias (inside the 3 deg deadband) | |
| Lost events | 3: look-back (lost 3.0 s, reacquired), turn-away, false lock | 5: look-back (3.3 s), turn-away, 3 false locks |
| Walker facing within 60 deg of the drone | 82 % of the follow | 81 % |
| Perception | 14.9 fps, frame age 31 ms mean / 48 ms p95, YuNet 11 ms | |

* **Face-only following loses the target every time the person turns away**: here 1 look-back (5.6 %
  of the walk) and the final turn. The look-back (4.5 s) fits inside the 5 s lost timeout, so the drone
  hovered and re-acquired; any longer and it would have landed. 18-19 % of each run had the face turned
  away.
* **Trails the target**: mean 3.0 m instead of 4 m. The walker approaches at 0.4 m/s and the controller
  reverses at most 0.5 m/s (P-only: `0.5 * (d - 3.5)` saturates near 2.7 m). Known limitation
  (feed-forward is a Jetson good-first-task).
* **False locks while the walker walks away** (score 0.5): 0.3-1.7 s of TRACKING at 7-10 m with the face
  turned away (probably the back of the head), each one restarting the 5 s lost timer and briefly
  commanding forward flight toward the walker. Landing came 2-6 s later than it should.
* Run 1 (before the actor-height fix) locked once and lost the face: logged, not counted.

### Round 3 (2026-11-13): body lock, long turn-away walker, 3 m altitude

Same stack plus the Jetson round-2 body path: `follow_person.py --person yolox` builds
`FollowPipeline(person_detector=YOLOX-nano, ...)`, which uses `jetson.lock.LockKeeper` (acquire by
face only; follow the target's body when the face is gone, up to `max_body_only_s` = 15 s; re-acquire
the face inside that body). Models come from `python -m jetson.fetch models` run with the sim venv
(into this checkout's gitignored `jetson/models/`); the round-2 fallback into the Jetson worktree is gone.

New world `quack_people_turnaway` (walker script `walker.TURNAWAY`): 25 s towards the drone, then
turns round and walks **away** for 12 s (~14 s without a face: longer than the 5 s lost timeout,
shorter than the 15 s body cap), faces the drone again for 30 s, then the 1.4 m/s walk back.
`CAM_PITCH_DEG=20` (run_sim.sh) pitches the camera for 3 m altitude (horizon moved from image row 175
to 71, as predicted). All runs: one live SITL run each, Intel Mac under heavy load from other
processes (load average ~8), YuNet score 0.5, lost timeout 5 s. Logs: `docs/runs/r3-*.{csv.gz,json}`.

| Run (`docs/runs/r3-...`) | alt / cam / target | Followed | Lost events | Landed early? | True dist while TRACKING mean (median) | Steered by body |
|---|---|---|---|---|---|---|
| `turnaway-face` | 2 m / 8 deg / 4 m | 31.7 s | 1 (the turn-away) | **yes**: at walker t = 30.9 s (final turn at 69 s) | 2.85 m (2.89) | - |
| `turnaway-body` (`--person yolox`) | 2 m / 8 deg / 4 m | **107.6 s** (hit `--max-seconds 110`) | **0** | no: followed the turn-away, the walk-back and into the next loop | 4.23 m (3.17) | 27 % of frames |
| `alt3-face` | 3 m / 20 deg / 4 m | 9.6 s | 2 | yes: face lost as the walker came within 2.4 m | 3.04 m (2.95) | - |
| `alt3-face-t5` | 3 m / 20 deg / 5 m | 27.6 s | 6 (5 reacquired) | yes: at walker t = 27.6 s (the look-back) | 3.79 m (3.69) | - |
| `alt3-body-t5` (`--person yolox`) | 3 m / 20 deg / 5 m | **83.2 s** | 1 | no: lost the body only in the 1.4 m/s walk-back | 5.10 m (4.25) | 37 % |

Bearing was within 5 deg of the walker 100 % of the follow in every run. Perception: 14.9 fps
face-only (YuNet 11 ms); 11-13 fps with YOLOX-nano on every frame (61-73 ms per person pass on this
CPU, under load). `turnaway-face-nolock.json` is the very first follow after a fresh world start: no
face was detected in 30 s (acquire timeout -> landed). This reproduced 3 of 3 times when the follow
started at sim time ~80 s (the walker's 2nd loop), and never when it started at >= 160 s. Camera
snapshots (`--snapshot-every`) of such a run show the actor rendered at inconsistent places from one
second to the next (about 1.5 m under the drone, then about 10 m away), not where its script puts it.
Workaround: `follow_person.py --skip-loops 2` (default) starts only from the walker's 3rd loop, and
the wait now polls Gazebo sim time (the sim runs ~0.9x real time here, so sleeping wall time arrived
early). Root cause in Gazebo's actor system is not investigated.

* **The body lock does what it is for**: same script, face-only lands 38 s before the walker's final
  turn; face+body never loses them. It also means the session no longer ends when the person walks
  off: with the body lock the drone follows a person walking *away* (at up to 1.5 m/s, our speed cap)
  until it loses the body or hits `--max-seconds`.
* **False faces on the back of the head are accepted as re-acquisitions.** During the turn-away, YuNet
  (score 0.5) fires on the back of the walker's head at 5-6 m for 0.1-0.4 s at a time; the lock treats
  each as "face re-acquired inside the target body" (rule 4) and steers by that box. Harmless here
  (same person) but it makes the distance estimate jump. Request to jetson/ below.
* **3 m needs a longer target distance and still wants the body lock.** At 3 m the controller's distance
  is slant range (+0.36 m vs horizontal at 2.5 m) and faces seen from above score low: with target 4 m
  the walker got within 2.4 m and the face vanished. Target 5 m keeps the face but detection is marginal
  (6 lost events); with YOLOX the gaps are bridged (95 % TRACKING).
* Body-lock distance while steering by body is the virtual face box from learned geometry: the
  estimate was 0.6 m short on average at 3 m (it learns the ratio on a foreshortened face).

Requests for jetson/ (not changed here): (1) don't re-acquire a "face" inside the target body unless it
is confirmed over a few frames or scores high (the round-3 `relock.py` may already cover this: not
tested here); (2) an altitude-aware distance (`horizontal = sqrt(slant^2 - (alt - face_h)^2)`) so 3 m
flights hold the commanded distance; (3) an option to end the session when the person turns away and
walks off (body seen, face absent for N s, distance growing), since following a departing person is
not always wanted.

## TODO / not verified

* LLM-planned run of the TypeFly demo (real OpenAI call, costs money); web UI frames by eye.
* SIH mode (`SIM=sih`) written but never started; Apple Silicon / Ubuntu / Windows bring-up; Docker
  (not evaluated: Docker Desktop is installed here but its daemon was not started).
* Follow-person: straight walker paths only; 2 m and 3 m, one live run per configuration (no
  repeat statistics); no turn-tolerant following (the Jetson round-3 head-yaw term is not tried here
  yet); real camera / Jetson latency. Gazebo renders the walker inconsistently for its first ~2 loops after a
  world start (worked around by `--skip-loops`, root cause unknown).
* Jetson round-3 (`relock.py`, `--head-lateral`, `--camera-preset`) not merged into this branch's runs.
* Hardware params file (QGC) and real Pixhawk bring-up; Jetson UART URL is an example, untested.
* `mavsdk>=4` wheels need macOS 15+ (Linux aarch64 4.1.0 needs glibc 2.28, i.e. JetPack 5+; JetPack 4 → `mavsdk==4.0.6`).
