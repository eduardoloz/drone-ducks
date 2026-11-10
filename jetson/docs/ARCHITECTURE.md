# Architecture: onboard face-follow

_2026-11-10. Companion to `WHAT_WE_NEED.md` (hardware/why) and `BRINGUP_CHECKLIST.md` (how)._

## 1. The picture

```
 ┌──────────────────────────────── Jetson (one Python process: python -m jetson.follow) ───────────────────────────────┐
 │                                                                                                                      │
 │  reader thread                 main thread (one iteration per new frame)                    mavsdk-loop thread       │
 │  ─────────────                 ─────────────────────────────────────────                    (asyncio, --fly only)    │
 │  FrameSource.read()            frame = reader.get(newer_than=last)                          startup: connect, wait   │
 │   csi:// usb sim http tcp       faces  = detector.detect(frame.image)    detectors/*          healthy, [takeoff],      │
 │        │                       tracks = tracker.update(faces, stamp)     tracker.py            5 hover setpoints,     │
 │        ▼                       target = selector.select(tracks)          tracker.py            offboard.start()       │
 │  [ single slot: newest  ]──►   obs    = TargetObservation(target box)    controller.py                                │
 │  [ frame, seq, stamp    ]      out    = controller.step(obs, now)        controller.py       every 50 ms (20 Hz):     │
 │   (older frames dropped)       sink.submit(out)  ─────────────────────────────────────────►  newest command, or       │
 │                                [HUD + blurred video if --out]           hud.py                hover if stale, or      │
 │                                                                                               land if stalled, or     │
 │                                                                                               stop if pilot took over │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────┘
                                                                                                     │ MAVLink 2
                                                       serial:///dev/ttyTHS1:921600  (or udpin://0.0.0.0:14540 in SITL)
                                                                                                     ▼
                              ┌────────────────── Pixhawk 6C running PX4 (TELEM2) ─────────────────────┐
   RC transmitter ──────────► │ OFFBOARD mode: velocity setpoint -> PX4's position/velocity/attitude/   │ ──► motors
   (pilot override, kill)     │ rate PID loops. Failsafes: RC loss, offboard loss, geofence, battery.   │
                              └───────────────────────────────┬────────────────────────────────────────┘
                                                              │ TELEM1: SiK radio
                                                              ▼
                                                QGroundControl (ground station)
```

Everything left of the MAVLink arrow is this package; everything right of it is PX4, which does all
the control loops. We send setpoints, never motor commands.

## 2. Modules and the data that flows between them

| Module | In | Out | Pure? |
|---|---|---|---|
| `sources.py` | camera spec string | `Frame(image BGR, seq, stamp=capture time, source)` | I/O |
| `detectors/` | BGR image | `list[Face(x1, y1, x2, y2, score, landmarks)]` in frame pixels | model |
| `tracker.py` `IouTracker` | faces + stamp | `list[Track(id, smoothed box, last_seen, confirmed...)]` | yes |
| `tracker.py` `TargetSelector` | tracks | the one `Track` to follow (sticky) | yes |
| `controller.py` | `TargetObservation(cx, cy, w, h normalised, stamp)` + `now` (+ altitude) | `ControlOutput(VelocityCommand, mode, reason, action)` | yes |
| `bridge.py` | `ControlOutput` | MAVSDK `set_velocity_body` at 20 Hz / dry-run log | I/O |
| `pipeline.py` | `Frame` | `StepResult` (all of the above + per-stage timings) | glue |
| `geometry.py` | FOV, widths, distances | pixels on face, range, bearing, distance | yes |
| `toysim.py` | `VelocityCommand` | kinematic drone + walking person + rendered frame | yes |

Conventions used everywhere: pixel coordinates x right / y down; normalised coordinates 0..1;
times are `time.monotonic()` seconds (or a simulated clock passed explicitly); body-frame velocities
are MAVSDK's `VelocityBodyYawspeed`: forward / right / **down positive** / yaw rate **positive =
clockwise seen from above**.

The *capture* stamp travels with the data. A `TargetObservation` carries the stamp of the last frame
the target was actually *detected* in (not the frame we are processing now), so the controller can
refuse to act on stale information even while the tracker coasts.

## 3. Controller in one paragraph

Bearing from the box centre (pinhole angle) -> yaw rate; distance from the box *width* (face ~0.16 m)
-> forward speed; optional elevation -> down speed (off by default). Each goes through a soft
deadband, a P gain, saturation and a slew limit; forward speed is scaled down when the face is off-centre
("turn first, then approach"); forward is forbidden inside `min_distance_m`. No target -> hover now;
LOST for `lost_timeout_s` -> TIMEOUT with an action (land/rtl/hold) that latches. Gains were picked so
the loop stays well damped with 0.3 s of perception latency in the toy closed-loop test
(`test_perception_latency_still_converges_without_growing_oscillation`). Known limitation: P-only
control trails a walking target by `v / k_forward` metres (`test_follows_a_walking_person...`);
a velocity feed-forward from the tracker is the planned fix.

## 4. Latency budget (camera to PX4)

| Stage | Mac, measured | Orin Nano (est.) | Original Nano (est.) | Notes |
|---|---:|---:|---:|---|
| Exposure + readout + ISP/driver | n/a | 20-40 ms | 30-60 ms | 33 ms frame period at 30 fps; measure glass-to-glass on HW |
| Waiting in our frame queue | **~0 ms** | 0-33 ms | 0-33 ms | latest-frame reader: frame age ~= processing time (measured mean 11.2 ms vs 10.9 ms detect) |
| Face detection, YuNet 640 px | **10.9 ms** mean, 12.3 p90 | ~40 ms CPU / a few ms TensorRT | ~90 ms CPU | Jetson figures scaled from opencv_zoo 160x120 rows (16x pixels) |
| Track + select + control | **~0.3 ms** | < 1 ms | ~1-2 ms | pure Python |
| Wait for next setpoint tick | 0-50 ms (avg 25) | same | same | 20 Hz stream; could send on every new command instead |
| MAVLink over UART | - | ~0.7 ms | ~0.7 ms | 65-byte SET_POSITION_TARGET_LOCAL_NED at 921600 baud |
| **Capture -> setpoint in PX4** | | **~70-120 ms** | **~150-220 ms** | |
| PX4 velocity response | - | time constant ~0.2-0.4 s | same | dominates; toy model uses 0.35 s |

Rules of thumb that follow:
- Detection rate matters less than detection *age*: 10 FPS of fresh frames beats 30 FPS of
  frames that are 300 ms old. That's why `LatestFrameReader` exists and why the controller treats
  observations older than 0.5 s as "no target".
- The perception loop and the 20 Hz setpoint stream are decoupled: PX4 sees a steady stream even if
  detection runs at 8 FPS on the original Nano.

## 5. Interfaces

### To the simulator (sim/ track, documented interfaces only)

| What | Interface | Our side |
|---|---|---|
| Camera | Gazebo camera bridge, `GET http://127.0.0.1:5601/frame`: raw RGB bytes, headers `X-Width`, `X-Height`, `X-Seq`, `X-Age-Ms`; `/health` JSON | `--source sim` (or `sim://host:port`): converts RGB->BGR, corrects the capture stamp by `X-Age-Ms`, skips repeated `X-Seq` |
| Camera model | quack_cam: HFOV 1.2 rad = 68.75 deg, 640x480 at 15 Hz | `--hfov 68.75` |
| Flight controller | PX4 SITL MAVLink, `udpin://0.0.0.0:14540` | `--connect udpin://0.0.0.0:14540` (default) |
| Face target | **not there yet**: the demo world has YOLO objects, no faces | ask sim/ for a billboard/actor textured with a public-domain face |

Run order for an end-to-end SITL test (the sim/ track starts and stops SITL; only one MAVSDK app can
bind UDP 14540 at a time, so don't run TypeFly's PX4 wrapper simultaneously):

```bash
# 1. sim/ track: PX4 SITL + Gazebo + camera bridge running (their README)
# 2. sanity: camera reachable, then a read-only link check
curl -s http://127.0.0.1:5601/health
python -m jetson.linkcheck --connect udpin://0.0.0.0:14540 --seconds 5
# 3. dry run (no commands sent): watch the HUD/logs
python -m jetson.follow --source sim --hfov 68.75 --show
# 4. fly in SITL: arm, take off to 3 m, follow; land when stopped
python -m jetson.follow --source sim --hfov 68.75 --fly --yes --auto-takeoff 3 --on-exit land --csv jetson/out/sitl.csv
```

### To the real PX4 drone

| What | Value |
|---|---|
| Wiring | Pixhawk 6C TELEM2 pin 2 TX -> Jetson pin 10, pin 3 RX <- Jetson pin 8, GND <-> pin 6 |
| Port | `serial:///dev/ttyTHS1:921600` (user in `dialout`; `nvgetty` disabled on the original Nano) |
| PX4 params | `MAV_1_CONFIG=102`, `MAV_1_MODE=2`, `SER_TEL2_BAUD=921600`, `COM_RC_OVERRIDE=3`, geofence, kill + return switches (WHAT_WE_NEED section 9) |
| Startup | pilot arms and takes off in Position mode -> app sees `in_air` -> streams hover setpoints -> requests OFFBOARD |
| Exit | app stop -> Hold (`--on-exit`); pilot mode switch -> app stops sending and never re-engages |

## 6. Threads, failures and who handles them

| Failure | Noticed by | What happens | Covered by test |
|---|---|---|---|
| Camera stalls / unplugged | reader (read errors), main loop (`get` timeout) | reader reconnects; no new commands -> bridge sends hover after 0.5 s, `--stall-action` (land) after 3 s | `test_reader_survives_errors_and_reconnects`, `test_stale_commands_become_hover_then_stall_lands` |
| Detection too slow | stamps | obs older than 0.5 s = no target -> hover | `test_stale_observation_counts_as_lost` |
| Python exception in the loop | `follow.py` `finally` | `sink.stop()` -> leave offboard via `--on-exit` (hold) | (by construction) |
| Hard crash / power loss / UART unplugged | PX4 | offboard loss after `COM_OF_LOSS_T` (1 s) -> `COM_OBL_RC_ACT` | PX4 (bench test stage 2) |
| Target lost | tracker / selector | hover; after 5 s land/rtl/hold, latched | `test_target_disappears_hover_then_land` |
| A bystander walks in | selector | keeps following the original track | `test_selector_prefers_large_central_face_then_sticks_with_it` |
| Tiny distant face jitters | tracker | centre-distance fallback keeps one ID | `test_tiny_noisy_face_keeps_one_identity` |
| Pilot takes over | bridge (mode != OFFBOARD after it was) | stops sending forever, no hold/land issued | `test_pilot_override_stops_streaming_and_never_reengages` |
| OFFBOARD refused / never engages | bridge | start fails, or session ends after 3 s | FakePX4 rule, `test_offboard_that_never_engages_ends_the_session` |

## 7. Why it's built this way

- **Pure core, thin I/O shell.** Controller, tracker, geometry and the YuNet/SCRFD decoders are pure
  functions/classes, so 150 tests run in ~20 s on a laptop with no camera, GPU or drone.
- **Lazy optional backends.** `import jetson.detectors` works anywhere; a missing TensorRT or
  jetson-inference becomes a clear `BackendUnavailable` message.
- **One process, three threads** keeps it debuggable for students. If perception ever needs isolation
  (e.g. a C++ detector), the seam is `CommandSink.submit()`: put a socket there.
- **Same code in SITL and on the drone:** only `--source` and `--connect` change.
