# Architecture: onboard face-follow

_2026-11-10, body path added 2026-11-13. Companion to `WHAT_WE_NEED.md` (hardware/why) and `BRINGUP_CHECKLIST.md` (how)._

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
| `detectors/person.py` (optional) | BGR image | `list[Face]` used as whole-person boxes (YOLOX-nano / HOG / TensorRT) | model |
| `lock.py` `LockKeeper` (optional) | face tracks + body tracks | `LockState(face, body, steering box, stamp, source="face"/"body"/"")` | yes |
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

### 2a. Keeping the lock when the face disappears (body path, `--person yolox`)

A person who turns round or walks away shows no face. With `FollowPipeline(..., person_detector=...)`
the pipeline also runs a person detector, tracks bodies with a second `IouTracker` (centre-gate
fallback off: it caused an identity swap in a group shot), and lets `lock.LockKeeper` decide what
to steer by:

```
 face tracks ─┐                    target face detected?  ── yes ─► steer by face; remember its body
              ├─► associate() ──►        │ no                          + learn face-in-body geometry
 body tracks ─┘  (face centre in         ▼
                  top of body box)  target body detected? ── yes ─► steer by a *virtual face box*
                                         │ no                       (learned geometry on the body box),
                                         ▼                          re-acquire when a face appears
                                    nothing: controller hovers      INSIDE that body; never another face
                                    -> TIMEOUT; no other face is    (cap: max_body_only_s = 15 s)
                                    adopted until lost_timeout_s
```

The virtual face box keeps the controller unchanged: its distance estimate still assumes a 0.16 m
face, the box is just synthesised from the body (from body *height* when the whole person is in
frame, which hardly changes as they turn; from body width when the frame cuts them off). A lock is
only ever *started* by a face, so the drone only follows someone who showed it their face first.
No recognition: no embeddings, nothing kept once a track ends. In body mode `StepResult.target` is
`None` (so the HUD blurs every visible face) and `target_body` / `target_source` say what is steered by.

Measured with `python -m jetson.lockeval` (closed loop, toy world, oracle detectors with misses and
noise, 5 seeds; numbers = share of frames after first acquisition):

| scenario | mode | on target | on a bystander | lost | timed out |
|---|---|---:|---:|---:|---:|
| walks away 8 s, back turned | face only | 19 % | 0 % | 81 % | 100 % |
| | face + body | **100 %** | 0 % | 0 % | 0 % |
| turns round twice, 6 s each | face only | 16 % | 0 % | 84 % | 100 % |
| | face + body | **100 %** | 0 % | 0 % | 0 % |
| walks away while a bystander faces the camera | face only | 19 % | **73 %** | 8 % | 0 % |
| | face + body | **100 %** | 0 % | 0 % | 0 % |
| face always visible (control) | both | 100 % | 0 % | 0 % | 0 % |

Real detectors on `crew_pan.mp4` (YuNet + YOLOX-nano, target's face painted out for 70 frames,
open loop, `python -m jetson.lockeval --clip ...`): face-only followed *another crew member* in 79 %
of those frames; face+body followed nobody (hover, 90 % lost, 0 % wrong). The body path did not keep
the lock there: the target is the front person of a tightly packed group and YOLOX's box for them
drops below 0.2. So in dense groups it fails safe but does not help; the toy world has no occlusion
between people, so its 100 % is an upper bound. Needs a real walk-away clip with a team member.

Cost: YOLOX-nano 416 px took ~95-140 ms per frame on the shared Intel Mac CPU under heavy load (not
a benchmark). On the Orin it should be a few ms with TensorRT FP16 (unmeasured); `--person-every N`
runs it on every Nth frame if needed.

### 2b. Confirmed re-lock after a loss (both paths, `jetson/relock.py`)

SITL run 2 (`sim/docs/runs/follow-sitl-run2.csv`): after the walker turned and walked away, YuNet hits
on the *back* of their head at 7-8.7 m (estimated 6.0-6.8 m) restarted tracking at 68.8 s and the
drone flew toward them for 1.6 s. `RelockGate` now sits in front of the controller. Once the target
has had no fresh detection for 0.5 s (the controller's `max_obs_age_s`, i.e. when it goes LOST), observations are withheld (the drone hovers and the lost timer
keeps running) until a re-lock is **confirmed**: 3 consecutive detections on one track (or one
scoring >= 0.9), each estimated within max(`target_distance + 1.5 m`, last followed distance + 1.5 m) (5.5 m when following at 4 m). An unconfirmed candidate
is not the target, so the HUD and `--out` blur it like a bystander. Flags: `--relock-hits`,
`--relock-score`, `--relock-max-distance`; `--relock-hits 1 --relock-score 0 --relock-max-distance 0`
restores round 2.

Why there's a distance cap and not just N hits: in run 2 the back-of-head detections came in streaks of 6, 7 and
16 consecutive frames, so a hit count would have to be > 16 (1.1 s at 14.8 fps) to block that
episode. Scores don't separate them either: YuNet on rendered *frontal* sim people scores 0.44-0.65 at
6 m. What does separate them is range: the genuine re-lock in run 2 (34.4 s) was estimated at
4.04 m, the false one at 6.0+ m. Cost: a person who walks beyond ~5.5 m while the face is lost is not
re-locked; the drone lands after `lost_timeout_s` (fails safe).

**Face-only mode does not wait out the lost timeout before adopting a new face** (body mode does).
In face-only mode the same person turning back round is *always* a new track: the tracker drops a
track after 1 s without a detection. SITL run 3 re-acquired the walker after 3.28 s and 2.87 s
losses (`follow-sitl-run3.json`), which were new tracks both times. With a holdoff equal to the lost timeout, both of those
would have become landings, i.e. face-only would never re-lock after a loss > 1 s. The bystander risk
that the body path's holdoff guards against is covered here by the confirmation + distance cap.
`--relock-holdoff S` adds a face-only holdoff if Eddie wants the stricter rule (tested).

Found while checking compatibility with the sim's fake-backend closed loop (`sim/tests/unit/test_jetson_follow.py`,
face 30 deg off and 8 m away, ~8 px wide): while the drone yaws, the tracker loses that small face every ~1 s
and spawns new tracks. Its motion model is in image space and ignores the drone's own rotation, and the
selector stays on the coasting old track until it expires (1 s). That was already true in round 2 and the
re-lock gate doesn't change it (the test passes 4/4), but it costs ~60 % of tracking time on far faces
while turning. Possible fix: shift predicted boxes by the commanded yaw rate (ego-motion compensation).

### 2c. Staying in front: head yaw -> sideways (`jetson/headpose.py`, `k_head_lateral`, off by default)

The controller only yaws and moves forward/back, so any point on the line of sight looks "centred".
A person walking at an angle to that line turns their face further and further away until it is
lost; the sim track saw this on 14-39 deg legs (20-50 s). Now YuNet's 5 landmarks give a yaw proxy:
where the nose tip sits between the eyes and between the mouth corners, `theta = atan(r / k)` with
k ~ 1 (head model in `headpose.py`; + = face points to the camera's right). It is skipped on faces
under 12 px. The controller smooths it (tau 0.5 s), drops it after 0.5 s without a sample (body-only
steering, small face, lost), and adds `right_m_s = k_head_lateral * deadband(yaw, 8 deg)`,
capped at `max_lateral_m_s` (0.5 m/s), alongside the yaw loop. Together that orbits the drone round to
the person's front. `--head-lateral 0.02` turns it on.

Closed loop without SITL (`python -m jetson.headeval`): toy world + a walker at 0.4 m/s who faces
where they walk (face detectable within 60 deg of the drone), sim camera preset (68.75 deg, 0.13 m face, 2 m altitude, 8 deg
tilt), landmark oracle with pixel noise, full pipeline (tracker, re-lock gate, head yaw, controller), 60 s:

| walker leg vs line of sight | lock %, off | in front (<=30 deg) %, off | timed out, off | lock %, k=0.02 | in front %, k=0.02 | timed out, k=0.02 |
|---|---|---|---|---|---|---|
| 0 deg | 35 | 24 | 5/5 (~25 s) | 100 | 99 | 0/5 |
| 15 deg | 39 | 31 | 5/5 (~28 s) | 100 | 99 | 0/5 |
| 25 deg | 19 | 9 | 5/5 (~16 s) | 100 | 99 | 0/5 |
| 35 deg | 13 | 1 | 5/5 (~12 s) | 100 | 95 | 0/5 |
| 45 deg | 8 | 0 | 5/5 (~10 s) | 100 | 91 | 0/5 |

(5 seeds, walker heading random walk 3 deg/sqrt(s); even the 0 deg leg drifts off without the term,
as the sim track predicted.) With no heading noise the off case keeps 0 deg perfectly but times out on every
leg >= 15 deg; k=0.02 keeps 100 % lock on all of them. k=0.01 also never times out but is in
front only 38-78 % of the time. The imx219 preset gives the same picture (91-99 % in front).

Not verified: real YuNet landmark noise on 12-20 px faces (the oracle adds 0.5 px + 4 % of the face width),
the head-model constant k, and any flight. Sideways flight is blind (no side camera), so keep the cap small, test it in SITL first,
then props-off. Moving sideways around someone who stands still and looks away is intended: it is an orbit to their front at <= 0.5 m/s.

## 3. Controller in one paragraph

Bearing from the box centre (pinhole angle) -> yaw rate; distance from the box *width* (face ~0.16 m)
-> forward speed; optional elevation -> down speed (off by default). Each goes through a soft
deadband, a P gain, saturation and a slew limit; forward speed is scaled down when the face is off-centre
("turn first, then approach"); forward is forbidden inside `min_distance_m`. No target -> hover now;
LOST for `lost_timeout_s` -> TIMEOUT with an action (land/rtl/hold) that latches. Gains were picked so
the loop stays well damped with 0.3 s of perception latency in the toy closed-loop test
(`test_perception_latency_still_converges_without_growing_oscillation`). Known limitation: P-only
control trails a walking target by `v / k_forward` metres (`test_follows_a_walking_person...`);
a velocity feed-forward from the tracker is the planned fix (mentee task M1). Optional sideways terms:
`k_lateral` (bearing) and `k_head_lateral` (head yaw, section 2c), both 0 by default, sharing `max_lateral_m_s`.

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
| Camera model | quack_cam: HFOV 1.2 rad = 68.75 deg, 640x480 at 15 Hz | `--camera-preset sim` (68.75 deg, face 0.13 m) |
| Flight controller | PX4 SITL MAVLink, `udpin://0.0.0.0:14540` | `--connect udpin://0.0.0.0:14540` (default) |
| Face target | `quack_people` world: a scripted walking actor (`sim/quackdrone/integrations/walker.py`) plus static people. YuNet boxes the walker's face at 0.13 m wide (sim agent, 3 frames at 2.3-3.9 m ground truth) and scores it 0.65-0.75 at 2.5-3.5 m; the walker's back of head also fires at 6-9 m (see 2b) | `--camera-preset sim`; the sim's `sim_follow_config()` sets the same values. Flown live by the sim track through `QuackdroneSink` (`sim/examples/follow_person.py`, results in `sim/docs/runs/`) |

Run order for an end-to-end SITL test (the sim/ track starts and stops SITL; only one MAVSDK app can
bind UDP 14540 at a time, so don't run TypeFly's PX4 wrapper simultaneously):

```bash
# 1. sim/ track: PX4 SITL + Gazebo + camera bridge running (their README)
# 2. sanity: camera reachable, then a read-only link check
curl -s http://127.0.0.1:5601/health
python -m jetson.linkcheck --connect udpin://0.0.0.0:14540 --seconds 5
# 3. dry run (no commands sent): watch the HUD/logs
python -m jetson.follow --source sim --camera-preset sim --show
# 4. fly in SITL: arm, take off to 3 m, follow; land when stopped
python -m jetson.follow --source sim --camera-preset sim --fly --yes --auto-takeoff 3 --on-exit land --csv jetson/out/sitl.csv
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
