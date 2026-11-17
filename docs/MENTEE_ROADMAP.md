# Mentee roadmap: fall 2026 through winter break

_Owner: Eddie. Last updated 2026-11-13 (end of round 3). This is a living doc; update it at each Friday meeting._

## The short version

ProjectQuack builds a drone that can see and act on what it sees. This semester we run three tracks, and they feed each other:

- **A. CCTV speed analytics** (`cctv/`): vision plus a website. No hardware needed, so results come fastest here.
- **B. Drone simulation** (`sim/`): PX4 + Gazebo + MAVSDK, with the TypeFly LLM planner on top. This is the drone's brain, tested in sim.
- **C. Onboard perception** (`jetson/`): face detection on a Jetson on the drone, closing the loop camera → detector → controller → PX4.

Everyone does Week 0 onboarding. Each mentee owns one track and helps on a second. Every Friday, everyone demos something that runs.

```
 Track A (CCTV)                 Track C (Jetson)                 Track B (Sim)
 YOLO + ByteTrack  ───────────► face detector + tracker ───────► follow-person in Gazebo
 (Stephen's scripts)            follow controller (P-control)    PX4 Offboard via MAVSDK
 homography → speed             MAVSDK velocity bridge ◄──────── camera stream + setpoint API
 website (map, events)          Jetson bring-up (hardware)       TypeFly: "find the chair"
```

## Where things stand (2026-11-13, end of round 2)

All three tracks are merged into `eduardo` and every suite passes from a clean checkout: CCTV 81 tests, sim 77 unit tests, Jetson 182 passed and 16 skipped (the skips need `python -m jetson.fetch models media`). Each folder's README has its quickstart.

| Track | Works today (verified on Eddie's Intel Mac) | Not yet |
|---|---|---|
| **A. CCTV** | 129-camera LIE manifest; R10_041 calibrated end to end (54 of 62 vehicles got speeds, median 57 mph, σ ≈ 1.7 mph). **New in round 2:** each camera can store a separate calibration for every direction an operator points it, and the pipeline picks the one matching the clip; if no stored direction matches, it gives counts only. Harness that compares our speeds with NYC DOT's public link speeds (`cctv/groundtruth/`). Validation-drive protocol (`cctv/docs/VALIDATION_DRIVE.md`) plus a GPS-log reader | The ground-truth table itself: no camera on a usable NYC DOT link is calibrated yet. The Queens links (84th–108th St) look unreliable; **use R10_007/R10_008 on the Little Neck Pkwy westbound link instead**. Live ingest (needs 511NY access), night accuracy |
| **B. Sim** | **Follow-person works in SITL.** Two live runs followed the walker for ~73 s each, with the face within 5° of centre the whole time, and landed by themselves when the walker turned away. Jetson pipeline → `QuackdroneSink` → `Drone.set_velocity_body` (one MAVSDK layer, with the sim's watchdog and limits). The sim now reports the walker's true position on :5602. 1 SITL follow test | Follower trails at 3.0 m instead of 4 m (needs feed-forward). Brief false locks on the back of the head at 7–10 m. Only a straight walker path works (no sideways control). Round-1 SITL/failsafe suite not re-run after the `start_sim.sh` change. LLM-planned run |
| **C. Jetson** | Six face backends (YuNet default). **New in round 2:** it matches the followed face to that person's body with a YOLOX-nano person detector (Apache-2.0), so the lock holds when the face turns away. In the toy world, lock retention went from 16–19% to 100%, and the drone never switched to a bystander. Python 3.8 / 3.10 / 3.12 compatibility checks, TensorRT export dry run, JetPack 6 setup script and Dockerfile | Anything on a real Jetson. On a real crowded clip the body path held off the wrong person but didn't keep the lock, because the person detector scores packed groups low. **Needs a recorded walk-away clip of consenting team members** |

### Round 3 (2026-11-13): sim + Jetson only, CCTV paused for daylight and mentee calibration

Suites on a clean checkout: sim 85 unit tests (+ the SITL suite: 10 passed, 1 skipped), Jetson 203 passed and 16 skipped, CCTV 81.

**Body lock is the big win in SITL.** One live run per row; the bearing stayed within 5° of the walker 100% of the time in every row:

| Walker | Altitude / camera pitch / target distance | Detector | Followed | Lost events | Landed early? |
|---|---|---|---|---|---|
| Turns away for 14 s (longer than the 5 s lost timeout) | 2 m / 8° / 4 m | face only | 31.7 s | 1 | yes |
| Same | 2 m / 8° / 4 m | face + body | 107.6 s (hit the time cap) | 0 | no |
| Walks ahead | 3 m / 20° / 5 m | face only | 27.6 s | 6 | yes |
| Same | 3 m / 20° / 5 m | face + body | 83.2 s | 1 | no |

Also in round 3:
- **Jetson:** confirmed re-lock (3 hits in a row, or a score of 0.9 or higher, plus a distance limit), which stops locking onto the back of the walker's head. An optional "stay in front" sideways term driven by head yaw: in the toy world, lock went from timeouts to 100% at walker angles of 0–45°. It's off by default and untested in SITL. Camera presets (`--camera-preset sim|imx219|webcam`).
- **Sim:** the round-1 regression suite passes (one flight failure was a heartbeat gap on an overloaded Mac). Fixed a Gazebo start-up bug where the walker was drawn in the wrong place for its first two loops. Run logs are gzipped (368 KB → 46 KB).

**Next integration step:** an angled-walker world to test `--head-lateral` and the re-lock in SITL. A distance estimate that accounts for altitude (at 3 m the drone stops ~0.36 m too close). Confirm the face before treating it as re-acquired inside the body lock. Feed-forward stays Max's M1 task.

### Round 4 (2026-11-13 to 2026-11-17): Jetson merged; sim SITL runs and CCTV round 3 in progress

Round 4 was interrupted on Nov 13 when the Mac's battery ran out, and resumed on Nov 17.

**Jetson (merged into `eduardo` on 2026-11-17; 268 tests pass, none skipped):**
- **Face hand-back confirmation.** Inside the body lock, a face must be confirmed (3 hits in a row, or a score of 0.9 or higher) before the drone steers by it again. A bug in the first version could leave the drone with no target while both a face and a body were visible; that is fixed, with a test.
- **Altitude-aware distance.** The controller now holds the horizontal distance. Toy check at 3 m altitude and 20° pitch: it holds 4.05 m for a 4 m target (was 3.80 m).
- **Ego-rotation compensation.** The tracker shifts tracks by the drone's own yaw between frames. In the toy world during a ±20°/s turn, the prediction error for a still face at 8 m went from about 3.7 px per frame to about 0. No sink provides yaw-rate telemetry yet, so the commanded rate through a 0.35 s lag is what runs.
- **Optional walk-away end** (`--end-on-walkaway`, off by default; `jetson/walkaway.py`). It ends the follow through the lost-target path (hover, then `--lost-action`) when, over 3 s, the drone steers by the body, the person recedes at 0.8 m/s or more, and they are more than 1 m beyond the target distance. Toy world: ends 1.9 s after the turn in 5/5 seeds; never ends for a person standing with their back turned. Replayed open loop on the round-3 SITL logs, it would have ended both walk-aways 2.3–2.8 s in, with no false triggers. **Not yet flown in SITL**: `sim/examples/follow_person.py` needs a flag for it.

Known gaps: the re-lock distance cap still compares straight-line distance with a target that is now horizontal (about 0.2–0.4 m tighter at 3 m, inferred); a body-track ID change while following by body alone is treated as a loss.

**Sim:** the round-4 code (angled-walker world, `--head-lateral`, re-lock flags, stay-in-front and back-of-head metrics, sideways soft-fence test) is merged, but its SITL runs are only happening now.

## Who does what (proposed; confirm at Friday's meeting)

| Mentee | Primary track | Why | Secondary |
|---|---|---|---|
| Stephen Kwong (`StephenSTK`) | **A**: tracking + speed estimation | Already has YOLO + ByteTrack running on webcam, phone stream and video. ByteTrack is the core of the speed pipeline | C: the tracker that picks which face to follow |
| Junwei Liu | **B**: PX4 wrapper + "find the chair" | Has MAVSDK 4.1.0 set up. The PX4 handoff names Junwei for wrapper features and the end-to-end demo | C: MAVSDK bridge from the face controller into the sim |
| Max Lashley | **C**: Jetson face detection (suggested) | Ask what they want. C suits someone who likes hardware, Linux and embedded work | B: failsafe tests |
| Donald Lei | **A**: website, API + map (suggested) | Ask what they want. The website is self-contained web work | A: validation-drive data |
| Ricky (`rickyliu3095-cell`) | ? | Confirm whether this is Junwei's GitHub account. If so, merge the rows | |

Rule of thumb: if two people want the same track, split it by layer (pipeline vs website, controller vs hardware), not by "you do half the files".

## Week 0: everyone, before or at Friday Nov 13

- [ ] Clone `drone-ducks`, make a branch named after yourself, and open a tiny PR into `eduardo`. For example, add your laptop (OS + chip) to the table in this doc. Goal: branch → commit → push → PR → review → merge, once, with nothing at stake.
- [ ] Run Stephen's `BTPC.py` on any video, or `TrackScript.py` on your webcam. If YOLO + ByteTrack runs, your Python/OpenCV setup works.
- [ ] Read your track's README (`cctv/`, `sim/` or `jetson/`) and run its quickstart. Post a screenshot in Discord.
- [ ] Take the FAA **TRUST** test. It's free, takes about 30 minutes, and is required before anyone flies a drone recreationally. Post the certificate in Discord.
- [ ] Read "Concepts" below and bring one question to the meeting.

## Concepts everyone should be able to explain by December

1. **Detection vs tracking vs recognition.** YOLO finds boxes in each frame. ByteTrack links boxes across frames into IDs. Recognition would say *who* someone is, and **we don't do it** (privacy). Tracks A and C both stop at detection + tracking.
2. **Coordinate frames.** Pixels → camera → world. A homography maps a flat road from pixels to metres (Track A). PX4 uses NED, north-east-*down*, so 5 m up is `z = -5` (Tracks B and C).
3. **Who does the PID.** PX4 runs the inner rate/attitude/velocity/position loops, and we never touch them. We send setpoints in Offboard mode. PX4 only accepts Offboard if setpoints are *already* streaming before `offboard.start()`, and it fails safe if the stream stops.
4. **Our own control loops sit one level up.** The face-follow controller turns "face is left of centre and small" into "yaw left, fly forward". It's a P-controller with a deadband, saturation and a lost-target timeout. That's where you learn control without risking the flight-critical loops.
5. **Latency budgets.** Camera → inference → controller → PX4: every 100 ms of lag makes following worse and speed estimates noisier. Measure it, don't guess.
6. **Honest numbers.** A speed or FPS number means nothing without how it was measured: hardware, resolution, load, sample size and error bars.

## Calendar

| Date | Milestone |
|---|---|
| **Fri Nov 13** | Kickoff: tracks assigned, Week 0 done, everyone's environment runs |
| **Fri Nov 20** | **M1**: first vertical slice per track runs on a mentee's own laptop |
| Nov 25–29 | Thanksgiving. Optional async work only |
| **Fri Dec 4** | **M2**: cross-track integration (face-follow in sim; CCTV pipeline on live feeds) |
| **Fri Dec 11** (or the last meeting before finals) | **M3**: demo day. A short video per track, README updated, PRs merged into `eduardo` |
| Winter break | Hardware bring-up if purchased (drone + Jetson), plus stretch goals |

## Track A: CCTV speed analytics on the Long Island Expressway

Folder `cctv/`, branch `eduardo-cctv-footage`. Design: `cctv/docs/ARCHITECTURE.md`.

**Goal:** a website that shows estimated vehicle speeds along the LIE (I-495) from public traffic cameras, with anonymous per-vehicle "over the limit" events and per-camera speed statistics.

| | Task | Suggested owner | Done when |
|---|---|---|---|
| M1 | Quickstart from `cctv/README.md`: record 60 s from R10_041 (`python -m cctv.ingest`), run `python -m cctv.pipeline`, and view the result on `:8010` | Stephen | Your own run shows up on the site |
| M1 | Calibrate a **second** camera (R10_011, R10_014, R10_017, R10_019 or R10_044): `/calibrate` page, then `python -m cctv.calibrate fit`, then `python -m cctv.error_budget` | Stephen | Reprojection error and error budget pasted in the PR |
| M1 | Get the site running and add a per-camera stats view: speed distribution by hour, % over the limit | Donald | Page shows real data from your DB |
| M2 | Ground truth. The harness is built (`cctv/scripts/groundtruth_session.sh R10_008 3`, daylight only). Calibrate R10_007 or R10_008 on the Little Neck Pkwy westbound link and run it against NYC DOT's link speeds (dataset `i4gi-tjb9`); the Queens links proved unreliable. Also do a validation drive: a passenger logs phone GPS while the car passes a calibrated camera at a steady, legal speed | Stephen + Donald | Error table, estimate vs truth |
| M2 | ~~Several calibrations per camera~~ (done in round 2). Calibrate R10_041's second preset `v2` from a daylight frame, and add `direction_labels` to `v1` | Stephen | `python -m cctv.calibrate views --camera R10_041` shows both calibrated |
| M2 | Live ingest workers for 4–5 cameras, **only after the 511NY developer request is approved** | Stephen | Runs unattended for 1 h |
| M3 | Accuracy write-up and demo video | Track | Numbers with error bars in the README |

**Rules:**
- No licence-plate reading and no identifying people or vehicles.
- Speeds are estimates, not enforcement.
- Follow the camera feeds' terms of use. NYC DOT cameras need a signed agreement, so we don't use them.
- Validation drives are always at legal speed, and the *passenger* handles the phone.

## Track B: drone simulation

Folder `sim/`, branch `eduardo-drone-sim`. Design: `sim/docs/ARCHITECTURE.md`.

**Goal:** a simulated x500 quadcopter in Gazebo, flown by our Python through PX4 Offboard mode (MAVSDK) and driven by the TypeFly LLM planner. It is also the test bed for Track C's perception, before anything flies for real.

| | Task | Suggested owner | Done when |
|---|---|---|---|
| M1 | Quickstart from `sim/README.md` on your own laptop: `scripts/start_sim.sh`, then `scripts/py.sh examples/01_takeoff_square_land.py`, then `scripts/stop_sim.sh`. Note your OS, chip and how long each step took | Junwei (everyone on B) | The square flight lands where it took off, and your notes are in the PR |
| M1 | Chair demo with the canned plan: `scripts/setup_typefly.sh`, then `scripts/py.sh typefly/e2e_check.py --no-llm`. It stops 2.8 m short today; get it within 1 m | Junwei | Within 1 m, recorded |
| M2 | Same demo via the LLM, with an API spending cap | Junwei | Demo video |
| M2 | Run the SITL failsafe suite on your machine and add RC-loss / `COM_OBL_RC_ACT` cases | Max or Stephen | New cases pass and the failsafe matrix is updated |
| M2 | **Follow-person in sim**: the baseline works (`sim/examples/follow_person.py`, results in `sim/docs/runs/`). Next: rerun it with `--person yolox`, try other altitudes and walker paths, and rerun the round-1 failsafe suite | Junwei + Track C | Numbers table updated in `sim/README.md` |
| M3 | Demo video | Track | |

**Heads-up:** native `mavsdk` 4.x needs macOS 15+ (and JetPack 5+ on a Jetson). On older Macs, writing a `mavsdk_grpc` backend for `quackdrone` is a good first task.

**Learn:** PX4 architecture, NED, async Python with MAVSDK, SDF worlds, QGroundControl. Optional: a PID-detuning demo in QGroundControl to *see* what PX4's loops do.

## Track C: face detection on a Jetson on the drone

Folder `jetson/`, branch `eduardo-jetson-face`. Research: `jetson/docs/WHAT_WE_NEED.md`. Design: `jetson/docs/ARCHITECTURE.md`.

**Goal:** a camera on the drone plus a Jetson running face detection, feeding a follow controller that sends velocity setpoints to PX4. Detection only, never recognition of strangers.

| | Task | Suggested owner | Done when |
|---|---|---|---|
| M1 | Quickstart from `jetson/README.md`: `python -m jetson.fetch models`, then `python -m jetson.demo --source 0 --show` on your webcam, then `python -m jetson.bench` | Max | Your FPS numbers in the PR, labelled with your hardware |
| M1 | Calibrate `--face-width` at 3 m and 5 m with a tape measure, so the distance estimates are right | Max | Table of measured vs estimated distance |
| M1 | Add velocity feed-forward to the controller. Proportional control alone trails a walking person: 6.1 m behind instead of the 4 m target at 0.8 m/s | Max | The closed-loop test shows the gap closing |
| M2 | Follow-person in SITL works through `quackdrone` (round 2). Next: with consenting team members, record a walk-away clip and measure the body-lock path on it (`python -m jetson.lockeval`) | Max + Junwei | Lock-retention numbers on a real clip |
| M2 | Hardware: SBU EH&S pre-approval, then order the parts (see `jetson/docs/WHAT_WE_NEED.md`) | Eddie | Parts ordered |
| M3 | Jetson bring-up when the hardware arrives, following `jetson/docs/BRINGUP_CHECKLIST.md`, ending with a props-off bench test | Max | Jetson FPS in the README |

## Open decisions for Eddie

1. **Track assignments**: confirm on Friday, ask Max and Donald what they want, and confirm whether Ricky (`rickyliu3095-cell`) is Junwei.
2. **SBU rules before buying or flying.** `jetson/docs/WHAT_WE_NEED.md` cites SBU's Drone Safety Policy (EH&S, Dec 2024):
   - Buying or flying a drone for university work needs **EH&S pre-approval**.
   - Project flights need a **Part 107** remote pilot. Who gets certified?
   - The SBU R&D Park has an FAA-recognised flying field (FRIA).
3. **Hardware and budget.**
   - Jetson: start on an original Jetson Nano if we already own one. Otherwise buy an **Orin Nano Super** ($399, often backordered; about $503 for the whole Orin kit).
   - Drone: Holybro X500 V2 + Pixhawk 6C, which matches the sim airframe.
   - Camera: buy a module that explicitly says **IMX219**. New RPi Camera v2 units use a sensor with no Jetson driver.
   - Remote ID: buy a module on the **FAA-accepted list**. Holybro's isn't accepted for home-built drones.
   - Who pays?
4. **511NY developer access** (free request form) before any continuous CCTV ingest. Also ask NYSDOT whether ~26 Mbit/s of continuous video pulls is OK.
5. **Before pushing `eduardo` (drone-ducks is a public repo):** is it OK to keep three small NYSDOT camera stills in git (a 52 KB empty-road reference frame and two doc images), plus the raw 511NY camera-list snapshot? 511NY's terms allow redistribution with attribution, a last-updated time and their disclaimer.
6. **Real-drone PX4 parameters:**
   - `COM_RC_OVERRIDE=3`. With the default, moving the RC sticks does *not* take control back from Offboard.
   - `MAV_1_CONFIG=102` so the Jetson can talk MAVLink on TELEM2.
   - An RTL altitude far below the 60 m default.
   - A choice for `COM_OBL_RC_ACT`.
7. **TypeFly:** keep our patch on a pinned upstream clone, or offer the PX4 platform to upstream TypeFly?
8. **Detector licences:** YuNet is MIT, SCRFD is research-only and YOLO-face is AGPL. Are the last two OK for internal experiments only?
9. **LLM budget** for TypeFly demos: an API key with a hard spending cap.
10. **GPU for CCTV at scale:** one T4 handles about 85 cameras at 7.5 fps, roughly $250/month if it only runs in daylight. Lab servers need their owners' OK.
11. **When `eduardo` gets pushed** and later merged into `main`.
12. **Round-2 calls:**
    - Face score threshold 0.5 (brief false locks on the back of the head) or 0.6 (misses the walker beyond ~3.5 m)?
    - Should `--person yolox` be the default, and should face-only mode also wait for the lost-target timeout before taking a new face?
    - Is a straight walker path OK for M2?
    - Keep the 368 KB SITL run CSV in git? (No people in it, just numbers.)
    - Lost target currently **lands**. Return-home on real hardware climbs to 60 m unless the parameter changes (see 6).
13. **Round-3 calls:**
    - With the body lock, the drone follows someone who walks away (up to 1.5 m/s) until it loses their body. Is that wanted, or should walking away end the session?
    - Keep the head-yaw sideways term off by default? Sideways flight is blind, because the camera only looks forward.
    - At 3 m altitude, use a 5 m target distance with the body lock on (the sim agent's recommendation).
    - With the re-lock distance limit, losing the face beyond about 5.5 m means a landing.
14. **Round-4 calls** (walk-away now exists as an option, off by default, which partly answers the first round-3 call):
    - Should walk-away **land** (the current lost-target action) or default to `hold`, so the pilot gets the drone back?
    - Is 0.8 m/s the right threshold, or should a slow drift away (0.6 m/s or lower) also end the session?
    - Should walk-away be on by default whenever the body lock is on?

## How we work

- **Friday meeting (~45 min):** each mentee demos for about 3 minutes by screen-sharing something *running*. Then blockers, then next week's task.
- **Async standup on Discord, Mon and Wed:** three lines, `done / next / blocked`.
- **PRs:** small, one thing each, with a screenshot or clip, reviewed by Eddie or by someone from *another* track. Cross-review is how everyone learns the whole system.
- **Definition of done:** PR merged, README updated, demo shown.
- **Stuck for more than 30 minutes?** Post the exact error and what you tried in Discord. Asking early is a skill, not a weakness.
- **Safety:** props off for every bench test, LiPo batteries charged in a fire-safe bag and never left unattended, and nobody flies real hardware without Eddie present.

## Paste-ready Discord kickoff

> **ProjectQuack: new plan for the rest of the semester** 🦆
> Three tracks: **A** CCTV speed analytics on the LIE (vision + website), **B** drone simulation (PX4/Gazebo/MAVSDK + LLM planner), **C** face detection on a Jetson on the drone.
> Before Friday: (1) open a tiny PR into the `eduardo` branch of drone-ducks, (2) run Stephen's `BTPC.py` or `TrackScript.py`, (3) take the free FAA TRUST test, (4) read `docs/MENTEE_ROADMAP.md` and tell me which track you want.
> Friday: everyone demos their setup running, and we lock in tracks. M1 is due Fri Nov 20.
