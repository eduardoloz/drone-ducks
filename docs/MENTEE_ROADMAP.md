# Mentee roadmap: fall 2026 through winter break

_Owner: Eddie. Last updated 2026-11-10. This is a living doc; update it at each Friday meeting._

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
| M1 | Run the pipeline on a recorded LIE clip; hand-calibrate one camera from the lane dashes (10 ft dash, 30 ft gap) | Stephen | Speed histogram for one clip looks plausible, with the median near the actual traffic speed |
| M1 | Website skeleton: map of LIE cameras and an events table read from the SQLite DB | Donald | Page loads at `localhost:8010` with real data |
| M2 | Continuous ingest from several cameras, with camera-moved (PTZ) detection | Stephen | Runs unattended for 1 h |
| M2 | Validation drive: a passenger logs phone GPS while the car passes a calibrated camera at a steady, legal speed | Stephen + Donald | Error table comparing estimate vs GPS |
| M3 | Accuracy write-up and demo video | Track | Numbers with error bars in the README |

**Rules:** no licence-plate reading and no identifying people or vehicles. Speeds are estimates, not enforcement. Follow the camera feeds' terms of use. Validation drives are always legal speed, and the *passenger* handles the phone.

## Track B: drone simulation

Folder `sim/`, branch `eduardo-drone-sim`. Design: `sim/docs/ARCHITECTURE.md`.

**Goal:** a simulated x500 quadcopter in Gazebo, flown by our Python through PX4 Offboard mode (MAVSDK) and driven by the TypeFly LLM planner. It is also the test bed for Track C's perception, before anything flies for real.

| | Task | Suggested owner | Done when |
|---|---|---|---|
| M1 | Sim runs on your own laptop; the basic-moves test passes | Junwei | Screen recording |
| M1 | "Find the chair and fly to it" through the skill layer, no LLM | Junwei | Drone stops within ~1 m of the chair |
| M2 | Same demo via the LLM, with an API spending cap | Junwei | Demo video |
| M2 | Failsafe tests: setpoint loss, geofence breach, our script crashing mid-flight | Max or Stephen | Each behaves as documented in the failsafe matrix |
| M2 | Person in the world + follow-person using Track C's controller | Junwei + Track C | Drone keeps the walking person centred |
| M3 | Demo video | Track | |

**Learn:** PX4 architecture, NED, async Python with MAVSDK, SDF worlds, QGroundControl. Optional: a PID-detuning demo in QGroundControl to *see* what PX4's loops do.

## Track C: face detection on a Jetson on the drone

Folder `jetson/`, branch `eduardo-jetson-face`. Research: `jetson/docs/WHAT_WE_NEED.md`. Design: `jetson/docs/ARCHITECTURE.md`.

**Goal:** a camera on the drone plus a Jetson running face detection, feeding a follow controller that sends velocity setpoints to PX4. Detection only, never recognition of strangers.

| | Task | Suggested owner | Done when |
|---|---|---|---|
| M1 | Face detection on your webcam (laptop-runnable backend), plus an FPS benchmark | Max | Numbers in the README, labelled with hardware |
| M1 | Face-follow controller dry run on the webcam: the HUD shows the commands it *would* send | Max | Moving your face left makes it say "yaw left" |
| M2 | Controller → SITL through the MAVSDK bridge (with Track B) | Max + Junwei | Sim drone yaws to keep the sim person centred |
| M2 | Hardware decision and order: Jetson model, camera, power, mount | Eddie | Parts ordered |
| M3 | Jetson bring-up when hardware arrives: fast backend, props-off bench test | Max | Jetson FPS in the README |

## Open decisions for Eddie

1. **Track assignments**: confirm on Friday and ask Max and Donald what they want.
2. **Hardware and budget**: the drone (Holybro X500 V2 + Pixhawk 6C matches the sim airframe), the Jetson (see Track C's doc), and who pays.
3. **Compute for CCTV at scale**: more than one or two live cameras needs a GPU box. Get the server owners' OK before using a lab server for a mentee project.
4. **LLM budget** for TypeFly demos: an API key with a hard spending cap.
5. **When `eduardo` gets pushed** and later merged into `main`.

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
