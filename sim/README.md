# sim/: ProjectQuack drone simulation

PX4 SITL + Gazebo Harmonic + MAVSDK, with `quackdrone`, our drone-control core. **PX4 runs every
control loop (no hand-written PID); our Python sends position / velocity / yaw setpoints in Offboard
mode.** The same code flies a real Pixhawk drone: only the connection URL and camera source change.

Architecture, interfaces for the Jetson track, failsafe matrix: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart (macOS or Ubuntu)

Prerequisite: PX4-Autopilot built for SITL (`make px4_sitl`; Eddie's `~/ProjectQuack/setup/01_px4_gazebo.sh`
does it on Intel Macs). Scripts look for it in `~/PX4-Autopilot` or `~/ProjectQuack/setup/PX4-Autopilot`;
otherwise `export PX4_DIR=...`.

```bash
cd sim
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,vision]"
.venv/bin/python -m pytest tests/unit            # ~65 s, no simulator needed

scripts/start_sim.sh                             # PX4 + Gazebo (quack_demo world) + camera bridge, in tmux
scripts/py.sh examples/01_takeoff_square_land.py # fly a 2 m square
.venv/bin/python -m pytest -m sitl -v -s tests/sitl   # flight + failsafe tests against SITL (~6 min)
scripts/stop_sim.sh                              # ALWAYS stop with this (cleans PX4 lock files, kills gz)
```

No PX4 at all? `Drone("fake://")` runs the same API on a pure-Python stand-in (good for logic, not physics).

| Option | Effect |
|---|---|
| `WORLD=quack_people scripts/start_sim.sh` | people for face detection (3 standing + 1 walking) |
| `WORLD=quack_people_turnaway` | same, walker with a 14 s turned-away stretch (body-lock test) |
| `CAM_PITCH_DEG=20` | camera pitched 20 deg down instead of 8 (for 3 m follow altitude) |
| `HEADLESS=0` | also open the Gazebo GUI |
| `SIM=sih` | PX4's built-in physics, no Gazebo, no camera: lightest option |
| `KEEP_PARAMS=1` | keep PX4 params between runs (default: fresh PX4 each start + `px4/sitl.params`) |

TypeFly ("find the chair and fly to it"):
```bash
scripts/setup_typefly.sh                         # pinned upstream TypeFly + our patch -> .deps/TypeFly
NO_WEBUI=1 scripts/start_all.sh                  # sim + camera + YOLO, no LLM web UI
scripts/py.sh typefly/e2e_check.py --no-llm      # canned plan, no OpenAI cost
```

## Follow-person (Jetson face follower flying quackdrone)

The Jetson track's pipeline (YuNet -> tracker -> `FaceFollowController`) reads the sim camera and
flies through `Drone.set_velocity_body`, so quackdrone's limits and watchdog apply. Glue:
`quackdrone/integrations/jetson_follow.py` (`QuackdroneSink`, a `jetson.bridge.CommandSink`).

```bash
# once, from the repo root with the sim venv: YuNet + YOLOX-nano (gitignored) -> jetson/models/
(cd .. && sim/.venv/bin/python -m jetson.fetch models)
WORLD=quack_people scripts/start_sim.sh          # also starts the ground-truth pose probe on :5602
scripts/py.sh examples/follow_person.py          # take off, fly to the walker's lane, follow, land
scripts/stop_sim.sh

WORLD=quack_people_turnaway scripts/start_sim.sh # walker turns away for ~14 s (> 5 s lost timeout)
scripts/py.sh examples/follow_person.py --person yolox     # face + body lock (jetson LockKeeper)
CAM_PITCH_DEG=20 WORLD=quack_people scripts/start_sim.sh   # camera pitched for 3 m
scripts/py.sh examples/follow_person.py --alt 3 --target-distance 5 --person yolox
WORLD=quack_people_angled scripts/start_sim.sh   # legs at 25 / 40 deg to the line of sight
scripts/py.sh examples/follow_person.py --person yolox --head-lateral 0.02   # "stay in front" (BLIND sideways)
```

`quack_people_angled` (round 4): after an 8 s straight start the walker walks 10 m at 25 deg west of
north, then 10 m at 40 deg east of north, then turns back. The JSON summary now also has
`stay_in_front` (sim seconds with the walker facing within 30 deg of the drone, per-leg facing angles,
from ground truth) and `back_of_head` (face locks while the walker faces away). Re-checked on old logs:
round-2 run 2 has 1 back-of-head re-lock (68.8 s, walker 180 deg away at 7 m); the round-3 logs have 0
re-locks, but `r3-turnaway-body` steered by a back-of-head "face" for 2.1 s while body-locked.
Relock flags: `--relock-hits 1 --relock-score 0 --relock-max-distance 0` restores round-2 behaviour.

**Round 5 (2026-11-17), angled walker A/B, measured in SITL.** Both arms use `--person yolox`, alt 2 m,
target 4 m, `--max-lateral 0.5`, everything else default. B adds `--head-lateral 0.02`. Two runs each,
alternating A/B/A/B. The Mac was heavily loaded the whole time (load average 14-39 on 16 threads:
Roblox ~270 % CPU, another agent's `cctv.pipeline` ~325 %, Chrome ~250 %), so perception ran at
3-8 fps instead of round 3's 11-13. Logs: `docs/runs/r5-angled-body-{nolat,lat002}-{1,2}.*`.

| Run | fps | Followed | Lost events | Landed early? (walker phase at end; final turn at 60 s) | In front (within 30 deg) | Leg A25 facing mean / max | Leg B40 facing mean / max (% tracking) | Back of head: re-locks / s | Sideways cmd max / measured max (p95) / clamp |
|---|---|---|---|---|---|---|---|---|---|
| A1 no lateral | 3.2 | 41.5 s | 2 | **yes** (37.5 s) | 10.2 s = 28 % | 80 / 138 deg | 78 / 83 deg (0 %) | 0 / 0.0 | 0 / 0.07 (0.04) / 0.5 m/s |
| A2 no lateral | 8.1 | 40.7 s | 1 | **yes** (39.7 s) | 10.7 s = 27 % | 82 / 137 deg | 83 / 94 deg (5 %) | 0 / 0.0 | 0 / 0.07 (0.04) / 0.5 m/s |
| B1 `--head-lateral 0.02` | 3.3 | 48.1 s | 4 | **yes** (45.1 s) | 32.3 s = 73 % | **10 / 18 deg** | 104 / 129 deg (42 %) | 0 / 0.0 | 0.50 / 0.49 (0.32) / 0.5 m/s |
| B2 `--head-lateral 0.02` | 3.6 | 57.4 s | 4 | **yes** (50.1 s) | 32.4 s = 65 % | **15 / 32 deg** | 129 / 150 deg (68 %) | 0 / 0.0 | 0.50 / 0.43 (0.35) / 0.5 m/s |

"Measured" sideways speed is the body-frame right velocity derived from logged PX4 position and yaw
(0.5 s finite difference). It never exceeded the 0.5 m/s clamp.

* **The head-yaw term keeps the drone in front on the 25 deg leg.** Without it the walker's facing
  creeps to ~137 deg by the end of leg A (the drone ends up behind him), and both runs land ~5 s after
  the leg B turn. With it, facing stays within 10-15 deg on leg A in both runs. Time in front roughly
  triples (10 s to 32 s), and the follow lasts 7-17 s longer.
* **It does not survive the 65 deg turn into leg B.** At the turn the face disappears within about a
  second. Head yaw is only measured while a face is detected, so the sideways command drops to 0. The
  body lock carries on with no sideways correction until facing passes about 120 deg and the body is
  lost. A bigger clamp would not help: the term has no input once the face is gone. Head-yaw
  estimates were also noisy (sign flips from frame to frame at 3 fps) and saturate near 35 deg.
* No run re-locked on the back of the head. In A1 the walker turned back towards the drone on leg B
  at 5.6-5.8 m (a face was detected) but no re-lock happened. That fits the re-lock distance limit
  (about 5.5 m), but this is inferred from the trace, not logged.
* All four runs landed before the final turn, so `quack_people_angled` is not followable yet with
  either setting.

`quack_people`: the walker walks straight towards the drone for ~62 s (with one 4.5 s look-back), then
turns away; the face-only follower hovers and lands 5 s after losing the face. Per-frame CSV.gz
(estimate vs Gazebo ground truth) and a JSON summary land in `.run/follow/`; re-derive a summary with
`.venv/bin/python -m quackdrone.integrations.follow_metrics <file.csv.gz> --walker turnaway`.

Measured live in SITL (one run each, 2026-11-13; bearing within 5 deg 100 % of the follow in all runs;
details and caveats in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#follow-person-jetson-pipeline--quackdrone)):

| Walker | Alt / camera pitch / target | Detector | Followed | Lost events | Landed early? | True dist (TRACKING), mean |
|---|---|---|---|---|---|---|
| lead (round 2, run 2) | 2 m / 8 deg / 4 m | face | 72.8 s | 3 | no | 2.99 m |
| turnaway | 2 m / 8 deg / 4 m | face | 31.7 s | 1 | **yes** (38 s before the final turn) | 2.85 m |
| turnaway | 2 m / 8 deg / 4 m | face + body | 107.6 s (cap) | 0 | no | 4.23 m |
| lead | 3 m / 20 deg / 4 m | face | 9.6 s | 2 | yes | 3.04 m |
| lead | 3 m / 20 deg / 5 m | face | 27.6 s | 6 | yes | 3.79 m |
| lead | 3 m / 20 deg / 5 m | face + body | 83.2 s | 1 | no | 5.10 m |

## Using the core

```python
from quackdrone import Drone
with Drone("udpin://0.0.0.0:14540") as drone:    # real drone: "serial:///dev/ttyTHS1:921600"
    drone.takeoff(2.0)                            # waits for PX4's estimator, arms, Offboard, climbs
    drone.move(forward=1.0); drone.turn(90)       # blocking position moves; turn + = clockwise
    drone.set_velocity_body(forward_m_s=0.5, yawspeed_deg_s=10)  # non-blocking; resend at 10-30 Hz
# leaving the block lands, also on exceptions / Ctrl-C
```

## Layout

| Path | What |
|---|---|
| `quackdrone/` | core: `drone.py` (Drone), `safety.py` (limits, watchdog), `backends/` (MAVSDK native, fake), `camera.py`, `integrations/` (TypeFly adapter, Jetson follow sink, walker ground truth) |
| `gazebo/` | `x500_quack` drone (x500 + light camera), worlds `quack_demo`, `quack_people` (+ `_turnaway`, `_angled`), `gz_camera_bridge.py`, `gz_pose_probe.py` (ground truth :5602) |
| `scripts/` | start/stop sim, camera bridge, TypeFly setup and launch |
| `typefly/` | patch against upstream TypeFly + `e2e_check.py` |
| `tests/unit` / `tests/sitl` | no-simulator tests / tests against live SITL |
| `examples/` | 01 square, 04 face-detector check in `quack_people`, `follow_person.py` (face / face+body follow) |
| `docs/runs/` | committed follow-run logs (gzipped CSV, floats rounded to 3 decimals) + JSON summaries |

## Good first tasks (mentees)

1. Run the quickstart on your machine; fix whatever breaks in the scripts, and note your OS and timings in `docs/ARCHITECTURE.md` (Verification log).
2. `examples/02_velocity_circle.py`: fly a circle with `set_velocity_body` (forward speed + yaw rate), print the radius you get versus `v / omega`.
3. A QGroundControl `.params` exporter for `px4/sitl.params` (path to hardware).
4. A `mavsdk_grpc` backend in `quackdrone/backends/` for Macs older than macOS 15 (no native `mavsdk` 4 wheel there).
