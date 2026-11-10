#!/usr/bin/env python3
""""Find the chair and fly to it", end to end, in Gazebo.

Same path as the web UI: LLMController -> planner -> plan validation -> PlanPolicy -> PX4Wrapper skills
-> quackdrone -> MAVSDK -> PX4 SITL, with YOLO on the Gazebo camera. Then checks the drone really
ended up near the target in quack_demo. Ported from Eddie's quack/test_e2e.py.

    NO_WEBUI=1 sim/scripts/start_all.sh                    # sim + camera + YOLO (no web UI!)
    sim/scripts/py.sh sim/typefly/e2e_check.py --no-llm    # canned plan: no OpenAI call, no cost
    sim/scripts/py.sh sim/typefly/e2e_check.py             # real LLM plan (needs OPENAI_API_KEY, costs money)
    sim/scripts/py.sh sim/typefly/e2e_check.py "find the car and fly to it" --target car --no-llm

Do not run it while the web UI is up: both would try to fly the drone.
"""
import argparse
import json
import math
import os
import sys
import time

TYPEFLY_DIR = os.environ.get("TYPEFLY_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                          ".deps", "TypeFly"))

# Object positions in quack_demo.sdf, Gazebo ENU (x=east, y=north) converted to PX4 NED (north, east).
# NED here is relative to where the drone spawned (Gazebo origin), which is also PX4's EKF origin.
TARGETS_NED = {"chair": (6.0, -1.0), "person": (0.0, 7.0), "car": (9.0, 8.0),
               "fire hydrant": (-6.0, -4.0), "stop sign": (-7.0, 0.0)}
RADIUS = {"car": 2.0}  # rough half-size, so "near" is measured to the object's side

CANNED = {"thoughts": "No {t} in view; scan for it, then fly to it.",
          "plan": "if scan('{t}'):\n    goto('{t}')\n    take_picture()\n    log('Reached the {t}')\n"
                  "else:\n    log('No {t} found')"}


class CannedPlanner:
    """Stands in for the LLM: returns a fixed MiniSpec plan, never touches the network."""

    def __init__(self, target):
        self.target = target

    def plan(self, instruction, error_message=None):
        return json.dumps({k: v.replace("{t}", self.target) for k, v in CANNED.items()})

    def probe(self, query, robot_info):
        return "False"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", nargs="?", default="find the chair and fly to it")
    ap.add_argument("--target", default="chair", choices=sorted(TARGETS_NED))
    ap.add_argument("--no-llm", action="store_true", help="canned plan instead of OpenAI (free)")
    ap.add_argument("--timeout", type=float, default=300.0, help="plan wall-clock budget (s)")
    ap.add_argument("--config", default=os.path.join(TYPEFLY_DIR, "typefly/config/robot_info.px4.json"))
    args = ap.parse_args()

    if args.no_llm:
        # LLMController insists on a key at construction time. Use a dummy one and point the client
        # at a closed local port, so even an accidental call fails fast instead of reaching OpenAI.
        os.environ["OPENAI_API_KEY"] = "sk-not-used-canned-plan"
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
    from typefly import llm_controller as lc
    from typefly.robot_info import RobotInfo

    with open(args.config) as f:
        info = RobotInfo.from_dict(json.load(f))
    ctl = lc.LLMController(info, plan_timeout=args.timeout)
    if args.no_llm:
        ctl.planner = CannedPlanner(args.target)
    robot = ctl.robot

    print(f"== instruction: {args.instruction!r}  ({'canned plan' if args.no_llm else 'LLM plan'})")
    robot.drone.connect(timeout_s=40)
    t = robot.drone.telemetry
    if math.hypot(t.north_m, t.east_m) > 1.5:
        print(f"FAIL: the drone is at N{t.north_m:.1f} E{t.east_m:.1f}, not at the spawn point, so the object "
              "positions above do not apply. Restart the sim first: NO_WEBUI=1 sim/scripts/start_all.sh")
        robot.drone.close(land=False)
        return 2
    robot.start()
    time.sleep(3)  # let a few detections arrive
    t0 = time.time()
    msgs = []
    try:
        ctl._execute_instruction(args.instruction)
    finally:
        elapsed = time.time() - t0
        while not lc._USER_LOG_QUEUE.empty():
            m = str(lc._USER_LOG_QUEUE.get())
            msgs.append(m if "<img" not in m else "<picture>")
        t = robot.drone.telemetry
        tn, te = TARGETS_NED[args.target]
        dist = math.dist((t.north_m, t.east_m), (tn, te))
        bearing = math.degrees(math.atan2(te - t.east_m, tn - t.north_m))
        facing_err = abs((t.yaw_deg - bearing + 180) % 360 - 180)
        visible = robot.is_visible(args.target)
        robot.stop()

    print("robot messages:", *msgs, sep="\n  ")
    print(f"plan took {elapsed:.0f} s; final N{t.north_m:.1f} E{t.east_m:.1f}; {args.target} at N{tn} E{te}: "
          f"{dist:.1f} m away, facing error {facing_err:.0f} deg, visible at end: {visible}")
    ok = dist - RADIUS.get(args.target, 0.3) < 3.5 and facing_err < 30
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
