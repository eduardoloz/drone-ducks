"""Follow the walking person in Gazebo with the Jetson face-follow pipeline, flown by quackdrone.

    WORLD=quack_people scripts/start_sim.sh         # PX4 + Gazebo + camera :5601 + pose probe :5602
    scripts/py.sh examples/follow_person.py          # take off, go to the lane, follow, land
    scripts/stop_sim.sh

camera /video|/frame -> jetson YuNet -> tracker -> FaceFollowController -> QuackdroneSink
                     -> Drone.set_velocity_body (quackdrone limits, 0.5 s watchdog) -> PX4 Offboard

Safety (also see docs/ARCHITECTURE.md "Follow-person"):
  * face lost -> controller commands hover at once; quackdrone's watchdog brakes/holds if frames stop
  * lost for --lost-timeout s (5) -> --lost-action (land); never resumes after that
  * no controller output for 3 s (camera/detector dead) -> land
  * soft fence --fence m around takeoff (velocity toward the edge is scaled down), altitude band,
    2 m/s cap; PX4's own failsafes (geofence params, battery, offboard loss) stay on top
  * leaving the `with Drone` block for any reason (Ctrl-C, exception) lands

Writes one CSV row per frame (controller estimate vs Gazebo ground truth) and a JSON summary to
sim/.run/follow/. The pose probe is optional: without it the truth columns are empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # sim/ (quackdrone) when run without pip install -e

from quackdrone import Drone, Limits  # noqa: E402
from quackdrone.integrations import follow_metrics as fm, walker  # noqa: E402
from quackdrone.integrations.jetson_follow import QuackdroneSink, sim_follow_config  # noqa: E402

from jetson.controller import FaceFollowController  # noqa: E402  (importable after jetson_follow)
from jetson.detectors import create_detector  # noqa: E402
from jetson.pipeline import FollowPipeline  # noqa: E402
from jetson.sources import LatestFrameReader, open_source  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIELDS = ["t", "sim_time", "mode", "faces", "est_bearing_deg", "est_dist_m", "req_fwd", "req_yaw", "sent_fwd",
          "sent_yaw", "px4_n", "px4_e", "alt_m", "px4_yaw", "drone_src", "walker_src", "walker_n", "walker_e",
          "true_dist_m", "true_bearing_deg", "walker_facing_deg", "frame_age_ms", "detect_ms", "reason"]


def find_yunet(path: str | None) -> str | None:
    if path:
        return path
    for d in (os.path.join(REPO, "jetson", "models"),
              os.path.expanduser("~/ProjectQuack/worktrees/jetson/jetson/models")):
        p = os.path.join(d, "face_detection_yunet_2023mar.onnx")
        if os.path.isfile(p):
            return p
    return None  # jetson's default lookup (python -m jetson.fetch models)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default="udpin://0.0.0.0:14540")
    ap.add_argument("--source", default="sim", help="camera: sim (raw /frame) or http://127.0.0.1:5601/video")
    ap.add_argument("--alt", type=float, default=2.0, help="follow altitude above takeoff (m)")
    ap.add_argument("--max-seconds", type=float, default=110.0, help="stop following after this long")
    ap.add_argument("--no-reposition", action="store_true", help="follow from the takeoff spot")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for the walker to restart its loop")
    ap.add_argument("--target-distance", type=float, default=4.0)
    ap.add_argument("--lost-timeout", type=float, default=5.0)
    ap.add_argument("--lost-action", default="land", choices=("land", "rtl", "hold"))
    ap.add_argument("--fence", type=float, default=25.0, help="quackdrone soft fence radius (m)")
    ap.add_argument("--max-speed", type=float, default=1.5, help="quackdrone horizontal speed cap (m/s)")
    ap.add_argument("--model", default=None, help="YuNet .onnx (default: jetson/models)")
    ap.add_argument("--score", type=float, default=0.5,
                    help="YuNet score threshold (sim walker scores 0.65-0.75 at 2.5-3.5 m; jetson default 0.6)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), ".run", "follow"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path, json_path = (os.path.join(args.out, f"follow-{stamp}.{x}") for x in ("csv", "json"))

    model = find_yunet(args.model)
    det = create_detector("yunet", **({"model": model} if model else {}), input_width=640,
                           score_threshold=args.score)
    reader = LatestFrameReader(open_source(args.source, realtime=True)).start()
    if reader.get(timeout=10.0) is None:
        print(f"no camera frames from {args.source}: is the camera bridge up? (curl :5601/health)")
        reader.stop()
        return 2
    if fm.fetch_truth() is None:
        print("note: no pose probe on :5602 -> no ground-truth columns (start_sim.sh starts it)")

    cfg = sim_follow_config(target_distance_m=args.target_distance, lost_timeout_s=args.lost_timeout,
                            lost_action=args.lost_action)
    limits = Limits(fence_radius_m=args.fence, max_speed_xy=args.max_speed)
    rows: list[dict] = []
    with Drone(args.connect, limits=limits) as drone:
        drone.takeoff(args.alt)
        if not args.no_reposition:
            sn, se = walker.enu_to_ned(*walker.DRONE_START_ENU)
            wn, we = walker.enu_to_ned(*walker.WAYPOINTS[0][1:3])
            yaw = math.degrees(math.atan2(we - se, wn - sn))
            print(f"repositioning to NED ({sn:.1f}, {se:.1f}) facing {yaw:.0f} deg")
            drone.goto_ned(sn, se, drone.home_ned[2] - args.alt, yaw, timeout_s=40)
        if not args.no_wait:
            truth = fm.fetch_truth()
            if truth and truth.get("sim_time") is not None:
                wait = (walker.PERIOD_S - walker.phase(truth["sim_time"]) - 2.0) % walker.PERIOD_S
                print(f"walker at phase {walker.phase(truth['sim_time']):.1f}/{walker.PERIOD_S} s: waiting {wait:.1f} s")
                time.sleep(wait)  # drone holds its Offboard position target meanwhile
        sink = QuackdroneSink(drone, takeoff_alt_m=args.alt)
        pipe = FollowPipeline(det, FaceFollowController(cfg), sink=sink)
        sink.start()
        t0, last = time.monotonic(), 0
        try:
            while not sink.finished and time.monotonic() - t0 < args.max_seconds:
                frame = reader.get(timeout=1.0, newer_than=last)
                if frame is None:
                    print("no camera frame for 1 s")   # watchdog brakes; sink lands after 3 s
                    continue
                last = frame.seq
                res = pipe.step(frame)
                rows.append(make_row(time.monotonic() - t0, res, sink, drone))
                if len(rows) % 30 == 0:
                    r = rows[-1]
                    print(f"t={r['t']:5.1f} {r['mode']:8s} est {fmt(r['est_dist_m'])} m "
                          f"true {fmt(r['true_dist_m'])} m  bearing est {fmt(r['est_bearing_deg'])} "
                          f"true {fmt(r['true_bearing_deg'])}  facing {fmt(r['walker_facing_deg'])}")
        finally:
            sink.stop()          # lands if the session is still ours
            reader.stop()
            write_outputs(rows, cfg.target_distance_m, csv_path, json_path, sink.finish_reason)
    return 0


def fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def make_row(t, res, sink, drone) -> dict:
    tel = drone.telemetry
    truth = fm.fetch_truth()
    w, wsrc = fm.walker_from_truth(truth)
    d = fm.drone_from_truth(truth)
    dsrc = "gazebo" if d else "px4"
    if d is None:
        d = (tel.north_m, tel.east_m, tel.yaw_deg)
    c = res.control
    last = sink.last
    row = {"t": round(t, 3), "sim_time": truth.get("sim_time") if truth else None, "mode": c.mode,
           "faces": len(res.faces), "est_bearing_deg": c.bearing_deg, "est_dist_m": c.distance_m,
           "req_fwd": last.requested.forward_m_s if last else None,
           "req_yaw": last.requested.yawspeed_deg_s if last else None,
           "sent_fwd": last.sent.forward_m_s if last and last.sent else None,
           "sent_yaw": last.sent.yawspeed_deg_s if last and last.sent else None,
           "px4_n": tel.north_m, "px4_e": tel.east_m, "alt_m": drone.altitude_m, "px4_yaw": tel.yaw_deg,
           "drone_src": dsrc, "walker_src": wsrc, "walker_n": w[0] if w else None, "walker_e": w[1] if w else None,
           "true_dist_m": None, "true_bearing_deg": None, "walker_facing_deg": None,
           "frame_age_ms": res.timings_ms.get("frame_age"), "detect_ms": res.timings_ms.get("detect"),
           "reason": c.reason}
    if w:
        row.update(fm.truth_geometry(d[:2], d[2], w[:2], w[2]))
    return row


def write_outputs(rows, target, csv_path, json_path, finish_reason) -> None:
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    summary = fm.summarize(rows, target)
    summary["finish_reason"] = finish_reason
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"log: {csv_path}\nsummary: {json_path}")


if __name__ == "__main__":
    sys.exit(main())
