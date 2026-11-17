"""Follow the walking person in Gazebo with the Jetson face-follow pipeline, flown by quackdrone.

    (cd .. && sim/.venv/bin/python -m jetson.fetch models)   # once: YuNet + YOLOX-nano -> jetson/models/
    WORLD=quack_people scripts/start_sim.sh         # PX4 + Gazebo + camera :5601 + pose probe :5602
    scripts/py.sh examples/follow_person.py          # take off, go to the lane, follow, land
    scripts/stop_sim.sh

    # face + body lock (keeps following when the face turns away), on the long turn-away walker:
    WORLD=quack_people_turnaway scripts/start_sim.sh
    scripts/py.sh examples/follow_person.py --person yolox
    # 3 m altitude: pitch the camera down so the face stays near the image centre
    CAM_PITCH_DEG=20 WORLD=quack_people scripts/start_sim.sh
    scripts/py.sh examples/follow_person.py --alt 3
    # angled legs (25 / 40 deg to the line of sight), with the "stay in front" sideways term:
    WORLD=quack_people_angled scripts/start_sim.sh
    scripts/py.sh examples/follow_person.py --person yolox --head-lateral 0.02

camera /video|/frame -> jetson YuNet (+ YOLOX-nano bodies + LockKeeper with --person) -> tracker
                     -> FaceFollowController -> QuackdroneSink
                     -> Drone.set_velocity_body (quackdrone limits, 0.5 s watchdog) -> PX4 Offboard

Safety (also see docs/ARCHITECTURE.md "Follow-person"):
  * face lost -> controller commands hover at once; quackdrone's watchdog brakes/holds if frames stop
  * lost for --lost-timeout s (5) -> --lost-action (land); never resumes after that
  * no controller output for 3 s (camera/detector dead) -> land
  * soft fence --fence m around takeoff (velocity toward the edge is scaled down), altitude band,
    --max-speed cap on |forward, right|; PX4's own failsafes (geofence params, battery, offboard loss)
    stay on top. The same clamps apply to sideways commands (tests/unit/test_safety.py)
  * --head-lateral flies SIDEWAYS, and sideways flight is BLIND: the only camera looks forward, so
    nothing sees what the drone moves into. Capped at --max-lateral (0.5 m/s); off by default
  * after a loss the controller only gets the target back after a confirmed re-lock (jetson
    RelockGate: --relock-hits consecutive detections or one >= --relock-score, within
    target distance + 1.5 m); --relock-hits 1 --relock-score 0 --relock-max-distance 0 = round 2
  * leaving the `with Drone` block for any reason (Ctrl-C, exception) lands

Writes one CSV row per frame (controller estimate vs Gazebo ground truth; gzipped, floats rounded
to 3 decimals) and a JSON summary to sim/.run/follow/. The pose probe is optional: without it the
truth columns are empty. Re-derive a summary: python -m quackdrone.integrations.follow_metrics X.csv.gz
"""
from __future__ import annotations

import argparse
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

from jetson import cli as jcli  # noqa: E402  (importable after jetson_follow)
from jetson.controller import FaceFollowController  # noqa: E402
from jetson.detectors import create_detector  # noqa: E402
from jetson.detectors.base import model_path  # noqa: E402
from jetson.pipeline import FollowPipeline  # noqa: E402
from jetson.sources import LatestFrameReader, open_source  # noqa: E402

FIELDS = ["t", "sim_time", "mode", "faces", "est_bearing_deg", "est_dist_m", "req_fwd", "req_yaw", "sent_fwd",
          "sent_yaw", "px4_n", "px4_e", "alt_m", "px4_yaw", "drone_src", "walker_src", "walker_n", "walker_e",
          "true_dist_m", "true_bearing_deg", "walker_facing_deg", "frame_age_ms", "detect_ms", "reason",
          "source", "bodies", "person_ms", "req_right", "sent_right", "head_yaw_deg", "gated"]
FETCH_HINT = "fetch it from the repo root with the sim venv: (cd .. && sim/.venv/bin/python -m jetson.fetch models)"


def find_model(path: str | None, fname: str) -> str:
    """--model if given, else jetson/models/<fname> (where `python -m jetson.fetch models` puts it)."""
    p = path or model_path(fname)
    if not os.path.isfile(p):
        sys.exit(f"model not found: {p}\n  {FETCH_HINT}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default="udpin://0.0.0.0:14540")
    ap.add_argument("--source", default="sim", help="camera: sim (raw /frame) or http://127.0.0.1:5601/video")
    ap.add_argument("--alt", type=float, default=2.0, help="follow altitude above takeoff (m)")
    ap.add_argument("--max-seconds", type=float, default=110.0, help="stop following after this long")
    ap.add_argument("--no-reposition", action="store_true", help="follow from the takeoff spot")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for the walker to restart its loop")
    ap.add_argument("--skip-loops", type=int, default=2, metavar="N",
                    help="don't start in the walker's first N loops after a world start: Gazebo renders the "
                         "actor at inconsistent positions then (seen 2026-11-13: no face found, 3 of 3 runs "
                         "started at sim time ~80 s; every run started at >= 160 s was fine)")
    ap.add_argument("--target-distance", type=float, default=4.0)
    ap.add_argument("--lost-timeout", type=float, default=5.0)
    ap.add_argument("--lost-action", default="land", choices=("land", "rtl", "hold"))
    ap.add_argument("--fence", type=float, default=25.0, help="quackdrone soft fence radius (m)")
    ap.add_argument("--max-speed", type=float, default=1.5, help="quackdrone horizontal speed cap (m/s)")
    ap.add_argument("--model", default=None, help="YuNet .onnx (default: jetson/models)")
    ap.add_argument("--person", default="none", choices=("none", "yolox", "hog"),
                    help="also detect bodies and keep the lock when the face turns away (jetson LockKeeper)")
    ap.add_argument("--person-every", type=int, default=1, metavar="N", help="person detector every Nth frame")
    ap.add_argument("--person-model", default=None, help="YOLOX-nano .onnx (default: jetson/models)")
    ap.add_argument("--walker", default="auto", help="walker script for ground truth: auto (from the pose "
                    "probe's world), lead (quack_people), turnaway (quack_people_turnaway) or angled "
                    "(quack_people_angled)")
    ap.add_argument("--camera-preset", default="sim", choices=sorted(jcli.CAMERA_PRESETS),
                    help="jetson camera preset (HFOV + face width); sim = quack_cam 68.75 deg + walker face "
                         "0.13 m, the same values as sim_follow_config()")
    ap.add_argument("--head-lateral", type=float, default=0.0, metavar="K",
                    help="jetson k_head_lateral: fly sideways K m/s per deg of head yaw to stay in front "
                         "of the face (0 = off; BLIND: the camera looks forward only)")
    ap.add_argument("--max-lateral", type=float, default=0.5, help="jetson max_lateral_m_s (m/s)")
    ap.add_argument("--relock-hits", type=int, default=3,
                    help="after a loss, resume after this many consecutive detections (1 = round 2)")
    ap.add_argument("--relock-score", type=float, default=0.9, help="...or one this good (<= 0 disables)")
    ap.add_argument("--relock-max-distance", type=float, default=None, metavar="M",
                    help="...and only within M m (default target distance + 1.5; 0 = any distance)")
    ap.add_argument("--score", type=float, default=0.5,
                    help="YuNet score threshold (sim walker scores 0.65-0.75 at 2.5-3.5 m; jetson default 0.6)")
    ap.add_argument("--snapshot-every", type=int, default=0, metavar="N",
                    help="also save every Nth camera frame as JPEG next to the log (debugging)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), ".run", "follow"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path, json_path = (os.path.join(args.out, f"follow-{stamp}.{x}") for x in ("csv.gz", "json"))

    det = create_detector("yunet", model=find_model(args.model, "face_detection_yunet_2023mar.onnx"),
                          input_width=640, score_threshold=args.score)
    person = None
    if args.person == "yolox":
        from jetson.detectors.person import YOLOX_NANO, create_person_detector
        person = create_person_detector("yolox", model=find_model(args.person_model, YOLOX_NANO))
    elif args.person == "hog":
        from jetson.detectors.person import create_person_detector
        person = create_person_detector("hog")
    reader = LatestFrameReader(open_source(args.source, realtime=True)).start()
    if reader.get(timeout=10.0) is None:
        print(f"no camera frames from {args.source}: is the camera bridge up? (curl :5601/health)")
        reader.stop()
        return 2
    truth0 = fm.fetch_truth()
    if truth0 is None:
        print("note: no pose probe on :5602 -> no ground-truth columns (start_sim.sh starts it)")
    script = fm.script_for(truth0, args.walker)
    print(f"walker script: {script.name} (period {script.period_s} s, final turn at {script.final_turn_s} s)")

    hfov, face_w = jcli.camera_params(args.camera_preset)
    cfg = sim_follow_config(target_distance_m=args.target_distance, lost_timeout_s=args.lost_timeout,
                            lost_action=args.lost_action, hfov_deg=hfov, face_width_m=face_w,
                            k_head_lateral=args.head_lateral, max_lateral_m_s=args.max_lateral)
    limits = Limits(fence_radius_m=args.fence, max_speed_xy=args.max_speed)
    rows: list[dict] = []
    with Drone(args.connect, limits=limits) as drone:
        drone.takeoff(args.alt)
        if not args.no_reposition:
            sn, se = walker.enu_to_ned(*script.drone_start_enu)
            wn, we = walker.enu_to_ned(*script.waypoints[0][1:3])
            yaw = math.degrees(math.atan2(we - se, wn - sn))
            print(f"repositioning to NED ({sn:.1f}, {se:.1f}) facing {yaw:.0f} deg")
            drone.goto_ned(sn, se, drone.home_ned[2] - args.alt, yaw, timeout_s=40)
        if not args.no_wait:
            truth = fm.fetch_truth()
            if truth and truth.get("sim_time") is not None:
                ph = walker.phase(truth["sim_time"], script)
                wait = (script.period_s - ph - 2.0) % script.period_s
                while truth["sim_time"] + wait + 2.0 < args.skip_loops * script.period_s - 0.5:
                    wait += script.period_s
                print(f"walker at phase {ph:.1f}/{script.period_s} s: waiting {wait:.1f} s of sim time")
                wait_until_sim_time(truth['sim_time'] + wait)  # drone holds its Offboard position target meanwhile
        sink = QuackdroneSink(drone, takeoff_alt_m=args.alt)
        pipe = FollowPipeline(det, FaceFollowController(cfg), sink=sink, person_detector=person,
                              person_every=args.person_every, **jcli.relock_kwargs(args))
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
                if args.snapshot_every and len(rows) % args.snapshot_every == 0:
                    import cv2
                    snap_dir = os.path.join(args.out, f"frames-{stamp}")
                    os.makedirs(snap_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(snap_dir, f"{len(rows):05d}.jpg"), frame.image)
                rows.append(make_row(time.monotonic() - t0, res, sink, drone, script, person is not None))
                if len(rows) % 30 == 0:
                    r = rows[-1]
                    print(f"t={r['t']:5.1f} {r['mode']:8s} est {fmt(r['est_dist_m'])} m "
                          f"true {fmt(r['true_dist_m'])} m  bearing est {fmt(r['est_bearing_deg'])} "
                          f"true {fmt(r['true_bearing_deg'])}  facing {fmt(r['walker_facing_deg'])}"
                          + (f"  right {fmt(r['sent_right'])}" if args.head_lateral else "")
                          + (f"  [{r['source'] or '-'}]" if person is not None else ""))
        finally:
            sink.stop()          # lands if the session is still ours
            reader.stop()
            config = {"alt_m": args.alt, "person": args.person, "person_every": args.person_every,
                      "walker": script.name, "target_distance_m": args.target_distance,
                      "lost_timeout_s": args.lost_timeout, "score": args.score, "source": args.source,
                      "cam_pitch_deg": float(os.environ.get("CAM_PITCH_DEG", 8.0)),
                      "camera_preset": args.camera_preset, "hfov_deg": hfov, "face_width_m": face_w,
                      "head_lateral": args.head_lateral, "max_lateral_m_s": args.max_lateral,
                      "relock_hits": args.relock_hits, "relock_score": args.relock_score,
                      "relock_max_distance_m": args.relock_max_distance}
            write_outputs(rows, cfg.target_distance_m, csv_path, json_path, sink.finish_reason, script, config)
    return 0


def wait_until_sim_time(target: float, timeout_s: float = 600.0) -> None:
    """Sleep until Gazebo's sim time reaches `target` (sim runs slower than real time on a busy laptop,
    so sleeping the difference in wall time arrives early)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        t = fm.fetch_truth()
        st = t.get("sim_time") if t else None
        if st is not None and st >= target:
            return
        time.sleep(0.1 if st is not None and target - st < 1.0 else 0.5)


def fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def make_row(t, res, sink, drone, script=walker.LEAD, body_path=False) -> dict:
    tel = drone.telemetry
    truth = fm.fetch_truth()
    w, wsrc = fm.walker_from_truth(truth, script)
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
           "reason": c.reason, "source": res.target_source if body_path else None,
           "bodies": len(res.bodies) if body_path else None, "person_ms": res.timings_ms.get("person"),
           "req_right": last.requested.right_m_s if last else None,
           "sent_right": last.sent.right_m_s if last and last.sent else None,
           "head_yaw_deg": c.head_yaw_deg, "gated": int(bool(getattr(res, "gated", False)))}
    if w:
        row.update(fm.truth_geometry(d[:2], d[2], w[:2], w[2]))
    return row


def write_outputs(rows, target, csv_path, json_path, finish_reason, script=walker.LEAD, config=None) -> None:
    fm.write_csv(csv_path, rows, FIELDS)
    summary = fm.summarize(fm.read_csv(csv_path), target)   # from the logged values: reproducible from the CSV
    summary["finish_reason"] = finish_reason
    summary["walker_end_phase_s"] = fm.end_phase(rows, script)
    summary["ended_early"] = fm.ended_early(rows, script)
    logged = fm.read_csv(csv_path)
    summary["stay_in_front"] = fm.stay_in_front(logged, script)
    summary["back_of_head"] = fm.back_of_head(logged)
    summary["config"] = config or {}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"log: {csv_path}\nsummary: {json_path}")


if __name__ == "__main__":
    sys.exit(main())
