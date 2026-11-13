#!/usr/bin/env python3
"""gz_pose_probe.py: serve Gazebo ground truth (model poses + sim time) as JSON over local HTTP.

Used by sim/examples/follow_person.py to log the true drone-to-walker distance next to the face
controller's estimate. Like gz_camera_bridge.py it must run with Gazebo's Python (it needs
gz.transport13 / gz.msgs10); everything else reads HTTP.

  GET /poses   {"sim_time": s, "age_ms": ms, "poses": {name: [x, y, z, yaw_rad], ...}}   (Gazebo ENU)
  GET /health  {"ok": bool, "msgs": n, "names": [...]}

Usage: sim/scripts/run_pose_probe.sh [--world quack_people] [--port 5602]

Note: Gazebo may not publish a pose for a scripted <actor> on /world/<w>/pose/info (the actor's
animation is evaluated from the script). If "walker" is missing, follow_person.py falls back to
evaluating the walker's trajectory from sim_time (quackdrone.integrations.walker).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"sim_time": None, "poses": {}, "t": 0.0, "msgs": 0}
LOCK = threading.Lock()


def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with LOCK:
            snap = {"world": STATE.get("world"), "sim_time": STATE["sim_time"], "poses": dict(STATE["poses"]),
                    "age_ms": (time.time() - STATE["t"]) * 1e3 if STATE["t"] else None, "msgs": STATE["msgs"]}
        if self.path.startswith("/poses"):
            self._json(snap)
        elif self.path.startswith("/health"):
            self._json({"ok": snap["msgs"] > 0, "msgs": snap["msgs"], "names": sorted(snap["poses"])})
        else:
            self._json({"error": "GET /poses or /health"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="quack_people")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5602)
    args = ap.parse_args()
    try:
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node
    except ImportError as e:
        sys.exit(f"[probe] Gazebo python bindings missing ({e}); run via sim/scripts/run_pose_probe.sh")

    def on_poses(msg):
        poses = {}
        for p in msg.pose:
            if p.name:
                q = p.orientation
                poses[p.name] = [p.position.x, p.position.y, p.position.z, yaw_from_quat(q.w, q.x, q.y, q.z)]
        with LOCK:
            STATE["poses"].update(poses)
            STATE["t"] = time.time()
            STATE["msgs"] += 1

    def on_clock(msg):
        with LOCK:
            STATE["sim_time"] = msg.sim.sec + msg.sim.nsec * 1e-9

    STATE["world"] = args.world
    node = Node()
    for topic, typ, cb in ((f"/world/{args.world}/pose/info", Pose_V, on_poses),
                           (f"/world/{args.world}/dynamic_pose/info", Pose_V, on_poses),
                           (f"/world/{args.world}/clock", Clock, on_clock)):
        ok = node.subscribe(typ, topic, cb)
        print(f"[probe] subscribe {topic}: {'ok' if ok else 'FAILED'}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"[probe] serving http://{args.host}:{args.port}/poses", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
