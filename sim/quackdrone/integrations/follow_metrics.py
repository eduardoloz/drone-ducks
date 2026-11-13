"""Ground truth + summary numbers for a follow run (pure functions; used by examples/follow_person.py).

One row per processed camera frame. Angles in degrees, distances in metres, positions in PX4
local NED (north, east) unless named *_enu.

Re-derive a summary from a logged (optionally gzipped) CSV, e.g. a committed run in docs/runs/:

    python -m quackdrone.integrations.follow_metrics docs/runs/follow-sitl-run2.csv.gz [--target 4]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
import sys
import urllib.request
from typing import Optional

from . import walker

POSE_PROBE_URL = "http://127.0.0.1:5602/poses"
DRONE_MODEL = "x500_quack_0"


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def fetch_truth(url: str = POSE_PROBE_URL, timeout: float = 0.2) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def truth_geometry(drone_ne: tuple[float, float], drone_yaw_deg: float, walker_ne: tuple[float, float],
                   walker_yaw_enu_rad: float) -> dict:
    """True distance (horizontal), bearing of the walker in the drone's view (+ = right of the
    camera axis), and the walker's facing angle (0 = looking straight at the drone)."""
    dn, de = walker_ne[0] - drone_ne[0], walker_ne[1] - drone_ne[1]
    dist = math.hypot(dn, de)
    bearing = wrap180(math.degrees(math.atan2(de, dn)) - drone_yaw_deg)
    # walker heading in NED degrees (ENU yaw 0 = east = NED 90, ENU + = counter-clockwise)
    walker_heading_ned = 90.0 - math.degrees(walker_yaw_enu_rad)
    to_drone = math.degrees(math.atan2(-de, -dn))
    facing = wrap180(to_drone - walker_heading_ned)
    return {"true_dist_m": dist, "true_bearing_deg": bearing, "walker_facing_deg": facing}


def script_for(truth: Optional[dict], override: Optional[str] = None) -> walker.Script:
    """The walker script of the running world (the probe reports it), or `override` ('lead',
    'turnaway' or a world name). Unknown/absent -> the lead walker."""
    name = override if override and override != "auto" else (truth or {}).get("world")
    try:
        return walker.get(name)
    except KeyError:
        return walker.LEAD


def walker_from_truth(truth: Optional[dict], script: walker.Script = walker.LEAD
                      ) -> tuple[Optional[tuple[float, float, float]], str]:
    """Walker (north, east, yaw_enu) from the pose probe if it publishes the actor, else from the
    scripted trajectory at the probe's sim time. Returns (pose or None, source)."""
    if not truth:
        return None, "none"
    p = truth.get("poses", {}).get("walker")
    if p:
        n, e = walker.enu_to_ned(p[0], p[1])
        return (n, e, p[3]), "gazebo"
    st = truth.get("sim_time")
    if st is None:
        return None, "none"
    x, y, yaw = walker.walker_pose(st, script)
    n, e = walker.enu_to_ned(x, y)
    return (n, e, yaw), "script"


def drone_from_truth(truth: Optional[dict]) -> Optional[tuple[float, float, float]]:
    """Drone (north, east, yaw NED deg) from Gazebo, if the probe has it."""
    p = (truth or {}).get("poses", {}).get(DRONE_MODEL)
    if not p:
        return None
    n, e = walker.enu_to_ned(p[0], p[1])
    return n, e, wrap180(90.0 - math.degrees(p[3]))


def summarize(rows: list[dict], target_distance_m: float, centred_deg: float = 5.0) -> dict:
    """Numbers for the report. rows need t, mode; optional true_bearing_deg, true_dist_m,
    est_dist_m, walker_facing_deg (None when unknown)."""
    if not rows:
        return {"frames": 0}
    track = [r for r in rows if r["mode"] == "TRACKING"]
    first = next((r["t"] for r in rows if r["mode"] == "TRACKING"), None)
    following = [r for r in rows if first is not None and r["t"] >= first and r["mode"] in ("TRACKING", "LOST")]

    def vals(rs, k):
        return [r[k] for r in rs if r.get(k) is not None and math.isfinite(r[k])]

    lost_events, lost_durations, start = 0, [], None
    prev = None
    for r in rows:
        if prev == "TRACKING" and r["mode"] in ("LOST", "TIMEOUT"):
            lost_events += 1
            start = r["t"]
        if start is not None and r["mode"] == "TRACKING":
            lost_durations.append(r["t"] - start)
            start = None
        prev = r["mode"]

    bearings = vals(following, "true_bearing_deg")
    dists = vals(following, "true_dist_m")
    est_err = [r["est_dist_m"] - r["true_dist_m"] for r in track
               if r.get("est_dist_m") is not None and r.get("true_dist_m") is not None]
    facing = vals(following, "walker_facing_deg")
    out = {
        "frames": len(rows),
        "duration_s": rows[-1]["t"] - rows[0]["t"],
        "follow_s": (following[-1]["t"] - first) if following else 0.0,
        "pct_frames_tracking_while_following": 100.0 * len([r for r in following if r["mode"] == "TRACKING"])
        / max(1, len(following)),
        "lost_events": lost_events,
        "reacquired": len(lost_durations),
        "lost_durations_s": [round(x, 2) for x in lost_durations],
        "final_mode": rows[-1]["mode"],
    }
    if bearings:
        out[f"pct_centred_{centred_deg:g}deg"] = 100.0 * sum(abs(b) <= centred_deg for b in bearings) / len(bearings)
        out["pct_centred_10deg"] = 100.0 * sum(abs(b) <= 10.0 for b in bearings) / len(bearings)
        out["pct_in_fov"] = 100.0 * sum(abs(b) <= 34.4 for b in bearings) / len(bearings)
        out["mean_abs_bearing_deg"] = statistics.fmean(abs(b) for b in bearings)
    if dists:
        out["true_dist_mean_m"] = statistics.fmean(dists)
        out["true_dist_median_m"] = statistics.median(dists)
        out["true_dist_minmax_m"] = [min(dists), max(dists)]
        out["target_distance_m"] = target_distance_m
    if est_err:
        out["est_minus_true_dist_mean_m"] = statistics.fmean(est_err)
        out["est_minus_true_dist_mae_m"] = statistics.fmean(abs(e) for e in est_err)
    if facing:
        out["pct_walker_facing_within_60deg"] = 100.0 * sum(abs(f) <= 60 for f in facing) / len(facing)
    if any(r.get("source") for r in rows):   # face+body runs log which box the controller steered by
        body = [r for r in following if r.get("source") == "body"]
        out["body_frames"] = len(body)
        out["pct_following_by_body"] = 100.0 * len(body) / max(1, len(following))
    return out


def end_phase(rows: list[dict], script: walker.Script) -> Optional[float]:
    """Walker script time at the last logged frame, counted from the start of the walker loop in which
    the target was first locked (so it exceeds the period if the follow ran into the next loop).
    None without sim time."""
    st = [r for r in rows if r.get("sim_time") is not None]
    if not st:
        return None
    first = next((r for r in st if r["mode"] == "TRACKING"), st[0])
    t0 = float(first["sim_time"])
    loop_start = t0 - walker.phase(t0, script)
    if walker.phase(t0, script) > script.final_turn_s:   # locked as the walker turned to start the next loop
        loop_start += script.period_s
    return float(st[-1]["sim_time"]) - loop_start


def ended_early(rows: list[dict], script: walker.Script, margin_s: float = 2.0) -> Optional[bool]:
    """True if the follow ended (lost timeout / landing) more than `margin_s` before the walker's final
    turn-away, i.e. the drone gave up on a person who was still there to follow."""
    ph = end_phase(rows, script)
    if ph is None:
        return None
    return ph < script.final_turn_s - margin_s


# ---------------------------------------------------------------- CSV logs
# Committed run logs stay small: floats rounded to what the numbers mean (mm, 0.01 deg/ms), gzip.
ROUND = 3


def compact(v):
    if isinstance(v, float):
        return round(v, ROUND) if math.isfinite(v) else v
    return v


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    if path.endswith(".gz"):   # mtime=0: same rows -> same bytes (no git churn on re-export)
        f = io.TextIOWrapper(gzip.GzipFile(path, "wb", mtime=0), newline="")
    else:
        f = open(path, "w", newline="")
    with f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows({k: compact(r.get(k)) for k in fields} for r in rows)


def _num(v: str):
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def read_csv(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return [{k: (v if k in ("mode", "reason", "source", "drone_src", "walker_src") else _num(v))
                 for k, v in r.items()} for r in csv.DictReader(f)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="summary of a follow_person CSV log (.csv or .csv.gz)")
    ap.add_argument("csv")
    ap.add_argument("--target", type=float, default=4.0, help="target distance used in the run (m)")
    ap.add_argument("--walker", default=None, help="walker script (lead/turnaway) for ended_early")
    args = ap.parse_args(argv)
    rows = read_csv(args.csv)
    out = summarize(rows, args.target)
    if args.walker:
        sc = walker.get(args.walker)
        out["walker_end_phase_s"] = end_phase(rows, sc)
        out["ended_early"] = ended_early(rows, sc)
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
