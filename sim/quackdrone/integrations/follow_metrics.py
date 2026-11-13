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


def _sim_dts(rows: list[dict], max_dt: float = 0.5) -> list[float]:
    """Sim-time duration each row stands for (gap to the next row, capped; 0 without sim time)."""
    out = []
    for r, nxt in zip(rows, rows[1:] + [None]):
        a, b = r.get("sim_time"), (nxt or {}).get("sim_time")
        out.append(min(max_dt, max(0.0, b - a)) if a is not None and b is not None else 0.0)
    return out


def _in_windows(ph: float, windows) -> bool:
    return any(w0 <= ph < w1 for w0, w1, *_ in windows)


def stay_in_front(rows: list[dict], script: walker.Script, in_front_deg: float = 30.0) -> dict:
    """How long the drone stayed in front of the walker, from ground truth (walker heading vs the
    direction to the drone: walker_facing_deg, 0 = looking straight at the drone).

    Counted over frames from the first lock to the end, while the walker's script says the walker
    faces the drone (script.facing_windows); durations in sim seconds. Per leg (script.legs): mean
    and max |facing|, facing at the end of the leg, and % of the leg's frames with a TRACKING mode.
    """
    first = next((i for i, r in enumerate(rows) if r["mode"] == "TRACKING"), None)
    if first is None:
        return {}
    rows = [r for r in rows[first:] if r.get("sim_time") is not None]
    dts = _sim_dts(rows)
    win, front = 0.0, 0.0
    legs: dict = {}
    for r, dt in zip(rows, dts):
        f = r.get("walker_facing_deg")
        ph = walker.phase(r["sim_time"], script)
        if f is None or not _in_windows(ph, script.facing_windows):
            continue
        win += dt
        front += dt if abs(f) <= in_front_deg else 0.0
        for t0, t1, label in script.legs:
            if t0 <= ph < t1:
                legs.setdefault(label, []).append((ph, f, r["mode"]))
    out = {"facing_window_s": round(win, 2), f"in_front_{in_front_deg:g}deg_s": round(front, 2),
           f"pct_in_front_{in_front_deg:g}deg": round(100.0 * front / win, 1) if win > 0 else None}
    for label, pts in legs.items():
        fs = [abs(f) for _ph, f, _m in pts]
        out[f"leg_{label}"] = {
            "mean_abs_facing_deg": round(statistics.fmean(fs), 1), "max_abs_facing_deg": round(max(fs), 1),
            "end_facing_deg": round(pts[-1][1], 1), "end_phase_s": round(pts[-1][0], 1),
            "pct_tracking": round(100.0 * sum(m == "TRACKING" for *_x, m in pts) / len(pts), 1)}
    return out


def back_of_head(rows: list[dict], away_deg: float = 120.0, settle_s: float = 1.0) -> dict:
    """False locks on the back of the walker's head, from ground truth.

    relocks_facing_away   transitions into TRACKING (first lock or re-lock) by a face (not the body
                          path) while the walker faces > 90 deg away from the drone
    tracking_facing_away_s  sim seconds TRACKING by a face while the walker has faced > away_deg
                          away for at least settle_s (skips the controller coasting on the last
                          frontal detection for max_obs_age_s right after a turn)
    """
    relocks, relock_at, away_s = 0, [], 0.0
    away_since = None
    prev = None
    dts = _sim_dts(rows)
    for r, dt in zip(rows, dts):
        f, st = r.get("walker_facing_deg"), r.get("sim_time")
        by_face = r.get("source") in (None, "", "face")
        if f is not None and st is not None:
            if abs(f) > away_deg:
                away_since = st if away_since is None else away_since
            else:
                away_since = None
            if r["mode"] == "TRACKING" and prev not in (None, "TRACKING") and by_face and abs(f) > 90.0:
                relocks += 1
                relock_at.append({"t": r["t"], "facing_deg": round(f, 1),
                                  "true_dist_m": None if r.get("true_dist_m") is None else round(r["true_dist_m"], 2)})
            if (r["mode"] == "TRACKING" and by_face and away_since is not None
                    and st - away_since >= settle_s):
                away_s += dt
        prev = r["mode"]
    return {"relocks_facing_away": relocks, "relocks_facing_away_at": relock_at,
            "tracking_facing_away_s": round(away_s, 2)}


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
    ap.add_argument("--walker", default=None, help="walker script (lead/turnaway/angled) for ended_early "
                    "and the stay-in-front numbers")
    args = ap.parse_args(argv)
    rows = read_csv(args.csv)
    out = summarize(rows, args.target)
    out["back_of_head"] = back_of_head(rows)
    if args.walker:
        sc = walker.get(args.walker)
        out["walker_end_phase_s"] = end_phase(rows, sc)
        out["ended_early"] = ended_early(rows, sc)
        out["stay_in_front"] = stay_in_front(rows, sc)
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
