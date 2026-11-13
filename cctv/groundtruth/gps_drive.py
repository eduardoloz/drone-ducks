"""Validation drive: line a phone GPS log up with camera clips and measure our speed error.

Protocol (who does what, safety, file formats): ``cctv/docs/VALIDATION_DRIVE.md``.

    # 1. passes: when did the car go by the camera, how fast, which way?
    python -m cctv.groundtruth.gps_drive passes --gps drive.gpx --camera R10_041
    # 2. candidates: which tracks in our runs could be the car? (match by eye on the overlay)
    python -m cctv.groundtruth.gps_drive candidates --gps drive.gpx --camera R10_041
    # 3. write matches.json by hand: [{"pass": 0, "run_id": 12, "track_id": 57}, ...]
    # 4. report: error table, bias, spread
    python -m cctv.groundtruth.gps_drive report --gps drive.gpx --camera R10_041 --matches matches.json

Formats read: GPX 1.0/1.1 (``<trkpt>`` with ``<time>``, optional ``<speed>`` in m/s,
also inside ``<extensions>``), and CSV with a header naming time, latitude and
longitude columns (GPSLogger for Android and most iOS loggers export one of these).
A CSV ``speed`` column is taken as m/s unless named ``speed_mph`` / ``speed_kmh``.
When the log has no Doppler speed we derive it from positions and say so: that is
noisier (~1 mph at 1 Hz) than Doppler (~0.5 mph).

Timing: our clip times come from when recording started, and the video content is
about 5-20 s older (HLS latency), so candidates are searched in a +-45 s window and
the car is picked by eye (lane, class, time order), never by plate. Each match also
measures that latency (``offset_s``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cctv.geometry import MPH_PER_MPS

KMH_PER_MPS = 3.6


@dataclass
class Fix:
    t: datetime          # UTC
    lat: float
    lon: float
    speed_mps: float | None = None   # Doppler speed from the receiver, if logged
    accuracy_m: float | None = None


# ---------------------------------------------------------------------- loading
def _parse_time(s: str) -> datetime:
    s = s.strip()
    if re.fullmatch(r"\d{10}(\.\d+)?", s):                   # unix seconds
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    if re.fullmatch(r"\d{13}", s):                           # unix milliseconds
        return datetime.fromtimestamp(int(s) / 1000, tz=timezone.utc)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T", 1))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {s!r} has no time zone; log in UTC or with an offset")
    return dt.astimezone(timezone.utc)


def load_gpx(path: Path | str) -> list[Fix]:
    root = ET.parse(path).getroot()
    out = []
    for el in root.iter():
        if not el.tag.endswith("trkpt"):
            continue
        t = speed = acc = None
        for ch in el.iter():
            tag = ch.tag.rsplit("}", 1)[-1].lower()
            if tag == "time" and ch.text:
                t = _parse_time(ch.text)
            elif tag == "speed" and ch.text:
                speed = float(ch.text)
            elif tag in ("hdop",) and ch.text and acc is None:
                acc = float(ch.text) * 5.0      # rough: HDOP x ~5 m
            elif tag in ("accuracy", "hacc") and ch.text:
                acc = float(ch.text)
        if t is not None:
            out.append(Fix(t, float(el.get("lat")), float(el.get("lon")), speed, acc))
    return sorted(out, key=lambda f: f.t)


def _col(header: list[str], *names: str) -> int | None:
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    return None


def load_csv(path: Path | str) -> list[Fix]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    it = _col(header, "time", "timestamp", "date time", "datetime", "time_utc", "utc")
    ila = _col(header, "lat", "latitude")
    ilo = _col(header, "lon", "lng", "long", "longitude")
    if None in (it, ila, ilo):
        raise ValueError(f"CSV needs time, lat, lon columns; header was {header}")
    isp, factor = _col(header, "speed", "speed_mps", "speed(m/s)"), 1.0
    if isp is None and (isp := _col(header, "speed_mph")) is not None:
        factor = 1 / MPH_PER_MPS
    if isp is None and (isp := _col(header, "speed_kmh", "speed(km/h)")) is not None:
        factor = 1 / KMH_PER_MPS
    iac = _col(header, "accuracy", "accuracy(m)", "hacc")
    out = []
    for r in body:
        if not r or not r[it].strip():
            continue
        sp = float(r[isp]) * factor if isp is not None and r[isp].strip() else None
        ac = float(r[iac]) if iac is not None and r[iac].strip() else None
        out.append(Fix(_parse_time(r[it]), float(r[ila]), float(r[ilo]), sp, ac))
    return sorted(out, key=lambda f: f.t)


def load_log(path: Path | str) -> list[Fix]:
    return load_gpx(path) if str(path).lower().endswith(".gpx") else load_csv(path)


# --------------------------------------------------------------------- geometry
def _enu(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    r = 6371000.0
    return np.c_[np.radians(lon - lon0) * r * math.cos(math.radians(lat0)), np.radians(lat - lat0) * r]


def bearing_to_direction(bearing_deg: float) -> str:
    """LIE runs roughly east-west: compass bearing -> EB / WB (or '?' when it is heading N/S)."""
    b = bearing_deg % 360
    if 30 <= b <= 150:
        return "EB"
    if 210 <= b <= 330:
        return "WB"
    return "?"


@dataclass
class Pass:
    index: int
    t_utc: datetime              # time of closest approach to the camera (interpolated)
    distance_m: float            # closest distance to the camera position
    speed_mph: float             # GPS speed averaged over +-window around closest approach
    speed_sd_mph: float          # spread of that speed in the window (was the car steady?)
    speed_source: str            # "doppler" or "positions"
    bearing_deg: float
    direction: str
    n_fixes: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["t_utc"] = self.t_utc.isoformat(timespec="milliseconds")
        return d


def find_passes(fixes: list[Fix], cam_lat: float, cam_lon: float, radius_m: float = 250.0,
                window_s: float = 3.0) -> list[Pass]:
    """Each time the log comes within ``radius_m`` of the camera, one Pass at the closest approach."""
    if len(fixes) < 3:
        return []
    t = np.array([(f.t - fixes[0].t).total_seconds() for f in fixes])
    xy = _enu(np.array([f.lat for f in fixes]), np.array([f.lon for f in fixes]), cam_lat, cam_lon)
    dist = np.linalg.norm(xy, axis=1)
    doppler = np.array([np.nan if f.speed_mps is None else f.speed_mps for f in fixes])
    has_doppler = np.isfinite(doppler).mean() > 0.8
    # position-derived speed: central differences
    v_pos = np.full(len(t), np.nan)
    dt = t[2:] - t[:-2]
    ok = dt > 0
    v_pos[1:-1][ok] = np.linalg.norm(xy[2:] - xy[:-2], axis=1)[ok] / dt[ok]
    speed = doppler if has_doppler else v_pos
    inside = dist <= radius_m
    passes, i = [], 0
    while i < len(t):
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and inside[j + 1] and t[j + 1] - t[j] < 10:
            j += 1
        k = i + int(np.argmin(dist[i:j + 1]))
        # refine closest approach on the segment through k (linear motion between fixes)
        tc, dc = t[k], dist[k]
        for a, b in ((k - 1, k), (k, k + 1)):
            if 0 <= a and b < len(t) and t[b] > t[a]:
                d = xy[b] - xy[a]
                s = float(np.clip(-np.dot(xy[a], d) / max(np.dot(d, d), 1e-9), 0, 1))
                dd = float(np.linalg.norm(xy[a] + s * d))
                if dd < dc:
                    tc, dc = t[a] + s * (t[b] - t[a]), dd
        win = (np.abs(t - tc) <= window_s) & np.isfinite(speed)
        sp = speed[win] if win.any() else speed[[k]]
        a, b = max(k - 1, 0), min(k + 1, len(t) - 1)
        dx, dy = xy[b] - xy[a]
        bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        passes.append(Pass(len(passes), fixes[0].t + timedelta(seconds=float(tc)), round(dc, 1),
                           round(float(np.nanmean(sp)) * MPH_PER_MPS, 2),
                           round(float(np.nanstd(sp)) * MPH_PER_MPS, 2),
                           "doppler" if has_doppler else "positions", round(bearing, 1),
                           bearing_to_direction(bearing), int(win.sum())))
        i = j + 1
    return passes


# ------------------------------------------------------------- matching tracks
def candidates(con: sqlite3.Connection, camera_id: str, p: Pass, window_s: float = 45.0) -> list[dict[str, Any]]:
    """Accepted vehicles whose (approximate) wall-clock time is within ``window_s`` of the pass."""
    lo = (p.t_utc - timedelta(seconds=window_s)).isoformat(timespec="seconds")
    hi = (p.t_utc + timedelta(seconds=window_s)).isoformat(timespec="seconds")
    rows = con.execute(
        "SELECT v.run_id, v.track_id, v.cls, v.ts_utc, v.speed_mph, v.sigma_mph, v.direction, v.accepted, "
        "v.reject_reason FROM vehicles v WHERE v.camera_id = ? AND v.ts_utc BETWEEN ? AND ? ORDER BY v.ts_utc",
        (camera_id, lo, hi)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["offset_s"] = round((datetime.fromisoformat(r["ts_utc"]) - p.t_utc).total_seconds(), 1)
        out.append(d)
    return out


def report(con: sqlite3.Connection, passes: list[Pass], matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Error of our estimate against GPS for each hand-made match, plus bias and spread."""
    by_idx = {p.index: p for p in passes}
    rows = []
    for m in matches:
        p = by_idx.get(int(m["pass"]))
        if p is None:
            raise KeyError(f"pass {m['pass']} not found (have {sorted(by_idx)})")
        v = con.execute("SELECT * FROM vehicles WHERE run_id = ? AND track_id = ?",
                        (int(m["run_id"]), int(m["track_id"]))).fetchone()
        if v is None:
            raise KeyError(f"run {m['run_id']} track {m['track_id']} not in the database")
        row = {"pass": p.index, "pass_utc": p.t_utc.isoformat(timespec="seconds"), "direction": p.direction,
               "gps_mph": p.speed_mph, "gps_sd_mph": p.speed_sd_mph, "run_id": v["run_id"], "track_id": v["track_id"],
               "accepted": bool(v["accepted"]), "est_mph": v["speed_mph"], "sigma_mph": v["sigma_mph"],
               "offset_s": round((datetime.fromisoformat(v["ts_utc"]) - p.t_utc).total_seconds(), 1)
               if v["ts_utc"] else None, "note": m.get("note", "")}
        if v["accepted"] and v["speed_mph"] is not None:
            row["error_mph"] = round(v["speed_mph"] - p.speed_mph, 2)
            row["within_2sigma"] = abs(row["error_mph"]) <= 2 * (v["sigma_mph"] or 0)
        rows.append(row)
    err = np.array([r["error_mph"] for r in rows if "error_mph" in r])
    summary: dict[str, Any] = {"matches": len(rows), "with_estimate": int(len(err))}
    if len(err):
        summary.update(bias_mph=round(float(err.mean()), 2),
                       sd_mph=round(float(err.std(ddof=1)), 2) if len(err) > 1 else None,
                       rmse_mph=round(float(np.sqrt(np.mean(err ** 2))), 2),
                       max_abs_mph=round(float(np.abs(err).max()), 2),
                       share_within_2sigma=round(float(np.mean([r["within_2sigma"] for r in rows if "error_mph" in r])), 2),
                       bias_se_mph=round(float(err.std(ddof=1) / np.sqrt(len(err))), 2) if len(err) > 1 else None)
    offs = [r["offset_s"] for r in rows if r["offset_s"] is not None]
    if offs:
        summary["median_clock_offset_s"] = float(np.median(offs))  # ~ HLS latency + logging clock error
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------- CLI
def _camera_pos(camera_id: str) -> tuple[float, float]:
    from cctv.registry import get_camera

    c = get_camera(camera_id)
    return float(c["lat"]), float(c["lon"])


def main(argv: list[str] | None = None) -> None:
    from cctv import store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("passes", "candidates", "report"):
        p = sub.add_parser(name)
        p.add_argument("--gps", type=Path, required=True, help="GPX or CSV log")
        p.add_argument("--camera", required=True)
        p.add_argument("--radius-m", type=float, default=250.0)
        p.add_argument("--db", type=Path, default=store.DEFAULT_DB)
        if name == "candidates":
            p.add_argument("--window-s", type=float, default=45.0)
        if name == "report":
            p.add_argument("--matches", type=Path, required=True)
            p.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    fixes = load_log(args.gps)
    lat, lon = _camera_pos(args.camera)
    passes = find_passes(fixes, lat, lon, args.radius_m)
    print(f"{len(fixes)} fixes {fixes[0].t:%Y-%m-%d %H:%M:%S}-{fixes[-1].t:%H:%M:%S} UTC; "
          f"{len(passes)} pass(es) within {args.radius_m:.0f} m of {args.camera}")
    if args.cmd == "passes":
        for p in passes:
            print(f"  #{p.index} {p.t_utc:%H:%M:%S.%f}"[:-3] + f" UTC  {p.direction}  {p.speed_mph:5.1f} ± "
                  f"{p.speed_sd_mph:.1f} mph ({p.speed_source})  closest {p.distance_m:.0f} m")
        return
    con = store.connect(args.db)
    if args.cmd == "candidates":
        for p in passes:
            print(f"#{p.index} {p.t_utc:%H:%M:%S} UTC {p.direction} GPS {p.speed_mph:.1f} mph:")
            for c in candidates(con, args.camera, p, args.window_s):
                spd = f"{c['speed_mph']:.1f} ± {c['sigma_mph']:.1f}" if c["accepted"] else f"({c['reject_reason']})"
                print(f"    run {c['run_id']} track {c['track_id']:4d} {c['cls']:10s} dir {c['direction']} "
                      f"offset {c['offset_s']:+6.1f} s  {spd}")
        return
    rep = report(con, passes, json.loads(args.matches.read_text()))
    print("| pass | dir | GPS mph | estimate ± σ | error | offset s |\n|---|---|---|---|---|---|")
    for r in rep["rows"]:
        est = f"{r['est_mph']:.1f} ± {r['sigma_mph']:.1f}" if "error_mph" in r else "no estimate"
        err = f"{r['error_mph']:+.1f}" if "error_mph" in r else "–"
        print(f"| {r['pass']} | {r['direction']} | {r['gps_mph']:.1f} | {est} | {err} | {r['offset_s']} |")
    print(json.dumps(rep["summary"], indent=1))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
