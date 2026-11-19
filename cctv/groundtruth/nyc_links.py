"""Compare our per-vehicle speed estimates with NYC DOT's published link speeds.

Source: NYC Open Data "DOT Traffic Speeds" (Socrata dataset ``i4gi-tjb9``), a
public open-data API. Each row is one *link* (a stretch of road a mile or two
long) with an average ``speed`` (mph) and ``travel_time`` (s), stamped
``data_as_of``, refreshed about every 5 minutes. The camera *video* of NYC DOT
needs a signed agreement and is not used; this open dataset is fine.

What can and cannot be compared (read before quoting a number):

* A link speed is a **space-mean speed over the whole link** (link length /
  travel time), averaged over minutes, mostly from probe vehicles / toll-tag
  readers. Our estimate is a **spot speed** at one camera over ~60 s.
* The fair comparison for a set of spot speeds is their **harmonic mean**
  (the space-mean speed of those vehicles if conditions were uniform), not the
  median. We report both.
* In congestion, speed varies along a link (queues), so a camera's spot
  speeds can differ from the link average by a lot and both can be right.
  Free-flow periods are where agreement is a meaningful test.
* ``data_as_of`` is a floating timestamp with no zone. We assume New York
  local time (``--data-tz``); check it against the live site if in doubt.
* Link quality varies a lot (measured 2026-11-12, 02:00-05:00, light traffic): the
  WB Little Neck Pkwy link read a plausible 54.7 mph median, but EB 84th-108th St
  read 20.8 mph (overnight work zone, or a bad link?) and WB 108th-84th St rows all
  had status -101 with 0-114 mph. Rows with status other than "0" are skipped
  by default; run ``quality`` before trusting a link.
* The API is sometimes stale (response header ``X-SODA2-Data-Out-Of-Date``).
  Rows further than ``--max-gap-min`` from the clip are not used.

Commands (run from the repo root)::

    python -m cctv.groundtruth.nyc_links links                 # LIE links + nearby cameras (writes a snapshot)
    python -m cctv.groundtruth.nyc_links fetch --start 2026-11-13T12:00 --end 2026-11-13T14:00
    python -m cctv.groundtruth.nyc_links quality               # is each link plausible (night free flow, status)?
    python -m cctv.groundtruth.nyc_links compare               # our runs vs link speeds -> table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np

from cctv.registry import DATA_DIR, load_cameras

DATASET = "i4gi-tjb9"
ENDPOINT = f"https://data.cityofnewyork.us/resource/{DATASET}.json"
USER_AGENT = "ProjectQuack-cctv-research/0.2 (open-data client)"   # generic on purpose: no personal info
GT_DIR = DATA_DIR / "groundtruth"
LINKS_SNAPSHOT = GT_DIR / "nyc_lie_links.json"        # small, committed: link ids, names, geometry
CACHE_DIR = GT_DIR / "cache"                          # fetched speed rows (gitignored)
REFERENCE_DIR = GT_DIR / "reference"                  # small, committed: link rows for windows we have clips of
LAG_LOG = GT_DIR / "feed_lag.json"                    # small, committed: measured publication lag of the API
LIE_NAME_PATTERNS = ("LIE ",)                         # link names look like "LIE E 84TH ST - 108TH ST"
NY_TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- Socrata client
def soql_get_with_headers(params: dict[str, str], timeout: float = 60.0
                          ) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """One GET against the dataset -> (rows, response headers). Optional app token from $SOCRATA_APP_TOKEN."""
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_headers = {k: v for k, v in dict(resp.headers).items()}
        rows = json.loads(resp.read().decode())
    if resp_headers.get("X-SODA2-Data-Out-Of-Date") == "true":
        print("note: API says its copy is out of date (X-SODA2-Data-Out-Of-Date: true)")
    return rows, resp_headers


def soql_get(params: dict[str, str], timeout: float = 60.0) -> list[dict[str, Any]]:
    """One GET against the dataset. Optional app token from $SOCRATA_APP_TOKEN (raises rate limits)."""
    return soql_get_with_headers(params, timeout)[0]


def _floating(dt: datetime, data_tz: ZoneInfo) -> str:
    """Datetime -> Socrata floating timestamp string in the dataset's zone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=data_tz)
    return dt.astimezone(data_tz).strftime("%Y-%m-%dT%H:%M:%S")


def fetch_link_rows(link_ids: Iterable[str], start: datetime, end: datetime,
                    data_tz: ZoneInfo = NY_TZ, limit: int = 50000) -> list[dict[str, Any]]:
    """Speed rows for some links in [start, end]. Always filtered by link and time (the table is huge)."""
    ids = ",".join(f"'{int(i)}'" for i in link_ids)   # int() guards the query against injection
    where = (f"link_id in ({ids}) AND data_as_of >= '{_floating(start, data_tz)}' "
             f"AND data_as_of <= '{_floating(end, data_tz)}'")
    return soql_get({"$select": "link_id,link_name,speed,travel_time,status,data_as_of",
                     "$where": where, "$order": "data_as_of", "$limit": str(limit)})


def fetch_lie_links(since: datetime, data_tz: ZoneInfo = NY_TZ) -> list[dict[str, Any]]:
    """Current LIE links (one row per link) with geometry, from rows newer than ``since``."""
    name_cond = " OR ".join(f"starts_with(upper(link_name), '{p}')" for p in LIE_NAME_PATTERNS)
    rows = soql_get({"$select": "link_id,link_name,borough,link_points,max(data_as_of) as last_seen",
                     "$where": f"data_as_of >= '{_floating(since, data_tz)}' AND ({name_cond})",
                     "$group": "link_id,link_name,borough,link_points", "$limit": "500"})
    return rows


def measure_lag(link_ids: Iterable[str], now: datetime | None = None, lookback_h: float = 12.0,
                data_tz: ZoneInfo = NY_TZ, get=soql_get_with_headers) -> dict[str, Any]:
    """How far behind real time the open-data API is: newest row per link vs now.

    Measured 2026-11-17: the newest Little Neck row was ~1 h 50 min old, and the dataset's
    ``Last-Modified`` header showed it is refreshed in batches (rows up to ~5 min before the batch).
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ids = ",".join(f"'{int(i)}'" for i in link_ids)
    since = _floating(now - timedelta(hours=lookback_h), data_tz)
    rows, headers = get({"$select": "link_id,max(data_as_of) as newest,count(*) as n",
                         "$where": f"link_id in ({ids}) AND data_as_of >= '{since}'", "$group": "link_id"})
    out: dict[str, Any] = {"measured_utc": now.isoformat(timespec="seconds"), "links": {},
                           "out_of_date_header": headers.get("X-SODA2-Data-Out-Of-Date")}
    lm = headers.get("Last-Modified") or headers.get("X-SODA2-Truth-Last-Modified")
    if lm:
        from email.utils import parsedate_to_datetime

        lm_dt = parsedate_to_datetime(lm).astimezone(timezone.utc)
        out["dataset_last_modified_utc"] = lm_dt.isoformat(timespec="seconds")
        out["minutes_since_last_modified"] = round((now - lm_dt).total_seconds() / 60, 1)
    for r in rows:
        newest = parse_data_as_of(r["newest"], data_tz)
        out["links"][str(r["link_id"])] = {"newest_as_of_utc": newest.isoformat(timespec="seconds"),
                                           "lag_min": round((now - newest).total_seconds() / 60, 1),
                                           f"rows_last_{lookback_h:g}h": int(r.get("n") or 0)}
    for i in link_ids:
        out["links"].setdefault(str(int(i)), {"newest_as_of_utc": None, "lag_min": None,
                                              "note": f"no row in the last {lookback_h:g} h"})
    return out


def load_link_rows(dirs: Iterable[Path] = (CACHE_DIR, REFERENCE_DIR)) -> list[dict[str, Any]]:
    """All cached + committed reference link rows, de-duplicated by (link_id, data_as_of)."""
    seen, out = set(), []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")) if Path(d).exists() else []:
            for r in json.loads(f.read_text()).get("rows", []):
                k = (str(r.get("link_id")), r.get("data_as_of"))
                if k not in seen:
                    seen.add(k)
                    out.append(r)
    return out


def clips_in_window(camera_ids: Iterable[str], start_utc: datetime, end_utc: datetime,
                    clips_dir: Path = DATA_DIR / "clips") -> list[dict[str, Any]]:
    """Recorded clips (sidecar JSON with ``started_utc`` and a non-empty .ts) of these cameras in a window."""
    cams, out = set(camera_ids), []
    for side in sorted(clips_dir.glob("*.json")) if clips_dir.exists() else []:
        try:
            d = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        ts = side.with_suffix(".ts")
        if d.get("camera_id") not in cams or not d.get("started_utc") or not ts.exists() or ts.stat().st_size == 0:
            continue
        t = datetime.fromisoformat(d["started_utc"]).astimezone(timezone.utc)
        if start_utc <= t <= end_utc:
            out.append({"camera_id": d["camera_id"], "clip": ts.name,
                        "started_utc": t.isoformat(timespec="seconds"), "returncode": d.get("returncode")})
    return out


def reference_record(link_id: str, link_name: str | None, rows: list[dict[str, Any]], start: str, end: str,
                     clips: list[dict[str, Any]], fetched_utc: str, lag: dict[str, Any] | None = None,
                     note: str = "") -> dict[str, Any]:
    """The committed reference file: link rows for a window we have (git-ignored) clips of, plus those clips."""
    keep = ("link_id", "link_name", "speed", "travel_time", "status", "data_as_of")
    return {"dataset": DATASET, "source": "NYC Open Data, NYC DOT Traffic Speeds (public open data)",
            "fetched_utc": fetched_utc, "window_local": [start, end], "data_tz": "America/New_York (assumed)",
            "link_id": str(link_id), "link_name": link_name,
            "rows": [{k: r.get(k) for k in keep} for r in sorted(rows, key=lambda r: r["data_as_of"])],
            "clips": clips, "lag_at_fetch": lag, "note": note}


# ------------------------------------------------------------------ geometry
def parse_link_points(s: str) -> np.ndarray:
    """'lat,lon lat,lon ...' -> (N, 2) array of (lat, lon). Skips truncated or malformed pairs."""
    out = []
    for tok in (s or "").split():
        parts = tok.split(",")
        if len(parts) != 2:
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if 39 < lat < 42 and -75 < lon < -72:   # NYC area sanity check (drops truncated numbers like "40.7")
            out.append((lat, lon))
    return np.asarray(out, float).reshape(-1, 2)


def _to_xy(latlon: np.ndarray, lat0: float) -> np.ndarray:
    """Local equirectangular metres (fine over a few km)."""
    r = 6371000.0
    return np.c_[np.radians(latlon[:, 1]) * r * math.cos(math.radians(lat0)), np.radians(latlon[:, 0]) * r]


def distance_to_polyline_m(lat: float, lon: float, poly: np.ndarray) -> tuple[float, float]:
    """(distance in metres from a point to a polyline, fraction along the polyline of the closest point)."""
    if len(poly) == 0:
        return float("inf"), float("nan")
    p = _to_xy(np.array([[lat, lon]]), lat)[0]
    xy = _to_xy(poly, lat)
    if len(xy) == 1:
        return float(np.linalg.norm(xy[0] - p)), 0.0
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg_len)]
    best = (float("inf"), 0.0)
    for i, (a, b) in enumerate(zip(xy[:-1], xy[1:])):
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
        d = float(np.linalg.norm(a + t * ab - p))
        if d < best[0]:
            best = (d, (cum[i] + t * seg_len[i]) / max(cum[-1], 1e-9))
    return best


def link_direction(link_name: str) -> str | None:
    """'LIE E 84TH ST - 108TH ST' -> 'EB'; 'LIE WB LITTLE NECK ...' -> 'WB'."""
    toks = link_name.upper().replace("-", " ").split()
    if len(toks) >= 2 and toks[0] == "LIE":
        return {"E": "EB", "EB": "EB", "W": "WB", "WB": "WB"}.get(toks[1])
    return None


@dataclass
class CameraLink:
    camera_id: str
    link_id: str
    link_name: str
    direction: str | None
    distance_m: float
    fraction_along: float


def cameras_near_links(links: list[dict[str, Any]], cameras: list[dict[str, Any]],
                       max_m: float = 150.0) -> list[CameraLink]:
    """Video cameras that sit on (within ``max_m`` of) a link's polyline."""
    out = []
    for link in links:
        poly = parse_link_points(link.get("link_points", ""))
        for c in cameras:
            if c.get("feed", {}).get("type") != "hls":
                continue        # NYC DOT stills: not used (agreement needed)
            d, f = distance_to_polyline_m(c["lat"], c["lon"], poly)
            if d <= max_m:
                out.append(CameraLink(c["id"], str(link["link_id"]), link["link_name"],
                                      link_direction(link["link_name"]), round(d, 1), round(f, 3)))
    return sorted(out, key=lambda x: (x.link_id, x.fraction_along))


# ---------------------------------------------------------------- comparison
def harmonic_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(len(x) / np.sum(1.0 / x)) if len(x) and np.all(x > 0) else float("nan")


def bootstrap_ci(x: np.ndarray, fn, n: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def parse_data_as_of(s: str, data_tz: ZoneInfo = NY_TZ) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", ""))
    return (dt if dt.tzinfo else dt.replace(tzinfo=data_tz)).astimezone(timezone.utc)


OK_STATUS = ("0",)   # observed: status "-101" rows carry 0-114 mph noise at 3 am; meaning undocumented


def link_speed_near(rows: list[dict[str, Any]], link_id: str, t0: datetime, t1: datetime,
                    max_gap_min: float = 10.0, data_tz: ZoneInfo = NY_TZ,
                    ok_status: tuple[str, ...] | None = OK_STATUS) -> dict[str, Any] | None:
    """Link speed for a clip spanning [t0, t1] (UTC): rows inside the clip window +- max_gap_min.

    Returns the mean of those rows, the row nearest the clip midpoint, and how far
    that row is from the midpoint. None when no usable row is close enough. Rows
    whose ``status`` is not in ``ok_status`` are skipped and counted.
    """
    mid = t0 + (t1 - t0) / 2
    lo, hi = t0 - timedelta(minutes=max_gap_min), t1 + timedelta(minutes=max_gap_min)
    sel, skipped = [], 0
    for r in rows:
        if str(r.get("link_id")) != str(link_id) or r.get("speed") in (None, ""):
            continue
        ts = parse_data_as_of(r["data_as_of"], data_tz)
        if not lo <= ts <= hi:
            continue
        if ok_status is not None and str(r.get("status")) not in ok_status:
            skipped += 1
            continue
        sel.append((ts, float(r["speed"]), r.get("status")))
    if not sel:
        return None
    nearest = min(sel, key=lambda s: abs((s[0] - mid).total_seconds()))
    return {"n_rows": len(sel), "mean_mph": float(np.mean([s[1] for s in sel])),
            "min_mph": float(min(s[1] for s in sel)), "max_mph": float(max(s[1] for s in sel)),
            "nearest_mph": nearest[1], "nearest_status": nearest[2],
            "nearest_offset_s": round((nearest[0] - mid).total_seconds()),
            "nearest_as_of_utc": nearest[0].isoformat(timespec="seconds"), "skipped_bad_status": skipped}


def link_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-link sanity summary of fetched rows: is this link usable as ground truth at all?"""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(f"{r['link_id']} {r.get('link_name', '')}".strip(), []).append(r)
    out = []
    for name, rs in sorted(by.items()):
        sp = np.array([float(r["speed"]) for r in rs if r.get("speed") not in (None, "")])
        status: dict[str, int] = {}
        for r in rs:
            status[str(r.get("status"))] = status.get(str(r.get("status")), 0) + 1
        out.append({"link": name, "rows": len(rs), "status_counts": status,
                    "median_mph": round(float(np.median(sp)), 1) if len(sp) else None,
                    "p10_mph": round(float(np.percentile(sp, 10)), 1) if len(sp) else None,
                    "p90_mph": round(float(np.percentile(sp, 90)), 1) if len(sp) else None,
                    "implausible_rows": int(np.sum((sp <= 0) | (sp > 90)))})
    return out


def our_runs(con: sqlite3.Connection, camera_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Runs with accepted speeds for these cameras, with their vehicles' speeds grouped by direction."""
    ids = list(camera_ids)
    if not ids:
        return []
    q = ",".join("?" for _ in ids)
    runs = con.execute(f"SELECT * FROM runs WHERE camera_id IN ({q}) AND clip_start_utc IS NOT NULL "
                       "AND COALESCE(camera_moved, 0) = 0 ORDER BY clip_start_utc", ids).fetchall()
    out = []
    for r in runs:
        rows = con.execute("SELECT speed_mph, direction FROM vehicles WHERE run_id = ? AND accepted = 1 "
                           "AND speed_mph IS NOT NULL", (r["id"],)).fetchall()
        if not rows:
            continue
        by_dir: dict[int, list[float]] = {}
        for v in rows:
            by_dir.setdefault(int(v["direction"] or 0), []).append(float(v["speed_mph"]))
        out.append({"run": dict(r), "by_direction": by_dir})
    return out


def direction_label(camera_id: str, view_id: str | None, sign: int) -> str | None:
    """Map our road-frame travel sign (+1/-1) to EB/WB using the view's calibration ``direction_labels``."""
    from cctv.views import get_view, primary_view

    try:
        v = get_view(camera_id, view_id) if view_id else primary_view(camera_id)
    except KeyError:
        return None
    cal = v.load_calibration() if v else None
    labels = (cal.direction_labels if cal else None) or {}
    return labels.get("+" if sign > 0 else "-")


def compare(runs: list[dict[str, Any]], cam_links: list[CameraLink], link_rows: list[dict[str, Any]],
            max_gap_min: float = 10.0, data_tz: ZoneInfo = NY_TZ, label_fn=direction_label) -> list[dict[str, Any]]:
    """One table row per (run, direction, matching link)."""
    by_cam: dict[str, list[CameraLink]] = {}
    for cl in cam_links:
        by_cam.setdefault(cl.camera_id, []).append(cl)
    table = []
    for item in runs:
        r = item["run"]
        t0 = datetime.fromisoformat(r["clip_start_utc"]).astimezone(timezone.utc)
        t1 = t0 + timedelta(seconds=float(r.get("duration_s") or 60))
        for sign, speeds in sorted(item["by_direction"].items()):
            lab = label_fn(r["camera_id"], r.get("view_id"), sign)
            for cl in by_cam.get(r["camera_id"], []):
                if lab is None or cl.direction is None or lab != cl.direction:
                    continue        # unknown or opposite direction: never compare across carriageways
                s = np.asarray(speeds, float)
                link = link_speed_near(link_rows, cl.link_id, t0, t1, max_gap_min, data_tz)
                hm = harmonic_mean(s)
                lo, hi = bootstrap_ci(s, harmonic_mean)
                row = {"camera_id": r["camera_id"], "run_id": r["id"], "view_id": r.get("view_id"),
                       "clip_start_utc": t0.isoformat(timespec="seconds"), "direction": lab, "n": int(len(s)),
                       "median_mph": round(float(np.median(s)), 1), "hmean_mph": round(hm, 1),
                       "hmean_ci95": [round(lo, 1), round(hi, 1)], "link_id": cl.link_id,
                       "link_name": cl.link_name, "link": link}
                if link:
                    row["delta_hmean_minus_link_mph"] = round(hm - link["mean_mph"], 1)
                table.append(row)
    return table


def format_table(table: list[dict[str, Any]]) -> str:
    hdr = ("| Camera | Clip start (local) | Dir | n | Our median | Our harmonic mean [95% CI] | Link (mean of rows) "
           "| Link rows (nearest offset) | Δ (hmean − link) |\n|---|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for r in table:
        local = datetime.fromisoformat(r["clip_start_utc"]).astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M")
        link = r["link"]
        lk = f"{link['mean_mph']:.1f} mph ({r['link_name']})" if link else f"no row within window ({r['link_name']})"
        rows = f"{link['n_rows']} ({link['nearest_offset_s']:+d} s)" if link else "–"
        d = f"{r['delta_hmean_minus_link_mph']:+.1f}" if link else "–"
        ci = r["hmean_ci95"]
        lines.append(f"| {r['camera_id']} | {local} | {r['direction']} | {r['n']} | {r['median_mph']:.1f} | "
                     f"{r['hmean_mph']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | {lk} | {rows} | {d} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI
def _load_links_snapshot(path: Path = LINKS_SNAPSHOT) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"no link snapshot at {path}: run `python -m cctv.groundtruth.nyc_links links` first")
    return json.loads(path.read_text())["links"]


def cmd_links(args) -> None:
    since = datetime.now(NY_TZ) - timedelta(hours=args.hours)
    links = fetch_lie_links(since)
    cams = load_cameras()
    cl = cameras_near_links(links, cams, args.max_m)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    snap = {"dataset": DATASET, "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "LIE links seen in the last hours; geometry for camera matching. Source: NYC Open Data, NYC DOT.",
            "links": sorted(links, key=lambda l: l["link_name"]),
            "camera_links": [cl_.__dict__ for cl_ in cl]}
    LINKS_SNAPSHOT.write_text(json.dumps(snap, indent=1) + "\n")
    for l in snap["links"]:
        print(f"{l['link_id']:>8s}  {l['link_name']:40s} last row {l['last_seen']}")
    print("\nNYSDOT video cameras on these links:")
    for c in cl:
        print(f"  {c.camera_id:8s} on {c.link_id} {c.link_name:34s} {c.distance_m:6.1f} m off, "
              f"{c.fraction_along:.0%} along")
    print(f"wrote {LINKS_SNAPSHOT}")


def cmd_fetch(args) -> None:
    snap = json.loads(LINKS_SNAPSHOT.read_text()) if LINKS_SNAPSHOT.exists() else None
    ids = args.link or sorted({c["link_id"] for c in (snap or {}).get("camera_links", [])})
    if not ids:
        raise SystemExit("no links: pass --link or run `links` first")
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    rows = fetch_link_rows(ids, start, end)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"speeds_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.write_text(json.dumps({"fetched_utc": fetched,
                               "start": args.start, "end": args.end, "links": ids, "rows": rows}, indent=1))
    print(f"{len(rows)} rows for links {ids} -> {out}")
    if args.keep:   # small committed copy + the clips it is for, so `compare` works on a fresh clone
        cams_by_link: dict[str, list[str]] = {}
        names: dict[str, str] = {}
        for c in (snap or {}).get("camera_links", []):
            cams_by_link.setdefault(str(c["link_id"]), []).append(c["camera_id"])
            names[str(c["link_id"])] = c["link_name"]
        s_utc = start.replace(tzinfo=start.tzinfo or NY_TZ).astimezone(timezone.utc)
        e_utc = end.replace(tzinfo=end.tzinfo or NY_TZ).astimezone(timezone.utc)
        lag = measure_lag(ids)
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        for lid in ids:
            lrows = [r for r in rows if str(r.get("link_id")) == str(lid)]
            clips = clips_in_window(cams_by_link.get(str(lid), []), s_utc, e_utc)
            rec = reference_record(str(lid), names.get(str(lid)) or (lrows[0].get("link_name") if lrows else None),
                                   lrows, args.start, args.end, clips, fetched,
                                   {"links": {str(lid): lag["links"].get(str(lid))},
                                    "measured_utc": lag["measured_utc"]}, args.note or "")
            ref = REFERENCE_DIR / f"link{lid}_{start:%Y%m%dT%H%M}_{end:%H%M}.json"
            ref.write_text(json.dumps(rec, indent=1) + "\n")
            print(f"kept {len(lrows)} rows + {len(clips)} clip(s) in window -> {ref}")


def cmd_lag(args) -> None:
    snap = json.loads(LINKS_SNAPSHOT.read_text()) if LINKS_SNAPSHOT.exists() else None
    ids = args.link or sorted({c["link_id"] for c in (snap or {}).get("camera_links", [])})
    lag = measure_lag(ids)
    for lid, v in lag["links"].items():
        local = (datetime.fromisoformat(v["newest_as_of_utc"]).astimezone(NY_TZ).strftime("%H:%M")
                 if v.get("newest_as_of_utc") else "none")
        print(f"link {lid}: newest row {local} local, {v.get('lag_min')} min behind")
    if "minutes_since_last_modified" in lag:
        print(f"dataset Last-Modified {lag['dataset_last_modified_utc']} ({lag['minutes_since_last_modified']} min ago); "
              f"out-of-date header: {lag['out_of_date_header']}")
    if args.record:
        log = json.loads(LAG_LOG.read_text()) if LAG_LOG.exists() else {
            "dataset": DATASET, "note": "Measured publication lag of the open-data API: newest data_as_of vs the "
            "time of the request (data_as_of assumed NY local). Written by `nyc_links lag --record`.",
            "measurements": []}
        log["measurements"].append(lag)
        LAG_LOG.write_text(json.dumps(log, indent=1) + "\n")
        print(f"appended to {LAG_LOG}")


def cmd_compare(args) -> None:
    from cctv import store

    snap = json.loads(LINKS_SNAPSHOT.read_text()) if LINKS_SNAPSHOT.exists() else None
    if not snap:
        raise SystemExit("run `links` first")
    cam_links = [CameraLink(**c) for c in snap["camera_links"]]
    con = store.connect(args.db)
    runs = our_runs(con, sorted({c.camera_id for c in cam_links}))
    if not runs:
        raise SystemExit("no runs with speeds for cameras on NYC DOT links yet (record + calibrate one first)")
    rows = load_link_rows()
    if args.fetch_missing:
        for item in runs:
            t0 = datetime.fromisoformat(item["run"]["clip_start_utc"])
            rows += fetch_link_rows({c.link_id for c in cam_links}, t0 - timedelta(minutes=30),
                                    t0 + timedelta(minutes=30))
    table = compare(runs, cam_links, rows, args.max_gap_min)
    print(format_table(table))
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
        print(f"wrote {args.json}")
    if not table:
        print("(no comparable rows: is direction_labels set in the calibration? are link rows near the clip times?)")


def cmd_quality(args) -> None:
    uniq = load_link_rows()      # de-duplicated across overlapping fetches and committed references
    if args.window:              # e.g. 20261112T0200 -> rows from 2026-11-12 02:00 onwards (local), for 3 h
        w0 = datetime.strptime(args.window, "%Y%m%dT%H%M")
        w1 = w0 + timedelta(hours=args.window_hours)
        uniq = [r for r in uniq if w0 <= datetime.fromisoformat(r["data_as_of"].replace("Z", "")) <= w1]
    for q in link_quality(uniq):
        print(f"{q['link']:44s} rows {q['rows']:4d}  median {q['median_mph']} mph (p10 {q['p10_mph']}, "
              f"p90 {q['p90_mph']})  status {q['status_counts']}  implausible {q['implausible_rows']}")


def main(argv: list[str] | None = None) -> None:
    from cctv import store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("links", help="find live LIE links and the NYSDOT cameras on them")
    a.add_argument("--hours", type=float, default=48, help="look for links with rows in the last N hours")
    a.add_argument("--max-m", type=float, default=150, help="camera must be within this many metres of the link")
    f = sub.add_parser("fetch", help="cache link speed rows for a time window (local NY time)")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--link", action="append", help="link id (repeatable); default: links with cameras on them")
    f.add_argument("--keep", action="store_true",
                   help="also write a small committed copy to data/groundtruth/reference/ with the clips in the window")
    f.add_argument("--note", help="free-text note stored in the --keep file")
    g = sub.add_parser("lag", help="how far behind real time is the API? (newest row per link vs now)")
    g.add_argument("--link", action="append", help="link id (repeatable); default: links with cameras on them")
    g.add_argument("--record", action="store_true", help=f"append the measurement to {LAG_LOG.name}")
    q = sub.add_parser("quality", help="sanity summary of cached link rows (is a link usable as ground truth?)")
    q.add_argument("--window", help="local start time, e.g. 20261112T0200 (with --window-hours)")
    q.add_argument("--window-hours", type=float, default=3.0)
    c = sub.add_parser("compare", help="our runs vs link speeds")
    c.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    c.add_argument("--max-gap-min", type=float, default=10.0)
    c.add_argument("--fetch-missing", action="store_true", help="also fetch +-30 min of link rows around each run")
    c.add_argument("--json", help="also write the table as JSON")
    args = ap.parse_args(argv)
    {"links": cmd_links, "fetch": cmd_fetch, "lag": cmd_lag, "quality": cmd_quality,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
