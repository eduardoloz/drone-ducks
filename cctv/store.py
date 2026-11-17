"""SQLite event store (MVP). Postgres/TimescaleDB later; same tables.

Tables
  cameras       copy of the manifest rows we care about (id, name, lat/lon, limit, ...)
  runs          one processed clip (or one live session) per row
  vehicles      one anonymous tracked vehicle per row, with its speed estimate
  track_points  per-frame positions for overlays/debugging (short retention)
View
  speed_events  accepted vehicles whose (speed - margin) exceeded the posted limit

No plates, no images of vehicles, no identities: a vehicle is (run, track id).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from cctv.registry import DATA_DIR

DEFAULT_DB = DATA_DIR / "db" / "cctv.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY, name TEXT, owner TEXT, lat REAL, lon REAL, county TEXT, kind TEXT,
  exit TEXT, chainage_km REAL, feed_type TEXT, feed_url TEXT, posted_limit_mph REAL,
  speed_suitability TEXT, measured_json TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  camera_id TEXT NOT NULL REFERENCES cameras(id),
  source TEXT, processed_utc TEXT, clip_start_utc TEXT, clip_start_note TEXT,
  duration_s REAL, frames INTEGER, fps REAL, model TEXT, tracker TEXT, ms_per_frame REAL,
  calibration_utc TEXT, calibration_rmse_m REAL, scale_uncertainty REAL,
  camera_moved INTEGER, motion_json TEXT, reference_frame TEXT, notes TEXT,
  view_id TEXT,                -- PTZ preset the clip matched (NULL: unknown view / not checked)
  sun_elev_deg REAL,           -- sun elevation at the camera at clip start (daylight.py)
  light TEXT,                  -- 'day' | 'twilight' | 'night' from sun_elev_deg
  luma REAL,                   -- mean brightness (0-255) of the clip's median frame
  view_shift_px REAL           -- small PTZ offset compensated against the view's reference (views.py)
);
CREATE TABLE IF NOT EXISTS vehicles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  camera_id TEXT NOT NULL, track_id INTEGER, cls TEXT,
  ts_utc TEXT,                 -- approx wall-clock time at the middle of the track
  t_start REAL, t_end REAL, n_points INTEGER, n_inliers INTEGER, distance_m REAL,
  speed_mph REAL, sigma_mph REAL, sigma_stat_mph REAL, sigma_cal_mph REAL,
  direction INTEGER, heading_deg REAL, accepted INTEGER, reject_reason TEXT,
  limit_mph REAL, margin_mph REAL, over_limit INTEGER
);
CREATE INDEX IF NOT EXISTS vehicles_cam_ts ON vehicles(camera_id, ts_utc);
CREATE TABLE IF NOT EXISTS track_points (
  vehicle_id INTEGER NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
  t REAL, u REAL, v REAL, x REAL, y REAL, in_roi INTEGER
);
CREATE INDEX IF NOT EXISTS track_points_vehicle ON track_points(vehicle_id);
CREATE VIEW IF NOT EXISTS speed_events AS
  SELECT v.*, c.name AS camera_name FROM vehicles v JOIN cameras c ON c.id = v.camera_id
  WHERE v.accepted = 1 AND v.over_limit = 1;
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    _migrate(con)
    return con


# Columns added after round 1; old databases get them via ALTER TABLE.
_ADDED_COLUMNS = {"runs": [("view_id", "TEXT"), ("sun_elev_deg", "REAL"), ("light", "TEXT"), ("luma", "REAL"),
                           ("view_shift_px", "REAL")]}


def _migrate(con: sqlite3.Connection) -> None:
    for table, cols in _ADDED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, typ in cols:
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
    con.commit()


def upsert_cameras(con: sqlite3.Connection, cameras: Iterable[dict[str, Any]]) -> int:
    rows = [(c["id"], c["name"], c.get("owner"), c["lat"], c["lon"], c.get("county"), c.get("kind"),
             c.get("exit"), c.get("chainage_km"), c["feed"]["type"], c["feed"]["url"],
             c.get("posted_limit_mph"), c.get("speed_suitability"),
             json.dumps(c.get("measured")) if c.get("measured") else None) for c in cameras]
    con.executemany(
        """INSERT INTO cameras VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET name=excluded.name, owner=excluded.owner, lat=excluded.lat,
             lon=excluded.lon, county=excluded.county, kind=excluded.kind, exit=excluded.exit,
             chainage_km=excluded.chainage_km, feed_type=excluded.feed_type, feed_url=excluded.feed_url,
             posted_limit_mph=excluded.posted_limit_mph, speed_suitability=excluded.speed_suitability,
             measured_json=excluded.measured_json""", rows)
    con.commit()
    return len(rows)


def insert_run(con: sqlite3.Connection, **fields: Any) -> int:
    cols = ", ".join(fields)
    qs = ", ".join("?" for _ in fields)
    cur = con.execute(f"INSERT INTO runs ({cols}) VALUES ({qs})", tuple(fields.values()))
    con.commit()
    return int(cur.lastrowid)


def insert_vehicle(con: sqlite3.Connection, points: list[tuple] | None = None, **fields: Any) -> int:
    cols = ", ".join(fields)
    qs = ", ".join("?" for _ in fields)
    cur = con.execute(f"INSERT INTO vehicles ({cols}) VALUES ({qs})", tuple(fields.values()))
    vid = int(cur.lastrowid)
    if points:
        con.executemany("INSERT INTO track_points VALUES (?,?,?,?,?,?,?)",
                        [(vid, *p) for p in points])
    return vid


def delete_runs_for_source(con: sqlite3.Connection, camera_id: str, source: str) -> int:
    """Re-processing the same clip replaces its previous results."""
    cur = con.execute("DELETE FROM runs WHERE camera_id = ? AND source = ?", (camera_id, source))
    con.commit()
    return cur.rowcount


def purge_track_points_older_than(con: sqlite3.Connection, days: int) -> int:
    """Retention: per-frame positions are only kept for a short time."""
    cur = con.execute(
        """DELETE FROM track_points WHERE vehicle_id IN (
             SELECT v.id FROM vehicles v WHERE julianday(v.ts_utc) < julianday('now', ?))""", (f"-{days} days",))
    con.commit()
    return cur.rowcount
