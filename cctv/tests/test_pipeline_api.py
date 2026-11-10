"""End-to-end without YOLO: synthetic tracks for a calibrated camera -> pipeline -> SQLite -> API."""

import json

import numpy as np
from fastapi.testclient import TestClient

from cctv.detect_track import TrackPoint, save_jsonl
from cctv.geometry import Calibration, MPH_PER_MPS, apply_homography
from cctv.pipeline import clip_start_utc, process
from cctv.registry import calibration_path
from cctv.web.app import create_app


def synthetic_tracks(cal, speeds_mph, fps=14.985):
    """One vehicle per speed, driving away from the camera down a lane centre."""
    pts = []
    Hi = np.linalg.inv(cal.H_np)
    for k, mph in enumerate(speeds_mph):
        v = mph / MPH_PER_MPS
        n = int(3.0 * fps)
        t = np.arange(n) / fps
        xy = np.c_[-5 + v * t, np.full(n, 1.83 + 3.66 * (k % 2))]
        uv = apply_homography(Hi, xy)
        for i, (u, vv) in enumerate(uv):
            w, h = 40.0, 30.0
            pts.append(TrackPoint(frame=i + 20 * k, t=float(t[i] + 2 * k), track_id=k + 1, cls="car", conf=0.8,
                                  x1=u - w / 2, y1=vv - h, x2=u + w / 2, y2=vv))
    return pts


def test_clip_start_from_filename(tmp_path):
    iso, note = clip_start_utc(tmp_path / "R10_041_20261110T213935Z.ts")
    assert iso == "2026-11-10T21:39:35+00:00" and "file name" in note


def test_pipeline_and_api_end_to_end(tmp_path):
    cal = Calibration.load(calibration_path("R10_041"))
    tracks = tmp_path / "tracks.jsonl"
    save_jsonl(synthetic_tracks(cal, [50.0, 70.0]), tracks, meta={"frames": 120, "model": "synthetic"})
    db = tmp_path / "t.sqlite"
    clip = tmp_path / "R10_041_20261110T213935Z.ts"   # does not exist: motion check is skipped
    s = process("R10_041", clip, db, tracks_path=tracks)
    assert s["vehicles"] == 2 and s["accepted"] == 2
    assert s["over_limit"] == 1                       # 70 mph flagged, 50 mph not (limit 55)
    assert abs(s["min_mph"] - 50) < 1.0 and abs(s["max_mph"] - 70) < 1.0

    client = TestClient(create_app(db_path=db))
    assert client.get("/api/health").json()["vehicles"] == 2
    cams = {c["id"]: c for c in client.get("/api/cameras").json()["cameras"]}
    assert cams["R10_041"]["calibrated"] and cams["R10_041"]["latest"]["with_speed"] == 2
    ev = client.get("/api/events?over_only=true").json()["events"]
    assert len(ev) == 1 and ev[0]["speed_mph"] > 65
    st = client.get("/api/stats?camera_id=R10_041").json()
    assert sum(b["count"] for b in st["histogram"]) == 2
    ov = client.get(f"/api/runs/{s['run_id']}/overlay").json()
    assert len(ov["tracks"]) == 2 and len(ov["tracks"][0]["points"][0]) == 3
    assert client.get("/api/cameras/NOPE").status_code == 404


def test_calibration_fit_endpoint_matches_committed_points(tmp_path):
    client = TestClient(create_app(db_path=tmp_path / "e.sqlite"))
    spec = client.get("/api/cameras/R10_041/calibration").json()["points"]
    r = client.post("/api/calibrate/fit", json=spec)
    assert r.status_code == 200
    out = r.json()
    assert out["rmse_m"] < 0.15 and len(out["errors_m"]) == len(spec["points"])
    assert client.post("/api/calibrate/fit", json={"points": []}).status_code == 422


def test_moved_camera_run_does_not_hide_last_valid_run(tmp_path):
    from cctv import store
    cal = Calibration.load(calibration_path("R10_041"))
    tracks = tmp_path / "tracks.jsonl"
    save_jsonl(synthetic_tracks(cal, [60.0]), tracks, meta={"frames": 60})
    db = tmp_path / "m.sqlite"
    first = process("R10_041", tmp_path / "R10_041_20261110T213935Z.ts", db, tracks_path=tracks)
    con = store.connect(db)
    moved = store.insert_run(con, camera_id="R10_041", source="night.ts", camera_moved=1, duration_s=30.0)
    store.insert_vehicle(con, run_id=moved, camera_id="R10_041", track_id=1, cls="car", accepted=0,
                         reject_reason="camera moved since calibration", over_limit=0)
    con.commit()
    client = TestClient(create_app(db_path=db))
    cam = {c["id"]: c for c in client.get("/api/cameras").json()["cameras"]}["R10_041"]
    assert cam["latest"]["camera_moved"] == 1
    assert cam["latest_valid"]["run_id"] == first["run_id"]
    detail = client.get("/api/cameras/R10_041").json()
    assert detail["runs"][0]["camera_moved"] == 1 and detail["latest_valid"]["run_id"] == first["run_id"]
