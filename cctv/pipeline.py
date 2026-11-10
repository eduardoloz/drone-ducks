"""Clip -> tracks -> speeds -> SQLite. The prototype's vertical slice.

    python -m cctv.pipeline --camera R10_041 --clip cctv/data/clips/R10_041_20261110T213935Z.ts

Steps
1. load the camera (manifest) and its calibration (data/calibration/<id>.json)
2. check the camera has not moved since calibration (PTZ!) using the committed
   reference frame; if it moved, speeds are stored as rejected
3. detect + track vehicles (or reuse a cached tracks JSONL: tracking is the slow part)
4. per track: keep points inside the calibrated ROI, estimate speed +- sigma
5. write run, vehicles and per-frame points to SQLite; print a summary

Without a calibration the run still stores vehicle counts (flow), with no speeds.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from cctv import store
from cctv.geometry import Calibration
from cctv.registry import CALIBRATION_DIR, DATA_DIR, calibration_path, get_camera, load_cameras
from cctv.speed import SpeedConfig, estimate_speed, is_speeding

RUNS_DIR = DATA_DIR / "runs"
MIN_ROI_POINTS_FOR_COUNT = 5


def clip_start_utc(clip: Path) -> tuple[str | None, str]:
    """Best guess of the wall-clock time of the clip's first frame."""
    side = clip.with_suffix(".json")
    if side.exists():
        d = json.loads(side.read_text())
        return d.get("started_utc"), "recording start (sidecar); content ~5-20 s older (HLS latency)"
    log = clip.with_suffix(".log")
    if log.exists():
        m = re.search(r"start_wall=([\d.]+)", log.read_text()[:200])
        if m:
            ts = datetime.fromtimestamp(float(m.group(1)), tz=timezone.utc)
            return ts.isoformat(timespec="seconds"), "recording start (log); content ~5-20 s older"
    m = re.search(r"(\d{8}T\d{6}Z)", clip.name)
    if m:
        ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return ts.isoformat(timespec="seconds"), "from file name; content ~5-20 s older"
    return None, "unknown"


def clip_median_frame(clip: Path, every: int = 30) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(f)
        i += 1
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None


def reference_frame_path(camera_id: str) -> Path:
    return CALIBRATION_DIR / f"{camera_id}.ref.jpg"


def check_camera_motion(camera_id: str, clip: Path, cal: Calibration | None) -> tuple[bool | None, dict]:
    """Compare the calibration reference frame with this clip's median frame."""
    import cv2

    from cctv.camera_motion import DEFAULT_OSD_BOXES, CameraMotionDetector

    ref_path = reference_frame_path(camera_id)
    if cal is None or not ref_path.exists():
        return None, {"reason": f"no reference frame at {ref_path.name}; motion not checked"}
    ref = cv2.imread(str(ref_path))
    cur = clip_median_frame(clip)
    if ref is None or cur is None:
        return None, {"reason": "could not read frames"}
    boxes = [tuple(b) for b in cal.osd_boxes] if cal.osd_boxes else DEFAULT_OSD_BOXES
    res = CameraMotionDetector(ref, osd_boxes=boxes).check(cur)
    return res.moved, res.to_dict()


def process(camera_id: str, clip: Path, db_path: Path, tracks_path: Path | None = None,
            model: str = "yolo11n.pt", imgsz: int = 640, max_frames: int | None = None,
            reuse_tracks: bool = True, k_sigma: float = 2.0, min_margin_mph: float = 3.0,
            limit_mph: float | None = None, store_points: bool = True,
            calibration_file: Path | None = None) -> dict:
    from cctv.detect_track import group_tracks, load_jsonl, run_tracker, save_jsonl

    cam = get_camera(camera_id)
    limit = float(limit_mph or cam.get("posted_limit_mph") or 55)
    cal_file = calibration_file or calibration_path(camera_id)
    cal = Calibration.load(cal_file) if Path(cal_file).exists() else None

    moved, motion = check_camera_motion(camera_id, clip, cal)
    if moved:
        print(f"!! camera view changed since calibration: {motion.get('reason')} -> speeds rejected")

    tracks_path = tracks_path or RUNS_DIR / f"{clip.stem}.{Path(model).stem}.tracks.jsonl"
    if reuse_tracks and tracks_path.exists():
        points, meta = load_jsonl(tracks_path)
        print(f"reusing {len(points)} tracked boxes from {tracks_path}")
    else:
        print(f"tracking {clip.name} with {model} (CPU, this is the slow part) ...")
        points, meta = run_tracker(clip, model, imgsz, max_frames=max_frames)
        meta["clip"] = str(clip)
        save_jsonl(points, tracks_path, meta)
    tracks = group_tracks(points)

    start_iso, start_note = clip_start_utc(clip)
    start_dt = datetime.fromisoformat(start_iso) if start_iso else None
    con = store.connect(db_path)
    store.upsert_cameras(con, load_cameras())
    store.delete_runs_for_source(con, camera_id, str(clip))
    frames = int(meta.get("frames") or (max(p.frame for p in points) + 1 if points else 0))
    duration = float(max((p.t for p in points), default=0.0))
    run_id = store.insert_run(
        con, camera_id=camera_id, source=str(clip),
        processed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        clip_start_utc=start_iso, clip_start_note=start_note, duration_s=duration, frames=frames,
        fps=round(frames / duration, 3) if duration else None, model=meta.get("model", model),
        tracker=meta.get("tracker", "bytetrack.yaml"), ms_per_frame=meta.get("ms_per_frame_mean"),
        calibration_utc=cal.created_utc if cal else None,
        calibration_rmse_m=cal.rmse_m if cal else None,
        scale_uncertainty=cal.scale_uncertainty if cal else None,
        camera_moved=None if moved is None else int(moved), motion_json=json.dumps(motion),
        reference_frame=str(reference_frame_path(camera_id).name) if cal else None,
        notes="speeds rejected: camera moved" if moved else None)

    cfg = SpeedConfig()
    summary = {"run_id": run_id, "camera_id": camera_id, "tracks_total": len(tracks), "vehicles": 0,
               "accepted": 0, "over_limit": 0, "rejected": {}, "limit_mph": limit}
    speeds = []
    for tid, tr in tracks.items():
        in_roi = cal.in_roi(tr["uv"]) if cal else np.ones(len(tr["t"]), bool)
        if in_roi.sum() < MIN_ROI_POINTS_FOR_COUNT:
            continue  # never really inside the measured stretch of road (other carriageway, far away)
        t, uv = tr["t"][in_roi], tr["uv"][in_roi]
        if cal is not None:
            est = estimate_speed(t, uv, cal.H_np, cal.scale_uncertainty, cfg, track_id=tid, cls=tr["cls"])
            if moved and est.accepted:
                est.accepted, est.reject_reason = False, "camera moved since calibration"
        else:
            est = None
        mid_t = float((t[0] + t[-1]) / 2)
        ts = (start_dt + timedelta(seconds=mid_t)).isoformat(timespec="seconds") if start_dt else None
        fields = dict(run_id=run_id, camera_id=camera_id, track_id=int(tid), cls=tr["cls"], ts_utc=ts,
                      t_start=float(t[0]), t_end=float(t[-1]), n_points=int(len(t)), limit_mph=limit)
        if est is not None:
            over = is_speeding(est, limit, k_sigma, min_margin_mph)
            fields.update(n_inliers=est.n_inliers, distance_m=_f(est.distance_m),
                          speed_mph=_f(est.speed_mph), sigma_mph=_f(est.sigma_mph),
                          sigma_stat_mph=_f(est.sigma_stat_mph), sigma_cal_mph=_f(est.sigma_cal_mph),
                          direction=est.direction, heading_deg=_f(est.heading_deg),
                          accepted=int(est.accepted), reject_reason=est.reject_reason,
                          margin_mph=_f(est.margin_mph(k_sigma, min_margin_mph)), over_limit=int(over))
            if est.accepted:
                summary["accepted"] += 1
                summary["over_limit"] += int(over)
                speeds.append(est.speed_mph)
            else:
                key = (est.reject_reason or "").split(" (")[0].split(":")[0]
                summary["rejected"][key] = summary["rejected"].get(key, 0) + 1
        else:
            fields.update(accepted=0, reject_reason="camera not calibrated", over_limit=0)
        pts = None
        if store_points:
            xy = cal.image_to_road(tr["uv"]) if cal else np.full((len(tr["t"]), 2), np.nan)
            pts = [(float(a), float(u), float(v), _f(x), _f(y), int(r))
                   for a, (u, v), (x, y), r in zip(tr["t"], tr["uv"], xy, in_roi)]
        store.insert_vehicle(con, points=pts, **fields)
        summary["vehicles"] += 1
    con.commit()
    if speeds:
        s = np.asarray(speeds)
        summary.update(median_mph=round(float(np.median(s)), 1), p85_mph=round(float(np.percentile(s, 85)), 1),
                       min_mph=round(float(s.min()), 1), max_mph=round(float(s.max()), 1))
    summary.update(camera_moved=moved, motion=motion, clip_start_utc=start_iso,
                   ms_per_frame=meta.get("ms_per_frame_mean"))
    return summary


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True, help="camera id from data/cameras_lie.json, e.g. R10_041")
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    ap.add_argument("--tracks", type=Path, help="cached tracks JSONL (default data/runs/<clip>.<model>.tracks.jsonl)")
    ap.add_argument("--retrack", action="store_true", help="ignore cached tracks and run YOLO again")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--limit-mph", type=float, help="override the posted limit from the manifest")
    ap.add_argument("--k-sigma", type=float, default=2.0, help="margin = max(k*sigma, min margin)")
    ap.add_argument("--min-margin-mph", type=float, default=3.0)
    ap.add_argument("--no-points", action="store_true", help="don't store per-frame track points")
    args = ap.parse_args(argv)
    summary = process(args.camera, args.clip, args.db, args.tracks, args.model, args.imgsz, args.max_frames,
                      reuse_tracks=not args.retrack, k_sigma=args.k_sigma, min_margin_mph=args.min_margin_mph,
                      limit_mph=args.limit_mph, store_points=not args.no_points)
    print(json.dumps(summary, indent=1, default=str))
    print(f"db: {args.db}")


if __name__ == "__main__":
    main()
