"""Web app for the prototype: serves the map, speed estimates and stats from SQLite.

    python -m cctv.web.app            # http://127.0.0.1:8010
    python -m cctv.web.app --port 8011 --db other.sqlite

JSON API (all read-only except the stateless calibration fit):
    GET  /api/health
    GET  /api/cameras                     all manifest cameras + latest-run stats
    GET  /api/cameras/{id}                one camera: manifest entry, calibration, runs, stats
    GET  /api/cameras/{id}/reference.jpg?view=   reference frame of a view (default: primary view)
    GET  /api/cameras/{id}/calibration?view=     calibration JSON + clicked points + list of views
    GET  /api/events?camera_id=&over_only=1&limit=50
    GET  /api/stats?camera_id=            histogram, by-hour aggregates, classes
    GET  /api/runs/{run_id}/overlay       track polylines for drawing on the reference frame
    POST /api/calibrate/fit               points spec -> H, errors, grid (nothing is saved)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cctv import store
from cctv.registry import MANIFEST_PATH, load_manifest
from cctv.views import get_view, list_views, primary_view

STATIC = Path(__file__).resolve().parent / "static"
LOCAL_TZ = ZoneInfo("America/New_York")


def _round(x: float | None, nd: int = 1) -> float | None:
    return None if x is None else round(float(x), nd)


def speed_summary(rows: list[sqlite3.Row], duration_s: float | None = None) -> dict[str, Any]:
    """Summary numbers for a set of vehicle rows (accepted speeds only)."""
    acc = [r for r in rows if r["accepted"] == 1 and r["speed_mph"] is not None]
    out: dict[str, Any] = {"vehicles": len(rows), "with_speed": len(acc)}
    if duration_s:
        out["flow_veh_per_hour"] = round(len(rows) / duration_s * 3600)
    if acc:
        s = np.array([r["speed_mph"] for r in acc])
        over = sum(r["over_limit"] for r in acc)
        out.update(median_mph=_round(np.median(s)), p85_mph=_round(np.percentile(s, 85)),
                   mean_sigma_mph=_round(np.mean([r["sigma_mph"] for r in acc])),
                   over_limit=int(over), pct_over_limit=_round(100.0 * over / len(acc)))
    return out


def create_app(db_path: Path | str = store.DEFAULT_DB, manifest_path: Path | str = MANIFEST_PATH) -> FastAPI:
    app = FastAPI(title="LIE traffic speed analytics (prototype)", version="0.1.0")
    con = store.connect(db_path)
    manifest = load_manifest(manifest_path)
    cams = {c["id"]: c for c in manifest["cameras"]}

    def latest_runs(valid_only: bool = False) -> dict[str, sqlite3.Row]:
        """Newest run per camera; with valid_only, the newest one that produced speeds."""
        where = ("WHERE COALESCE(camera_moved, 0) = 0 AND EXISTS (SELECT 1 FROM vehicles v "
                 "WHERE v.run_id = runs.id AND v.accepted = 1)") if valid_only else ""
        rows = con.execute(f"SELECT r.* FROM runs r JOIN (SELECT camera_id, MAX(id) AS id FROM runs {where} "
                           "GROUP BY camera_id) m ON m.id = r.id").fetchall()
        return {r["camera_id"]: r for r in rows}

    def run_summary(run: sqlite3.Row) -> dict[str, Any]:
        rows = con.execute("SELECT * FROM vehicles WHERE run_id = ?", (run["id"],)).fetchall()
        s = speed_summary(rows, run["duration_s"])
        s.update(run_id=run["id"], clip_start_utc=run["clip_start_utc"], duration_s=_round(run["duration_s"]),
                 camera_moved=run["camera_moved"], view_id=run["view_id"], model=run["model"],
                 ms_per_frame=run["ms_per_frame"], light=run["light"], sun_elev_deg=_round(run["sun_elev_deg"]),
                 luma=_round(run["luma"]), view_shift_px=_round(run["view_shift_px"]))
        return s

    def calibrated(cid: str) -> bool:
        return any(v.calibrated for v in list_views(cid))

    def pick_view(cid: str, view: str | None):
        if view:
            try:
                return get_view(cid, view)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
        return primary_view(cid)

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/about", include_in_schema=False)
    def about() -> FileResponse:
        return FileResponse(STATIC / "about.html")

    @app.get("/calibrate", include_in_schema=False)
    def calibrate_page() -> FileResponse:
        return FileResponse(STATIC / "calibrate.html")

    # -------------------------------------------------------------------- api
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        n = con.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        return {"ok": True, "cameras": len(cams), "vehicles": n, "manifest_generated": manifest["generated_at"]}

    @app.get("/api/cameras")
    def cameras() -> dict[str, Any]:
        runs, valid = latest_runs(), latest_runs(valid_only=True)
        out = []
        for c in manifest["cameras"]:
            meas = c.get("measured") or {}
            item = {k: c.get(k) for k in ("id", "name", "owner", "lat", "lon", "county", "kind", "exit",
                                          "chainage_km", "posted_limit_mph", "speed_suitability")}
            item.update(feed_type=c["feed"]["type"], calibrated=calibrated(c["id"]),
                        width=meas.get("width"), height=meas.get("height"),
                        unique_fps=meas.get("unique_fps"))
            if c["id"] in runs:
                item["latest"] = run_summary(runs[c["id"]])
            if c["id"] in valid:
                item["latest_valid"] = run_summary(valid[c["id"]])
            out.append(item)
        return {"generated_at": manifest["generated_at"], "corridor": manifest["corridor"],
                "summary": manifest["summary"], "cameras": out}

    @app.get("/api/cameras/{camera_id}")
    def camera(camera_id: str) -> dict[str, Any]:
        if camera_id not in cams:
            raise HTTPException(404, f"unknown camera {camera_id}")
        runs = con.execute("SELECT * FROM runs WHERE camera_id = ? ORDER BY id DESC", (camera_id,)).fetchall()
        cal = None
        pv = primary_view(camera_id)
        if pv is not None and pv.calibrated:
            d = json.loads(pv.calibration_path.read_text())
            cal = {k: d.get(k) for k in ("created_utc", "rmse_m", "max_err_m", "scale_uncertainty",
                                         "image_size", "roi_polygon", "method")}
            cal["view_id"] = pv.view_id
        summaries = [run_summary(r) for r in runs]
        valid = next((s for s in summaries if not s["camera_moved"] and s["with_speed"]), None)
        return {"camera": cams[camera_id], "calibration": cal,
                "views": [v.summary() for v in list_views(camera_id)],
                "has_reference_frame": bool(pv and pv.reference_paths),
                "runs": summaries, "latest_valid": valid}

    @app.get("/api/cameras/{camera_id}/reference.jpg", include_in_schema=False)
    def reference(camera_id: str, view: str | None = None) -> FileResponse:
        v = pick_view(camera_id, view) if camera_id in cams else None
        if v is None or not v.reference_paths:
            raise HTTPException(404, "no reference frame for this camera/view")
        return FileResponse(v.reference_paths[0], media_type="image/jpeg")

    @app.get("/api/cameras/{camera_id}/calibration")
    def calibration(camera_id: str, view: str | None = None) -> dict[str, Any]:
        v = pick_view(camera_id, view)
        out: dict[str, Any] = {"camera_id": camera_id, "view_id": v.view_id if v else None,
                               "views": [x.summary() for x in list_views(camera_id)]}
        for key, p in (("calibration", v and v.calibration_path), ("points", v and v.points_path)):
            out[key] = json.loads(p.read_text()) if p else None
        if v is None and camera_id not in cams:
            raise HTTPException(404, f"unknown camera {camera_id}")
        return out

    @app.get("/api/events")
    def events(camera_id: str | None = None, over_only: bool = False,
               limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        sql = ("SELECT v.id, v.ts_utc, v.camera_id, c.name AS camera_name, v.track_id, v.cls, v.speed_mph, "
               "v.sigma_mph, v.limit_mph, v.margin_mph, v.over_limit, v.run_id, v.n_points, v.distance_m "
               "FROM vehicles v JOIN cameras c ON c.id = v.camera_id WHERE v.accepted = 1")
        args: list[Any] = []
        if camera_id:
            sql += " AND v.camera_id = ?"
            args.append(camera_id)
        if over_only:
            sql += " AND v.over_limit = 1"
        sql += " ORDER BY v.ts_utc DESC, v.speed_mph DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        rows = [dict(r) for r in con.execute(sql, args).fetchall()]
        for r in rows:
            for k in ("speed_mph", "sigma_mph", "margin_mph", "distance_m"):
                r[k] = _round(r[k])
        return {"events": rows, "limit": limit, "offset": offset}

    @app.get("/api/stats")
    def stats(camera_id: str | None = None, bin_mph: int = Query(5, ge=1, le=20)) -> dict[str, Any]:
        sql, args = "SELECT * FROM vehicles", []
        if camera_id:
            sql += " WHERE camera_id = ?"
            args.append(camera_id)
        rows = con.execute(sql, args).fetchall()
        acc = [r for r in rows if r["accepted"] == 1 and r["speed_mph"] is not None]
        speeds = np.array([r["speed_mph"] for r in acc])
        hist = []
        if len(speeds):
            lo = int(np.floor(speeds.min() / bin_mph) * bin_mph)
            hi = int(np.ceil((speeds.max() + 1e-9) / bin_mph) * bin_mph)
            counts, edges = np.histogram(speeds, bins=np.arange(lo, hi + bin_mph, bin_mph))
            hist = [{"from": int(a), "to": int(b), "count": int(n)} for a, b, n in zip(edges, edges[1:], counts)]
        by_hour: dict[int, list[sqlite3.Row]] = {}
        for r in acc:
            if r["ts_utc"]:
                h = datetime.fromisoformat(r["ts_utc"]).astimezone(LOCAL_TZ).hour
                by_hour.setdefault(h, []).append(r)
        hours = [{"hour_local": h, **speed_summary(rs)} for h, rs in sorted(by_hour.items())]
        limits = sorted({r["limit_mph"] for r in rows if r["limit_mph"] is not None})
        return {"camera_id": camera_id, "summary": speed_summary(rows), "histogram": hist,
                "by_hour_local": hours, "classes": dict(Counter(r["cls"] for r in rows)),
                "reject_reasons": dict(Counter((r["reject_reason"] or "").split(" (")[0].split(":")[0]
                                               for r in rows if not r["accepted"])),
                "posted_limits_mph": limits, "timezone": "America/New_York"}

    @app.get("/api/runs/{run_id}/overlay")
    def overlay(run_id: int, roi_only: bool = True) -> dict[str, Any]:
        run = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise HTTPException(404, f"unknown run {run_id}")
        cid = run["camera_id"]
        vid = run["view_id"]
        try:
            view = get_view(cid, vid) if vid else primary_view(cid)
        except KeyError:
            view = None
        cal = json.loads(view.calibration_path.read_text()) if view and view.calibration_path else {}
        vehicles = con.execute("SELECT * FROM vehicles WHERE run_id = ?", (run_id,)).fetchall()
        tracks = []
        for v in vehicles:
            q = "SELECT t, u, v FROM track_points WHERE vehicle_id = ?" + (" AND in_roi = 1" if roi_only else "")
            pts = con.execute(q + " ORDER BY t", (v["id"],)).fetchall()
            tracks.append({"vehicle_id": v["id"], "track_id": v["track_id"], "cls": v["cls"],
                           "accepted": v["accepted"], "over_limit": v["over_limit"],
                           "speed_mph": _round(v["speed_mph"]), "sigma_mph": _round(v["sigma_mph"]),
                           "reject_reason": v["reject_reason"], "t_start": v["t_start"], "t_end": v["t_end"],
                           "points": [[round(p["u"], 1), round(p["v"], 1), round(p["t"], 2)] for p in pts]})
        return {"run_id": run_id, "camera_id": cid, "image_size": cal.get("image_size", [640, 480]),
                "duration_s": run["duration_s"],
                "roi_polygon": cal.get("roi_polygon"), "view_id": view.view_id if view else None,
                "reference_url": f"/api/cameras/{cid}/reference.jpg" + (f"?view={view.view_id}" if view else ""),
                "camera_moved": run["camera_moved"], "tracks": tracks}

    @app.post("/api/calibrate/fit")
    def calibrate_fit(spec: dict[str, Any] = Body(...)) -> JSONResponse:
        from cctv.calibrate import bootstrap_scale_uncertainty, fit_points, DEFAULT_SYSTEMATIC_SCALE
        from cctv.geometry import FT, LANE_FT, apply_homography, reprojection_errors

        pts = spec.get("points") or []
        if len(pts) < 4:
            raise HTTPException(422, "need at least 4 points")
        try:
            H, offsets, road = fit_points(spec)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"could not fit: {exc}") from exc
        img = np.asarray([p["img"] for p in pts], float)
        err = reprojection_errors(H, img, road)
        boot = bootstrap_scale_uncertainty(img, road, H, n=150)
        Hi = np.linalg.inv(H)
        lane_w = spec.get("lane_width_ft", LANE_FT) * FT
        x0, x1 = road[:, 0].min() - 6, road[:, 0].max() + 12
        ys = np.arange(road[:, 1].min() - lane_w, road[:, 1].max() + 2 * lane_w + 1e-6, lane_w)
        grid = [apply_homography(Hi, np.c_[np.linspace(x0, x1, 30), np.full(30, y)]).round(1).tolist() for y in ys]
        grid += [apply_homography(Hi, np.array([[x, ys[0]], [x, ys[-1]]])).round(1).tolist()
                 for x in np.arange(np.floor(x0 / (10 * FT)) * 10 * FT, x1, 10 * FT)]
        return JSONResponse({
            "H": H.tolist(), "rmse_m": float(np.sqrt(np.mean(err ** 2))), "max_err_m": float(err.max()),
            "errors_m": err.round(3).tolist(), "road_pts": road.round(3).tolist(),
            "phase_offsets_m": {str(k): v for k, v in offsets.items()},
            "bootstrap_scale_sigma": boot,
            "scale_uncertainty": float(np.hypot(boot, DEFAULT_SYSTEMATIC_SCALE)) if np.isfinite(boot) else None,
            "reprojected": apply_homography(Hi, road).round(1).tolist(), "grid": grid})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args(argv)
    uvicorn.run(create_app(args.db), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
