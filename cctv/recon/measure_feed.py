"""Measure real feed characteristics for cameras in the manifest.

For HLS video cameras it records a short clip (default 20 s, ~0.5 MB) and reports:
resolution, codec, bitrate, nominal vs effective fps, frame-timestamp regularity,
duplicate-frame ratio ("unique fps"), segment length and publish cadence, the
on-screen clock tick spacing, and whether the view stayed still.
For NYC DOT JPEG cameras it polls the snapshot once per second and reports how
often the image actually changes.

    python -m cctv.recon.measure_feed --ids R10_041 R10_017 NYCDOT_43e3d020
    python -m cctv.recon.measure_feed --sample 10 --seconds 20

Results are merged (by camera id) into data/feed_measurements.json. Run
``python -m cctv.recon.build_manifest`` afterwards to fold them into the manifest.
One camera at a time, with a pause between cameras: be a polite viewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cctv.ingest import (decode_gray_small, fetch_bytes, fetch_text, frame_times,
                         parse_master_playlist, parse_media_playlist, probe_video, record_hls)
from cctv.registry import DATA_DIR, load_cameras

OUT = DATA_DIR / "feed_measurements.json"
CLIPS = DATA_DIR / "clips"
FRAMES = DATA_DIR / "frames"


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {}
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)),
            "p99": float(np.percentile(x, 99)), "max": float(np.max(x))}


def osd_clock_events(gray: np.ndarray, clip_seconds: float) -> dict | None:
    """How often the burned-in clock's seconds digit changes, per stream-second.

    Most NYSDOT cameras draw a clock in the top-right corner. We look at the
    right-most 5% x top 15% of the picture (the seconds digit) and count change
    events. On the 2026-11-10 clips this region changed exactly twice per
    second (digit change + a second refresh), so a rate of 2.00/s measured in
    *stream time* means the stream's timestamps run at real-world speed. A
    rate that is not close to a whole number suggests dropped or mistimed
    frames. Weak evidence on its own (it depends on the overlay layout), so we
    report it next to the PTS and segment-cadence checks.
    """
    h, w = gray.shape[1:]
    box = gray[:, : int(0.15 * h), int(0.95 * w):].astype(np.int16)
    d = np.abs(np.diff(box, axis=0)).mean(axis=(1, 2))
    if len(d) < 30 or clip_seconds <= 0:
        return None
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    spikes = np.where(d > max(3 * med, med + 8 * mad))[0]
    ticks: list[int] = []
    for s in spikes:  # merge spikes closer than 3 frames
        if not ticks or s - ticks[-1] > 3:
            ticks.append(int(s))
    if len(ticks) < 3:
        return None
    rate = len(ticks) / clip_seconds
    return {"events_per_s": round(rate, 3), "median_interval_frames": float(np.median(np.diff(ticks))),
            "n": len(ticks), "near_integer_rate": bool(abs(rate - round(rate)) < 0.05 and round(rate) > 0)}


def measure_hls(cam: dict, seconds: float, keep_clip: bool) -> dict:
    url = cam["feed"]["url"]
    res: dict = {"camera_id": cam["id"], "feed_type": "hls", "url": url}
    master = fetch_text(url)
    variants = parse_master_playlist(master)
    res["variants"] = [{k: v for k, v in var.items() if k != "uri"} for var in variants]
    if variants:
        base = url.rsplit("/", 1)[0]
        media = parse_media_playlist(fetch_text(f"{base}/{variants[0]['uri']}"))
        segs = [d for d, _ in media["segments"]]
        res["target_duration_s"] = media["target_duration"]
        res["segment_duration_s"] = float(np.median(segs)) if segs else None
        res["playlist_window_s"] = float(np.sum(segs)) if segs else None
        res["playlist_segments"] = len(segs)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clip = CLIPS / f"{cam['id']}_{stamp}.ts"
    rec = record_hls(url, clip, seconds, camera_id=cam["id"])
    res["recorded_utc"] = rec.started_utc
    if rec.returncode != 0 or not clip.exists() or clip.stat().st_size < 10_000:
        res.update(ok=False, error=f"ffmpeg rc={rec.returncode}: {rec.stderr_tail[-300:]}")
        return res

    info = probe_video(clip)
    res.update(width=info["width"], height=info["height"], codec=f"{info['codec']} {info['profile']}",
               nominal_fps=round(info["nominal_fps"], 3) if info["nominal_fps"] else None)
    pts = frame_times(clip)
    dt = np.diff(pts)
    nominal_dt = 1.0 / info["nominal_fps"] if info["nominal_fps"] else float(np.median(dt))
    res["frames"] = int(len(pts))
    res["clip_seconds"] = float(pts[-1] - pts[0]) if len(pts) > 1 else 0.0
    res["effective_fps"] = round((len(pts) - 1) / res["clip_seconds"], 3) if res["clip_seconds"] else None
    res["frame_dt_ms"] = {k: round(v * 1000, 2) for k, v in _stats(dt).items()}
    res["timestamp_gaps_over_1_5x"] = int(np.sum(dt > 1.5 * nominal_dt))
    res["bitrate_kbps"] = round(clip.stat().st_size * 8 / 1000 / max(res["clip_seconds"], 1e-6), 1)

    gray = decode_gray_small(clip, 320, 240)
    res["mean_luma"] = round(float(gray.mean()), 1)   # < 35: too dark to judge (night)
    diffs = np.abs(np.diff(gray.astype(np.int16), axis=0)).mean(axis=(1, 2))
    dup = diffs < 0.3
    res["dup_frame_ratio"] = round(float(dup.mean()), 3) if len(dup) else None
    res["unique_fps"] = round(res["effective_fps"] * (1 - res["dup_frame_ratio"]), 2) if res["effective_fps"] else None
    res["osd_clock"] = osd_clock_events(gray, res["clip_seconds"])

    # Segment publish cadence from live arrivals (skip the initial backlog burst).
    arr = [t for t, _ in rec.segment_arrivals]
    live = np.diff(arr)
    live = live[live > 1.0]
    res["segment_publish_interval_s"] = round(float(np.median(live)), 2) if len(live) else None

    # Did the view move during the clip? (first vs last frame, full resolution)
    try:
        import cv2

        from cctv.camera_motion import CameraMotionDetector

        cap = cv2.VideoCapture(str(clip))
        ok, first = cap.read()
        last = first
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            last = fr
        cap.release()
        FRAMES.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(FRAMES / f"{cam['id']}_{stamp}.jpg"), first)
        mres = CameraMotionDetector(first).check(last)
        res["view_static_during_clip"] = not mres.moved
        res["view_check"] = {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in mres.to_dict().items()}
    except Exception as exc:  # pragma: no cover - diagnostics only
        res["view_check_error"] = repr(exc)

    res["ok"] = True
    if not keep_clip:
        clip.unlink(missing_ok=True)
    else:
        rec.save_sidecar()
        res["clip"] = str(clip.relative_to(DATA_DIR.parent))
    return res


def measure_jpeg(cam: dict, seconds: float) -> dict:
    url = cam["feed"]["url"]
    res: dict = {"camera_id": cam["id"], "feed_type": "jpeg", "url": url,
                 "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    changes, prev, first = [], None, None
    t_end = time.time() + seconds
    while time.time() < t_end:
        t = time.time()
        data = fetch_bytes(url)
        first = first or data
        h = hashlib.md5(data).hexdigest()
        if h != prev:
            changes.append(t)
            prev = h
        time.sleep(max(0.0, 1.0 - (time.time() - t)))
    iv = np.diff(changes)
    res["refresh_interval_s"] = {k: round(v, 2) for k, v in _stats(iv).items()} if len(iv) else None
    res["effective_fps"] = round(1.0 / float(np.median(iv)), 3) if len(iv) else 0.0
    res["unique_fps"] = res["effective_fps"]
    res["poll_interval_s"] = 1.0
    try:
        import cv2

        img = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
        res["height"], res["width"] = img.shape[:2]
    except Exception:  # pragma: no cover
        pass
    res["note"] = "Polled at 1 Hz, so refresh intervals are quantised to ~1 s."
    res["ok"] = True
    return res


def pick_sample(cams: list[dict], n: int) -> list[dict]:
    """Spread n mainline video cameras along the corridor, plus 2 NYC DOT stills."""
    video = [c for c in cams if c["kind"] == "mainline" and c["feed"]["type"] == "hls"]
    idx = np.linspace(0, len(video) - 1, n).round().astype(int)
    stills = [c for c in cams if c["feed"]["type"] == "jpeg" and c["kind"] == "mainline"][:2]
    return [video[i] for i in sorted(set(idx))] + stills


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--sample", type=int, default=0, help="measure N cameras spread along the LIE")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--keep-clips", action="store_true")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between cameras")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    cams = load_cameras()
    by_id = {c["id"]: c for c in cams}
    todo = [by_id[i] for i in args.ids] + (pick_sample(cams, args.sample) if args.sample else [])
    existing = json.loads(args.out.read_text()) if args.out.exists() else {"results": []}
    results = {r["camera_id"]: r for r in existing["results"]}
    for cam in todo:
        print(f"-> {cam['id']} {cam['name']} ({cam['feed']['type']})", flush=True)
        try:
            r = measure_hls(cam, args.seconds, args.keep_clips) if cam["feed"]["type"] == "hls" \
                else measure_jpeg(cam, args.seconds)
        except Exception as exc:
            r = {"camera_id": cam["id"], "ok": False, "error": repr(exc),
                 "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            traceback.print_exc()
        results[cam["id"]] = r
        brief = {k: r.get(k) for k in ("ok", "width", "height", "effective_fps", "unique_fps",
                                       "segment_duration_s", "view_static_during_clip", "error")}
        print("   ", brief, flush=True)
        args.out.write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                        "results": sorted(results.values(), key=lambda x: x["camera_id"])},
                                       indent=1) + "\n")
        time.sleep(args.pause)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
