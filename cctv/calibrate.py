"""Calibration helper: clicked lane-marking points -> homography + honest error.

Workflow for a new camera (see README "Calibrating a camera"):

1. ``python -m cctv.calibrate background --clip <clip.ts> --out <bg.png>``
   median of many frames = empty road, easy to click dash corners on.
2. Write ``data/calibration/<CAM>.points.json``: for each dash corner its pixel
   position, which lane line (0, 1, ...), which dash along that line (0, 1, ...)
   and whether it is the near ("start") or far ("end") end. X grows away from
   the camera along the lane lines; Y grows from line 0 to line 1.
3. ``python -m cctv.calibrate fit --camera <CAM>`` -> ``data/calibration/<CAM>.json``
   with H, reprojection error, and a *speed* scale uncertainty from bootstrap.
4. ``python -m cctv.calibrate show --camera <CAM> --frame <bg.png> --out check.png``
   draws a 10 ft road grid over the image: if the grid does not sit on the lane
   lines, the calibration is wrong.

Dashes on neighbouring lines are often not painted in phase, so lines listed in
``unknown_phase_lines`` get their along-road offset solved by a 1-D search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cctv.geometry import (DASH_FT, FT, GAP_FT, LANE_FT, Calibration, apply_homography,
                           fit_homography, reprojection_errors)
from cctv.registry import CALIBRATION_DIR, calibration_path
from cctv.views import camera_dir, list_views, match_view, set_view_label, validate_view_id

DEFAULT_SYSTEMATIC_SCALE = 0.025  # paint tolerance: 40 ft dash cycle +-1 ft is ~2.5%


def road_coords(points: list[dict], lane_w: float, dash: float, gap: float,
                offsets: dict[int, float] | None = None) -> np.ndarray:
    offsets = offsets or {}
    out = []
    for p in points:
        if "road" in p:  # explicit road coordinates (metres) win
            out.append(p["road"])
            continue
        x = p["dash"] * (dash + gap) + (dash if p["end"] == "end" else 0.0) + offsets.get(p["line"], 0.0)
        out.append([x, p["line"] * lane_w])
    return np.asarray(out, float)


def _transverse_cost(H: np.ndarray, lines: list[list[list[float]]]) -> float:
    """Sum of squared dX/dY slopes of mapped transverse lines (0 = perpendicular)."""
    cost = 0.0
    for line in lines:
        r = apply_homography(H, np.asarray(line, float))
        cost += float(np.polyfit(r[:, 1], r[:, 0], 1)[0]) ** 2
    return cost


def fit_points(spec: dict, step: float = 0.01) -> tuple[np.ndarray, dict[int, float], np.ndarray]:
    """Fit H, solving the dash phase of ``unknown_phase_lines``.

    Why this needs care: with two parallel lane lines, sliding one line along
    the road is the same as *shearing* the road frame, which a homography
    absorbs perfectly, so the reprojection error barely changes (it is flat to
    within noise). Along-lane speeds don't care about shear, but lane changes
    and headings do. If the spec lists ``transverse_lines`` (image points on
    lines known to cross the road at right angles, e.g. concrete pavement joints
    or stop bars) we choose the phase that makes them perpendicular. Otherwise
    we fall back to minimum reprojection error and say so.
    """
    lane_w = spec.get("lane_width_ft", LANE_FT) * FT
    dash = spec.get("dash_ft", DASH_FT) * FT
    gap = spec.get("gap_ft", GAP_FT) * FT
    pts = spec["points"]
    img = np.asarray([p["img"] for p in pts], float)
    transverse = spec.get("transverse_lines") or []
    offsets = {int(k): 0.0 for k in spec.get("unknown_phase_lines", [])}
    cycle = dash + gap
    grid = np.arange(-cycle, cycle + 1e-9, step)
    for _ in range(3):  # a few passes of coordinate descent over the unknown offsets
        for line in offsets:
            best = None
            for d in grid:
                trial = {**offsets, line: float(d)}
                road = road_coords(pts, lane_w, dash, gap, trial)
                try:
                    H = fit_homography(img, road)
                except ValueError:
                    continue
                if transverse:
                    cost = _transverse_cost(H, transverse)
                else:
                    cost = float(np.sqrt(np.mean(reprojection_errors(H, img, road) ** 2)))
                if best is None or cost < best[0]:
                    best = (cost, float(d))
            offsets[line] = best[1]
    road = road_coords(pts, lane_w, dash, gap, offsets)
    return fit_homography(img, road), offsets, road


def bootstrap_scale_uncertainty(img: np.ndarray, road: np.ndarray, H: np.ndarray,
                                n: int = 300, seed: int = 0) -> float:
    """Relative 1-sigma error of *speeds* caused by calibration point noise.

    Refit H on bootstrap resamples of the points and measure how much the
    length of a reference 30 m path (through the middle of the calibrated
    area, along X) changes. That spread is what a speed would inherit.
    """
    rng = np.random.default_rng(seed)
    cx, cy = road[:, 0].mean(), road[:, 1].mean()
    ref_road = np.array([[cx - 15, cy], [cx + 15, cy]])
    ref_img = apply_homography(np.linalg.inv(H), ref_road)
    lengths = []
    for _ in range(n):
        idx = rng.integers(0, len(img), len(img))
        if len(set(idx.tolist())) < 5:
            continue
        try:
            Hb = fit_homography(img[idx], road[idx])
            p = apply_homography(Hb, ref_img)
        except (ValueError, np.linalg.LinAlgError):
            continue
        lengths.append(np.linalg.norm(p[1] - p[0]))
    lengths = np.asarray(lengths)
    return float(np.std(lengths) / 30.0) if len(lengths) > 10 else float("nan")


def _view_paths(camera: str, view: str | None) -> tuple[Path, Path]:
    """(points.json, calibration .json) for a view, or the round-1 flat files when no view is given."""
    if view:
        d = camera_dir(camera)
        validate_view_id(view)
        return d / f"{view}.points.json", d / f"{view}.json"
    return CALIBRATION_DIR / f"{camera}.points.json", CALIBRATION_DIR / f"{camera}.json"


def cmd_fit(args) -> None:
    pts_default, out_default = _view_paths(args.camera, args.view)
    spec_path = args.points or pts_default
    spec = json.loads(Path(spec_path).read_text())
    H, offsets, road = fit_points(spec)
    img = np.asarray([p["img"] for p in spec["points"]], float)
    err = reprojection_errors(H, img, road)
    boot = bootstrap_scale_uncertainty(img, road, H)
    scale_unc = float(np.hypot(boot, DEFAULT_SYSTEMATIC_SCALE))
    cal = Calibration.fit(
        spec["camera_id"], spec["image_size"], img, road,
        scale_uncertainty=round(scale_unc, 4), reference_frame=spec.get("source_frame"),
        osd_boxes=spec.get("osd_boxes"), roi_polygon=spec.get("roi_polygon"),
        method=f"lane-dash homography; phase offsets {offsets} "
               f"({'from transverse joints' if spec.get('transverse_lines') else 'min reprojection error: shear unconstrained'})",
        notes=f"bootstrap scale sigma {boot:.4f}; systematic paint tolerance {DEFAULT_SYSTEMATIC_SCALE}; "
              + spec.get("notes", ""),
        view_id=args.view, direction_labels=spec.get("direction_labels"))
    out = args.out or out_default
    cal.save(out)
    print(f"points: {len(img)}  rmse {cal.rmse_m:.3f} m  max {cal.max_err_m:.3f} m")
    for p, e in zip(spec["points"], err):
        print(f"  line {p.get('line')} dash {p.get('dash')} {p.get('end', ''):5s} img {p['img']}  err {e:.3f} m")
    print(f"phase offsets (m): {offsets}")
    for k, line in enumerate(spec.get("transverse_lines") or []):
        r = apply_homography(H, np.asarray(line, float))
        print(f"  transverse line {k}: road X spread {np.ptp(r[:, 0]):.2f} m over {np.ptp(r[:, 1]):.1f} m of width")
    print(f"bootstrap speed-scale sigma: {boot:.2%}; total used: {scale_unc:.2%}")
    print(f"wrote {out}")


def draw_overlay(frame, cal: Calibration, spec: dict | None = None):
    import cv2

    img = frame.copy()
    H_inv = np.linalg.inv(cal.H_np)
    road = np.asarray(cal.road_pts)
    x0, x1 = road[:, 0].min() - 6, road[:, 0].max() + 12
    y_lines = sorted(set(np.round(road[:, 1], 3)))
    lane_w = LANE_FT * FT
    ys = np.arange(min(y_lines) - lane_w, max(y_lines) + 2 * lane_w + 1e-6, lane_w)

    def to_px(xy):
        return [tuple(int(round(c)) for c in p) for p in apply_homography(H_inv, xy)]

    for y in ys:  # lane lines (cyan) along X
        pts = to_px(np.c_[np.linspace(x0, x1, 40), np.full(40, y)])
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (255, 255, 0), 1)
    for x in np.arange(np.floor(x0 / (10 * FT)) * 10 * FT, x1, 10 * FT):  # every 10 ft across (yellow)
        a, b = to_px(np.array([[x, ys[0]], [x, ys[-1]]]))
        cv2.line(img, a, b, (0, 255, 255), 1)
    for (u, v), r in zip(cal.img_pts, apply_homography(H_inv, road)):
        cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 0, 255), 1)       # clicked
        cv2.circle(img, (int(round(r[0])), int(round(r[1]))), 1, (0, 255, 0), -1)  # reprojected
    if cal.roi_polygon:
        cv2.polylines(img, [np.asarray(cal.roi_polygon, np.int32)], True, (255, 0, 255), 1)
    return img


def cmd_show(args) -> None:
    import cv2

    cal = Calibration.load(args.calibration or calibration_path(args.camera, args.view))
    frame = cv2.imread(str(args.frame))
    out = draw_overlay(frame, cal)
    if args.scale != 1:
        out = cv2.resize(out, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(args.out), out)
    print(f"wrote {args.out}")


def cmd_background(args) -> None:
    import cv2

    cap = cv2.VideoCapture(str(args.clip))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % args.every == 0:
            frames.append(f)
        i += 1
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), bg)
    print(f"median of {len(frames)} frames -> {args.out}")


def cmd_views(args) -> None:
    views = list_views(args.camera)
    if not views:
        print(f"{args.camera}: no views (no calibration, no reference frame)")
        return
    for v in views:
        s = v.summary()
        cal = f"rmse {s['rmse_m']:.3f} m, scale ±{s['scale_uncertainty']:.1%}" if v.calibrated else "NOT calibrated"
        print(f"{v.view_id:12s} {cal:32s} refs: {', '.join(s['references']) or '-':38s} {v.label}"
              + ("  [flat round-1 files]" if v.legacy else ""))


def cmd_add_view(args) -> None:
    """Register a (new) PTZ preset: store a reference frame for it, built from a clip or copied from an image."""
    import cv2

    validate_view_id(args.view)
    d = camera_dir(args.camera)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{args.view}.ref.jpg" if not args.tag else f"{args.view}.ref-{args.tag}.jpg"
    out = d / name
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force to replace it)")
    if args.clip:
        from cctv.pipeline import clip_median_frame
        img = clip_median_frame(args.clip, every=args.every)
    else:
        img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit("could not read the clip/image")
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if args.label:
        set_view_label(args.camera, args.view, args.label)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB). Now click points for it and run: "
          f"python -m cctv.calibrate fit --camera {args.camera} --view {args.view}")


def cmd_match(args) -> None:
    from cctv.pipeline import clip_median_frame

    frame = clip_median_frame(args.clip)
    if frame is None:
        raise SystemExit(f"could not read {args.clip}")
    m = match_view(args.camera, frame)
    for c in m.candidates:
        print(f"  {c['view_id']:12s} {c['reference']:28s} matches {c['n_matches']:4d}  inliers {c['inlier_ratio']:.2f}  "
              f"shift {c['shift_px']:6.1f}px  zoom x{c['scale']:.3f}  -> {c['reason']}")
    print(f"{m.status.upper()}: {m.reason}")


def cmd_migrate(args) -> None:
    """Move the round-1 flat files <CAM>.json/.points.json/.ref.jpg into <CAM>/<VIEW>.*"""
    validate_view_id(args.view)
    d = camera_dir(args.camera)
    d.mkdir(parents=True, exist_ok=True)
    moved = []
    for suffix in (".json", ".points.json", ".ref.jpg"):
        src, dst = CALIBRATION_DIR / f"{args.camera}{suffix}", d / f"{args.view}{suffix}"
        if not src.exists():
            continue
        if dst.exists():
            raise SystemExit(f"{dst} already exists; not overwriting")
        src.rename(dst)
        moved.append(dst.name)
    cal_p = d / f"{args.view}.json"
    if cal_p.exists():
        cal = Calibration.load(cal_p)
        cal.view_id = args.view
        cal.save(cal_p)
    if args.label:
        set_view_label(args.camera, args.view, args.label)
    print(f"moved {moved or 'nothing'} into {d}/")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--camera", required=True)
    f.add_argument("--points", type=Path)
    f.add_argument("--out", type=Path)
    f.add_argument("--view", help="PTZ preset id: reads <CAM>/<VIEW>.points.json, writes <CAM>/<VIEW>.json "
                                  "(omit for the round-1 flat files <CAM>.points.json -> <CAM>.json)")
    s = sub.add_parser("show")
    s.add_argument("--camera", required=True)
    s.add_argument("--view")
    s.add_argument("--calibration", type=Path)
    s.add_argument("--frame", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--scale", type=float, default=1.0)
    b = sub.add_parser("background")
    b.add_argument("--clip", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--every", type=int, default=15)
    v = sub.add_parser("views", help="list a camera's known views (PTZ presets)")
    v.add_argument("--camera", required=True)
    a = sub.add_parser("add-view", help="register a PTZ preset by saving its reference frame")
    a.add_argument("--camera", required=True)
    a.add_argument("--view", required=True)
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", type=Path, help="median frame of this clip becomes the reference")
    src.add_argument("--image", type=Path, help="use this image as the reference")
    a.add_argument("--tag", help="extra reference for the same view, e.g. 'night' -> <VIEW>.ref-night.jpg")
    a.add_argument("--label", help="human description, stored in <CAM>/views.json")
    a.add_argument("--every", type=int, default=30)
    a.add_argument("--force", action="store_true")
    m = sub.add_parser("match", help="which known view does this clip show?")
    m.add_argument("--camera", required=True)
    m.add_argument("--clip", type=Path, required=True)
    g = sub.add_parser("migrate", help="move round-1 flat calibration files into <CAM>/<VIEW>.*")
    g.add_argument("--camera", required=True)
    g.add_argument("--view", required=True)
    g.add_argument("--label")
    args = ap.parse_args(argv)
    {"fit": cmd_fit, "show": cmd_show, "background": cmd_background, "views": cmd_views,
     "add-view": cmd_add_view, "match": cmd_match, "migrate": cmd_migrate}[args.cmd](args)


if __name__ == "__main__":
    main()
