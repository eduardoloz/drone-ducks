"""Monte Carlo error budget for one calibrated camera.

Simulates vehicles driving straight down each calibrated lane at a known speed,
projects them into the image with the camera's homography, injects one error
source at a time, runs the real estimator, and reports bias and spread.

    python -m cctv.error_budget --camera R10_041 --mph 60

Error sources (each alone, then all together):
- bbox jitter: Gaussian noise on the bottom-centre point (px)
- occlusion: a fraction of frames where the box bottom jumps (another car hides it)
- timestamp jitter: frame-time noise (PTS measured at ~0.4 ms; arrival times would be tens of ms)
- duplicated frames: encoder repeats the previous picture with a new timestamp
- calibration scale: the 40 ft dash cycle is really 39 or 41 ft (paint tolerance)
- undetected camera nudge: the view shifted by a few px after calibration
- loose boxes: box bottom sits a fixed fraction of box height below the tyres
"""

from __future__ import annotations

import argparse

import numpy as np

from cctv.geometry import MPH_PER_MPS, Calibration, apply_homography
from cctv.registry import calibration_path
from cctv.speed import SpeedConfig, estimate_speed


def lane_tracks(cal: Calibration, mph: float, fps: float = 14.985):
    """Constant-speed tracks along each lane centre, clipped to the ROI. Yields (t, uv, xy)."""
    road = np.asarray(cal.road_pts)
    lines = sorted(set(np.round(road[:, 1], 3)))
    lane_w = lines[1] - lines[0] if len(lines) > 1 else 3.6576
    centres = [lines[0] - lane_w / 2] + [y + lane_w / 2 for y in lines]
    v = mph / MPH_PER_MPS
    x = np.arange(road[:, 0].min() - 25, road[:, 0].max() + 25, v / fps)
    for yc in centres:
        xy = np.c_[x, np.full_like(x, yc)]
        uv = apply_homography(np.linalg.inv(cal.H_np), xy)
        keep = cal.in_roi(uv)
        if keep.sum() >= 8:
            idx = np.where(keep)[0]
            yield np.arange(len(idx)) / fps, uv[idx], xy[idx]


def run_scenario(cal: Calibration, mph: float, n: int, seed: int, perturb) -> dict:
    rng = np.random.default_rng(seed)
    errs, rejected = [], 0
    for _ in range(n):
        for t, uv, xy in lane_tracks(cal, mph):
            t2, uv2, H = perturb(rng, t.copy(), uv.copy(), cal)
            est = estimate_speed(t2, uv2, H, cal.scale_uncertainty, SpeedConfig())
            if est.accepted:
                errs.append(est.speed_mph - mph)
            else:
                rejected += 1
    e = np.asarray(errs)
    return {"bias": float(e.mean()) if len(e) else np.nan, "std": float(e.std()) if len(e) else np.nan,
            "p95_abs": float(np.percentile(np.abs(e), 95)) if len(e) else np.nan,
            "accepted": len(e), "rejected": rejected}


def scenarios(mph: float):
    """name -> perturbation function (rng, t, uv, cal) -> (t, uv, H used by the estimator)."""
    def base(rng, t, uv, cal):
        return t, uv, cal.H_np

    def jitter(px):
        return lambda rng, t, uv, cal: (t, uv + rng.normal(0, px, uv.shape), cal.H_np)

    def occlusion(frac, px):
        def f(rng, t, uv, cal):
            k = rng.random(len(t)) < frac
            uv[k, 1] -= rng.uniform(5, px, k.sum())  # bottom hidden -> box bottom jumps up
            return t, uv + rng.normal(0, 1.0, uv.shape), cal.H_np
        return f

    def tjitter(s):
        return lambda rng, t, uv, cal: (t + rng.normal(0, s, len(t)), uv + rng.normal(0, 1.0, uv.shape), cal.H_np)

    def duplicates(frac):
        def f(rng, t, uv, cal):
            for i in range(1, len(t)):
                if rng.random() < frac:
                    uv[i] = uv[i - 1]  # same picture, new timestamp
            return t, uv + rng.normal(0, 1.0, uv.shape), cal.H_np
        return f

    def scale(rel):
        # True dash cycle differs from 40 ft by `rel`: the world is (1+rel) bigger than we think.
        def f(rng, t, uv, cal):
            S = np.diag([1 / (1 + rel), 1 / (1 + rel), 1.0])
            return t, uv + rng.normal(0, 1.0, uv.shape), S @ cal.H_np
        return f

    def nudge(dv):
        return lambda rng, t, uv, cal: (t, uv + np.array([0.0, dv]) + rng.normal(0, 1.0, uv.shape), cal.H_np)

    def loose(frac):
        def f(rng, t, uv, cal):
            # box height ~ projected length of 4.5 m of road at the vehicle's position
            xy = apply_homography(cal.H_np, uv)
            ahead = apply_homography(np.linalg.inv(cal.H_np), xy + np.array([4.5, 0.0]))
            box_h = np.abs(uv[:, 1] - ahead[:, 1]) * 1.4
            uv[:, 1] += frac * box_h
            return t, uv + rng.normal(0, 1.0, uv.shape), cal.H_np
        return f

    def everything(rng, t, uv, cal):
        t, uv, _ = occlusion(0.1, 25)(rng, t, uv, cal)
        t, uv, _ = duplicates(0.05)(rng, t, uv, cal)
        uv[:, 1] += 2.0
        rel = rng.normal(0, 0.025)
        S = np.diag([1 / (1 + rel), 1 / (1 + rel), 1.0])
        return t + rng.normal(0, 0.001, len(t)), uv + rng.normal(0, 1.0, uv.shape), S @ cal.H_np

    return [
        ("no noise (sanity check)", base),
        ("bbox jitter 1 px", jitter(1.0)), ("bbox jitter 2 px", jitter(2.0)), ("bbox jitter 4 px", jitter(4.0)),
        ("occlusion 10% of frames, jumps 5-25 px", occlusion(0.10, 25)),
        ("occlusion 25% of frames, jumps 5-40 px", occlusion(0.25, 40)),
        ("timestamp jitter 0.5 ms (measured PTS)", tjitter(0.0005)),
        ("timestamp jitter 30 ms (arrival times)", tjitter(0.030)),
        ("duplicated frames 10%", duplicates(0.10)), ("duplicated frames 50%", duplicates(0.50)),
        ("dash cycle really 41 ft (+2.5%)", scale(0.025)), ("dash cycle really 39 ft (-2.5%)", scale(-0.025)),
        ("camera nudged 2 px down, undetected", nudge(2.0)), ("camera nudged 8 px down, undetected", nudge(8.0)),
        ("loose boxes: bottom 5% of box height low", loose(0.05)),
        ("all of the above, realistic mix", everything),
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="R10_041")
    ap.add_argument("--mph", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=40, help="repetitions per lane")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cal = Calibration.load(calibration_path(args.camera))
    lanes = list(lane_tracks(cal, args.mph))
    print(f"camera {args.camera}: {len(lanes)} lanes in ROI, {np.mean([len(t) for t, _, _ in lanes]):.0f} frames per "
          f"track at {args.mph:.0f} mph (track length {np.mean([np.ptp(xy[:, 0]) for _, _, xy in lanes]):.0f} m)")
    print(f"{'scenario':44s} {'bias':>6s} {'std':>6s} {'p95|e|':>7s} {'acc':>5s} {'rej':>4s}   (mph)")
    for name, fn in scenarios(args.mph):
        r = run_scenario(cal, args.mph, 1 if name.startswith("no noise") else args.n, args.seed, fn)
        print(f"{name:44s} {r['bias']:+6.2f} {r['std']:6.2f} {r['p95_abs']:7.2f} {r['accepted']:5d} {r['rejected']:4d}")


if __name__ == "__main__":
    main()
