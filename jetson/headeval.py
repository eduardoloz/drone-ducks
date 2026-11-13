"""Closed-loop check of the head-yaw lateral term (k_head_lateral) on angled walker paths. No PX4.

ToyWorld + a walker who faces where they walk (like the sim's quack_people walker, "leading"
follow: the drone stays in front of a person walking toward it). The face is detectable only
while the walker faces within `visible_deg` (60) of the drone and is in the camera's view. The
oracle detector returns the box plus 5 landmarks from jetson.headpose's head model, with pixel
noise, so the whole perception path runs: tracker -> RelockGate -> head yaw -> controller.

    python -m jetson.headeval                      # table: with vs without the lateral term
    python -m jetson.headeval --preset sim --k 0.02 --seeds 5

Metrics per run (60 s):
  lock %       frames the controller was TRACKING
  front %      frames the walker faced within 30 deg of the drone (the drone is "in front")
  timeout_s    when the controller gave up (TIMEOUT -> land), or "-" if it never did

Kinematic toy, NOT a flight sim: no attitude, wind or camera tilt. The head model and landmark
noise are assumptions; real YuNet landmarks on 12-20 px faces were not measured.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .bridge import DryRunSink
from .controller import TIMEOUT, TRACKING, FaceFollowController, FollowConfig
from .detectors.base import FaceDetector
from .face import Face
from .headpose import synth_landmarks
from .pipeline import FollowPipeline
from .sources import Frame
from .toysim import ToyWorld, wrap_deg


@dataclass
class WalkerWorld(ToyWorld):
    heading_deg: float = 180.0      # walker's walking/facing direction, NED (0 = north)
    walk_speed: float = 0.4         # m/s (sim walker: 0.4)
    heading_noise_deg: float = 0.0  # random walk of the heading, deg per sqrt(s)
    visible_deg: float = 60.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def facing_deg(self) -> float:
        """Signed head yaw as the camera sees it: angle between the walker's facing and the
        direction from the walker to the drone; + = face points to the camera's right."""
        dn, de = self.north - self.person_n, self.east - self.person_e
        h = math.radians(self.heading_deg)
        hn, he = math.cos(h), math.sin(h)
        psi = math.radians(self.yaw_deg)
        rn, re = -math.sin(psi), math.cos(psi)       # camera's right in world (N, E)
        d = math.hypot(dn, de) or 1e-9
        return math.degrees(math.atan2(hn * rn + he * re, (hn * dn + he * de) / d))

    def step(self, cmd, dt: float) -> None:
        if self.heading_noise_deg:
            self.heading_deg = wrap_deg(self.heading_deg + self._rng.gauss(0.0, self.heading_noise_deg * math.sqrt(dt)))
        h = math.radians(self.heading_deg)
        self.person_vn, self.person_ve = self.walk_speed * math.cos(h), self.walk_speed * math.sin(h)
        super().step(cmd, dt)

    def face_box(self):
        if abs(self.facing_deg()) > self.visible_deg:
            return None
        return super().face_box()


class LandmarkOracle(FaceDetector):
    name = "oracle-landmarks"

    def __init__(self, world: WalkerWorld, noise_px: float = 0.5, noise_rel: float = 0.04, seed: int = 0) -> None:
        self.world, self.noise_px, self.noise_rel = world, noise_px, noise_rel
        self.rng = random.Random(seed + 1000)

    def detect(self, frame: np.ndarray) -> List[Face]:
        box = self.world.face_box()
        if box is None:
            return []
        x1, y1, x2, y2 = box
        w = x2 - x1
        sd = self.noise_px + self.noise_rel * w
        lm = [(x + self.rng.gauss(0, sd), y + self.rng.gauss(0, sd))
              for x, y in synth_landmarks((x1 + x2) / 2, (y1 + y2) / 2, w, self.world.facing_deg())]
        return [Face(x1, y1, x2, y2, 0.8, tuple(lm))]

    def info(self):
        return {"backend": self.name}


def run(angle_deg: float, k: float, preset: str = "sim", seconds: float = 60.0, seed: int = 0,
        heading_noise_deg: float = 0.0, start_m: float = 4.0, fps: float = 15.0) -> dict:
    from .cli import camera_params

    hfov, face_w = camera_params(preset)
    cfg = FollowConfig(hfov_deg=hfov, face_width_m=face_w, k_head_lateral=k, lost_timeout_s=5.0)
    # Drone at the origin facing north; walker start_m ahead, walking back toward the drone's
    # position but angle_deg off the line of sight.
    # Altitude 2 m and an 8 deg down-tilt like the sim's quack_cam (0.14 rad) flown at 2 m.
    world = WalkerWorld(hfov_deg=hfov, face_width_m=face_w, person_n=start_m, person_e=0.0,
                        altitude=2.0, cam_pitch_deg=8.0,
                        heading_deg=180.0 + angle_deg, heading_noise_deg=heading_noise_deg, seed=seed)
    pipe = FollowPipeline(LandmarkOracle(world, seed=seed), FaceFollowController(cfg),
                          sink=DryRunSink(log_every_s=1e9))
    img = np.zeros((world.height, world.width, 3), np.uint8)
    dt = 1.0 / fps
    n = int(seconds * fps)
    lock = front = 0
    timeout_s: Optional[float] = None
    for i in range(n):
        res = pipe.step(Frame(img, i, world.t), now=world.t)
        if res.control.mode == TRACKING:
            lock += 1
        if abs(world.facing_deg()) <= 30.0:
            front += 1
        if res.control.mode == TIMEOUT and timeout_s is None:
            timeout_s = world.t
            break
        world.step(res.control.command, dt)
    return {"lock_pct": 100.0 * lock / n, "front_pct": 100.0 * front / n, "timeout_s": timeout_s}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="sim")
    ap.add_argument("--k", type=float, default=0.02, help="k_head_lateral to compare against 0")
    ap.add_argument("--angles", default="0,15,25,35,45")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--heading-noise", type=float, default=3.0, help="walker heading random walk, deg/sqrt(s)")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args(argv)
    print(f"preset {args.preset}, {args.seconds:g} s, walker 0.4 m/s, heading noise {args.heading_noise:g} deg/sqrt(s), "
          f"{args.seeds} seeds; mean lock % / front % / runs that timed out")
    print(f"| leg angle | k=0 lock | k=0 front | k=0 timeouts | k={args.k:g} lock | k={args.k:g} front | k={args.k:g} timeouts |")
    print("|---|---|---|---|---|---|---|")
    for a in (float(x) for x in args.angles.split(",")):
        cells = []
        for k in (0.0, args.k):
            rs = [run(a, k, args.preset, args.seconds, s, args.heading_noise) for s in range(args.seeds)]
            to = [r["timeout_s"] for r in rs if r["timeout_s"] is not None]
            cells += [f"{np.mean([r['lock_pct'] for r in rs]):.0f}", f"{np.mean([r['front_pct'] for r in rs]):.0f}",
                      f"{len(to)}/{len(rs)}" + (f" (~{np.mean(to):.0f} s)" if to else "")]
        print(f"| {a:g} deg | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
