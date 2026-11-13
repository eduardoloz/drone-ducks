"""Measure lock retention: does the follower stay on the consenting person when their face disappears?

    python -m jetson.lockeval                  # all scenarios, face-only vs face+body, 5 seeds each
    python -m jetson.lockeval --seeds 20 --scenario walk_away

Closed loop in a kinematic toy world (jetson/toysim.py): the real FollowPipeline, tracker,
LockKeeper and controller drive a simulated drone; only the detectors are replaced by oracles
that return the true boxes, with random misses and pixel noise. People have a facing direction:
their face is only "detectable" when they face the camera within `face_cone_deg`, so walking
away or turning side-on hides it, exactly the case face-only following cannot handle.

Per run we report, over all frames after the target was first acquired:
  on_target  steering by the target person (face or body)
  on_wrong   steering by a bystander (the failure we must never have)
  lost       steering by nothing (hovering)
  timeout    the controller gave up (lost too long -> land/rtl/hold)
  dist_err   mean |distance - standoff| while on target, metres

Recorded clip (real detectors, open loop):

    python -m jetson.lockeval --clip jetson/media/crew_pan.mp4 --mask-from 30 --mask-to 100

runs YuNet face-only once to find the followed person, then replays the clip with that person's
face painted out for frames [mask-from, mask-to) (a stand-in for "turned their head away") and
counts, inside that window, how often each mode steers by the target vs by somebody else.

Limitations: no occlusion between people, perfect association geometry (boxes are exact up to
noise), no camera tilt during acceleration. It shows the *logic* works; recorded clips and the
bench test are still needed for the detector side.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bridge import DryRunSink
from .controller import TIMEOUT, FaceFollowController, FollowConfig
from .detectors.base import FaceDetector
from .detectors.person import PersonDetector
from .face import Box, Face
from .pipeline import FollowPipeline
from .sources import Frame
from .toysim import ToyWorld, wrap_deg


@dataclass
class Walker:
    """A person on flat ground. Waypoint script: list of (t_start, vn, ve, facing_deg or None).

    facing None = faces the drone. facing is NED degrees (0 = north, 90 = east)."""

    n: float
    e: float
    script: List[Tuple[float, float, float, Optional[float]]] = field(default_factory=list)
    face_height_m: float = 1.6
    height_m: float = 1.75
    body_width_m: float = 0.5
    vn: float = 0.0
    ve: float = 0.0
    facing: Optional[float] = None

    def update(self, t: float, dt: float) -> None:
        for t0, vn, ve, facing in self.script:
            if t >= t0:
                self.vn, self.ve, self.facing = vn, ve, facing
        self.n += self.vn * dt
        self.e += self.ve * dt


class LockWorld(ToyWorld):
    """ToyWorld with several walkers that have a facing direction and a body box."""

    def __init__(self, walkers: Sequence[Walker], face_cone_deg: float = 70.0, **kw) -> None:
        super().__init__(**kw)
        self.walkers = list(walkers)
        self.face_cone_deg = face_cone_deg
        self._sync_target()

    def _sync_target(self) -> None:
        tgt = self.walkers[0]
        self.person_n, self.person_e, self.face_height_m = tgt.n, tgt.e, tgt.face_height_m

    def step(self, cmd, dt: float) -> None:
        for w in self.walkers:
            w.update(self.t, dt)
        self._sync_target()
        self.person_vn = self.person_ve = 0.0  # walkers move themselves
        super().step(cmd, dt)

    def _project(self, n: float, e: float, h: float) -> Optional[Tuple[float, float, float]]:
        """World point -> (u, v, forward depth) in pixels, or None if behind the camera."""
        dn, de = n - self.north, e - self.east
        psi = math.radians(self.yaw_deg)
        fwd, right, down = dn * math.cos(psi) + de * math.sin(psi), -dn * math.sin(psi) + de * math.cos(psi), \
            self.altitude - h
        th = math.radians(self.cam_pitch_deg)
        fwd, down = fwd * math.cos(th) + down * math.sin(th), -fwd * math.sin(th) + down * math.cos(th)
        if fwd <= 0.2:
            return None
        f = self.focal_px
        return self.width / 2.0 + f * right / fwd, self.height / 2.0 + f * down / fwd, fwd

    def faces_camera(self, w: Walker) -> bool:
        if w.facing is None:
            return True
        to_drone = math.degrees(math.atan2(self.east - w.e, self.north - w.n))
        return abs(wrap_deg(to_drone - w.facing)) <= self.face_cone_deg

    def walker_face(self, w: Walker) -> Optional[Box]:
        if not self.faces_camera(w):
            return None
        p = self._project(w.n, w.e, w.face_height_m)
        if p is None:
            return None
        u, v, d = p
        fw = self.focal_px * self.face_width_m / d
        fh = 1.25 * fw
        if fw < self.min_face_px or not (0 <= u <= self.width and 0 <= v <= self.height):
            return None
        return (u - fw / 2, v - fh / 2, u + fw / 2, v + fh / 2)

    def walker_body(self, w: Walker) -> Optional[Box]:
        top, bot = self._project(w.n, w.e, w.height_m), self._project(w.n, w.e, 0.0)
        if top is None or bot is None:
            return None
        bw = self.focal_px * w.body_width_m / top[2]
        u = top[0]
        x1, x2 = u - bw / 2, u + bw / 2
        y1, y2 = max(0.0, top[1]), min(float(self.height), bot[1])
        if bw < 6 or not (0 <= u <= self.width) or y2 - y1 < 0.25 * (bot[1] - top[1]) or y2 <= y1:
            return None
        return (max(0.0, x1), y1, min(float(self.width), x2), y2)

    def face_box(self) -> Optional[Box]:  # target's face, for the ToyWorld API
        return self.walker_face(self.walkers[0])


class _Oracle:
    def __init__(self, world: LockWorld, kind: str, dropout: float, noise_px: float, seed: int) -> None:
        self.world, self.kind, self.dropout, self.noise = world, kind, dropout, noise_px
        self.rng = random.Random(seed)

    def detect(self, frame: np.ndarray) -> List[Face]:
        out = []
        for w in self.world.walkers:
            b = self.world.walker_face(w) if self.kind == "face" else self.world.walker_body(w)
            if b is None or self.rng.random() < self.dropout:
                continue
            s = self.noise * (b[2] - b[0]) / 40.0  # noise scales a little with size
            n = [self.rng.gauss(0.0, self.noise + s) for _ in range(4)]
            out.append(Face(b[0] + n[0], b[1] + n[1], b[2] + n[2], b[3] + n[3], 0.9))
        return out

    def info(self):
        return {"backend": f"oracle-{self.kind}"}

    def close(self) -> None:
        pass


class OracleFaces(_Oracle, FaceDetector):
    name = "oracle-face"

    def __init__(self, world, dropout=0.1, noise_px=1.0, seed=0):
        _Oracle.__init__(self, world, "face", dropout, noise_px, seed)


class OraclePeople(_Oracle, PersonDetector):
    name = "oracle-person"

    def __init__(self, world, dropout=0.05, noise_px=2.0, seed=1):
        _Oracle.__init__(self, world, "body", dropout, noise_px, seed)


# ----------------------------------------------------------------------------- scenarios
def scenario(name: str) -> Tuple[List[Walker], float]:
    """(walkers, seconds). walkers[0] is always the consenting target."""
    if name == "walk_away":
        # Faces the drone, turns and walks away for 8 s, turns back, then walks off side-on.
        tgt = Walker(6.0, 0.0, [(0, 0, 0, None), (4, 0.8, 0, 0.0), (12, 0, 0, None), (16, 0, 0.6, 90.0)])
        return [tgt], 24.0
    if name == "turn_around":
        # Stands still but turns their back to the drone for 6 s, twice.
        tgt = Walker(5.0, 0.5, [(0, 0, 0, None), (3, 0, 0, 0.0), (9, 0, 0, None), (12, 0, 0, 180.0 + 90.0),
                                (18, 0, 0, None)])
        return [tgt], 22.0
    if name == "bystander":
        # Target walks away (back turned) while a bystander facing the camera crosses nearby.
        tgt = Walker(6.0, -0.5, [(0, 0, 0, None), (3, 0.6, 0, 0.0), (13, 0, 0, None)])
        by = Walker(5.0, 4.0, [(0, 0, 0, None), (2, 0, -0.7, None)])
        return [tgt, by], 18.0
    if name == "face_visible":
        # Control: the face is always visible; both modes should be ~100 %.
        tgt = Walker(6.0, 1.0, [(0, 0, 0, None), (2, 0.3, 0.4, None), (10, -0.3, -0.3, None)])
        return [tgt], 16.0
    raise ValueError(f"unknown scenario {name!r}")


SCENARIOS = ("walk_away", "turn_around", "bystander", "face_visible")


def _who(world: LockWorld, box: Box) -> int:
    """Index of the walker the steering box belongs to (-1 = nobody): the walker whose (face or)
    body box contains the steering box centre; closest body centre wins ties."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    best, best_d = -1, float("inf")
    for i, w in enumerate(world.walkers):
        b = world.walker_body(w)
        if b is None:
            continue
        pad = 0.15 * (b[2] - b[0])
        if b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3]:
            d = abs(cx - (b[0] + b[2]) / 2.0)
            if d < best_d:
                best, best_d = i, d
    return best


def run(name: str, use_body: bool, seed: int = 0, dt: float = 1 / 15,
        config: Optional[FollowConfig] = None) -> Dict[str, float]:
    walkers, seconds = scenario(name)
    world = LockWorld(walkers, altitude=2.5, cam_pitch_deg=10.0)
    cfg = config or FollowConfig(hfov_deg=world.hfov_deg)
    pipe = FollowPipeline(OracleFaces(world, seed=seed), FaceFollowController(cfg), sink=DryRunSink(log_every_s=1e9),
                          person_detector=OraclePeople(world, seed=seed + 1000) if use_body else None)
    image = np.zeros((world.height, world.width, 3), np.uint8)
    counts = {"on_target": 0, "on_wrong": 0, "lost": 0}
    frames, acquired, timeout, body_frames, errs = 0, False, False, 0, []
    for i in range(int(round(seconds / dt))):
        res = pipe.step(Frame(image, i + 1, world.t), now=world.t)
        if res.observation is not None:
            acquired = True
        if acquired:
            frames += 1
            if res.control.mode == TIMEOUT:
                timeout = True
            if res.observation is None or res.control.mode != "TRACKING":
                counts["lost"] += 1
            else:
                who = _who(world, (res.observation.cx * world.width - 1, res.observation.cy * world.height - 1,
                                   res.observation.cx * world.width + 1, res.observation.cy * world.height + 1))
                if who == 0:
                    counts["on_target"] += 1
                    errs.append(abs(world.distance_m - cfg.target_distance_m))
                    body_frames += res.target_source == "body"
                else:
                    counts["on_wrong"] += 1
        world.step(res.control.command, dt)
    n = max(1, frames)
    return {"on_target": counts["on_target"] / n, "on_wrong": counts["on_wrong"] / n, "lost": counts["lost"] / n,
            "timeout": float(timeout), "body_share": body_frames / max(1, counts["on_target"]),
            "dist_err": float(np.mean(errs)) if errs else float("nan")}


def evaluate(names: Sequence[str] = SCENARIOS, seeds: int = 5) -> List[Dict[str, object]]:
    rows = []
    for name in names:
        for use_body in (False, True):
            runs = [run(name, use_body, seed=s) for s in range(seeds)]
            row: Dict[str, object] = {"scenario": name, "mode": "face+body" if use_body else "face-only"}
            for k in runs[0]:
                vals = [r[k] for r in runs if not math.isnan(r[k])]
                row[k] = float(np.mean(vals)) if vals else float("nan")
            rows.append(row)
    return rows


def format_rows(rows: List[Dict[str, object]]) -> str:
    hdr = f"{'scenario':<13} {'mode':<10} {'on_target':>9} {'on_wrong':>8} {'lost':>6} {'timeout':>7} " \
          f"{'via_body':>8} {'dist_err':>8}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['scenario']:<13} {r['mode']:<10} {r['on_target']:>9.0%} {r['on_wrong']:>8.0%} "
                     f"{r['lost']:>6.0%} {r['timeout']:>7.0%} {r['body_share']:>8.0%} {r['dist_err']:>7.2f}m")
    return "\n".join(lines)


def _paint_out(img: np.ndarray, box: Box, grow: float = 1.4) -> None:
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * grow, (y2 - y1) * grow
    H, W = img.shape[:2]
    a, b = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    c, d = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    if c > a and d > b:
        img[b:d, a:c] = img[b:d, a:c].reshape(-1, 3).mean(0).astype(np.uint8)


def clip_eval(path: str, mask_from: int = 30, mask_to: int = 100, max_frames: int = 0,
              person: str = "yolox", face_backend: str = "yunet") -> Dict[str, Dict[str, float]]:
    """Real detectors on a recorded clip; see the module docstring. Returns {mode: metrics}."""
    import cv2

    from .detectors import create_detector
    from .detectors.person import create_person_detector

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(img)
    cap.release()
    if not frames:
        raise ValueError(f"could not read {path}")
    fps = 15.0
    det = create_detector(face_backend)
    ref = FollowPipeline(det, sink=DryRunSink(log_every_s=1e9))
    target_boxes: List[Optional[Box]] = []
    target_id = None
    for i, img in enumerate(frames):
        r = ref.step(Frame(img, i + 1, i / fps), now=i / fps)
        if target_id is None and r.target is not None and i < mask_from:
            target_id = r.target.track_id
        target_boxes.append(r.target.box if r.target is not None and r.target.track_id == target_id else None)
    pdet = create_person_detector(person)
    results = {}
    for use_body in (False, True):
        pipe = FollowPipeline(create_detector(face_backend), sink=DryRunSink(log_every_s=1e9),
                              person_detector=pdet if use_body else None)
        on, wrong, lost, n, last_ref = 0, 0, 0, 0, None
        for i, img in enumerate(frames):
            ref_box = target_boxes[i] or last_ref
            if target_boxes[i] is not None:
                last_ref = target_boxes[i]
            im = img
            if mask_from <= i < mask_to and ref_box is not None:
                im = img.copy()
                _paint_out(im, ref_box)
            r = pipe.step(Frame(im, i + 1, i / fps), now=i / fps)
            if not (mask_from <= i < mask_to) or ref_box is None:
                continue
            n += 1
            if r.observation is None or r.control.mode != "TRACKING":
                lost += 1
                continue
            ox, oy = r.observation.cx * img.shape[1], r.observation.cy * img.shape[0]
            fw = ref_box[2] - ref_box[0]
            if abs(ox - (ref_box[0] + ref_box[2]) / 2) <= fw and abs(oy - (ref_box[1] + ref_box[3]) / 2) <= 1.5 * fw:
                on += 1
            else:
                wrong += 1
        k = max(1, n)
        results["face+body" if use_body else "face-only"] = {"frames": n, "on_target": on / k,
                                                             "on_wrong": wrong / k, "lost": lost / k}
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", choices=SCENARIOS, action="append", help="default: all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--clip", default=None, help="recorded clip: real detectors, target face painted out")
    ap.add_argument("--mask-from", type=int, default=30)
    ap.add_argument("--mask-to", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--person", default="yolox", choices=("yolox", "hog"))
    args = ap.parse_args(argv)
    if args.clip:
        res = clip_eval(args.clip, args.mask_from, args.mask_to, args.max_frames, args.person)
        for mode, m in res.items():
            print(f"{mode:<10} masked frames {m['frames']:>4}  on_target {m['on_target']:>4.0%}  "
                  f"on_wrong {m['on_wrong']:>4.0%}  lost {m['lost']:>4.0%}")
        return 0
    print(format_rows(evaluate(args.scenario or SCENARIOS, args.seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
