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

Walk-away end (FollowConfig.end_on_walkaway, jetson/walkaway.py):

    python -m jetson.lockeval --end-on-walkaway          # toy scenarios with the setting on
    python -m jetson.lockeval --replay-sitl sim/docs/runs/r3-turnaway-body.csv.gz

The replay feeds a sim follow log (gzipped CSV) through the current LockKeeper (hand-back rule) and
WalkAwayDetector and prints when the follow would have ended, next to the ground truth.

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
    if name == "leaves":
        # Faces the drone, then turns and walks off at 1.3 m/s for good (the walk-away end's case).
        tgt = Walker(5.0, 0.0, [(0, 0, 0, None), (4, 1.3, 0, 0.0)])
        return [tgt], 16.0
    if name == "face_visible":
        # Control: the face is always visible; both modes should be ~100 %.
        tgt = Walker(6.0, 1.0, [(0, 0, 0, None), (2, 0.3, 0.4, None), (10, -0.3, -0.3, None)])
        return [tgt], 16.0
    raise ValueError(f"unknown scenario {name!r}")


SCENARIOS = ("walk_away", "turn_around", "bystander", "leaves", "face_visible")


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
    timeout_t, timeout_reason = float("nan"), ""
    for i in range(int(round(seconds / dt))):
        res = pipe.step(Frame(image, i + 1, world.t), now=world.t)
        if res.observation is not None:
            acquired = True
        if acquired:
            frames += 1
            if res.control.mode == TIMEOUT:
                if not timeout:
                    timeout_t, timeout_reason = world.t, res.control.reason
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
            "dist_err": float(np.mean(errs)) if errs else float("nan"), "timeout_t": timeout_t,
            "walked_away": float("walked away" in timeout_reason)}


def evaluate(names: Sequence[str] = SCENARIOS, seeds: int = 5,
             config_kw: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    rows = []
    for name in names:
        for use_body in (False, True):
            cfg = FollowConfig(hfov_deg=ToyWorld().hfov_deg, **config_kw) if config_kw else None
            runs = [run(name, use_body, seed=s, config=cfg) for s in range(seeds)]
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


def handback_replay(csv_path: str, confirm_hits: int = 3, confirm_score: Optional[float] = 0.9,
                    face_score: float = 0.65, fps_hint: float = 13.0) -> Dict[str, float]:
    """Replay the face-detection pattern of a sim follow log through IouTracker + LockKeeper.

    sim/docs/runs/r3-turnaway-body.csv.gz has, per frame, how many faces YuNet found (`faces`), whether
    a body was found (`bodies`) and where the walker was facing (`walker_facing_deg`, 0 = towards
    the drone, +-180 = away). A face while the walker faces away (|facing| > 90) is a false face on
    the back of the head. Geometry is fixed (one face box inside one body box), detections score
    `face_score` (unknown in the log), so this measures only the *timing* rule: how many frames the
    lock would steer by a back-of-head face, and how much a genuine hand-back is delayed.
    Returns frame counts: false_face (steering by a back-of-head face), genuine_face, handbacks,
    false_handbacks (body -> face switches while facing away).
    """
    import csv
    import gzip

    from .lock import LockKeeper
    from .tracker import IouTracker

    rows = read_run_log(csv_path)
    out = {"frames": 0, "false_face": 0, "genuine_face": 0, "handbacks": 0, "false_handbacks": 0}
    prev = ""
    for r, source in zip(rows, replay_lock_sources(rows, confirm_hits, confirm_score, face_score)):
        away = r["walker_facing_deg"] not in ("", None) and abs(float(r["walker_facing_deg"])) > 90.0
        out["frames"] += 1
        if source == "face":
            out["false_face" if away else "genuine_face"] += 1
            if prev == "body":
                out["handbacks"] += 1
                out["false_handbacks"] += int(away)
        prev = source
    return out


def read_run_log(csv_path: str) -> List[Dict[str, str]]:
    """Rows of a sim follow log (sim/docs/runs/*.csv or *.csv.gz)."""
    import csv
    import gzip

    opener = gzip.open if csv_path.endswith(".gz") else open
    with opener(csv_path, "rt") as fh:
        return list(csv.DictReader(fh))


def replay_lock_sources(rows: Sequence[Dict[str, str]], confirm_hits: int = 3,
                        confirm_score: Optional[float] = 0.9, face_score: float = 0.65) -> List[str]:
    """What the current LockKeeper would steer by ("face", "body", "") on each logged frame, from the
    log's detection counts only (`faces`, `bodies`; fixed geometry: one face box inside one body
    box, scores unknown -> `face_score`). Logs without a `bodies` column are face-only runs."""
    from .lock import LockKeeper
    from .tracker import IouTracker

    face_box, body_box = (232.0, 105.0, 268.0, 150.0), (200.0, 100.0, 300.0, 400.0)
    ft = IouTracker()
    bt = IouTracker(high_score=0.5, low_score=0.2, max_age_s=1.0, center_gate=0.0)
    keeper = LockKeeper(confirm_hits=confirm_hits, confirm_score=confirm_score)
    out = []
    for r in rows:
        t = float(r["t"])
        faces = [Face(*face_box, face_score)] if int(float(r["faces"] or 0)) > 0 else []
        bodies = [Face(*body_box, 0.8)] if int(float(r.get("bodies") or 0)) > 0 else []
        out.append(keeper.update(ft.update(faces, t), bt.update(bodies, t), 640, 480, t).source)
    return out


def _true_recession(rows: Sequence[Dict[str, str]], i: int, half: int = 7) -> Optional[float]:
    """Ground truth: walker speed along the drone->walker line (+ = away), central difference."""
    try:
        a, b, r = rows[max(0, i - half)], rows[min(len(rows) - 1, i + half)], rows[i]
        dn, de = float(r["walker_n"]) - float(r["px4_n"]), float(r["walker_e"]) - float(r["px4_e"])
        span = float(b["t"]) - float(a["t"])
        L = math.hypot(dn, de)
        if span <= 0 or L < 1e-6:
            return None
        vn = (float(b["walker_n"]) - float(a["walker_n"])) / span
        ve = (float(b["walker_e"]) - float(a["walker_e"])) / span
        return (vn * dn + ve * de) / L
    except (KeyError, ValueError):
        return None


def walkaway_replay(csv_path: str, seconds: float = 3.0, speed_m_s: float = 0.8, margin_m: float = 1.0,
                    target_distance_m: Optional[float] = None, lock: str = "replay") -> Dict[str, object]:
    """When would `end_on_walkaway` have ended this SITL follow? (open loop: the log keeps going.)

    Inputs per frame are what the Jetson had: the steering source (lock="replay": recomputed by the
    current LockKeeper with the hand-back confirmation, see replay_lock_sources; lock="logged": the
    source the round-3 code actually used), the controller's distance `est_dist_m` and the forward
    command sent `sent_fwd`. After each trigger the detector is reset, as if a new session started,
    so every walk-away in the log is visible. Ground truth (never fed to the detector): the walker's
    facing (`walker_facing_deg`, |deg| > 90 = back turned) and speed away from the drone.

    Returns {"triggers": [{t, speed, dist, true_dist, true_recede, facing}], "false_triggers": n
    (over the detector's window the walker faced the drone most of the time or truly receded at
    < 0.3 m/s on average), "episodes": [{start, end, peak_recede,
    detected_at}] (facing away and truly receding >= speed_m_s for >= seconds), "target_distance_m"}.
    """
    import json
    import os

    from .walkaway import WalkAwayDetector

    rows = read_run_log(csv_path)
    if target_distance_m is None:
        target_distance_m = 4.0
        side = csv_path[:-3] if csv_path.endswith(".gz") else csv_path
        side = os.path.splitext(side)[0] + ".json"
        if os.path.exists(side):
            with open(side) as fh:
                target_distance_m = float(json.load(fh).get("config", {}).get("target_distance_m", 4.0))
    if lock == "replay":
        sources = replay_lock_sources(rows)
    elif lock == "logged":
        sources = [r.get("source") or ("face" if r.get("est_dist_m") else "") for r in rows]
    else:
        raise ValueError("lock must be 'replay' or 'logged'")
    det = WalkAwayDetector(seconds=seconds, speed_m_s=speed_m_s, min_distance_m=target_distance_m + margin_m)
    triggers, false_triggers = [], 0
    episodes: List[Dict[str, object]] = []
    ep: Optional[Dict[str, object]] = None
    for i, (r, src) in enumerate(zip(rows, sources)):
        t = float(r["t"])
        est = float(r["est_dist_m"]) if r.get("est_dist_m") not in ("", None) else None
        dist = est if src in ("face", "body") else None
        res = det.update(t, float(r.get("sent_fwd") or 0.0), dist, face_visible=src == "face")
        facing = float(r["walker_facing_deg"]) if r.get("walker_facing_deg") not in ("", None) else None
        rec = _true_recession(rows, i)
        away = facing is not None and abs(facing) > 90.0
        if away and rec is not None and rec >= speed_m_s:
            if ep is None:
                ep = {"start": t, "end": t, "peak_recede": rec, "detected_at": None}
            ep["end"], ep["peak_recede"] = t, max(float(ep["peak_recede"]), rec)
        elif ep is not None:
            if float(ep["end"]) - float(ep["start"]) >= seconds:
                episodes.append(ep)
            ep = None
        if res.triggered:
            triggers.append({"t": t, "speed": res.speed_m_s, "dist": res.distance_m,
                             "true_dist": float(r["true_dist_m"]) if r.get("true_dist_m") else None,
                             "true_recede": rec, "facing": facing})
            # False = the walker was not walking away during the window that fired it (mean true
            # speed away < 0.3 m/s, or facing the drone for most of it).
            win = [j for j in range(i + 1) if float(rows[j]["t"]) >= t - seconds]
            recs = [x for x in (_true_recession(rows, j) for j in win) if x is not None]
            aways = [abs(float(rows[j]["walker_facing_deg"])) > 90.0 for j in win
                     if rows[j].get("walker_facing_deg") not in ("", None)]
            if not recs or sum(recs) / len(recs) < 0.3 or not aways or sum(aways) < 0.5 * len(aways):
                false_triggers += 1
            if ep is not None and ep["detected_at"] is None:
                ep["detected_at"] = t
            det.reset()
    if ep is not None and float(ep["end"]) - float(ep["start"]) >= seconds:
        episodes.append(ep)
    body_lock = any(int(float(r.get("bodies") or 0)) > 0 for r in rows)
    return {"triggers": triggers, "false_triggers": false_triggers, "episodes": episodes,
            "target_distance_m": target_distance_m, "frames": len(rows), "body_lock": body_lock}


def format_walkaway(path: str, res: Dict[str, object]) -> str:
    lines = [f"{path}: {res['frames']} frames, target {res['target_distance_m']:g} m, "
             f"{len(res['triggers'])} trigger(s), {res['false_triggers']} false"]
    for tr in res["triggers"]:
        rec = "n/a" if tr["true_recede"] is None else f"{tr['true_recede']:.2f}"
        lines.append(f"  ends at t={tr['t']:6.1f} s: est {tr['speed']:.2f} m/s at {tr['dist']:.1f} m | truth: "
                     f"{rec} m/s, {tr['true_dist']:.1f} m, facing {tr['facing']:+.0f} deg")
    for ep in res["episodes"]:
        det = ep["detected_at"]
        if det is not None:
            verdict = f"detected after {det - ep['start']:.1f} s"
        elif not res.get("body_lock", True):
            verdict = "n/a (face-only run: the lost timeout ends it)"
        else:
            verdict = "MISSED"
        lines.append(f"  true walk-away {ep['start']:6.1f}-{ep['end']:6.1f} s (peak {ep['peak_recede']:.2f} m/s): "
                     + verdict)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", choices=SCENARIOS, action="append", help="default: all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--clip", default=None, help="recorded clip: real detectors, target face painted out")
    ap.add_argument("--mask-from", type=int, default=30)
    ap.add_argument("--mask-to", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--person", default="yolox", choices=("yolox", "hog"))
    ap.add_argument("--end-on-walkaway", action="store_true", help="toy scenarios with the walk-away end on")
    ap.add_argument("--replay-sitl", nargs="+", metavar="CSV", default=None,
                    help="sim follow logs (csv/csv.gz): when would --end-on-walkaway have ended them?")
    ap.add_argument("--walkaway-seconds", type=float, default=FollowConfig.walkaway_s)
    ap.add_argument("--walkaway-speed", type=float, default=FollowConfig.walkaway_speed_m_s)
    ap.add_argument("--walkaway-margin", type=float, default=FollowConfig.walkaway_margin_m)
    args = ap.parse_args(argv)
    if args.replay_sitl:
        for path in args.replay_sitl:
            print(format_walkaway(path, walkaway_replay(path, args.walkaway_seconds, args.walkaway_speed,
                                                        args.walkaway_margin)))
        return 0
    if args.clip:
        res = clip_eval(args.clip, args.mask_from, args.mask_to, args.max_frames, args.person)
        for mode, m in res.items():
            print(f"{mode:<10} masked frames {m['frames']:>4}  on_target {m['on_target']:>4.0%}  "
                  f"on_wrong {m['on_wrong']:>4.0%}  lost {m['lost']:>4.0%}")
        return 0
    kw = None
    if args.end_on_walkaway:
        kw = {"end_on_walkaway": True, "walkaway_s": args.walkaway_seconds,
              "walkaway_speed_m_s": args.walkaway_speed, "walkaway_margin_m": args.walkaway_margin}
    rows = evaluate(args.scenario or SCENARIOS, args.seeds, kw)
    print(format_rows(rows))
    if kw:
        print("\nwalk-away end: share of runs ended by it, mean time it fired")
        for r in rows:
            print(f"  {r['scenario']:<13} {r['mode']:<10} {r['walked_away']:>5.0%}  "
                  + ("-" if math.isnan(r["timeout_t"]) or not r["walked_away"] else f"t={r['timeout_t']:.1f} s"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
