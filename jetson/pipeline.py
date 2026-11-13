"""Glue: frame -> detector -> tracker -> target selector -> controller -> command sink.

One FollowPipeline.step() per camera frame. It records how long each stage took and how old the
frame was when its command went out ("frame age"), which is the number the latency budget in
docs/ARCHITECTURE.md is about.

Optional body path: pass `person_detector=` (jetson.detectors.person) and the pipeline also tracks
whole-person boxes and uses jetson.lock.LockKeeper, so it keeps following the target's body when
the face turns away and re-acquires by face. Without it, behaviour is exactly the face-only path.

Both paths go through jetson.relock.RelockGate: once the target has been lost (no fresh detection
for `lost_after_s`), the controller only gets observations again after a *confirmed* re-lock
(N consecutive detections, or one high-score one). The followed face's landmarks also give a head
yaw estimate (jetson.headpose) that the controller can use to stay in front (k_head_lateral).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import geometry
from .bridge import CommandSink, DryRunSink
from .controller import ControlOutput, FaceFollowController, TargetObservation
from .detectors.base import FaceDetector
from .face import Face
from .headpose import head_yaw_deg
from .lock import LockKeeper
from .relock import RelockGate
from .sources import Frame
from .tracker import IouTracker, TargetSelector, Track


@dataclass
class StepResult:
    frame: Frame
    faces: List[Face]
    tracks: List[Track]
    target: Optional[Track]
    observation: Optional[TargetObservation]
    control: ControlOutput
    timings_ms: Dict[str, float] = field(default_factory=dict)
    # body path (empty / None / "face" when no person detector is configured)
    bodies: List[Face] = field(default_factory=list)
    body_tracks: List[Track] = field(default_factory=list)
    target_body: Optional[Track] = None
    target_source: str = "face"   # "face", "body" (steering by the body, face not visible) or ""
    gated: bool = False           # a target was there but RelockGate withheld it (re-lock not confirmed)


class FollowPipeline:
    def __init__(self, detector: FaceDetector, controller: Optional[FaceFollowController] = None,
                 tracker: Optional[IouTracker] = None, selector: Optional[TargetSelector] = None,
                 sink: Optional[CommandSink] = None, person_detector=None,
                 body_tracker: Optional[IouTracker] = None, lock: Optional[LockKeeper] = None,
                 person_every: int = 1, relock: Optional[RelockGate] = None,
                 relock_holdoff_s: float = 0.0, ego_compensation: bool = True) -> None:
        self.detector = detector
        self.controller = controller or FaceFollowController()
        self.tracker = tracker or IouTracker()
        self.selector = selector or TargetSelector()
        self.sink = sink or DryRunSink()
        self.person_detector = person_detector
        # Bodies are big, so plain IoU matching works; min_hits=2 kills one-frame false positives.
        # center_gate=0: the small-face centre-distance fallback let a coasting body track jump to
        # the next person in a group (seen on crew_pan.mp4), an identity swap we must not have.
        self.body_tracker = body_tracker or IouTracker(high_score=0.5, low_score=0.2, max_age_s=1.0,
                                                        center_gate=0.0)
        self.lock = lock or LockKeeper(self.selector)
        self.lock.selector = self.selector
        if self.lock.reacquire_holdoff_s is None:
            # After losing the consenting person, don't switch to anyone else before the controller
            # has given up (TIMEOUT latches land/rtl/hold), unless it is them again (via their body).
            self.lock.reacquire_holdoff_s = self.controller.config.lost_timeout_s
        self.person_every = max(1, int(person_every))  # run the person detector every Nth frame
        self._n = 0
        # Confirmed re-lock after a loss (both paths). RelockGate(confirm_hits=1, confirm_score=None)
        # restores the round-2 behaviour.
        cfg = self.controller.config
        self.relock = relock if relock is not None else RelockGate(lost_after_s=cfg.max_obs_age_s)
        if self.relock.max_distance_m is None and relock is None:
            self.relock.max_distance_m = cfg.target_distance_m + 1.5
        # Face-only path: after losing the target, adopt no *new* face track for this long. Default 0:
        # in face-only mode the same person turning back round is always a new track (see
        # docs/ARCHITECTURE.md "Re-lock"), so a holdoff = lost_timeout_s would mean "never re-lock".
        self.relock_holdoff_s = relock_holdoff_s
        self._face_lost_at: Optional[float] = None
        self._last_target_id: Optional[int] = None
        # Ego-rotation compensation: shift tracks by the drone's own yaw between frames (measured
        # yaw rate passed to step() or the sink's yaw_rate_deg_s, else the last commanded yaw rate).
        self.ego_compensation = ego_compensation
        self.ego_tau_s = 0.35   # commanded -> actual yaw-rate lag model (PX4-ish first order)
        self._prev_stamp: Optional[float] = None
        self._yaw_rate_est = 0.0

    def _ego_rotate(self, frame: Frame, yaw_rate_deg_s: Optional[float]) -> None:
        prev, self._prev_stamp = self._prev_stamp, frame.stamp
        if not self.ego_compensation or prev is None:
            return
        dt = min(max(0.0, frame.stamp - prev), 0.5)
        rate = yaw_rate_deg_s
        if rate is None:
            rate = getattr(self.sink, "yaw_rate_deg_s", None)
        if rate is None:
            # No measurement: the last command through a first-order lag (the drone is slower
            # than the command; raw commands over-rotate the prediction at the start of a turn).
            a = dt / (self.ego_tau_s + dt) if self.ego_tau_s > 0 else 1.0
            self._yaw_rate_est += a * (self.controller.last_command.yawspeed_deg_s - self._yaw_rate_est)
            rate = self._yaw_rate_est
        d_yaw = math.radians(rate) * dt
        if d_yaw:
            f = (frame.width / 2.0) / math.tan(math.radians(self.controller.config.hfov_deg) / 2.0)
            self.tracker.compensate_rotation(d_yaw, f, frame.width / 2.0)
            self.body_tracker.compensate_rotation(d_yaw, f, frame.width / 2.0)

    def step(self, frame: Frame, now: Optional[float] = None,
             altitude_m: Optional[float] = None, yaw_rate_deg_s: Optional[float] = None) -> StepResult:
        """Process one frame. `now` defaults to time.monotonic(); pass a simulated clock for files/toys.
        altitude_m / yaw_rate_deg_s: telemetry if the caller has it (else taken from the sink)."""
        t0 = time.perf_counter()
        faces = self.detector.detect(frame.image)
        t1 = time.perf_counter()
        self._ego_rotate(frame, yaw_rate_deg_s)
        tracks = self.tracker.update(faces, frame.stamp)
        bodies: List[Face] = []
        body_tracks: List[Track] = []
        target_body, source = None, "face"
        t_person = 0.0
        if self.person_detector is None:
            target = self._select_face(tracks, frame)
            box = target.box if target is not None else None
            box_stamp = target.last_seen if target is not None else None
        else:
            self._n += 1
            if (self._n - 1) % self.person_every == 0:
                tp = time.perf_counter()
                bodies = self.person_detector.detect(frame.image)
                t_person = time.perf_counter() - tp
                body_tracks = self.body_tracker.update(bodies, frame.stamp)
            else:
                body_tracks = list(self.body_tracker.tracks)
            st = self.lock.update(tracks, body_tracks, frame.width, frame.height, frame.stamp)
            target, target_body, source = st.face, st.body, st.source
            box, box_stamp = st.box, st.stamp
            if source == "body":
                target = None  # the HUD/blur treat every visible face as a bystander
        gated = False
        if box is not None:
            src_track = target if source == "face" else target_body
            key = (source, src_track.track_id if src_track is not None else None)
            fresh = box_stamp is not None and box_stamp >= frame.stamp - 1e-9
            score = src_track.score if src_track is not None else 0.0
            c = self.controller.config
            bw = (box[2] - box[0]) / float(frame.width)
            dist = geometry.distance_from_width(bw, c.hfov_deg, c.face_width_m) if bw > 0 else None
            if not self.relock.update(key, fresh, score, frame.stamp, dist):
                # Not confirmed yet: not the target, so the HUD and --out blur treat it as a bystander.
                box, box_stamp, gated, target = None, None, True, None
        else:
            self.relock.update(None, False, 0.0, frame.stamp)
        obs = None
        if box is not None:
            # Stamp = capture time of the frame the target was last *detected* in, so a target the
            # tracker is only coasting on goes stale in the controller after max_obs_age_s.
            yaw = None
            if source == "face" and target is not None and target.last_seen >= frame.stamp - 1e-9:
                yaw = head_yaw_deg(target.landmarks, face_width_px=target.w)
            obs = TargetObservation.from_box(box, frame.width, frame.height, stamp=box_stamp, head_yaw_deg=yaw)
        t2 = time.perf_counter()
        now = time.monotonic() if now is None else now
        alt = altitude_m if altitude_m is not None else self.sink.altitude_m
        out = self.controller.step(obs, now, altitude_m=alt)
        self.sink.submit(out)
        t3 = time.perf_counter()
        timings = {"detect": (t1 - t0) * 1e3, "track": (t2 - t1 - t_person) * 1e3, "control": (t3 - t2) * 1e3,
                   "total": (t3 - t0) * 1e3, "frame_age": (now - frame.stamp) * 1e3}
        if self.person_detector is not None:
            timings["person"] = t_person * 1e3
        if box is None:
            source = ""
        return StepResult(frame, faces, tracks, target, obs, out, timings, bodies, body_tracks,
                          target_body, source, gated)

    def _select_face(self, tracks: List[Track], frame: Frame) -> Optional[Track]:
        """Face-only target selection, with the optional new-face holdoff after a loss."""
        prev = self._last_target_id
        target = self.selector.select(tracks, frame.width, frame.height)
        if target is None:
            if prev is not None and self._face_lost_at is None:
                self._face_lost_at = frame.stamp
            return None
        if (self.relock_holdoff_s > 0 and self._face_lost_at is not None and target.track_id != prev
                and frame.stamp - self._face_lost_at < self.relock_holdoff_s):
            self.selector.target_id = prev  # keep waiting for the old track id; adopt nobody else
            return None
        self._face_lost_at = None
        self._last_target_id = target.track_id
        return target
