"""Glue: frame -> detector -> tracker -> target selector -> controller -> command sink.

One FollowPipeline.step() per camera frame. It records how long each stage took and how old the
frame was when its command went out ("frame age"), which is the number the latency budget in
docs/ARCHITECTURE.md is about.

Optional body path: pass `person_detector=` (jetson.detectors.person) and the pipeline also tracks
whole-person boxes and uses jetson.lock.LockKeeper, so it keeps following the target's body when
the face turns away and re-acquires by face. Without it, behaviour is exactly the face-only path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bridge import CommandSink, DryRunSink
from .controller import ControlOutput, FaceFollowController, TargetObservation
from .detectors.base import FaceDetector
from .face import Face
from .lock import LockKeeper
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


class FollowPipeline:
    def __init__(self, detector: FaceDetector, controller: Optional[FaceFollowController] = None,
                 tracker: Optional[IouTracker] = None, selector: Optional[TargetSelector] = None,
                 sink: Optional[CommandSink] = None, person_detector=None,
                 body_tracker: Optional[IouTracker] = None, lock: Optional[LockKeeper] = None,
                 person_every: int = 1) -> None:
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

    def step(self, frame: Frame, now: Optional[float] = None,
             altitude_m: Optional[float] = None) -> StepResult:
        """Process one frame. `now` defaults to time.monotonic(); pass a simulated clock for files/toys."""
        t0 = time.perf_counter()
        faces = self.detector.detect(frame.image)
        t1 = time.perf_counter()
        tracks = self.tracker.update(faces, frame.stamp)
        bodies: List[Face] = []
        body_tracks: List[Track] = []
        target_body, source = None, "face"
        t_person = 0.0
        if self.person_detector is None:
            target = self.selector.select(tracks, frame.width, frame.height)
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
        obs = None
        if box is not None:
            # Stamp = capture time of the frame the target was last *detected* in, so a target the
            # tracker is only coasting on goes stale in the controller after max_obs_age_s.
            obs = TargetObservation.from_box(box, frame.width, frame.height, stamp=box_stamp)
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
                          target_body, source)
