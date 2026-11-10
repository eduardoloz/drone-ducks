"""Glue: frame -> detector -> tracker -> target selector -> controller -> command sink.

One FollowPipeline.step() per camera frame. It records how long each stage took and how old the
frame was when its command went out ("frame age"), which is the number the latency budget in
docs/ARCHITECTURE.md is about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bridge import CommandSink, DryRunSink
from .controller import ControlOutput, FaceFollowController, TargetObservation
from .detectors.base import FaceDetector
from .face import Face
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


class FollowPipeline:
    def __init__(self, detector: FaceDetector, controller: Optional[FaceFollowController] = None,
                 tracker: Optional[IouTracker] = None, selector: Optional[TargetSelector] = None,
                 sink: Optional[CommandSink] = None) -> None:
        self.detector = detector
        self.controller = controller or FaceFollowController()
        self.tracker = tracker or IouTracker()
        self.selector = selector or TargetSelector()
        self.sink = sink or DryRunSink()

    def step(self, frame: Frame, now: Optional[float] = None,
             altitude_m: Optional[float] = None) -> StepResult:
        """Process one frame. `now` defaults to time.monotonic(); pass a simulated clock for files/toys."""
        t0 = time.perf_counter()
        faces = self.detector.detect(frame.image)
        t1 = time.perf_counter()
        tracks = self.tracker.update(faces, frame.stamp)
        target = self.selector.select(tracks, frame.width, frame.height)
        obs = None
        if target is not None:
            # Stamp = capture time of the frame the target was last *detected* in, so a target the
            # tracker is only coasting on goes stale in the controller after max_obs_age_s.
            obs = TargetObservation.from_box(target.box, frame.width, frame.height, stamp=target.last_seen)
        t2 = time.perf_counter()
        now = time.monotonic() if now is None else now
        alt = altitude_m if altitude_m is not None else self.sink.altitude_m
        out = self.controller.step(obs, now, altitude_m=alt)
        self.sink.submit(out)
        t3 = time.perf_counter()
        timings = {"detect": (t1 - t0) * 1e3, "track": (t2 - t1) * 1e3, "control": (t3 - t2) * 1e3,
                   "total": (t3 - t0) * 1e3, "frame_age": (now - frame.stamp) * 1e3}
        return StepResult(frame, faces, tracks, target, obs, out, timings)
