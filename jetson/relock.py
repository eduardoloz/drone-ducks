"""Confirmed re-lock: after the target is lost, only resume on a target that is *really* back.

Seen in SITL (round 2): the face was lost when the walker turned away, then single YuNet hits on
the back of the walker's head at 7-10 m (score ~0.5) refreshed the coasting track, the controller
saw a fresh observation, restarted its lost timer and flew toward the walker.

RelockGate sits between "what to steer by" (FollowPipeline / LockKeeper) and the controller:

  ENGAGED    the target was detected within `lost_after_s`: everything passes through (normal
             one- or two-frame detector flicker is not a loss, the controller already rides it out).
  RELOCKING  no fresh detection of the target for `lost_after_s`: observations are withheld from
             the controller (it keeps hovering and its lost timer keeps running) until the target
             is detected in `confirm_hits` consecutive frames on the same track, or once with a
             score >= `confirm_score` (None disables that fast path), and only from detections
             no farther than max(`max_distance_m`, last distance seen while engaged +
             `distance_margin_m`) (distances estimated from the box width). Then ENGAGED again.

Why the distance cap: the N-hit rule alone does not catch the SITL failure. In
sim/docs/runs/follow-sitl-run2.csv the false re-lock at 68.8 s came from back-of-head detections in
runs of 6, 7 and 16 consecutive frames (walker facing 180 deg away, 7-8.7 m true, 6.0-6.8 m
estimated), so N would have to be > 16 (1.1 s) to block it. Scores don't separate either: YuNet on
rendered frontal sim people at 6 m scores 0.44-0.65. What does separate them is range: the genuine
re-lock in the same run was at ~3.5 m. FollowPipeline sets max_distance_m = target distance + 1.5 m.
The cap grows with the distance the target was last followed at, so a target followed far away
(approaching from 8 m) can still be re-locked at that range: in the sim's fake-backend closed loop
(sim/tests/unit/test_jetson_follow.py) a 8 px face at 8 m loses its track for ~0.5 s while the drone
yaws (the tracker's motion model ignores ego-rotation); a fixed 5.5 m cap would never re-lock it.
lost_after_s defaults to the controller's max_obs_age_s (0.5 s): arming earlier (0.3 s) cut off the
last stale-but-useful yaw commands in that loop and made its 30 deg turn test flaky.

The first acquisition is left to the tracker (min_hits) as before; the gate only arms after the
target has been followed once. Pure, clock-free, unit-tested in tests/test_relock.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

ENGAGED = "ENGAGED"
RELOCKING = "RELOCKING"
IDLE = "IDLE"


@dataclass
class RelockGate:
    confirm_hits: int = 3
    confirm_score: Optional[float] = 0.9
    lost_after_s: float = 0.5   # = FollowConfig.max_obs_age_s: the gate arms when the controller goes LOST
    max_distance_m: Optional[float] = None   # re-lock only on targets estimated this close (None = any)
    distance_margin_m: float = 1.5           # ...or within this much of the last engaged distance

    def __post_init__(self) -> None:
        if self.confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        if self.lost_after_s < 0:
            raise ValueError("lost_after_s must be >= 0")
        self.reset()

    def reset(self) -> None:
        self.state = IDLE
        self._last_fresh: Optional[float] = None
        self._key: Optional[Hashable] = None
        self._streak = 0
        self._last_distance: Optional[float] = None
        self.blocked = 0          # observations withheld so far (for logs/tests)
        self.relocks = 0          # times we went RELOCKING -> ENGAGED

    def distance_cap(self) -> float:
        """Farthest estimated distance a re-lock is accepted from (inf = no cap)."""
        if self.max_distance_m is None:
            return float("inf")
        last = self._last_distance
        return self.max_distance_m if last is None else max(self.max_distance_m, last + self.distance_margin_m)

    def update(self, key: Optional[Hashable], fresh: bool, score: float, stamp: float,
               distance_m: Optional[float] = None) -> bool:
        """One frame. key: identity of the candidate target track (None = nothing to steer by);
        fresh: it was detected in this frame; score: that detection's score; stamp: frame capture
        time. Returns True if the controller may use this frame's observation."""
        if self.state == ENGAGED and (self._last_fresh is None or stamp - self._last_fresh > self.lost_after_s):
            self.state, self._streak, self._key = RELOCKING, 0, None
        if key is None:
            self._streak = 0
            return False
        if self.state in (IDLE, ENGAGED):
            if fresh:
                self.state, self._last_fresh = ENGAGED, stamp
                if distance_m is not None:
                    self._last_distance = distance_m
            return True
        # RELOCKING
        if fresh and distance_m is not None and distance_m > self.distance_cap():
            fresh = False   # too far to trust as the person coming back (see module doc)
        if not fresh:
            self._streak = 0
            self.blocked += 1
            return False
        self._streak = self._streak + 1 if key == self._key else 1
        self._key = key
        if self._streak >= self.confirm_hits or (self.confirm_score is not None and score >= self.confirm_score):
            self.state, self._last_fresh, self._streak = ENGAGED, stamp, 0
            self.relocks += 1
            return True
        self.blocked += 1
        return False
