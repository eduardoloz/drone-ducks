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
             score >= `confirm_score` (None disables that fast path). Then ENGAGED again.

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
    lost_after_s: float = 0.3

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
        self.blocked = 0          # observations withheld so far (for logs/tests)
        self.relocks = 0          # times we went RELOCKING -> ENGAGED

    def update(self, key: Optional[Hashable], fresh: bool, score: float, stamp: float) -> bool:
        """One frame. key: identity of the candidate target track (None = nothing to steer by);
        fresh: it was detected in this frame; score: that detection's score; stamp: frame capture
        time. Returns True if the controller may use this frame's observation."""
        if self.state == ENGAGED and (self._last_fresh is None or stamp - self._last_fresh > self.lost_after_s):
            self.state, self._streak, self._key = RELOCKING, 0, None
        if key is None:
            self._streak = 0
            return False
        if self.state == IDLE:
            if fresh:
                self.state, self._last_fresh = ENGAGED, stamp
            return True
        if self.state == ENGAGED:
            if fresh:
                self._last_fresh = stamp
            return True
        # RELOCKING
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
