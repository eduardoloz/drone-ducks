"""End the follow when the person walks away from the drone (optional, off by default).

With the body lock (`--person yolox`) the drone keeps following someone whose face it cannot see,
so a person who turns round and walks off gets chased until their body is lost (SITL round 3:
followed out to 11-13 m at the walker's 1.4 m/s). `FollowConfig(end_on_walkaway=True)`
(`--end-on-walkaway`) makes that end the session instead, through the same TIMEOUT -> land / rtl /
hold path as losing the target (`lost_action`), so the bridges and the sim sink need no changes.

What counts as "walking away": over the last `seconds` (default 3 s) of tracking, with no gap
longer than 1 s, all of:
  1. back turned: the controller steers by the body (observation source "body") now and in at
     least 2/3 of the window's frames. A face that is only a hand-back candidate (jetson.lock rule
     4: maybe the back of the head) counts as not seen. Why not "no face at all": the hand-back rule
     still lets some back-of-head streaks through (3 hits in a row; SITL run 2 had streaks of 6-16
     frames at 6-9 m), and one of those must not restart the clock. Someone who really turns back
     to the drone shows their face for good and stops receding.
  2. the person moves away over the ground at >= `speed_m_s` (default 0.8 m/s). The box-width
     distance alone cannot show this because the drone follows: at 0.8 m/s the P controller settles
     ~2 m behind and the distance stops growing. So we add back how far the drone itself flew
     toward the person (its forward command through the same 0.35 s first-order lag the toy world
     and ego compensation use; in the SITL logs this matched the drone's real travel within 0.2 m):
         p(t) = distance(t) + drone forward travel(t)       (person position along the line of sight)
     Speed = displacement of p over the body-steered runs of the window / their duration, each
     run's displacement taken between the medians of its first and last thirds (one bad box cannot
     fake it). Runs are split at face frames because every face frame re-learns the face-in-body
     geometry, which can step the body distance by ~1 m (SITL r3-turnaway-body, 34.8 s).
  3. the person is farther than target_distance + `margin_m` (default 1 m), i.e. the drone really
     is behind them. A guard against small estimator biases while the drone flies forward.
In practice it fires ~2-3 s after the person turns and walks off.

Why these defaults (3 s, 0.8 m/s, +1.0 m), measured on the round-3 SITL logs (lockeval
--replay-sitl) and the toy world (lockeval --end-on-walkaway):
  - Standing with the back turned, or turning round to look at something: ~0 m/s, never fires
    (toy turn_around scenario, 5/5 seeds).
  - People leaving walk at 1.2-1.4 m/s (the sim walker: 1.37 m/s). The estimator read 0.9-1.3 m/s
    for it when it first fired (1.2-1.5 m/s once under way); both SITL walk-aways were caught
    2.3-2.8 s after they began.
  - The sim's turnaway walker also drifts away at 0.39 m/s with its back turned for 13 s and then
    comes back. Body-box noise made that read up to ~0.8 m/s, so 0.6 m/s fired on it and 0.8 does
    not. Lower --walkaway-speed if a slow drift away should end the session too.
  - 3 s: a step or two back is not a walk-away, and the person is still well inside person-detector
    range when it fires (SITL: 7.8-8.2 m true distance).
Face-only mode never triggers it (the face disappears first and the normal lost timeout applies).

Pure and clock-free (times are passed in); unit-tested in tests/test_walkaway.py and replayed
against the gzipped SITL run logs by jetson.lockeval.walkaway_replay.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class WalkAwayResult:
    triggered: bool
    speed_m_s: Optional[float]     # estimated ground speed away from the drone (None: window not full)
    distance_m: Optional[float]    # median distance over the newest third of the window
    reason: str = ""


class WalkAwayDetector:
    def __init__(self, seconds: float = 3.0, speed_m_s: float = 0.8, min_distance_m: float = 5.0,
                 drone_lag_s: float = 0.35, max_gap_s: float = 1.0, min_samples: int = 6,
                 max_face_share: float = 1.0 / 3.0) -> None:
        if seconds <= 0 or speed_m_s <= 0:
            raise ValueError("seconds and speed_m_s must be > 0")
        self.seconds = seconds
        self.speed_m_s = speed_m_s
        self.min_distance_m = min_distance_m   # = target distance + margin
        self.drone_lag_s = drone_lag_s
        self.max_gap_s = max_gap_s
        self.min_samples = max(3, int(min_samples))
        self.max_face_share = max_face_share
        self.reset()

    def reset(self) -> None:
        self._t: Optional[float] = None
        self._v = 0.0          # drone forward speed estimate (lagged command)
        self._travel = 0.0     # drone forward travel since start, m
        # (t, distance + travel, distance, steered by a face)
        self._samples: Deque[Tuple[float, float, float, bool]] = deque()
        self._since: Optional[float] = None   # start of the current run of samples (no gap > max_gap_s)
        self.last = WalkAwayResult(False, None, None)

    def _clear(self) -> None:
        self._samples.clear()
        self._since = None

    def update(self, t: float, flying_forward_m_s: float, distance_m: Optional[float] = None,
               face_visible: bool = False) -> WalkAwayResult:
        """One controller step at time t.

        flying_forward_m_s: the forward command the drone has been flying since the previous call
        distance_m:         horizontal distance to the target this step (None: no usable target)
        face_visible:       steering by a confirmed face this step
        """
        if self._t is not None:
            dt = min(max(0.0, t - self._t), self.max_gap_s)
            a = dt / (self.drone_lag_s + dt) if self.drone_lag_s > 0 and dt > 0 else (1.0 if dt > 0 else 0.0)
            self._v += a * (flying_forward_m_s - self._v)
            self._travel += self._v * dt
        self._t = t

        if distance_m is None:
            if self._samples and t - self._samples[-1][0] > self.max_gap_s:
                self._clear()
            self.last = WalkAwayResult(False, None, None)
            return self.last
        if self._samples and t - self._samples[-1][0] > self.max_gap_s:
            self._clear()
        if self._since is None:
            self._since = t
        self._samples.append((t, distance_m + self._travel, distance_m, bool(face_visible)))
        while self._samples and self._samples[0][0] < t - self.seconds:
            self._samples.popleft()

        faces = sum(1 for x in self._samples if x[3])
        if (face_visible or t - self._since < self.seconds or len(self._samples) < self.min_samples
                or faces > self.max_face_share * len(self._samples)):
            self.last = WalkAwayResult(False, None, distance_m)
            return self.last
        # Slope from body-steered samples only, run by run: every frame steered by a face re-learns
        # the face-in-body geometry (jetson.lock), which can step the body's distance scale by
        # ~1 m (SITL r3-turnaway-body at 34.8 s: 4.7 -> 5.7 m with the true distance flat). So the
        # window is split at face frames and only motion *within* each body run counts:
        #   speed = sum(run displacement) / sum(run duration), displacement = median of the run's
        #   last third minus median of its first third (one bad box cannot fake it).
        runs, cur = [], []
        for x in self._samples:
            if x[3]:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(x)
        if cur:
            runs.append(cur)
        moved, span = 0.0, 0.0
        for r in runs:
            n = len(r) // 3
            if n < 1 or len(r) < 4:
                continue
            dt_run = statistics.mean(x[0] for x in r[-n:]) - statistics.mean(x[0] for x in r[:n])
            if dt_run <= 0:
                continue
            moved += statistics.median(x[1] for x in r[-n:]) - statistics.median(x[1] for x in r[:n])
            span += dt_run
        if span < self.seconds / 3.0:
            self.last = WalkAwayResult(False, None, distance_m)
            return self.last
        speed = moved / span
        tail = runs[-1][-max(1, len(runs[-1]) // 3):]
        dist = statistics.median(x[2] for x in tail)
        hit = speed >= self.speed_m_s and dist >= self.min_distance_m
        reason = (f"walked away: receding {speed:.1f} m/s over {self.seconds:g} s, face seen in "
                  f"{faces}/{len(self._samples)} frames, {dist:.1f} m away") if hit else ""
        self.last = WalkAwayResult(hit, speed, dist, reason)
        return self.last
