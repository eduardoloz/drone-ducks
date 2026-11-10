"""Keep face identities stable across frames, and pick ONE face to follow.

IouTracker (ByteTrack-style, simplified):
  1. Predict where each track's box is now (constant velocity since its last detection).
  2. First pass: match high-score detections to tracks by IoU (greedy, best overlap first).
  3. Second pass: match the remaining *low-score* detections to the still-unmatched tracks.
     This is ByteTrack's key idea: a blurry/turned face scores low but is still the same face,
     so it should keep its track alive instead of being thrown away.
  4. Unmatched high-score detections start new tracks. Low-score ones never do (too noisy).
  5. A track is "confirmed" after `min_hits` matches; unconfirmed tracks die on their first miss
     (kills one-frame false positives), confirmed ones survive misses for up to `max_age_s`.

The tracker smooths each box (exponential moving average) so the controller sees less jitter.

TargetSelector: chooses which confirmed track to follow and *sticks with it* while the tracker
keeps it alive, so a bigger face walking into view doesn't steal the drone.

Pure Python, no OpenCV needed. Greedy matching instead of the Hungarian algorithm is fine for
the handful of faces we expect in a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .face import Box, Face, box_area, iou

Velocity = Tuple[float, float, float, float]  # d(cx, cy, w, h)/dt in px/s


@dataclass
class Track:
    track_id: int
    box: Box                 # smoothed (x1, y1, x2, y2)
    score: float
    first_seen: float
    last_seen: float         # capture stamp of the last frame this track was matched in
    hits: int = 1
    misses: int = 0          # consecutive updates without a match
    confirmed: bool = False
    velocity: Velocity = (0.0, 0.0, 0.0, 0.0)
    landmarks: Optional[tuple] = None

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    def predicted_box(self, stamp: float) -> Box:
        dt = max(0.0, stamp - self.last_seen)
        vx, vy, vw, vh = self.velocity
        cx, cy = self.cx + vx * dt, self.cy + vy * dt
        w, h = max(1.0, self.w + vw * dt), max(1.0, self.h + vh * dt)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _cxcywh(b: Box) -> Tuple[float, float, float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b[2] - b[0], b[3] - b[1])


def _greedy_match(tracks: Sequence[Track], dets: Sequence[Face], stamp: float,
                  threshold: float) -> List[Tuple[int, int]]:
    """Return (track_index, det_index) pairs, best IoU first, each index used once."""
    pairs = []
    for ti, t in enumerate(tracks):
        pb = t.predicted_box(stamp)
        for di, d in enumerate(dets):
            overlap = iou(pb, d.box)
            if overlap >= threshold:
                pairs.append((overlap, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d, out = set(), set(), []
    for _, ti, di in pairs:
        if ti not in used_t and di not in used_d:
            used_t.add(ti)
            used_d.add(di)
            out.append((ti, di))
    return out


class IouTracker:
    def __init__(self, iou_threshold: float = 0.3, low_iou_threshold: float = 0.5,
                 high_score: float = 0.6, low_score: float = 0.3, min_hits: int = 2,
                 max_age_s: float = 1.0, smoothing: float = 0.6, max_tracks: int = 32) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing is the weight of the new detection, in (0, 1]")
        if low_score > high_score:
            raise ValueError("low_score must be <= high_score")
        self.iou_threshold = iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.high_score = high_score
        self.low_score = low_score
        self.min_hits = min_hits
        self.max_age_s = max_age_s
        self.smoothing = smoothing
        self.max_tracks = max_tracks
        self.tracks: List[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]

    def get(self, track_id: Optional[int]) -> Optional[Track]:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def _absorb(self, t: Track, d: Face, stamp: float) -> None:
        dt = stamp - t.last_seen
        a = self.smoothing
        pred = t.predicted_box(stamp)
        new_box = tuple(a * dv + (1.0 - a) * pv for dv, pv in zip(d.box, pred))
        if dt > 1e-6:
            old, new = _cxcywh(t.box), _cxcywh(new_box)
            measured = tuple((n - o) / dt for n, o in zip(new, old))
            t.velocity = tuple(0.5 * v + 0.5 * m for v, m in zip(t.velocity, measured))
        t.box = new_box
        t.score = d.score
        t.landmarks = d.landmarks
        t.last_seen = stamp
        t.hits += 1
        t.misses = 0
        if t.hits >= self.min_hits:
            t.confirmed = True

    def update(self, faces: Sequence[Face], stamp: float) -> List[Track]:
        """Feed this frame's detections (capture time `stamp`); returns all live tracks."""
        high = [f for f in faces if f.score >= self.high_score]
        low = [f for f in faces if self.low_score <= f.score < self.high_score]

        # Pass 1: high-score detections.
        matched_t, used_high = set(), set()
        for ti, di in _greedy_match(self.tracks, high, stamp, self.iou_threshold):
            self._absorb(self.tracks[ti], high[di], stamp)
            matched_t.add(ti)
            used_high.add(di)

        # Pass 2: low-score detections may only extend tracks that pass 1 left unmatched.
        remaining = [i for i in range(len(self.tracks)) if i not in matched_t]
        rem_tracks = [self.tracks[i] for i in remaining]
        for ri, di in _greedy_match(rem_tracks, low, stamp, self.low_iou_threshold):
            self._absorb(rem_tracks[ri], low[di], stamp)
            matched_t.add(remaining[ri])

        survivors = []
        for i, t in enumerate(self.tracks):
            if i in matched_t:
                survivors.append(t)
                continue
            t.misses += 1
            if t.confirmed and stamp - t.last_seen <= self.max_age_s:
                survivors.append(t)
        self.tracks = survivors

        for di, d in enumerate(high):
            if di in used_high or len(self.tracks) >= self.max_tracks:
                continue
            self.tracks.append(Track(self._next_id, d.box, d.score, first_seen=stamp, last_seen=stamp,
                                     confirmed=self.min_hits <= 1, landmarks=d.landmarks))
            self._next_id += 1
        return list(self.tracks)


SELECT_POLICIES = ("largest", "center", "largest_center")


class TargetSelector:
    """Pick one confirmed track to follow and keep it until the tracker drops it."""

    def __init__(self, policy: str = "largest_center") -> None:
        if policy not in SELECT_POLICIES:
            raise ValueError(f"policy must be one of {SELECT_POLICIES}")
        self.policy = policy
        self.target_id: Optional[int] = None
        self._locked = False

    def lock(self, track_id: int) -> None:
        """Follow this track id no matter what (e.g. the operator clicked it)."""
        self.target_id, self._locked = track_id, True

    def unlock(self) -> None:
        self._locked = False

    def _score(self, t: Track, frame_w: int, frame_h: int) -> float:
        area = box_area(t.box) / float(frame_w * frame_h)
        dx, dy = t.cx / frame_w - 0.5, t.cy / frame_h - 0.5
        off_centre = (dx * dx + dy * dy) ** 0.5 / 0.7071  # 0 at centre, 1 in a corner
        if self.policy == "largest":
            return area
        if self.policy == "center":
            return -off_centre
        return area * (1.0 - 0.5 * off_centre)

    def select(self, tracks: Sequence[Track], frame_w: int, frame_h: int) -> Optional[Track]:
        for t in tracks:
            if t.track_id == self.target_id:
                return t
        if self._locked:
            return None  # locked target is gone: follow nothing rather than a stranger
        candidates = [t for t in tracks if t.confirmed]
        if not candidates:
            self.target_id = None
            return None
        best = max(candidates, key=lambda t: self._score(t, frame_w, frame_h))
        self.target_id = best.track_id
        return best
