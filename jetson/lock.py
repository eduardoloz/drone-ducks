"""Keep following the same person when their face disappears: face <-> body association.

Face-only following drops the target as soon as the person turns round or walks away (the back of
a head is not a face). This module ties the followed face to that person's *body* box, follows the
body while the face is gone, and switches back to the face when it reappears.

    keeper = LockKeeper(selector)
    lock = keeper.update(face_tracks, body_tracks, frame_w, frame_h, stamp)
    lock.box          # what to steer by: the face box, or a face-sized box synthesised from the body
    lock.source       # "face", "body" or "" (nothing to follow)

Rules (all here, all unit-tested in tests/test_lock.py):
  1. A target is only ever *acquired* by face (TargetSelector over face tracks). A body alone never
     starts a lock, so the drone only follows someone who showed it their face first.
  2. While the target face is detected, remember which body track it sits in (face centre inside
     the body box, in its upper part) and learn the face-to-body geometry (size ratio, offset).
  3. Face gone (track dead, or only coasting) but its body still detected -> follow the body. The
     controller gets a *virtual face box* = learned geometry applied to the body box, so its
     distance estimate (which assumes a face width) keeps working unchanged.
  4. While following the body, a face that appears inside that body becomes the target face again
     (re-acquired), but only once it is *confirmed*: linked to the body and freshly detected in
     `confirm_hits` consecutive frames, or once with score >= `confirm_score` (the same rule as
     jetson.relock.RelockGate). Until then the body keeps steering and the candidate is treated as
     a bystander. Why: in SITL (sim/docs/runs/r3-turnaway-body.csv.gz) YuNet fired on the back of
     the walker's head at 5-6 m and each one-to-few-frame hit was taken as re-acquisition.
     Faces outside the target body are ignored: a bystander never steals the lock.
  5. Body following is capped at `max_body_only_s` without seeing the face. Body tracks can swap
     identity when two people cross; the cap bounds how long a swap could be followed. Then LOST.
  6. Face and body both gone -> nothing (the controller hovers, then times out). For
     `reacquire_holdoff_s` after that no *other* face is acquired (FollowPipeline sets it to the
     controller's lost_timeout_s, so the drone lands/holds instead of adopting a stranger); then
     acquisition goes back to rule 1.

This is detection + tracking continuity only. Nothing here recognises anyone: there are no
embeddings, no identities, nothing stored once the track ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .face import Box
from .tracker import TargetSelector, Track

# Priors for the face-to-body geometry before we have learned it from this person.
DEFAULT_FACE_PER_BODY_W = 0.33   # face width / body-box width   (0.16 m face, ~0.45-0.5 m box)
DEFAULT_FACE_PER_BODY_H = 0.09   # face width / body-box height  (0.16 m face, ~1.75 m person)


def face_body_score(face: Box, body: Box, max_rel_y: float = 0.45) -> float:
    """How plausible it is that `face` belongs to `body` (0 = not at all, up to 1).

    The face centre must be inside the body box horizontally, in the top `max_rel_y` of it
    vertically (or within one face height of its top, for head-and-shoulders boxes close up),
    and the face must be narrower than the body. Score = how central it sits.
    """
    fx1, fy1, fx2, fy2 = face
    bx1, by1, bx2, by2 = body
    bw, bh = bx2 - bx1, by2 - by1
    fw, fh = fx2 - fx1, fy2 - fy1
    if bw <= 0 or bh <= 0 or fw <= 0 or fh <= 0:
        return 0.0
    fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    if not bx1 <= fcx <= bx2:
        return 0.0
    rel_y = (fcy - by1) / bh
    # Close up, the box is head-and-shoulders (cut off by the frame) and the face fills its top
    # half, so also accept a face whose centre is within one face height of the box top.
    if rel_y > max_rel_y and fcy - by1 <= fh:
        rel_y = max_rel_y
    if rel_y < -0.05 or rel_y > max_rel_y or fy1 < by1 - fh:
        return 0.0
    if fw > 0.95 * bw or fw < 0.05 * bw:
        return 0.0
    horiz = 1.0 - abs(fcx - (bx1 + bx2) / 2.0) / (bw / 2.0)
    return max(0.0, horiz) * (1.0 - 0.5 * max(0.0, rel_y) / max_rel_y)


def associate(faces: Sequence[Track], bodies: Sequence[Track], min_score: float = 0.15) -> Dict[int, int]:
    """face track id -> body track id, greedy best score first, each body used once."""
    pairs = []
    for f in faces:
        for b in bodies:
            s = face_body_score(f.box, b.box)
            if s >= min_score:
                pairs.append((s, f.track_id, b.track_id))
    pairs.sort(reverse=True)
    used_f, used_b, out = set(), set(), {}
    for _, fid, bid in pairs:
        if fid not in used_f and bid not in used_b:
            used_f.add(fid)
            used_b.add(bid)
            out[fid] = bid
    return out


@dataclass
class BodyGeometry:
    """Where the face sits in its body box, learned while both are seen (exponential average).

    dx      face-centre x offset from body-centre, in body widths
    dy      face-centre y below body top, in body widths (the top of a person box is the head,
            which is stable; the bottom is often cut off by the frame)
    per_w   face width / body width
    per_h   face width / body height (only learned when the whole body is in frame)
    """

    dx: float = 0.0
    dy: float = 0.5 * DEFAULT_FACE_PER_BODY_W * 1.25
    per_w: float = DEFAULT_FACE_PER_BODY_W
    per_h: float = DEFAULT_FACE_PER_BODY_H
    learned_w: bool = False
    learned_h: bool = False

    def learn(self, face: Box, body: Box, body_whole: bool, alpha: float) -> None:
        bw, bh = body[2] - body[0], body[3] - body[1]
        fw = face[2] - face[0]
        if bw <= 0 or bh <= 0 or fw <= 0:
            return
        a = 1.0 if not self.learned_w else alpha
        self.dx += a * (((face[0] + face[2]) / 2.0 - (body[0] + body[2]) / 2.0) / bw - self.dx)
        self.dy += a * (((face[1] + face[3]) / 2.0 - body[1]) / bw - self.dy)
        self.per_w += a * (fw / bw - self.per_w)
        self.learned_w = True
        if body_whole:
            a = 1.0 if not self.learned_h else alpha
            self.per_h += a * (fw / bh - self.per_h)
            self.learned_h = True

    def virtual_face(self, body: Box, body_whole: bool, aspect: float = 1.25) -> Box:
        """A face-sized box where the face would be, sized from body height when the whole body
        is visible (height barely changes as a person turns) and from body width otherwise."""
        bw, bh = body[2] - body[0], body[3] - body[1]
        fw = self.per_h * bh if body_whole else self.per_w * bw
        cx = (body[0] + body[2]) / 2.0 + self.dx * bw
        cy = body[1] + self.dy * bw
        fh = aspect * fw
        return (cx - fw / 2.0, cy - fh / 2.0, cx + fw / 2.0, cy + fh / 2.0)


@dataclass
class LockState:
    face: Optional[Track]      # followed face track, if it is alive
    body: Optional[Track]      # the followed person's body track, if known and alive
    box: Optional[Box]         # what to steer by (face box or virtual face box), None = nothing
    stamp: Optional[float]     # capture time of the detection `box` comes from
    source: str                # "face", "body" or ""
    reacquired: bool = False   # True on the frame the face came back after body-only following


def _whole(body: Box, frame_h: int, margin: float = 2.0) -> bool:
    return body[1] > margin and body[3] < frame_h - margin


class LockKeeper:
    def __init__(self, selector: Optional[TargetSelector] = None, max_body_only_s: float = 15.0,
                 learn_rate: float = 0.2, min_link_score: float = 0.15,
                 reacquire_holdoff_s: Optional[float] = None, confirm_hits: int = 3,
                 confirm_score: Optional[float] = 0.9) -> None:
        if confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        self.selector = selector or TargetSelector()
        # Body -> face hand-back confirmation (rule 4). confirm_hits=1, confirm_score=None = round 2.
        self.confirm_hits = confirm_hits
        self.confirm_score = confirm_score
        self.reacquire_holdoff_s = reacquire_holdoff_s
        self.max_body_only_s = max_body_only_s
        self.learn_rate = learn_rate
        self.min_link_score = min_link_score
        self.reset()

    def reset(self) -> None:
        self.body_id: Optional[int] = None
        self.geometry = BodyGeometry()
        self.face_last_seen: Optional[float] = None
        self._was_body = False
        self._lost_at: Optional[float] = None   # when face and body were both lost
        self._cand_id: Optional[int] = None     # face track waiting for hand-back confirmation
        self._cand_streak = 0
        self.handback_blocked = 0               # frames a candidate face was held back (logs/tests)

    @property
    def face_id(self) -> Optional[int]:
        return self.selector.target_id

    def _adopt_face(self, track: Track) -> None:
        if self.selector._locked:
            self.selector.lock(track.track_id)
        else:
            self.selector.target_id = track.track_id

    def _confirm(self, cand: Track) -> bool:
        """Rule 4: count consecutive fresh frames of the same candidate face inside the body."""
        if cand.misses > 0:
            self._cand_streak = 0
            return False
        self._cand_streak = self._cand_streak + 1 if cand.track_id == self._cand_id else 1
        self._cand_id = cand.track_id
        return self._cand_streak >= self.confirm_hits or (
            self.confirm_score is not None and cand.score >= self.confirm_score)

    def update(self, faces: Sequence[Track], bodies: Sequence[Track], frame_w: int, frame_h: int,
               stamp: float) -> LockState:
        faces_by_id = {t.track_id: t for t in faces}
        bodies_by_id = {t.track_id: t for t in bodies if t.confirmed}
        links = associate([t for t in faces if t.confirmed], list(bodies_by_id.values()), self.min_link_score)
        face = faces_by_id.get(self.face_id) if self.face_id is not None else None
        body = bodies_by_id.get(self.body_id) if self.body_id is not None else None
        if self.body_id is not None and body is None:
            self.body_id = None  # body track died

        # Face gone: try to re-acquire it through the body before anything else. A face inside the
        # target body only takes over once confirmed (rule 4); so does the old face track coming
        # back to life after the body has been steering (it may have latched onto a false face).
        reacquired = False
        held = False   # the face track is alive but its hand-back is not confirmed yet
        candidate: Optional[Track] = None
        if face is None and body is not None:
            for fid, bid in links.items():
                if bid == body.track_id:
                    candidate = faces_by_id[fid]
                    break
        elif face is not None and self._was_body and face.misses == 0:
            candidate = face
        if candidate is not None:
            if self._confirm(candidate):
                if face is None:
                    face = candidate
                    self._adopt_face(face)
                reacquired = self._was_body
            else:
                self.handback_blocked += 1
                held = face is not None
        elif face is not None and self._was_body and face.track_id == self._cand_id:
            held = True   # coasting on an unconfirmed hit: still not the target
            self._cand_streak = 0
        if reacquired or (candidate is None and not held):
            self._cand_id, self._cand_streak = None, 0

        # Never had a face (or lost face and body): acquire by face only (rule 1).
        if face is None and body is None:
            self.body_id = None
            self.geometry = BodyGeometry()
            if self.face_last_seen is not None and self._lost_at is None:
                self._lost_at = stamp
            if self._lost_at is not None and self.reacquire_holdoff_s \
                    and stamp - self._lost_at < self.reacquire_holdoff_s:
                return LockState(None, None, None, None, "")
            face = self.selector.select(list(faces), frame_w, frame_h)
            if face is not None:
                self._lost_at = None

        face_fresh = face is not None and face.misses == 0 and not held
        if face_fresh:
            self.face_last_seen = face.last_seen
            bid = links.get(face.track_id)
            if bid is not None:
                if self.body_id is not None and bid != self.body_id:
                    self.geometry = BodyGeometry()  # new body for this face: relearn
                self.body_id = bid
                body = bodies_by_id[bid]
                if body.misses == 0:
                    self.geometry.learn(face.box, body.box, _whole(body.box, frame_h), self.learn_rate)
            self._was_body = False
            return LockState(face, body, face.box, face.last_seen, "face", reacquired)

        body_ok = (body is not None and body.misses == 0 and self.face_last_seen is not None
                   and stamp - self.face_last_seen <= self.max_body_only_s)
        if body_ok:
            self._was_body = True
            box = self.geometry.virtual_face(body.box, _whole(body.box, frame_h))
            return LockState(face, body, box, body.last_seen, "body")
        if body is not None and self.face_last_seen is not None \
                and stamp - self.face_last_seen > self.max_body_only_s:
            self.body_id = None  # rule 5: give up on the body
            body = None
        if face is not None and not held:  # face only coasting, no fresh body: controller ages it out
            return LockState(face, body, face.box, face.last_seen, "face")
        if body is not None:  # body coasting too
            box = self.geometry.virtual_face(body.box, _whole(body.box, frame_h))
            return LockState(None, body, box, body.last_seen, "body")
        return LockState(None, None, None, None, "")
