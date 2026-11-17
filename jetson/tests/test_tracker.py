import random
import statistics

import pytest

from jetson.face import Face, iou, nms
from jetson.tracker import IouTracker, TargetSelector


def face(cx, cy, w, score=0.9):
    return Face(cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2, score)


def test_iou_basics():
    a = (0, 0, 10, 10)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, (10, 10, 20, 20)) == 0.0
    assert iou(a, (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_nms_keeps_best_of_overlapping_group():
    kept = nms([face(50, 50, 20, 0.7), face(52, 50, 20, 0.9), face(200, 50, 20, 0.8)])
    assert sorted(f.score for f in kept) == [0.8, 0.9]


def test_moving_face_keeps_its_id_and_gets_confirmed():
    tr = IouTracker(min_hits=2)
    ids = set()
    for i in range(20):
        tracks = tr.update([face(100 + 3 * i, 100, 40)], stamp=i * 0.05)
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert ids == {1}
    assert tr.tracks[0].confirmed and tr.tracks[0].hits == 20


def test_two_faces_two_ids():
    tr = IouTracker()
    for i in range(5):
        tr.update([face(100, 100, 40), face(400, 100, 40)], stamp=i * 0.05)
    assert sorted(t.track_id for t in tr.tracks) == [1, 2]


def test_one_frame_false_positive_dies_immediately():
    tr = IouTracker(min_hits=2)
    tr.update([face(300, 300, 30)], stamp=0.0)
    assert len(tr.tracks) == 1 and not tr.tracks[0].confirmed
    tr.update([], stamp=0.05)
    assert tr.tracks == []


def test_confirmed_track_survives_short_dropout_but_not_long_one():
    tr = IouTracker(min_hits=2, max_age_s=1.0)
    for i in range(3):
        tr.update([face(100, 100, 40)], stamp=i * 0.05)
    last = 0.10
    tr.update([], stamp=last + 0.5)
    assert len(tr.tracks) == 1 and tr.tracks[0].misses == 1
    assert tr.tracks[0].last_seen == pytest.approx(last)  # stamp tells the controller it's stale
    tr.update([face(100, 100, 40)], stamp=last + 0.6)     # re-appears: same id
    assert tr.tracks[0].track_id == 1 and tr.tracks[0].misses == 0
    tr.update([], stamp=last + 0.6 + 1.01)
    assert tr.tracks == []


def test_low_score_detection_extends_track_but_never_starts_one():
    tr = IouTracker(high_score=0.6, low_score=0.3, min_hits=2)
    for i in range(3):
        tr.update([face(100, 100, 40, 0.9)], stamp=i * 0.05)
    tr.update([face(101, 100, 40, 0.4)], stamp=0.15)       # blurry but same face
    assert len(tr.tracks) == 1 and tr.tracks[0].misses == 0
    tr.update([face(101, 100, 40, 0.4), face(400, 400, 40, 0.4)], stamp=0.20)
    assert len(tr.tracks) == 1                             # the low-score stranger made no track
    tr.update([face(101, 100, 40, 0.1)], stamp=0.25)       # below low_score: ignored
    assert tr.tracks[0].misses == 1


def test_smoothing_reduces_jitter():
    rng = random.Random(7)
    tr = IouTracker(smoothing=0.4)
    raw, smooth = [], []
    for i in range(200):
        cx = 320 + rng.gauss(0, 4)
        tr.update([face(cx, 240, 60)], stamp=i * 0.05)
        raw.append(cx)
        smooth.append(tr.tracks[0].cx)
    assert statistics.pstdev(smooth[20:]) < 0.75 * statistics.pstdev(raw[20:])


def test_prediction_follows_velocity_for_matching():
    tr = IouTracker(smoothing=1.0)
    for i in range(10):  # 10 px per 0.05 s = 200 px/s
        tr.update([face(100 + 10 * i, 100, 20)], stamp=i * 0.05)
    t = tr.tracks[0]
    assert t.velocity[0] == pytest.approx(200.0, rel=0.05)
    pb = t.predicted_box(t.last_seen + 0.1)
    assert (pb[0] + pb[2]) / 2 == pytest.approx(t.cx + 20.0, rel=0.05)
    # Skip a frame: the new box (20 px further) has zero overlap with the last box, so only the
    # constant-velocity prediction can associate it with the existing track.
    last_box = t.box
    nxt = face(100 + 10 * 11, 100, 20)
    assert iou(last_box, nxt.box) == 0.0
    tr.update([nxt], stamp=11 * 0.05)
    assert [tk.track_id for tk in tr.tracks] == [1]


def test_tiny_noisy_face_keeps_one_identity():
    """A ~9 px face (a person ~9 m away at 640 px) jittering by 1 px and drifting 3 px/frame
    barely overlaps itself frame to frame; the centre-distance fallback keeps one track."""
    rng = random.Random(5)
    tr = IouTracker(center_gate=1.0)
    for i in range(60):
        cx = 200 + 3.0 * i + rng.gauss(0, 1.0)
        tr.update([face(cx, 240 + rng.gauss(0, 1.0), 9 + rng.gauss(0, 0.5))], stamp=i / 15)
    assert [t.track_id for t in tr.tracks] == [1]
    tr_no_gate = IouTracker(center_gate=0.0, smoothing=1.0)
    rng = random.Random(5)
    for i in range(60):
        cx = 200 + 6.0 * i + rng.gauss(0, 1.0)
        tr_no_gate.update([face(cx, 240, 9)], stamp=i / 15)
    assert tr_no_gate.tracks[-1].track_id > 1        # pure IoU fragments into many tracks


def test_centre_gate_does_not_merge_far_apart_faces():
    tr = IouTracker(center_gate=1.0)
    for i in range(5):
        tr.update([face(100, 100, 20), face(160, 100, 20)], stamp=i * 0.05)
    assert sorted(t.track_id for t in tr.tracks) == [1, 2]


def test_bad_tracker_args():
    with pytest.raises(ValueError):
        IouTracker(smoothing=0.0)
    with pytest.raises(ValueError):
        IouTracker(high_score=0.3, low_score=0.6)


# ---------------------------------------------------------------- target selection
def confirmed_tracker(faces, frames=3):
    tr = IouTracker(min_hits=2)
    for i in range(frames):
        tr.update(faces, stamp=i * 0.05)
    return tr


def test_selector_prefers_large_central_face_then_sticks_with_it():
    tr = confirmed_tracker([face(320, 240, 50), face(60, 60, 60)])
    sel = TargetSelector("largest_center")
    first = sel.select(tr.tracks, 640, 480)
    assert first.cx == pytest.approx(320)
    # A much bigger face walks in: we keep following the original target.
    for i in range(3, 6):
        tr.update([face(320, 240, 50), face(60, 60, 60), face(500, 300, 150)], stamp=i * 0.05)
    assert sel.select(tr.tracks, 640, 480).track_id == first.track_id


def test_selector_switches_only_after_target_is_dropped():
    tr = confirmed_tracker([face(320, 240, 50), face(100, 100, 40)])
    sel = TargetSelector("largest")
    target = sel.select(tr.tracks, 640, 480)
    other = [t for t in tr.tracks if t.track_id != target.track_id][0]
    tr.update([face(100, 100, 40)], stamp=0.2)                 # target missing, still alive
    assert sel.select(tr.tracks, 640, 480).track_id == target.track_id
    tr.update([face(100, 100, 40)], stamp=0.2 + 1.5)            # dropped by the tracker
    assert sel.select(tr.tracks, 640, 480).track_id == other.track_id


def test_selector_policies_and_lock():
    tr = confirmed_tracker([face(320, 240, 30), face(80, 80, 90)])
    assert TargetSelector("largest").select(tr.tracks, 640, 480).w == pytest.approx(90)
    assert TargetSelector("center").select(tr.tracks, 640, 480).w == pytest.approx(30)
    sel = TargetSelector()
    sel.lock(tr.tracks[1].track_id)
    assert sel.select(tr.tracks, 640, 480).track_id == tr.tracks[1].track_id
    assert sel.select([tr.tracks[0]], 640, 480) is None        # locked target gone: no stranger
    with pytest.raises(ValueError):
        TargetSelector("random")


def test_unconfirmed_tracks_are_never_selected():
    tr = IouTracker(min_hits=3)
    tr.update([face(320, 240, 50)], stamp=0.0)
    assert TargetSelector().select(tr.tracks, 640, 480) is None


# ----------------------------------------------------------------------------- ego-rotation compensation
def test_compensate_rotation_sign_turning_right_moves_tracks_left():
    import math
    f, cx0 = 500.0, 320.0
    t = IouTracker(min_hits=1)
    t.update([face(320, 240, 20), face(420, 240, 20)], 0.0)
    before = [tr.cx for tr in t.tracks]
    t.compensate_rotation(math.radians(5.0), f, cx0)   # + = drone yawed right (clockwise from above)
    after = [tr.cx for tr in t.tracks]
    assert after[0] == pytest.approx(cx0 - f * math.tan(math.radians(5.0)))
    ang = math.atan2(before[1] - cx0, f) - math.radians(5.0)
    assert after[1] == pytest.approx(cx0 + f * math.tan(ang))
    assert all(a < b for a, b in zip(after, before))
    assert [tr.w for tr in t.tracks] == pytest.approx([20.0, 20.0])   # pure shift, size kept
    t.compensate_rotation(-math.radians(5.0), f, cx0)                  # turning back undoes it
    assert [tr.cx for tr in t.tracks] == pytest.approx(before)
