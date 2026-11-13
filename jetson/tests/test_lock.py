"""Face <-> body association, LockKeeper rules, the pipeline body path and the lock-retention eval."""

import os

import numpy as np
import pytest

from jetson.face import Face
from jetson.lock import BodyGeometry, LockKeeper, associate, face_body_score
from jetson.tracker import IouTracker, TargetSelector, Track

from .conftest import MEDIA_DIR, MODELS_DIR

W, H = 640, 480


def trk(tid, box, misses=0, confirmed=True, last_seen=0.0):
    return Track(tid, box, 0.9, first_seen=0.0, last_seen=last_seen, misses=misses, confirmed=confirmed)


BODY = (200.0, 100.0, 300.0, 400.0)
FACE = (232.0, 105.0, 268.0, 150.0)       # top centre of BODY
BYSTANDER_BODY = (450.0, 100.0, 550.0, 400.0)
BYSTANDER_FACE = (482.0, 105.0, 518.0, 150.0)


# ----------------------------------------------------------------------------- association
def test_face_inside_body_top_scores_high():
    assert face_body_score(FACE, BODY) > 0.8


@pytest.mark.parametrize("face", [
    (232.0, 300.0, 268.0, 345.0),   # face at the body's waist
    (400.0, 105.0, 436.0, 150.0),   # outside horizontally
    (201.0, 105.0, 299.0, 200.0),   # wider than 95 % of the body
    (232.0, 20.0, 268.0, 65.0),     # well above the head
])
def test_implausible_face_body_pairs_score_zero(face):
    assert face_body_score(face, BODY) == 0.0


def test_associate_is_one_to_one_and_picks_the_right_body():
    faces = [trk(1, FACE), trk(2, BYSTANDER_FACE)]
    bodies = [trk(10, BYSTANDER_BODY), trk(11, BODY)]
    assert associate(faces, bodies) == {1: 11, 2: 10}


def test_geometry_learns_and_reconstructs_the_face_box():
    g = BodyGeometry()
    g.learn(FACE, BODY, body_whole=True, alpha=0.2)
    vf = g.virtual_face(BODY, body_whole=False)
    assert vf == pytest.approx(FACE, abs=0.6)
    vh = g.virtual_face(BODY, body_whole=True)
    assert (vh[2] - vh[0]) == pytest.approx(FACE[2] - FACE[0], rel=1e-6)
    # person walks away: body half the size -> virtual face half the size, same relative place
    small = (225.0, 150.0, 275.0, 300.0)
    v2 = g.virtual_face(small, body_whole=True)
    assert (v2[2] - v2[0]) == pytest.approx((FACE[2] - FACE[0]) / 2, rel=1e-6)
    assert (v2[0] + v2[2]) / 2 == pytest.approx(250.0)


# ----------------------------------------------------------------------------- LockKeeper
def test_body_alone_never_starts_a_lock():
    k = LockKeeper()
    st = k.update([], [trk(10, BODY)], W, H, 0.0)
    assert st.source == "" and st.box is None


def test_follow_face_then_body_then_reacquire_by_face():
    k = LockKeeper()
    st = k.update([trk(1, FACE)], [trk(10, BODY)], W, H, 0.0)
    assert st.source == "face" and st.face.track_id == 1 and k.body_id == 10
    # face gone (turned round); body still detected
    st = k.update([], [trk(10, BODY, last_seen=0.1)], W, H, 0.1)
    assert st.source == "body" and st.face is None and st.body.track_id == 10
    assert st.box == pytest.approx(FACE, abs=0.6) and st.stamp == 0.1
    # face comes back with a new track id inside the same body
    st = k.update([trk(7, FACE, last_seen=0.2)], [trk(10, BODY, last_seen=0.2)], W, H, 0.2)
    assert st.source == "face" and st.face.track_id == 7 and st.reacquired
    assert k.face_id == 7


def test_bystander_face_cannot_steal_while_following_the_body():
    k = LockKeeper()
    k.update([trk(1, FACE)], [trk(10, BODY), trk(11, BYSTANDER_BODY)], W, H, 0.0)
    st = k.update([trk(2, BYSTANDER_FACE)], [trk(10, BODY), trk(11, BYSTANDER_BODY)], W, H, 0.1)
    assert st.source == "body" and st.body.track_id == 10 and k.face_id == 1


def test_face_only_selector_switches_to_bystander_for_contrast():
    sel = TargetSelector()
    sel.select([trk(1, FACE)], W, H)
    assert sel.select([trk(2, BYSTANDER_FACE)], W, H).track_id == 2


def test_coasting_face_hands_over_to_fresh_body():
    k = LockKeeper()
    k.update([trk(1, FACE)], [trk(10, BODY)], W, H, 0.0)
    st = k.update([trk(1, FACE, misses=1)], [trk(10, BODY, last_seen=0.1)], W, H, 0.1)
    assert st.source == "body"


def test_body_only_following_is_capped():
    k = LockKeeper(max_body_only_s=2.0)
    k.update([trk(1, FACE)], [trk(10, BODY)], W, H, 0.0)
    assert k.update([], [trk(10, BODY, last_seen=1.9)], W, H, 1.9).source == "body"
    st = k.update([], [trk(10, BODY, last_seen=2.5)], W, H, 2.5)
    assert st.source == "" and k.body_id is None


def test_operator_lock_follows_the_reacquired_face():
    sel = TargetSelector()
    sel.lock(1)
    k = LockKeeper(sel)
    k.update([trk(1, FACE)], [trk(10, BODY)], W, H, 0.0)
    k.update([], [trk(10, BODY)], W, H, 0.1)
    st = k.update([trk(5, FACE)], [trk(10, BODY)], W, H, 0.2)
    assert st.face.track_id == 5 and sel.target_id == 5 and sel._locked


def test_unconfirmed_body_is_ignored():
    k = LockKeeper()
    k.update([trk(1, FACE)], [trk(10, BODY, confirmed=False)], W, H, 0.0)
    assert k.body_id is None


# ----------------------------------------------------------------------------- pipeline
class _Fixed:
    name = "fixed"

    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def detect(self, frame):
        out = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return out

    def info(self):
        return {"backend": self.name}


def _frames(n):
    from jetson.sources import Frame

    img = np.zeros((H, W, 3), np.uint8)
    return [Frame(img, i + 1, i / 15.0) for i in range(n)]


def test_pipeline_without_person_detector_is_unchanged():
    from jetson.pipeline import FollowPipeline

    face = Face(*FACE, 0.9)
    p = FollowPipeline(_Fixed([[face]] * 3 + [[]] * 3))
    results = [p.step(f, now=f.stamp) for f in _frames(6)]
    assert results[2].target_source == "face" and results[2].observation is not None
    assert results[2].bodies == [] and results[2].target_body is None and "person" not in results[2].timings_ms


def test_pipeline_body_path_keeps_observation_when_face_vanishes():
    from jetson.pipeline import FollowPipeline

    face, body = Face(*FACE, 0.9), Face(*BODY, 0.9)
    p = FollowPipeline(_Fixed([[face]] * 4 + [[]] * 6), person_detector=_Fixed([[body]] * 10))
    results = [p.step(f, now=f.stamp) for f in _frames(10)]
    assert results[3].target_source == "face" and results[3].target_body is not None
    last = results[-1]
    assert last.target_source == "body" and last.target is None and last.observation is not None
    assert last.control.mode == "TRACKING" and "person" in last.timings_ms


def test_pipeline_person_every_reuses_body_tracks():
    from jetson.pipeline import FollowPipeline

    det = _Fixed([[Face(*BODY, 0.9)]] * 10)
    p = FollowPipeline(_Fixed([[Face(*FACE, 0.9)]] * 10), person_detector=det, person_every=3)
    for f in _frames(9):
        p.step(f, now=f.stamp)
    assert det.i == 3


# ----------------------------------------------------------------------------- lock-retention eval
def test_lockeval_body_path_keeps_the_target_when_they_walk_away():
    from jetson.lockeval import run

    face_only = run("walk_away", use_body=False, seed=0)
    body = run("walk_away", use_body=True, seed=0)
    assert face_only["timeout"] == 1.0 and face_only["on_target"] < 0.5
    assert body["on_target"] > 0.95 and body["timeout"] == 0.0 and body["on_wrong"] == 0.0


def test_lockeval_bystander_never_followed_with_body_path():
    from jetson.lockeval import run

    assert run("bystander", use_body=False, seed=0)["on_wrong"] > 0.3   # the failure we fix
    r = run("bystander", use_body=True, seed=0)
    assert r["on_wrong"] == 0.0 and r["on_target"] > 0.95


def test_lockeval_cli_prints_a_table(capsys):
    from jetson.lockeval import main

    assert main(["--scenario", "face_visible", "--seeds", "1"]) == 0
    assert "face+body" in capsys.readouterr().out


# ----------------------------------------------------------------------------- person detectors
def test_yolox_grid_count_and_order():
    from jetson.detectors.person import yolox_grids

    g, s = yolox_grids((416, 416))
    assert g.shape == (3549, 2) and s.shape == (3549, 1)
    assert tuple(g[1]) == (1, 0) and s[0, 0] == 8 and s[-1, 0] == 32


def test_decode_yolox_synthetic_person_mapped_back():
    from jetson.detectors.person import decode_yolox, yolox_grids

    g, s = yolox_grids()
    raw = np.zeros((1, 3549, 85), np.float32)
    i = int(np.where((s[:, 0] == 16) & (g[:, 0] == 10) & (g[:, 1] == 5))[0][0])
    raw[0, i, :4] = (0.5, 0.5, np.log(4.0), np.log(8.0))   # centre (168, 88), size 64x128 in blob px
    raw[0, i, 4], raw[0, i, 5] = 0.9, 0.9                    # person
    j = i + 1
    raw[0, j, 4], raw[0, j, 5 + 2] = 0.9, 0.9                # a car: ignored
    out = decode_yolox(raw, ratio=0.5)
    assert len(out) == 1
    f = out[0]
    assert (f.cx, f.cy, f.w, f.h) == pytest.approx((336.0, 176.0, 128.0, 256.0))
    assert f.score == pytest.approx(0.81)


def test_decode_yolox_rejects_wrong_input_size():
    from jetson.detectors.person import decode_yolox

    with pytest.raises(ValueError):
        decode_yolox(np.zeros((1, 100, 85), np.float32), 1.0)


def test_person_letterbox():
    pytest.importorskip("cv2")
    from jetson.detectors.person import letterbox

    blob, r = letterbox(np.zeros((480, 640, 3), np.uint8))
    assert blob.shape == (1, 3, 416, 416) and r == pytest.approx(0.65)
    assert blob[0, 0, -1, 0] == 114  # padded bottom


def test_unknown_person_backend_and_missing_model(tmp_path):
    from jetson.detectors.base import BackendUnavailable
    from jetson.detectors.person import create_person_detector

    with pytest.raises(ValueError):
        create_person_detector("nope")
    with pytest.raises(BackendUnavailable, match="jetson.fetch models"):
        create_person_detector("yolox", model=str(tmp_path / "missing.onnx"))


def test_hog_runs_without_a_download():
    pytest.importorskip("cv2")
    from jetson.detectors.person import create_person_detector

    det = create_person_detector("hog")
    assert det.detect(np.full((240, 320, 3), 127, np.uint8)) == []
    assert det.info()["licence"].startswith("Apache")


@pytest.fixture
def yolox_model():
    path = os.path.join(MODELS_DIR, "yolox_nano.onnx")
    if not os.path.exists(path):
        pytest.skip("YOLOX-nano missing: run `python -m jetson.fetch models`")
    return path


def test_yolox_finds_people_whose_faces_yunet_finds(media_image, yunet_model, yolox_model):
    import cv2

    from jetson.detectors import create_detector
    from jetson.detectors.person import create_person_detector

    img = cv2.imread(media_image)
    people = create_person_detector("yolox", model=yolox_model).detect(img)
    faces = create_detector("yunet", model=yunet_model).detect(img)
    assert len(people) >= 3
    ft = [trk(i, f.box) for i, f in enumerate(faces)]
    bt = [trk(100 + i, p.box) for i, p in enumerate(people)]
    assert len(associate(ft, bt)) >= len(faces) - 1  # (almost) every crew face sits in a body box


def test_yolox_onnxruntime_matches_opencv(media_image, yolox_model):
    import cv2

    pytest.importorskip("onnxruntime")
    from jetson.detectors.person import create_person_detector

    img = cv2.imread(media_image)
    a = create_person_detector("yolox", model=yolox_model).detect(img)
    b = create_person_detector("yolox", model=yolox_model, engine="onnxruntime").detect(img)
    assert len(a) == len(b) and all(x.box == pytest.approx(y.box, abs=1.0) for x, y in zip(a, b))
