"""Per-PTZ-preset calibrations: layout, legacy compatibility, view matching, pipeline selection."""

import json

import cv2
import numpy as np
import pytest

from cctv import views as views_mod
from cctv.calibrate import main as calibrate_main
from cctv.geometry import Calibration
from cctv.registry import calibration_path
from cctv.tests.test_camera_motion import scene, shift
from cctv.views import LEGACY_VIEW, get_view, list_views, match_view, primary_view, validate_view_id


def _cal(camera_id="CAM", view_id=None):
    img = [[100, 400], [500, 400], [200, 250], [420, 250], [300, 300]]
    road = [[0, 0], [0, 10], [30, 0], [30, 10], [15, 5]]
    return Calibration.fit(camera_id, (640, 480), img, road, view_id=view_id)


def _make_view(d, cam, vid, img=None, cal=True, tag=None):
    (d / cam).mkdir(parents=True, exist_ok=True)
    if cal:
        _cal(cam, vid).save(d / cam / f"{vid}.json")
    if img is not None:
        cv2.imwrite(str(d / cam / (f"{vid}.ref.jpg" if not tag else f"{vid}.ref-{tag}.jpg")), img)


def test_list_views_directory_and_legacy(tmp_path):
    _make_view(tmp_path, "CAM", "b", scene(1))
    _make_view(tmp_path, "CAM", "a", scene(0), cal=False)
    _make_view(tmp_path, "CAM", "a", scene(2), cal=False, tag="night")
    _cal("CAM").save(tmp_path / "CAM.json")               # round-1 flat file -> view "default"
    cv2.imwrite(str(tmp_path / "CAM.ref.jpg"), scene(3))
    vs = list_views("CAM", tmp_path)
    assert [v.view_id for v in vs] == ["a", "b", LEGACY_VIEW]
    a = vs[0]
    assert not a.calibrated and [p.name for p in a.reference_paths] == ["a.ref.jpg", "a.ref-night.jpg"]
    assert vs[2].legacy and vs[2].calibrated
    assert primary_view("CAM", tmp_path).view_id == "b"   # first *calibrated* view
    assert get_view("CAM", "a", tmp_path).view_id == "a"
    with pytest.raises(KeyError):
        get_view("CAM", "zzz", tmp_path)


def test_legacy_only_camera_still_works(tmp_path):
    _cal("OLD").save(tmp_path / "OLD.json")
    vs = list_views("OLD", tmp_path)
    assert len(vs) == 1 and vs[0].view_id == LEGACY_VIEW and vs[0].load_calibration().camera_id == "OLD"
    assert list_views("NONE", tmp_path) == []


def test_old_calibration_json_without_new_fields_loads(tmp_path):
    d = json.loads(json.dumps(_cal().__dict__, default=list))
    for k in ("view_id", "direction_labels"):
        d.pop(k)
    d["some_future_key"] = 1
    p = tmp_path / "x.json"
    p.write_text(json.dumps(d))
    cal = Calibration.load(p)
    assert cal.view_id is None and cal.direction_labels is None


def test_view_id_validation():
    assert validate_view_id("v1") == "v1"
    for bad in ("", "../x", "a b", "views", "x" * 41):
        with pytest.raises(ValueError):
            validate_view_id(bad)


def test_match_picks_the_right_preset(tmp_path):
    _make_view(tmp_path, "CAM", "east", scene(0))
    _make_view(tmp_path, "CAM", "west", scene(1))
    m = match_view("CAM", shift(scene(1), 1, 0), cal_dir=tmp_path)
    assert m.status == "matched" and m.view.view_id == "west" and m.usable
    assert m.calibration.view_id == "west"
    assert len(m.candidates) == 2
    m = match_view("CAM", scene(0), cal_dir=tmp_path)
    assert m.view.view_id == "east"


def test_unknown_view_is_rejected(tmp_path):
    _make_view(tmp_path, "CAM", "east", scene(0))
    _make_view(tmp_path, "CAM", "west", scene(1))
    m = match_view("CAM", scene(42), cal_dir=tmp_path)
    assert m.status == "no_match" and m.view is None and not m.usable and m.calibration is None
    m = match_view("CAM", shift(scene(0), 15, 0), cal_dir=tmp_path)  # known preset but nudged 15 px
    assert m.status == "no_match" and "shift" in m.reason


def test_known_but_uncalibrated_preset(tmp_path):
    _make_view(tmp_path, "CAM", "east", scene(0))
    _make_view(tmp_path, "CAM", "turned", None, cal=False)
    _make_view(tmp_path, "CAM", "turned", scene(5), cal=False, tag="night")
    m = match_view("CAM", scene(5), cal_dir=tmp_path)
    assert m.status == "matched" and m.view.view_id == "turned" and not m.usable
    assert "not calibrated" in m.reason


def test_no_reference_frames(tmp_path):
    _make_view(tmp_path, "CAM", "east", None)
    assert match_view("CAM", scene(0), cal_dir=tmp_path).status == "no_views"


def test_pipeline_select_view(tmp_path, monkeypatch):
    from cctv import pipeline

    _make_view(tmp_path, "CAM", "east", scene(0))
    _make_view(tmp_path, "CAM", "west", scene(1))
    monkeypatch.setattr(views_mod, "CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "list_views", lambda cid: list_views(cid, tmp_path))
    monkeypatch.setattr(pipeline, "match_view", lambda cid, f, v: match_view(cid, f, v, cal_dir=tmp_path))
    clip = tmp_path / "clip.ts"
    clip.write_bytes(b"x")
    for frame, want in ((scene(1), "west"), (scene(0), "east")):
        monkeypatch.setattr(pipeline, "clip_median_frame", lambda c, every=30, f=frame: f)
        cal, m, moved = pipeline.select_view("CAM", clip)
        assert moved is False and m.view.view_id == want and cal.view_id == want
    monkeypatch.setattr(pipeline, "clip_median_frame", lambda c, every=30: scene(77))
    cal, m, moved = pipeline.select_view("CAM", clip)
    assert moved is True and cal is None and m.status == "no_match"
    # unreadable clip with two calibrated views: refuse to guess
    cal, m, moved = pipeline.select_view("CAM", tmp_path / "missing.ts")
    assert cal is None and moved is None and m.status == "unreadable"


def test_migrate_cli_moves_flat_files(tmp_path, monkeypatch):
    import cctv.calibrate as calmod

    _cal("CAM").save(tmp_path / "CAM.json")
    (tmp_path / "CAM.points.json").write_text("{}")
    cv2.imwrite(str(tmp_path / "CAM.ref.jpg"), scene(0))
    monkeypatch.setattr(calmod, "CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(calmod, "camera_dir", lambda cam: tmp_path / cam)
    monkeypatch.setattr(views_mod, "CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(calmod, "set_view_label",
                        lambda c, v, label: views_mod.set_view_label(c, v, label, cal_dir=tmp_path))
    calibrate_main(["migrate", "--camera", "CAM", "--view", "p1", "--label", "first preset"])
    assert not (tmp_path / "CAM.json").exists()
    vs = list_views("CAM", tmp_path)
    assert [v.view_id for v in vs] == ["p1"] and vs[0].label == "first preset"
    assert vs[0].load_calibration().view_id == "p1" and vs[0].points_path and vs[0].reference_paths


def test_committed_r10_041_layout():
    vs = {v.view_id: v for v in list_views("R10_041")}
    assert vs["v1"].calibrated and vs["v1"].reference_paths
    assert not vs["v2"].calibrated and vs["v2"].reference_paths[0].name == "v2.ref-night.jpg"
    assert calibration_path("R10_041").name == "v1.json"


def test_committed_r10_041_references_tell_views_apart():
    """The two real presets of camera 041 (day v1, night v2) must not match each other."""
    vs = {v.view_id: v for v in list_views("R10_041")}
    v1 = cv2.imread(str(vs["v1"].reference_paths[0]))
    m = match_view("R10_041", v1)
    assert m.view.view_id == "v1" and m.usable
    v2 = cv2.imread(str(vs["v2"].reference_paths[0]))
    m = match_view("R10_041", v2)
    assert m.view.view_id == "v2" and not m.usable
    assert all(c["moved"] for c in m.candidates if c["view_id"] == "v1")


def test_store_migrates_old_runs_table(tmp_path):
    import sqlite3

    from cctv import store

    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, source TEXT)")
    con.commit()
    con.close()
    con = store.connect(p)
    assert "view_id" in {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    np.testing.assert_equal(con.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 0)


def test_api_lists_views_and_serves_each_reference(tmp_path):
    from fastapi.testclient import TestClient

    from cctv.web.app import create_app

    client = TestClient(create_app(db_path=tmp_path / "v.sqlite"))
    d = client.get("/api/cameras/R10_041").json()
    assert d["calibration"]["view_id"] == "v1" and {v["view_id"] for v in d["views"]} >= {"v1", "v2"}
    assert client.get("/api/cameras/R10_041/reference.jpg?view=v2").status_code == 200
    assert client.get("/api/cameras/R10_041/reference.jpg?view=nope").status_code == 404
    c = client.get("/api/cameras/R10_041/calibration?view=v2").json()
    assert c["view_id"] == "v2" and c["calibration"] is None
    assert client.get("/api/cameras/R10_041/calibration").json()["points"]["camera_id"] == "R10_041"
