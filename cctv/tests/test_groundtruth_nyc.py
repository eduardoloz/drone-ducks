"""NYC DOT link-speed harness: geometry, direction matching, time windows, comparison (no network)."""

import io
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from cctv.groundtruth import nyc_links as nl

EB_POINTS = "40.72966,-73.87775 40.730165,-73.87643  40.7304,-73.87567 40.73241,-73.86913001 40.7"  # truncated tail


def test_parse_link_points_skips_truncated_pairs():
    p = nl.parse_link_points(EB_POINTS)
    assert p.shape == (4, 2) and p[0, 0] == pytest.approx(40.72966)


def test_distance_to_polyline():
    poly = nl.parse_link_points(EB_POINTS)
    d, f = nl.distance_to_polyline_m(40.72966, -73.87775, poly)      # on the first vertex
    assert d < 1 and f == pytest.approx(0, abs=1e-6)
    d, _ = nl.distance_to_polyline_m(40.7400, -73.8700, poly)        # ~1 km north
    assert 700 < d < 1300


@pytest.mark.parametrize("name,want", [("LIE E 84TH ST - 108TH ST", "EB"), ("LIE W 108TH ST - 84TH ST", "WB"),
                                       ("LIE WB LITTLE NECK PKWY - NB CIP", "WB"), ("CIP S LIE - Hempstead Tpk", None)])
def test_link_direction(name, want):
    assert nl.link_direction(name) == want


def test_cameras_near_links_skips_stills_and_far_cameras():
    links = [{"link_id": "1", "link_name": "LIE E 84TH ST - 108TH ST", "link_points": EB_POINTS}]
    cams = [{"id": "ON", "lat": 40.7302, "lon": -73.8760, "feed": {"type": "hls"}},
            {"id": "STILL", "lat": 40.7302, "lon": -73.8760, "feed": {"type": "image"}},
            {"id": "FAR", "lat": 40.7500, "lon": -73.8760, "feed": {"type": "hls"}}]
    out = nl.cameras_near_links(links, cams)
    assert [c.camera_id for c in out] == ["ON"] and out[0].direction == "EB"


def test_harmonic_mean_and_ci():
    s = np.array([30.0, 60.0])
    assert nl.harmonic_mean(s) == pytest.approx(40.0)     # space-mean of 30 and 60 mph is 40, not 45
    lo, hi = nl.bootstrap_ci(np.full(20, 50.0), nl.harmonic_mean)
    assert lo == pytest.approx(50) and hi == pytest.approx(50)
    assert np.isnan(nl.bootstrap_ci(np.array([1.0, 2.0]), nl.harmonic_mean)[0])


def _rows():
    # floating local NY timestamps (EST = UTC-5): 12:00 local = 17:00 UTC
    return [{"link_id": "1", "speed": str(s), "status": "0", "data_as_of": f"2026-11-13T{hm}:00.000"}
            for hm, s in (("11:40", 20.0), ("11:58", 48.0), ("12:03", 52.0), ("12:30", 10.0))]


def test_link_speed_near_uses_only_rows_in_window_and_local_time():
    t0 = datetime(2026, 11, 13, 17, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 13, 17, 2, 0, tzinfo=timezone.utc)
    r = nl.link_speed_near(_rows(), "1", t0, t1, max_gap_min=10)
    assert r["n_rows"] == 2 and r["mean_mph"] == pytest.approx(50.0)
    assert r["nearest_mph"] == 52.0 and r["nearest_offset_s"] == 120
    assert nl.link_speed_near(_rows(), "1", t0, t1, max_gap_min=0.5) is None
    assert nl.link_speed_near(_rows(), "2", t0, t1) is None


def test_compare_only_matches_same_direction():
    runs = [{"run": {"id": 7, "camera_id": "CAM", "view_id": "v1", "clip_start_utc": "2026-11-13T17:00:00+00:00",
                     "duration_s": 60.0},
             "by_direction": {1: [50.0, 55.0, 45.0, 52.0], -1: [20.0, 22.0, 25.0]}}]
    cl = [nl.CameraLink("CAM", "1", "LIE E 84TH ST - 108TH ST", "EB", 3.0, 0.1),
          nl.CameraLink("CAM", "2", "LIE W 108TH ST - 84TH ST", "WB", 15.0, 0.9)]
    labels = {1: "EB", -1: "WB"}
    table = nl.compare(runs, cl, _rows(), label_fn=lambda c, v, s: labels[s])
    assert len(table) == 2
    eb = next(t for t in table if t["direction"] == "EB")
    assert eb["n"] == 4 and eb["link"]["mean_mph"] == pytest.approx(50.0)
    assert eb["delta_hmean_minus_link_mph"] == pytest.approx(eb["hmean_mph"] - 50.0, abs=0.06)
    wb = next(t for t in table if t["direction"] == "WB")
    assert wb["link"] is None                         # no WB rows: reported, not invented
    md = nl.format_table(table)
    assert "no row within window" in md and "| CAM |" in md
    # without direction labels nothing is compared (never mix carriageways)
    assert nl.compare(runs, cl, _rows(), label_fn=lambda c, v, s: None) == []


def test_our_runs_groups_accepted_speeds_by_direction(tmp_path):
    from cctv import store

    con = store.connect(tmp_path / "g.sqlite")
    con.execute("INSERT INTO cameras (id, name, lat, lon) VALUES ('CAM', 'test', 40.7, -73.9)")
    rid = store.insert_run(con, camera_id="CAM", clip_start_utc="2026-11-13T17:00:00+00:00", duration_s=60.0,
                           camera_moved=0, view_id="v1")
    moved = store.insert_run(con, camera_id="CAM", clip_start_utc="2026-11-13T17:10:00+00:00", camera_moved=1)
    for tid, (spd, d, acc) in enumerate([(50, 1, 1), (55, 1, 1), (30, -1, 1), (99, 1, 0)]):
        store.insert_vehicle(con, run_id=rid, camera_id="CAM", track_id=tid, speed_mph=spd, direction=d, accepted=acc)
    store.insert_vehicle(con, run_id=moved, camera_id="CAM", track_id=1, speed_mph=40, direction=1, accepted=1)
    con.commit()
    out = nl.our_runs(con, ["CAM"])
    assert len(out) == 1 and out[0]["by_direction"] == {1: [50.0, 55.0], -1: [30.0]}


def test_client_sends_generic_user_agent_and_optional_token(monkeypatch):
    seen = {}

    class Resp(io.BytesIO):
        headers = {"X-SODA2-Data-Out-Of-Date": "false"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        seen["token"] = req.get_header("X-app-token")
        seen["url"] = req.full_url
        return Resp(json.dumps([{"link_id": "1"}]).encode())

    monkeypatch.setattr(nl.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("SOCRATA_APP_TOKEN", "abc")
    rows = nl.fetch_link_rows(["4362342"], datetime(2026, 11, 13, 12), datetime(2026, 11, 13, 13))
    assert rows == [{"link_id": "1"}]
    assert "@" not in seen["ua"] and seen["token"] == "abc"
    assert "i4gi-tjb9" in seen["url"] and "4362342" in seen["url"]
    with pytest.raises(ValueError):
        nl.fetch_link_rows(["1') OR ('1'='1"], datetime(2026, 11, 13), datetime(2026, 11, 13, 1))


def test_committed_link_snapshot_has_queens_cameras():
    snap = json.loads(nl.LINKS_SNAPSHOT.read_text())
    cams = {(c["camera_id"], c["direction"]) for c in snap["camera_links"]}
    assert ("R11_140", "EB") in cams and ("R11_141", "WB") in cams


def test_bad_status_rows_are_skipped_and_quality_flags_them():
    t0 = datetime(2026, 11, 13, 17, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 13, 17, 2, 0, tzinfo=timezone.utc)
    rows = _rows() + [{"link_id": "1", "speed": "114.3", "status": "-101", "data_as_of": "2026-11-13T12:01:00.000"}]
    r = nl.link_speed_near(rows, "1", t0, t1)
    assert r["skipped_bad_status"] == 1 and r["mean_mph"] == pytest.approx(50.0)
    assert nl.link_speed_near(rows, "1", t0, t1, ok_status=None)["n_rows"] == 3
    q = nl.link_quality(rows)[0]
    assert q["rows"] == 5 and q["status_counts"] == {"0": 4, "-101": 1} and q["implausible_rows"] == 1


def test_measure_lag_reports_newest_row_and_last_modified():
    now = datetime(2026, 11, 18, 2, 22, 51, tzinfo=timezone.utc)          # 21:22 local
    seen = {}

    def fake_get(params):
        seen.update(params)
        return ([{"link_id": "4362251", "newest": "2026-11-17T19:32:06.000", "n": "91"}],
                {"X-SODA2-Data-Out-Of-Date": "true", "Last-Modified": "Wed, 18 Nov 2026 00:37:38 GMT"})

    lag = nl.measure_lag(["4362251", "4362249"], now=now, get=fake_get)
    assert "4362251" in seen["$where"] and "max(data_as_of)" in seen["$select"]
    link = lag["links"]["4362251"]
    assert link["lag_min"] == pytest.approx(110.75, abs=0.1)              # 19:32:06 local = 00:32:06 UTC
    assert lag["minutes_since_last_modified"] == pytest.approx(105.2, abs=0.1)
    assert lag["out_of_date_header"] == "true"
    assert lag["links"]["4362249"]["lag_min"] is None                      # no row: reported, not invented
    with pytest.raises(ValueError):
        nl.measure_lag(["1) OR (1=1"], now=now, get=fake_get)


def test_load_link_rows_merges_cache_and_reference_without_duplicates(tmp_path):
    a, b = tmp_path / "cache", tmp_path / "reference"
    a.mkdir()
    b.mkdir()
    r1 = {"link_id": "1", "data_as_of": "2026-11-17T16:02:03.000", "speed": "33.5", "status": "0"}
    r2 = {"link_id": "1", "data_as_of": "2026-11-17T16:07:04.000", "speed": "31.7", "status": "0"}
    (a / "speeds_x.json").write_text(json.dumps({"rows": [r1]}))
    (b / "link1.json").write_text(json.dumps({"rows": [r1, r2]}))
    rows = nl.load_link_rows([a, b, tmp_path / "missing"])
    assert [r["speed"] for r in rows] == ["33.5", "31.7"]


def test_clips_in_window_and_reference_record(tmp_path):
    def clip(name, cam, start, size=10):
        (tmp_path / f"{name}.json").write_text(json.dumps({"camera_id": cam, "started_utc": start, "returncode": 0}))
        (tmp_path / f"{name}.ts").write_bytes(b"x" * size)

    clip("R10_008_a", "R10_008", "2026-11-17T21:27:41.000+00:00")
    clip("R10_008_empty", "R10_008", "2026-11-17T21:30:00.000+00:00", size=0)     # failed recording
    clip("R10_041_a", "R10_041", "2026-11-17T21:26:43.000+00:00")                 # not on the link
    clip("R10_007_late", "R10_007", "2026-11-17T23:51:00.000+00:00")              # outside the window
    t0 = datetime(2026, 11, 17, 21, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 17, 22, 0, tzinfo=timezone.utc)
    clips = nl.clips_in_window(["R10_007", "R10_008"], t0, t1, clips_dir=tmp_path)
    assert [c["clip"] for c in clips] == ["R10_008_a.ts"]
    rows = [{"link_id": "1", "data_as_of": "2026-11-17T16:07:04.000", "speed": "31.7", "status": "0", "extra": 1},
            {"link_id": "1", "data_as_of": "2026-11-17T16:02:03.000", "speed": "33.5", "status": "0"}]
    rec = nl.reference_record("1", "LIE WB X", rows, "2026-11-17T16:00", "2026-11-17T17:00", clips, "now")
    assert [r["speed"] for r in rec["rows"]] == ["33.5", "31.7"] and "extra" not in rec["rows"][0]
    assert rec["dataset"] == nl.DATASET and rec["clips"][0]["camera_id"] == "R10_008"
