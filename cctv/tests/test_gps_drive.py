"""Validation-drive tooling on a synthetic GPS log (no phone, no network)."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from cctv.geometry import MPH_PER_MPS
from cctv.groundtruth import gps_drive as gd

CAM = (40.80, -73.10)
T0 = datetime(2026, 11, 14, 17, 0, 0, tzinfo=timezone.utc)


def drive(mph=55.0, bearing_deg=90.0, offset_m=20.0, seconds=60, hz=1.0, start_m=-800.0):
    """Straight drive past CAM at constant speed; returns (t, lat, lon, speed m/s) rows."""
    v = mph / MPH_PER_MPS
    r = 6371000.0
    b = math.radians(bearing_deg)
    ux, uy = math.sin(b), math.cos(b)          # east, north
    px, py = -uy * offset_m, ux * offset_m      # lateral offset to the left
    out = []
    for k in range(int(seconds * hz)):
        s = start_m + v * k / hz
        x, y = px + ux * s, py + uy * s
        lat = CAM[0] + math.degrees(y / r)
        lon = CAM[1] + math.degrees(x / (r * math.cos(math.radians(CAM[0]))))
        out.append((T0 + timedelta(seconds=k / hz), lat, lon, v))
    return out


def write_gpx(path, rows, with_speed=True):
    pts = "".join(
        f'<trkpt lat="{la:.7f}" lon="{lo:.7f}"><time>{t.isoformat().replace("+00:00", "Z")}</time>'
        + (f"<extensions><speed>{v:.3f}</speed></extensions>" if with_speed else "") + "</trkpt>"
        for t, la, lo, v in rows)
    path.write_text('<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
                    f"<trk><trkseg>{pts}</trkseg></trk></gpx>")


def test_gpx_pass_speed_time_direction(tmp_path):
    p = tmp_path / "d.gpx"
    write_gpx(p, drive(55, 90))
    fixes = gd.load_log(p)
    assert len(fixes) == 60 and fixes[0].speed_mps == pytest.approx(55 / MPH_PER_MPS, rel=1e-3)
    passes = gd.find_passes(fixes, *CAM)
    assert len(passes) == 1
    ps = passes[0]
    assert ps.speed_source == "doppler" and ps.speed_mph == pytest.approx(55, abs=0.05)
    assert ps.direction == "EB" and ps.distance_m == pytest.approx(20, abs=1.0)
    expect = T0 + timedelta(seconds=800 / (55 / MPH_PER_MPS))   # closest approach after 800 m
    assert abs((ps.t_utc - expect).total_seconds()) < 0.2


def test_position_derived_speed_when_no_doppler(tmp_path):
    p = tmp_path / "d.gpx"
    write_gpx(p, drive(48, 270), with_speed=False)
    ps = gd.find_passes(gd.load_log(p), *CAM)[0]
    assert ps.speed_source == "positions" and ps.speed_mph == pytest.approx(48, abs=0.2) and ps.direction == "WB"


def test_csv_formats_and_units(tmp_path):
    rows = drive(60, 90)
    p = tmp_path / "d.csv"
    p.write_text("time,latitude,longitude,speed_mph,accuracy\n" + "".join(
        f"{t.isoformat()},{la},{lo},{v * MPH_PER_MPS:.3f},4\n" for t, la, lo, v in rows))
    fixes = gd.load_log(p)
    assert fixes[0].speed_mps == pytest.approx(60 / MPH_PER_MPS, rel=1e-3) and fixes[0].accuracy_m == 4
    q = tmp_path / "unix.csv"
    q.write_text("timestamp,lat,lon\n" + "".join(f"{t.timestamp():.0f},{la},{lo}\n" for t, la, lo, _ in rows))
    assert gd.load_log(q)[0].t == T0
    bad = tmp_path / "naive.csv"
    bad.write_text("time,lat,lon\n2026-11-14 12:00:00,40.8,-73.1\n")
    with pytest.raises(ValueError, match="time zone"):
        gd.load_log(bad)


def test_two_passes_both_directions():
    out = drive(55, 90, seconds=60) + [(t + timedelta(seconds=300), la, lo, v) for t, la, lo, v in drive(50, 270, seconds=60)]
    fixes = [gd.Fix(t, la, lo, v) for t, la, lo, v in out]
    ps = gd.find_passes(fixes, *CAM)
    assert [p.direction for p in ps] == ["EB", "WB"]
    assert [round(p.speed_mph) for p in ps] == [55, 50]


@pytest.mark.parametrize("b,d", [(90, "EB"), (60, "EB"), (270, "WB"), (300, "WB"), (0, "?"), (180, "?")])
def test_bearing_to_direction(b, d):
    assert gd.bearing_to_direction(b) == d


def test_candidates_and_report(tmp_path):
    from cctv import store

    con = store.connect(tmp_path / "v.sqlite")
    con.execute("INSERT INTO cameras (id, name, lat, lon) VALUES ('CAM', 'x', 40.8, -73.1)")
    fixes = [gd.Fix(t, la, lo, v) for t, la, lo, v in drive(55, 90)]
    ps = gd.find_passes(fixes, *CAM)[0]
    rid = store.insert_run(con, camera_id="CAM", clip_start_utc=T0.isoformat())
    ts = lambda dt: (ps.t_utc + timedelta(seconds=dt)).isoformat(timespec="seconds")
    store.insert_vehicle(con, run_id=rid, camera_id="CAM", track_id=1, cls="car", ts_utc=ts(-12), speed_mph=56.5,
                         sigma_mph=1.5, direction=1, accepted=1)
    store.insert_vehicle(con, run_id=rid, camera_id="CAM", track_id=2, cls="truck", ts_utc=ts(-10), speed_mph=None,
                         accepted=0, reject_reason="too short")
    store.insert_vehicle(con, run_id=rid, camera_id="CAM", track_id=3, cls="car", ts_utc=ts(-200), speed_mph=40,
                         sigma_mph=1.5, direction=1, accepted=1)
    con.commit()
    c = gd.candidates(con, "CAM", ps)
    assert [x["track_id"] for x in c] == [1, 2] and c[0]["offset_s"] == pytest.approx(-12, abs=1)
    rep = gd.report(con, [ps], [{"pass": 0, "run_id": rid, "track_id": 1}, {"pass": 0, "run_id": rid, "track_id": 2}])
    s = rep["summary"]
    assert s["matches"] == 2 and s["with_estimate"] == 1
    assert s["bias_mph"] == pytest.approx(1.5, abs=0.05) and s["share_within_2sigma"] == 1.0
    assert s["median_clock_offset_s"] < 0
    with pytest.raises(KeyError):
        gd.report(con, [ps], [{"pass": 5, "run_id": rid, "track_id": 1}])
