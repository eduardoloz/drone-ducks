"""Sun position / light condition (no network)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from cctv import daylight
from cctv.daylight import NY_TZ, camera_light, light_condition, sun_elevation_deg, sun_times

NYC = (40.7128, -74.0060)


def _hm(t: datetime) -> float:
    t = t.astimezone(NY_TZ)
    return t.hour + t.minute / 60 + t.second / 3600


def test_sunrise_sunset_nyc_match_almanac():
    # Published almanac values for New York City (NOAA / USNO), within 3 minutes.
    t = sun_times(*NYC, date(2026, 11, 17))
    assert _hm(t["sunrise"]) == pytest.approx(6 + 44 / 60, abs=3 / 60)
    assert _hm(t["sunset"]) == pytest.approx(16 + 35 / 60, abs=3 / 60)
    t = sun_times(*NYC, date(2026, 6, 21))          # EDT; longest day
    assert _hm(t["sunrise"]) == pytest.approx(5 + 25 / 60, abs=3 / 60)
    assert _hm(t["sunset"]) == pytest.approx(20 + 31 / 60, abs=3 / 60)


def test_noon_elevation_solstice():
    # Max elevation at the June solstice is 90 - lat + 23.44 deg.
    best = max(sun_elevation_deg(*NYC, datetime(2026, 6, 21, 16, m, tzinfo=timezone.utc)) for m in range(0, 60, 2))
    assert best == pytest.approx(90 - NYC[0] + 23.44, abs=0.2)


def test_light_condition_thresholds():
    assert light_condition(30) == "day"
    assert light_condition(daylight.DAY_MIN_ELEV) == "day"
    assert light_condition(-3) == "twilight"
    assert light_condition(-20) == "night"


def test_camera_light_on_recorded_clips():
    # R10_008 was recorded at 16:27 local on 2026-11-17 (sun just above the horizon, clear light)
    # and the night clips after 17:15 (civil twilight over).
    cam = {"lat": 40.759786, "lon": -73.734113}
    assert camera_light(cam, datetime(2026, 11, 17, 16, 27, tzinfo=NY_TZ))["light"] == "day"
    assert camera_light(cam, datetime(2026, 11, 17, 16, 50, tzinfo=NY_TZ))["light"] == "twilight"
    assert camera_light(cam, datetime(2026, 11, 17, 17, 30, tzinfo=NY_TZ))["light"] == "night"
    assert camera_light(cam, datetime(2026, 11, 18, 11, 0, tzinfo=NY_TZ))["light"] == "day"


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        sun_elevation_deg(*NYC, datetime(2026, 11, 17, 12, 0))


def test_cli_check_exit_code(capsys):
    assert daylight.main(["--camera", "R10_041", "--at", "2026-11-18T11:00", "--check"]) == 0
    assert daylight.main(["--camera", "R10_041", "--at", "2026-11-18T22:00", "--check"]) == 1
    assert "sunset" in capsys.readouterr().out
