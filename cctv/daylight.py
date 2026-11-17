"""Is there daylight at a camera right now? Sun elevation, sunrise/sunset, light condition.

Why: the detector (YOLO11n, COCO) and the calibration (lane dashes) both need
daylight. At night the LIE cameras show headlights and light streaks, and the
pipeline's numbers mean something different (see README, "Night"). So:

* the recording script refuses to record outside daylight unless forced,
* the pipeline stores the sun elevation and a light condition with every run,
* the site labels night/twilight runs instead of mixing them silently with day runs.

The sun position uses the NOAA solar calculator equations (accurate to well under
0.1 degree for 1900-2100); no network, no extra dependency.

    python -m cctv.daylight                         # today at R10_041: sunrise, sunset, light now
    python -m cctv.daylight --camera R10_008 --check  # exit 0 in daylight, 1 otherwise (for scripts)
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# Light condition thresholds on the sun's elevation (degrees, geometric, no refraction):
#   day       >= DAY_MIN_ELEV   (observed 2026-11-17: R10_007/008/041 clips at sun +0.4 to +1 deg
#                                still show lane dashes and well-lit vehicles)
#   twilight  between            (headlights on, contrast falling: treat with care)
#   night     <  NIGHT_MAX_ELEV  (civil twilight over: headlights only)
DAY_MIN_ELEV = 0.0
NIGHT_MAX_ELEV = -6.0
SUNRISE_ELEV = -0.833        # standard sunrise/sunset: refraction + solar radius


def _julian_day(dt: datetime) -> float:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.timestamp() / 86400.0 + 2440587.5


def sun_elevation_deg(lat: float, lon: float, when: datetime) -> float:
    """Geometric sun elevation (degrees above the horizon) at lat/lon (east positive) and time."""
    jc = (_julian_day(when) - 2451545.0) / 36525.0
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * jc) + math.sin(3 * mr) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * jc)
    app_long = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(omega))
    eps0 = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(eps) * math.sin(app_long))
    y = math.tan(eps / 2) ** 2
    l0r = math.radians(l0)
    eot_min = 4.0 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr)
                                 + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                                 - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    u = when.astimezone(timezone.utc)
    minutes = u.hour * 60 + u.minute + (u.second + u.microsecond / 1e6) / 60.0
    tst = (minutes + eot_min + 4.0 * lon) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    latr = math.radians(lat)
    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))


def light_condition(elev_deg: float) -> str:
    """'day', 'twilight' or 'night' from the sun elevation."""
    if elev_deg >= DAY_MIN_ELEV:
        return "day"
    if elev_deg < NIGHT_MAX_ELEV:
        return "night"
    return "twilight"


def _crossings(lat: float, lon: float, day: date, tz: ZoneInfo, elev: float) -> list[tuple[datetime, bool]]:
    """Times on a local calendar day when the sun crosses ``elev``; bool = rising."""
    t0 = datetime(day.year, day.month, day.day, tzinfo=tz)
    step = timedelta(minutes=10)
    out = []
    a, fa = t0, sun_elevation_deg(lat, lon, t0) - elev
    for _ in range(int(timedelta(days=1) / step)):
        b = a + step
        fb = sun_elevation_deg(lat, lon, b) - elev
        if (fa < 0) != (fb < 0):
            lo, hi, flo = a, b, fa
            for _ in range(30):              # bisection to well under a second
                mid = lo + (hi - lo) / 2
                fm = sun_elevation_deg(lat, lon, mid) - elev
                if (fm < 0) == (flo < 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            out.append((lo + (hi - lo) / 2, fa < 0))
        a, fa = b, fb
    return out


def sun_times(lat: float, lon: float, day: date, tz: ZoneInfo = NY_TZ) -> dict[str, datetime | None]:
    """Sunrise/sunset and the start/end of usable daylight (sun >= DAY_MIN_ELEV) on a local day."""
    out: dict[str, datetime | None] = {"sunrise": None, "sunset": None, "day_start": None, "day_end": None}
    for t, rising in _crossings(lat, lon, day, tz, SUNRISE_ELEV):
        out["sunrise" if rising else "sunset"] = t
    for t, rising in _crossings(lat, lon, day, tz, DAY_MIN_ELEV):
        out["day_start" if rising else "day_end"] = t
    return out


def camera_light(camera: dict, when: datetime) -> dict:
    """Sun elevation and light condition for a manifest camera dict at ``when``."""
    elev = sun_elevation_deg(float(camera["lat"]), float(camera["lon"]), when)
    return {"sun_elev_deg": round(elev, 2), "light": light_condition(elev)}


def main(argv: list[str] | None = None) -> int:
    from cctv.registry import get_camera

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="R10_041")
    ap.add_argument("--at", help="local NY time to check instead of now, e.g. 2026-11-18T10:00")
    ap.add_argument("--check", action="store_true", help="exit 0 if it is 'day' at the camera, else 1")
    args = ap.parse_args(argv)
    cam = get_camera(args.camera)
    when = datetime.fromisoformat(args.at).replace(tzinfo=NY_TZ) if args.at else datetime.now(NY_TZ)
    info = camera_light(cam, when)
    times = sun_times(cam["lat"], cam["lon"], when.astimezone(NY_TZ).date())
    fmt = lambda t: t.astimezone(NY_TZ).strftime("%H:%M") if t else "--"   # noqa: E731
    print(f"{args.camera} {when.astimezone(NY_TZ):%Y-%m-%d %H:%M} NY: sun {info['sun_elev_deg']:+.1f} deg -> "
          f"{info['light']}; sunrise {fmt(times['sunrise'])}, sunset {fmt(times['sunset'])}, "
          f"usable daylight (sun >= {DAY_MIN_ELEV:g} deg) {fmt(times['day_start'])}-{fmt(times['day_end'])}")
    return 0 if (not args.check or info["light"] == "day") else 1


if __name__ == "__main__":
    sys.exit(main())
