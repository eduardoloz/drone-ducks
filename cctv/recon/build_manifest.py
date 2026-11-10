"""Build ``data/cameras_lie.json``: every NY camera along I-495 (Long Island Expressway).

Input is a saved snapshot of the official 511NY camera layer (``data/raw/``), so
this script makes **no network requests**. Feed measurements produced by
``cctv.recon.measure_feed`` (``data/feed_measurements.json``) are merged in when
present.

    python -m cctv.recon.build_manifest            # from repo root
    python -m cctv.recon.build_manifest --raw other_snapshot.json

Classification (mainline / ramp / service road / ...) is name-based and therefore
approximate; fix individual cameras with ``OVERRIDES`` below and re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cctv.registry import DATA_DIR, MANIFEST_PATH

RAW_511 = DATA_DIR / "raw" / "511ny_cameras_lie_corridor_2026-11-10.json"
RAW_OPENCCTV = DATA_DIR / "raw" / "opencctv_i495_listing_2026-11-10.json"
MEASUREMENTS = DATA_DIR / "feed_measurements.json"

LIE_NAME = re.compile(r"(I-?495|\bLIE\b|Long Island Exp|\b495\b)", re.I)

# Approximate county lines where they cross the LIE (longitude, degrees).
# Queens|Nassau: between Exit 32 (Little Neck Pkwy, Queens) and Exit 33 (Lakeville Rd, Nassau).
# Nassau|Suffolk: between Exit 48 (Round Swamp Rd, Nassau) and Exit 49 (NY 110, Suffolk).
QUEENS_NASSAU_LON = -73.7015
NASSAU_SUFFOLK_LON = -73.4265

# Posted limits used for the "over limit" flag (see docs/SOURCES.md, accessed 2026-11-10):
# - Queens: NYC DOT posted-speed dataset (5mad-ntua) codes most of the LIE at 50 mph and
#   45 mph from the Queens-Midtown Tunnel plaza east to about 48th St.
# - Nassau/Suffolk to Riverhead: 55 mph (statewide maximum, VTL 1180(b); per-segment
#   NYSDOT confirmation still missing, only secondary sources).
POSTED_LIMIT_MPH = {"Queens": 50, "Nassau": 55, "Suffolk": 55}
LIC_45_MPH_EAST_LIMIT_LON = -73.919  # ~48th St: cameras west of this get 45 mph


def posted_limit(county: str, lon: float) -> int:
    if county == "Queens" and lon <= LIC_45_MPH_EAST_LIMIT_LON:
        return 45
    return POSTED_LIMIT_MPH[county]

# Manual fixes for name-based classification: camera id -> fields to override.
OVERRIDES: dict[str, dict[str, Any]] = {
    "R10_314": {"ptz": "fixed"},  # name says "(Fixed)"
    "R10_357": {"notes": "Seen zoomed onto an incident on 2026-11-10 (PTZ moves)."},
    "R10_362": {"exit": "72"},  # name "Exit 72 - 1" means camera 1 at Exit 72
    "R10_363": {"exit": "72"},
}

ROUTE_PREFIX = re.compile(
    r"^(I-278|I-678|NSP|SAG|NY ?25|NY ?27|Grand Ave|Main St|Kissena Blvd|OVD)", re.I
)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def county_for(lon: float) -> str:
    if lon < QUEENS_NASSAU_LON:
        return "Queens"
    if lon < NASSAU_SUFFOLK_LON:
        return "Nassau"
    return "Suffolk"


def classify_kind(name: str) -> str:
    n = name or ""
    if "welcome center" in n.lower():
        return "rest_area" if "east bound" in n.lower() else "indoor"
    if re.search(r"rest area", n, re.I):
        return "rest_area"
    if re.search(r"S/R|\bSSR\b|service r", n, re.I):
        return "service_road"
    if ROUTE_PREFIX.search(n):
        return "cross_route"
    if re.search(r"ramp|connect|interchange|shelter-\d|\bOVD\b", n, re.I):
        return "ramp_or_connector"
    return "mainline"


def parse_exit(name: str) -> str | None:
    m = re.search(r"(?:\bExits?\s*|-Ex)(\d+[A-Z]?(?:\s*(?:-|and)\s*\d+[A-Z]?)?)", name or "", re.I)
    return re.sub(r"\s+", "", m.group(1)).replace("and", "-") if m else None


def parse_direction(name: str) -> str | None:
    n = name or ""
    if re.search(r"((?:^|[-_\s])EB(?:$|[-_\s])|\bE/B\b|eastbound|east bound)", n, re.I):
        return "EB"
    if re.search(r"((?:^|[-_\s])WB(?:$|[-_\s])|\bW/B\b|westbound|west bound)", n, re.I):
        return "WB"
    return None


def camera_id(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    view = (raw.get("views") or [{}])[0]
    url = view.get("u") or ""
    m = re.search(r"skyvdn\.com/rtplive/([A-Za-z0-9_]+)/", url)
    if m:
        host = re.search(r"https://([^/]+)/", url).group(1)
        return m.group(1), {"type": "hls", "url": url, "host": host, "stream_id": m.group(1)}
    m = re.search(r"nyctmc\.org/api/cameras/([0-9a-f-]{36})/image", url)
    if m:
        return f"NYCDOT_{m.group(1)[:8]}", {"type": "jpeg", "url": url, "host": "nyctmc.org"}
    return f"511NY_{raw['id']}", {"type": "unknown", "url": url}


def norm_name(s: str) -> str:
    s = s.lower().replace("street", "st").replace("avenue", "ave").replace("road", "rd")
    return re.sub(r"[^a-z0-9]", "", s)


def match_opencctv(cam: dict[str, Any], occ: list[dict[str, Any]]) -> str | None:
    """Match an opencctv.org listing entry by name + position (no extra requests)."""
    best = None
    for o in occ:
        d = haversine_m(cam["lat"], cam["lon"], o["lat"], o["lon"])
        if d < 150 and norm_name(o["name"]) == norm_name(cam["name"]):
            if best is None or d < best[0]:
                best = (d, o["url"])
    return best[1] if best else None


def suitability(cam: dict[str, Any]) -> str:
    """How useful is this camera for *per-vehicle* speed? (see docs/SOURCES.md)."""
    if cam["kind"] in ("indoor",):
        return "excluded"
    if cam["feed"]["type"] == "jpeg":
        return "snapshot_only"
    meas = cam.get("measured")
    if meas and meas.get("ok"):
        fps = meas.get("effective_fps") or 0
        unique = meas.get("unique_fps") or 0
        h = meas.get("height") or 0
        if fps >= 10 and unique >= 10 and h >= 400:
            return "measured_good"
        if fps >= 5 and unique >= 5:
            return "measured_marginal"
        return "measured_unusable"
    if meas and not meas.get("ok"):
        return "offline_when_measured"
    return "candidate_video" if cam["kind"] == "mainline" else "candidate_video_secondary"


def build(raw_path: Path, occ_path: Path | None, meas_path: Path | None) -> dict[str, Any]:
    raw = json.loads(Path(raw_path).read_text())
    occ = json.loads(Path(occ_path).read_text()) if occ_path and Path(occ_path).exists() else []
    meas = json.loads(Path(meas_path).read_text()) if meas_path and Path(meas_path).exists() else {}
    meas_by_id = {m["camera_id"]: m for m in meas.get("results", [])}

    cams: list[dict[str, Any]] = []
    excluded_indoor = 0
    for r in raw["cams"]:
        if not LIE_NAME.search(r.get("name") or ""):
            continue
        cid, feed = camera_id(r)
        kind = classify_kind(r["name"])
        if kind == "indoor":
            excluded_indoor += 1  # building interiors at the LIE Welcome Center: not traffic, people present
            continue
        cam = {
            "id": cid,
            "name": r["name"],
            "owner": r.get("owner"),
            "ids": {"511ny": r["id"]},
            "lat": round(r["lat"], 6),
            "lon": round(r["lon"], 6),
            "county": county_for(r["lon"]),
            "kind": kind,
            "exit": parse_exit(r["name"]),
            "direction_hint": parse_direction(r["name"]),
            "ptz": "likely" if feed["type"] == "hls" else "unknown",
            "feed": feed,
            "broken_flag_511ny": bool((r.get("views") or [{}])[0].get("b")),
        }
        cam["posted_limit_mph"] = posted_limit(cam["county"], cam["lon"])
        url = match_opencctv(cam, occ)
        if url:
            cam["opencctv_url"] = url
            cam["ids"]["opencctv"] = int(url.rsplit("-", 1)[-1])
        cam.update(OVERRIDES.get(cid, {}))
        if cid in meas_by_id:
            cam["measured"] = meas_by_id[cid]
        cams.append(cam)

    # Approximate chainage (km from the Queens-Midtown Tunnel) along mainline cameras.
    main = sorted((c for c in cams if c["kind"] == "mainline"), key=lambda c: c["lon"])
    chain = 0.0
    for prev, cur in zip([None] + main[:-1], main):
        if prev is not None:
            chain += haversine_m(prev["lat"], prev["lon"], cur["lat"], cur["lon"]) / 1000
        cur["chainage_km"] = round(chain, 2)
    for c in cams:
        if "chainage_km" not in c and main:
            near = min(main, key=lambda m: haversine_m(c["lat"], c["lon"], m["lat"], m["lon"]))
            c["chainage_km"] = near["chainage_km"]
        c["co_located"] = sorted(
            o["id"] for o in cams
            if o is not c and haversine_m(c["lat"], c["lon"], o["lat"], o["lon"]) < 60
        )
        c["speed_suitability"] = suitability(c)

    cams.sort(key=lambda c: (c["chainage_km"], c["lon"]))
    summary = {
        "total": len(cams),
        "by_owner": dict(Counter(c["owner"] for c in cams)),
        "by_feed_type": dict(Counter(c["feed"]["type"] for c in cams)),
        "by_kind": dict(Counter(c["kind"] for c in cams)),
        "by_county": dict(Counter(c["county"] for c in cams)),
        "by_speed_suitability": dict(Counter(c["speed_suitability"] for c in cams)),
        "excluded_indoor_welcome_center": excluded_indoor,
        "matched_opencctv": sum(1 for c in cams if "opencctv_url" in c),
        "opencctv_listing_size": len(occ),
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corridor": "I-495 Long Island Expressway, Queens-Midtown Tunnel (Queens) to Riverhead (Suffolk)",
        "sources": {
            "camera_list": raw.get("source"),
            "camera_list_fetched_at": raw.get("fetched_at"),
            "opencctv_listing": "https://opencctv.org/cameras/united-states/new-york/roadway/i-495 (pages 1-6)",
            "measurements": str(meas_path.name) if meas else None,
            "terms": "511NY Developer Access Agreement https://511ny.org/help/24 ; video hosted by NYSDOT (skyvdn.com); NYC DOT stills at nyctmc.org",
        },
        "county_boundaries_note": "county is approximate: longitude cut-offs at the LIE "
        f"(Queens|Nassau {QUEENS_NASSAU_LON}, Nassau|Suffolk {NASSAU_SUFFOLK_LON})",
        "posted_limit_mph_by_county": POSTED_LIMIT_MPH,
        "posted_limit_note": "45 mph for Queens cameras west of ~48th St (lon <= "
        f"{LIC_45_MPH_EAST_LIMIT_LON}); service-road and ramp limits are lower and not modelled",
        "summary": summary,
        "cameras": cams,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_511)
    ap.add_argument("--opencctv", type=Path, default=RAW_OPENCCTV)
    ap.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    ap.add_argument("--out", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args(argv)
    manifest = build(args.raw, args.opencctv, args.measurements)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["summary"], indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
