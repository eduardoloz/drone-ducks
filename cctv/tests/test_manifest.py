"""Data tests: the committed camera manifest and the name-parsing rules that built it."""

import re

from cctv.recon.build_manifest import (classify_kind, county_for, parse_direction, parse_exit,
                                       posted_limit)
from cctv.registry import load_manifest


def test_name_rules():
    assert classify_kind("I-495 at Glen Cove Rd") == "mainline"
    assert classify_kind("I-495 at Shelter Rock Rd") == "mainline"
    assert classify_kind("I-278 at LIE Interchange/Shelter-2") == "cross_route"
    assert classify_kind("Main St @ LIE N S/R") == "service_road"
    assert classify_kind("I-495 ramp to W/B I-278") == "ramp_or_connector"
    assert classify_kind("LIE Welcome Center Kitchen") == "indoor"
    assert parse_exit("I-495 at Manetto Hill Rd (Exit 45)") == "45"
    assert parse_exit("C5-LIE-17-EB_at_GCP-Ex22A_GCP") == "22A"
    assert parse_exit("I-495 between Exits 40 and 39") == "40-39"
    assert parse_direction("C5-LIE-02-EB_at_27th_St-Ex15") == "EB"
    assert parse_direction("I-495 ramp to W/B I-278") == "WB"
    assert parse_direction("I-495 at Glen Cove Rd") is None


def test_counties_and_limits():
    assert county_for(-73.80) == "Queens" and county_for(-73.60) == "Nassau" and county_for(-73.20) == "Suffolk"
    assert posted_limit("Queens", -73.94) == 45     # Long Island City approach to the tunnel
    assert posted_limit("Queens", -73.80) == 50
    assert posted_limit("Suffolk", -73.20) == 55


def test_committed_manifest_is_consistent():
    m = load_manifest()
    cams = m["cameras"]
    ids = [c["id"] for c in cams]
    assert len(ids) == len(set(ids)), "camera ids must be unique"
    assert m["summary"]["total"] == len(cams) >= 100
    for c in cams:
        assert 40.6 < c["lat"] < 41.0 and -74.0 < c["lon"] < -72.6, c["id"]
        assert c["kind"] != "indoor"
        if c["feed"]["type"] == "hls":
            assert re.match(r"https://s\d+\.nysdot\.skyvdn\.com/rtplive/[A-Z0-9_]+/playlist\.m3u8$", c["feed"]["url"])
        assert c["posted_limit_mph"] in (45, 50, 55)
    # chainage increases eastward along the mainline
    main = [c for c in cams if c["kind"] == "mainline"]
    assert max(c["chainage_km"] for c in main) > 100
