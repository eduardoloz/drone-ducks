"""Regression: real YOLO11n + ByteTrack tracks from R10_041 (view v1) -> pipeline -> fixed numbers.

The fixture holds only bounding boxes (no imagery) of the 62 tracks that entered the
calibrated stretch in the 60 s clip of 2026-11-10 16:39 local (the clip behind the README
figures). If a change to speed.py, geometry.py, the calibration or the pipeline moves these
numbers, this test says so; update the expectations only on purpose and note why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cctv.detect_track import group_tracks, load_jsonl
from cctv.pipeline import process

FIXTURE = Path(__file__).parent / "data" / "r10_041_v1_20261110T213935Z.tracks.jsonl.gz"


def test_fixture_is_boxes_only():
    pts, meta = load_jsonl(FIXTURE)
    assert len(group_tracks(pts)) == 62 and meta["model"] == "yolo11n.pt"
    assert set(vars(pts[0])) == {"frame", "t", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"}


def test_r10_041_v1_clip_numbers_are_stable(tmp_path):
    clip = tmp_path / "R10_041_20261110T213935Z.ts"     # not present: view check skipped, v1 is the only calibration
    s = process("R10_041", clip, tmp_path / "r.sqlite", tracks_path=FIXTURE)
    assert s["view_id"] == "v1"
    assert (s["vehicles"], s["accepted"], s["over_limit"]) == (62, 54, 21)
    assert s["rejected"] == {"too short": 3, "too few points": 4, "jumpy track": 1}
    assert s["median_mph"] == pytest.approx(57.2, abs=0.05)
    assert s["p85_mph"] == pytest.approx(62.0, abs=0.05)
    assert (s["min_mph"], s["max_mph"]) == (pytest.approx(46.5, abs=0.05), pytest.approx(71.9, abs=0.05))
    assert s["light"] == "twilight" and -1.5 < s["sun_elev_deg"] < 0   # recorded just after sunset
