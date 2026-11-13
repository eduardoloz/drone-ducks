"""Follow-person end to end in Gazebo SITL: examples/follow_person.py as a subprocess.

Needs the camera bridge (:5601) + pose probe (:5602) from start_sim.sh and the models
(python -m jetson.fetch models). Which test runs depends on the world that is up:

  WORLD=quack_people start_sim.sh            face-only follow of the lead walker
  WORLD=quack_people_turnaway start_sim.sh   face+body (YOLOX-nano) follow through a 14 s turn-away

Each takes 2-4 min: the script waits for the walker's loop to restart, follows, and lands after the
walker finally turns away.

Pass bars (deliberately below the live runs, see docs/ARCHITECTURE.md "Results"):
  lead:     followed >= 50 s, face centred within 10 deg for >= 80 % of the follow, ended by the
            lost-target timeout with a landing
  turnaway: did NOT land early (still following at the walker's final turn), followed >= 50 s,
            some frames steered by the body, landed at the end. (With the body lock the drone also
            follows the 1.4 m/s walk-back and the next loop, so the session usually ends at
            --max-seconds rather than by the lost timeout.)
"""
import json
import os
import subprocess
import sys
import urllib.request

import pytest

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(os.path.dirname(SIM), "jetson", "models")


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _world_or_skip(world):
    poses = _get("http://127.0.0.1:5602/poses")
    if not _get("http://127.0.0.1:5601/health") or not poses:
        pytest.skip(f"needs the camera bridge (:5601) and pose probe (:5602): WORLD={world} start_sim.sh")
    if "x500_quack_0" not in poses.get("poses", {}):
        pytest.skip("pose probe has no drone yet")
    if poses.get("world") != world:
        pytest.skip(f"world {poses.get('world')!r} is up, this test needs WORLD={world}")


def _follow(tmp_path, *extra):
    proc = subprocess.run([sys.executable, os.path.join(SIM, "examples", "follow_person.py"),
                           "--out", str(tmp_path), *extra], capture_output=True, text=True, timeout=480)
    print(proc.stdout[-3000:])
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.load(open(next(tmp_path.glob("follow-*.json"))))


def test_follow_walker_then_land_when_lost(tmp_path):
    _world_or_skip("quack_people")
    summary = _follow(tmp_path)
    assert summary["follow_s"] >= 50
    assert summary["pct_centred_10deg"] >= 80
    assert summary["final_mode"] == "TIMEOUT" and "LAND" in summary["finish_reason"]


def test_body_lock_follows_through_long_turn_away(tmp_path):
    _world_or_skip("quack_people_turnaway")
    if not os.path.isfile(os.path.join(MODELS, "yolox_nano.onnx")):
        pytest.skip("YOLOX-nano missing: (cd .. && sim/.venv/bin/python -m jetson.fetch models)")
    summary = _follow(tmp_path, "--person", "yolox")
    assert summary["ended_early"] is False, f"landed at walker phase {summary['walker_end_phase_s']}"
    assert summary["follow_s"] >= 50
    assert summary["body_frames"] > 0
    assert "LAND" in summary["finish_reason"]
