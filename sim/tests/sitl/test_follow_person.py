"""Follow-person end to end in Gazebo SITL: examples/follow_person.py as a subprocess.

Needs `WORLD=quack_people scripts/start_sim.sh` (camera :5601 + pose probe :5602) and the YuNet
model (python -m jetson.fetch models). Takes 2-4 min: the script waits for the walker's loop to
restart, follows it for ~65 s, and lands after the walker turns away.

Pass bar (deliberately below the first live run: 73 s followed, 100 % within 5 deg, 2 reacquired):
followed >= 50 s, the face centred within 10 deg for >= 80 % of the follow, the session ended by
the lost-target timeout with a landing.
"""
import json
import os
import subprocess
import sys
import urllib.request

import pytest

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def test_follow_walker_then_land_when_lost(tmp_path):
    poses = _get("http://127.0.0.1:5602/poses")
    if not _get("http://127.0.0.1:5601/health") or not poses:
        pytest.skip("needs the camera bridge (:5601) and pose probe (:5602): WORLD=quack_people start_sim.sh")
    if "x500_quack_0" not in poses.get("poses", {}) or not os.path.exists(os.path.join(SIM, "gazebo")):
        pytest.skip("pose probe has no drone yet")
    proc = subprocess.run([sys.executable, os.path.join(SIM, "examples", "follow_person.py"),
                           "--out", str(tmp_path)], capture_output=True, text=True, timeout=420)
    print(proc.stdout[-3000:])
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.load(open(next(tmp_path.glob("follow-*.json"))))
    assert summary["follow_s"] >= 50
    assert summary["pct_centred_10deg"] >= 80
    assert summary["final_mode"] == "TIMEOUT" and "LAND" in summary["finish_reason"]
