"""Ground-truth geometry and the summary numbers reported for a follow run."""
import math

import pytest

from quackdrone.integrations import follow_metrics as fm


def test_truth_geometry_signs():
    # drone at origin facing north; walker 4 m north-east-ish, right of the camera axis, facing south-west
    g = fm.truth_geometry((0.0, 0.0), 0.0, (4.0, 1.0), math.radians(-90 - 14))   # ENU yaw: facing ~south
    assert g["true_dist_m"] == pytest.approx(math.hypot(4, 1))
    assert g["true_bearing_deg"] == pytest.approx(math.degrees(math.atan2(1, 4)))
    assert abs(g["walker_facing_deg"]) < 5
    away = fm.truth_geometry((0.0, 0.0), 0.0, (4.0, 0.0), math.radians(90))     # ENU 90 = facing north
    assert abs(away["walker_facing_deg"]) == pytest.approx(180)


def test_walker_falls_back_to_script_and_converts_frames():
    w, src = fm.walker_from_truth({"sim_time": 0.0, "poses": {}})
    assert src == "script" and w[:2] == pytest.approx((-12.0, -5.0))           # ENU (-5,-12) -> NED
    w, src = fm.walker_from_truth({"sim_time": 0.0, "poses": {"walker": [1.0, 2.0, 0.0, 0.5]}})
    assert src == "gazebo" and w == pytest.approx((2.0, 1.0, 0.5))
    assert fm.walker_from_truth(None) == (None, "none")
    d = fm.drone_from_truth({"poses": {fm.DRONE_MODEL: [3.0, 4.0, 2.0, 0.0]}})   # facing east
    assert d == pytest.approx((4.0, 3.0, 90.0))


def test_summary_counts_lost_events_and_centring():
    rows = []
    modes = ["IDLE"] * 2 + ["TRACKING"] * 10 + ["LOST"] * 3 + ["TRACKING"] * 5 + ["LOST"] * 2 + ["TIMEOUT"]
    for i, m in enumerate(modes):
        rows.append({"t": i * 0.1, "mode": m, "true_bearing_deg": 2.0 if m == "TRACKING" else 20.0,
                     "true_dist_m": 4.0, "est_dist_m": 4.5 if m == "TRACKING" else None,
                     "walker_facing_deg": 10.0})
    s = fm.summarize(rows, 4.0)
    assert s["lost_events"] == 2 and s["reacquired"] == 1
    assert s["lost_durations_s"] == [pytest.approx(0.3)]
    assert s["pct_centred_5deg"] == pytest.approx(100 * 15 / 20)
    assert s["est_minus_true_dist_mean_m"] == pytest.approx(0.5)
    assert s["final_mode"] == "TIMEOUT"
    assert fm.summarize([], 4.0) == {"frames": 0}
