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


def test_script_for_uses_the_probe_world_or_override():
    from quackdrone.integrations import walker
    assert fm.script_for({"world": "quack_people_turnaway"}) is walker.TURNAWAY
    assert fm.script_for({"world": "quack_demo"}) is walker.LEAD          # unknown world -> lead
    assert fm.script_for(None, "turnaway") is walker.TURNAWAY
    assert fm.script_for({"world": "quack_people_turnaway"}, "lead") is walker.LEAD
    w, src = fm.walker_from_truth({"sim_time": 30.0, "poses": {}}, walker.TURNAWAY)
    assert src == "script" and w[2] == pytest.approx(walker.S)           # walking away at t = 30


def test_ended_early_and_body_share():
    from quackdrone.integrations import walker
    sc = walker.TURNAWAY
    rows = [{"t": i * 0.1, "sim_time": 1000 * sc.period_s + 20 + i * 0.1, "mode": "TRACKING",
             "source": "face" if i < 50 else "body"} for i in range(100)]
    assert fm.end_phase(rows, sc) == pytest.approx(29.9)
    assert fm.ended_early(rows, sc) is True                    # landed at 30 s, final turn at 69 s
    late = [dict(r, sim_time=r["sim_time"] + 45) for r in rows]
    assert fm.ended_early(late, sc) is False
    # followed into the next walker loop (body lock through the walk-back): not early either
    wrapped = rows + [dict(rows[-1], sim_time=rows[-1]["sim_time"] + 70)]
    assert fm.end_phase(wrapped, sc) == pytest.approx(99.9) and fm.ended_early(wrapped, sc) is False
    # first lock during the turn at the very end of the previous loop counts from the next loop
    tail = [dict(r, sim_time=r["sim_time"] - 21.0) for r in rows]   # first lock at phase 82.3
    assert fm.end_phase(tail, sc) == pytest.approx(8.9)
    assert fm.ended_early([{"t": 0, "mode": "IDLE"}], sc) is None
    s = fm.summarize(rows, 4.0)
    assert s["body_frames"] == 50 and s["pct_following_by_body"] == pytest.approx(50.0)
    assert "body_frames" not in fm.summarize([{"t": 0, "mode": "TRACKING"}], 4.0)


@pytest.mark.parametrize("name", ["log.csv", "log.csv.gz"])
def test_csv_roundtrip_reproduces_summary(tmp_path, name):
    rows = [{"t": i / 15, "mode": "TRACKING" if i % 7 else "LOST", "true_bearing_deg": math.sin(i) * 3.123456,
             "true_dist_m": 3.0 + i / 97, "est_dist_m": 3.1 + i / 101, "walker_facing_deg": 1.23456,
             "reason": "x, with a comma", "source": "face", "sim_time": None} for i in range(200)]
    fields = list(rows[0])
    path = str(tmp_path / name)
    fm.write_csv(path, rows, fields)
    back = fm.read_csv(path)
    assert back[0]["sim_time"] is None and back[0]["reason"] == "x, with a comma"
    a, b = fm.summarize(rows, 4.0), fm.summarize(back, 4.0)
    assert a["lost_events"] == b["lost_events"] and a["frames"] == b["frames"]
    assert b["true_dist_mean_m"] == pytest.approx(a["true_dist_mean_m"], abs=1e-3)
    if name.endswith(".gz"):
        first = open(path, "rb").read()
        fm.write_csv(path, rows, fields)
        assert open(path, "rb").read() == first                  # deterministic bytes (gzip mtime=0)


def _rows(facings, modes=None, sources=None, dt=0.1, t0=0.0):
    modes = modes or ["TRACKING"] * len(facings)
    sources = sources or [None] * len(facings)
    return [{"t": i * dt, "sim_time": t0 + i * dt, "mode": m, "walker_facing_deg": f, "source": s,
             "true_dist_m": 5.0} for i, (f, m, s) in enumerate(zip(facings, modes, sources))]


def test_stay_in_front_counts_time_and_per_leg_numbers():
    from quackdrone.integrations import walker
    sc = walker.ANGLED
    # 9.0-34.0 s (leg A): facing grows 0 -> 50 deg; rows every 0.1 s
    n = 250
    rows = _rows([50.0 * i / n for i in range(n)], t0=9.0)
    out = fm.stay_in_front(rows, sc)
    assert out["facing_window_s"] == pytest.approx(24.9, abs=0.05)
    assert out["in_front_30deg_s"] == pytest.approx(15.0, abs=0.15)          # |facing| <= 30 for 60 %
    assert out["pct_in_front_30deg"] == pytest.approx(60.0, abs=0.7)
    leg = out["leg_A25"]
    assert leg["max_abs_facing_deg"] == pytest.approx(49.8, abs=0.1) and leg["pct_tracking"] == 100.0
    assert "leg_B40" not in out
    # nothing locked -> nothing to report; turned-away script time is not counted
    assert fm.stay_in_front(_rows([0.0] * 5, modes=["IDLE"] * 5), sc) == {}
    assert fm.stay_in_front(_rows([180.0] * 20, t0=62.0), sc)["facing_window_s"] == 0.0


def test_back_of_head_counts_relocks_and_settled_tracking():
    # facing away from the start; LOST, then a re-lock while still facing away (the round-2 failure)
    f = [10.0] * 5 + [180.0] * 30
    modes = ["TRACKING"] * 7 + ["LOST"] * 13 + ["TRACKING"] * 15
    out = fm.back_of_head(_rows(f, modes))
    assert out["relocks_facing_away"] == 1 and out["relocks_facing_away_at"][0]["facing_deg"] == 180.0
    # TRACKING frames after >= 1 s facing away: rows 20..34 (15 frames, last one has no successor)
    assert out["tracking_facing_away_s"] == pytest.approx(1.4, abs=0.01)
    # the same frames steered by the body path are not false face locks
    body = fm.back_of_head(_rows(f, modes, sources=["face"] * 7 + [""] * 13 + ["body"] * 15))
    assert body["relocks_facing_away"] == 0 and body["tracking_facing_away_s"] == 0.0
    # coasting right after a turn (< settle_s) is not counted
    assert fm.back_of_head(_rows([10.0] * 5 + [180.0] * 5))["tracking_facing_away_s"] == 0.0


def test_committed_round2_log_has_the_known_back_of_head_relock():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "runs", "follow-sitl-run2.csv.gz")
    out = fm.back_of_head(fm.read_csv(path))
    assert out["relocks_facing_away"] == 1                    # 68.8 s, walker 180 deg away at 7 m
    assert out["relocks_facing_away_at"][0]["t"] == pytest.approx(68.8, abs=0.1)
