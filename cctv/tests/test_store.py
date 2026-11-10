from cctv import store

CAM = {"id": "CAM1", "name": "Test cam", "owner": "NYSDOT", "lat": 40.8, "lon": -73.2, "county": "Suffolk",
       "kind": "mainline", "exit": "57", "chainage_km": 68.0, "posted_limit_mph": 55,
       "speed_suitability": "measured_good", "feed": {"type": "hls", "url": "https://example/playlist.m3u8"}}


def make_db():
    con = store.connect(":memory:")
    store.upsert_cameras(con, [CAM])
    run = store.insert_run(con, camera_id="CAM1", source="clip.ts", duration_s=60.0, frames=900)
    for i, (mph, over) in enumerate([(52.0, 0), (66.0, 1), (None, 0)]):
        store.insert_vehicle(con, points=[(0.0, 100.0, 400.0, 1.0, 2.0, 1), (0.1, 101.0, 395.0, 3.0, 2.0, 1)],
                             run_id=run, camera_id="CAM1", track_id=i, cls="car", ts_utc="2026-11-10T21:40:00+00:00",
                             speed_mph=mph, sigma_mph=1.5 if mph else None, accepted=int(mph is not None),
                             limit_mph=55, over_limit=over)
    con.commit()
    return con, run


def test_upsert_is_idempotent():
    con, _ = make_db()
    store.upsert_cameras(con, [dict(CAM, name="Renamed")])
    rows = con.execute("SELECT name FROM cameras").fetchall()
    assert [r["name"] for r in rows] == ["Renamed"]


def test_speed_events_view_only_has_flagged_accepted():
    con, _ = make_db()
    ev = con.execute("SELECT speed_mph, camera_name FROM speed_events").fetchall()
    assert [(r["speed_mph"], r["camera_name"]) for r in ev] == [(66.0, "Test cam")]


def test_deleting_a_run_cascades_to_vehicles_and_points():
    con, run = make_db()
    assert store.delete_runs_for_source(con, "CAM1", "clip.ts") == 1
    assert con.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM track_points").fetchone()[0] == 0


def test_retention_purges_old_track_points_only():
    con, _ = make_db()
    old = con.execute("SELECT id FROM vehicles ORDER BY id LIMIT 1").fetchone()["id"]
    con.execute("UPDATE vehicles SET ts_utc = datetime('now', '-30 days') WHERE id = ?", (old,))
    con.execute("UPDATE vehicles SET ts_utc = strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now') WHERE id != ?", (old,))
    assert store.purge_track_points_older_than(con, days=7) == 2      # the old vehicle's 2 points
    assert con.execute("SELECT COUNT(*) FROM track_points").fetchone()[0] == 4
    assert con.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0] == 3   # estimates are kept
