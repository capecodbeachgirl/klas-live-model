from klas_model.collectors.radar import compare_radar_scans, radar_sample_points, summarize_radar_samples


def test_radar_ring_points_include_three_rings():
    pts = radar_sample_points(bearings=4)
    assert len(pts) == 13
    assert {p["radius_miles"] for p in pts} == {0.0, 10.0, 25.0, 50.0}


def test_radar_sample_summary_and_approach():
    meta = [
        {"radius_miles": 0.0},
        {"radius_miles": 10.0},
        {"radius_miles": 25.0},
    ]
    no_echo = [{"value": "0,0,0,0"}, {"value": "0,0,0,0"}, {"value": "10,200,20,255"}]
    near_echo = [{"value": "0,0,0,0"}, {"value": "10,200,20,255"}, {"value": "10,200,20,255"}]
    prior = summarize_radar_samples(no_echo, meta)
    now = summarize_radar_samples(near_echo, meta)
    trend = compare_radar_scans(now, prior)
    assert prior["nearest_echo_miles"] == 25.0
    assert now["nearest_echo_miles"] == 10.0
    assert trend["approaching"] is True
    assert trend["risk"] == "HIGH"
