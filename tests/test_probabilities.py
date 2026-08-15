from klas_model.probabilities import (
    central_range,
    empirical_integer_probabilities,
    parse_temperature_bucket_label,
    probability_for_bounds,
    probability_for_market,
)


def test_empirical_distribution_respects_floor():
    dist = empirical_integer_probabilities(109.4, [-1, 0, 0, 1, 2], floor_f=109)
    assert abs(sum(dist.values()) - 1) < 1e-9
    assert min(dist) >= 109
    low, high = central_range(dist)
    assert low >= 109 and high >= low


def test_probability_for_bounds():
    dist = {108: .1, 109: .4, 110: .3, 111: .2}
    assert abs(probability_for_bounds(dist, 109, 110) - .7) < 1e-9


def test_parse_displayed_temperature_buckets():
    assert parse_temperature_bucket_label("96° to 97°") == (96, 97)
    assert parse_temperature_bucket_label("91° or below") == (None, 91)
    assert parse_temperature_bucket_label("100° or above") == (100, None)


def test_displayed_buckets_are_mutually_exclusive_and_sum_to_one():
    dist = {91: .05, 92: .10, 93: .15, 94: .20, 95: .15, 96: .15, 97: .10, 98: .05, 99: .03, 100: .02}
    markets = [
        {"subtitle": "91° or below"},
        {"subtitle": "92° to 93°"},
        {"subtitle": "94° to 95°"},
        {"subtitle": "96° to 97°"},
        {"subtitle": "98° to 99°"},
        {"subtitle": "100° or above"},
    ]
    probs = [probability_for_market(dist, m) for m in markets]
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[3] == .25
