from anomaly_rules import average_delta, classify_event, direction_vector, entered_zone, point_in_polygon, sample_due


ZONE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_point_in_polygon():
    assert point_in_polygon((50, 50), ZONE)
    assert not point_in_polygon((150, 50), ZONE)


def test_zone_obstacle_event():
    assert classify_event("person", (10, 10, 30, 30), [], ZONE, direction_vector("down"), 25, 650, 0.75) == "masked_zone_obstacle"


def test_fast_incoming_outside_zone_event():
    assert classify_event("person", (200, 80, 240, 120), [(220, 20), (220, 120)], ZONE, direction_vector("down"), 25, 650, 0.75) == "fast_incoming"


def test_fast_incoming_ignores_non_obstacle_classes():
    assert classify_event("chair", (200, 80, 240, 120), [(220, 20), (220, 120)], ZONE, direction_vector("down"), 25, 650, 0.75) == ""


def test_average_delta_uses_history_span():
    assert average_delta([(0, 0), (0, 10), (0, 30)]) == (0, 30, 2)


def test_entry_only_warns_only_when_crossing_into_zone():
    assert entered_zone([(150, 50), (50, 50)], ZONE)
    assert classify_event("person", (40, 40, 60, 60), [(150, 50), (50, 50)], ZONE, direction_vector("down"), 25, 650, 0.75, True) == "zone_entry_obstacle"
    assert classify_event("person", (40, 40, 60, 60), [(60, 50), (50, 50)], ZONE, direction_vector("down"), 25, 650, 0.75, True) == ""


def test_sampling_selects_the_nearest_frames_to_four_fps():
    assert sample_due(0, 25, 0)
    assert not sample_due(5, 25, 0.25)
    assert sample_due(6, 25, 0.25)
