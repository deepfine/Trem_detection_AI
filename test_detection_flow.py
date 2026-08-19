from detect_anomalies import track
from process_blur_anomalies import confirm_alert, load_areas, predicted_area, show_danger_box, visible_people_count


def test_track_survives_one_missing_frame():
    tracks = {}
    first = list(track([("person", 0.9, (0, 0, 20, 20))], tracks, 100, 5))
    list(track([], tracks, 100, 5))
    second = list(track([("person", 0.9, (2, 2, 22, 22))], tracks, 100, 5))
    assert first[0][0] == second[0][0]
    assert len(second[0][4]) == 2


def test_confirm_alert_emits_once_after_three_frames():
    alerts = {}
    assert confirm_alert(alerts, (1, "danger", "zone_1"), 0, 3) == (False, False)
    assert confirm_alert(alerts, (1, "danger", "zone_1"), 1, 3) == (False, False)
    assert confirm_alert(alerts, (1, "danger", "zone_1"), 2, 3) == (True, True)
    assert confirm_alert(alerts, (1, "danger", "zone_1"), 3, 3) == (True, False)


def test_load_areas_removes_consecutive_duplicate_points(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text('{"zones":[{"name":"a","level":"danger","points":[[0,0],[1,0],[1,0],[1,1]]}]}')
    areas, legacy = load_areas(path)
    assert legacy is None
    assert areas[0]["points"] == [[0, 0], [1, 0], [1, 1]]


def test_predicted_area_requires_future_path_to_reach_zone():
    areas = [{"name": "track", "level": "danger", "points": [[100, 0], [200, 0], [200, 100], [100, 100]]}]
    assert predicted_area([(0, 50), (20, 50)], areas, 25, 1.5)["name"] == "track"
    assert predicted_area([(0, 150), (20, 150)], areas, 25, 1.5) is None


def test_only_confirmed_danger_zone_objects_get_boxes():
    assert show_danger_box("danger_zone_object", True)
    assert not show_danger_box("warning_zone_object", True)
    assert not show_danger_box("fast_incoming", True)
    assert not show_danger_box("danger_zone_object", False)


def test_visible_people_count_uses_current_tracks_only():
    tracked = [(1, "person", 0.9, (0, 0, 1, 1), []), (2, "bicycle", 0.9, (0, 0, 1, 1), []), (3, "person", 0.9, (0, 0, 1, 1), [])]
    assert visible_people_count(tracked) == 2
