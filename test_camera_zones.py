from camera_zones import (
    latest_jpeg,
    list_cameras,
    load_zone_file,
    parse_camera_folder,
    polygons_from_payload,
    save_zone_file,
    validate_zone_payload,
    zones_from_payload,
    zone_file,
)
from frame_objects import apply_tram_policy, objects_in_areas, objects_in_polygons, zone_areas, zone_polygon, zone_polygons


def test_parse_camera_folder():
    assert parse_camera_folder("33_4") == (4, "33")
    assert parse_camera_folder("정문_118") == (118, "정문")
    assert parse_camera_folder("nope") is None


def test_polygons_from_legacy_levels():
    payload = {
        "zones": [
            {"name": "track", "level": "danger", "points": [[0, 0], [10, 0], [10, 10]]},
            {"name": "safe", "level": "safe", "points": [[20, 20], [30, 20], [30, 30]]},
        ]
    }
    assert polygons_from_payload(payload) == [
        [[0, 0], [10, 0], [10, 10]],
        [[20, 20], [30, 20], [30, 30]],
    ]


def test_validate_and_save_roundtrip(tmp_path):
    payload = validate_zone_payload(
        {"width": 1920, "height": 1080, "polygons": [[[120, 180], [80, 650], [1200, 650]]]}
    )
    path = save_zone_file(zone_file(tmp_path, 33), payload)
    loaded = load_zone_file(tmp_path / "33.json")
    assert loaded == path
    assert loaded["zones"][0]["level"] == "danger"
    assert loaded["zones"][0]["points"][0] == [120, 180]
    assert loaded["zoneGapThreshold"] == 2


def test_level_zones_and_tram_gap_policy():
    payload = {
        "width": 100,
        "height": 100,
        "zoneGapThreshold": 2,
        "zones": [
            {"name": "track-z8", "level": "danger", "index": 8, "points": [[0, 0], [40, 0], [40, 100], [0, 100]]},
            {"name": "rail-z8", "level": "warning", "index": 8, "points": [[40, 0], [70, 0], [70, 100], [40, 100]]},
        ],
    }
    assert zones_from_payload(payload)[0]["index"] == 8
    areas = zone_areas(payload, 100, 100)
    objects = objects_in_areas(
        [("person", 0.9, (10, 10, 30, 80)), ("person", 0.8, (45, 10, 65, 80)), ("person", 0.7, (75, 10, 95, 80))],
        areas,
        {"person"},
        include_outside=True,
    )
    result = apply_tram_policy(objects, tram_zone=6, threshold=2)
    assert [(item["zoneLevel"], item["zoneGap"], item["alert"]) for item in result[:2]] == [
        ("danger", 2, True),
        ("warning", 2, True),
    ]
    assert result[2]["zoneLevel"] == "safe" and result[2]["alert"] is False
    assert apply_tram_policy(objects, tram_zone=5, threshold=2)[0]["alert"] is False
    assert apply_tram_policy([{**objects[0], "class": "bicycle"}], tram_zone=6, threshold=2)[0]["alert"] is False


def test_tram_policy_uses_uncertainty_direction_and_fail_safe():
    person = {"class": "person", "zoneLevel": "danger", "zoneIndex": 9}
    uncertain = apply_tram_policy([person], 6, 2, uncertainty=1)[0]
    assert uncertain["tramZoneCandidates"] == [5, 6, 7]
    assert uncertain["zoneGap"] == 2 and uncertain["alertReason"] == "TRAM_POSITION_UNCERTAIN"

    assert apply_tram_policy([person], 6, 2, direction=1, forward_threshold=3, rear_threshold=1)[0]["alert"] is True
    behind = {**person, "zoneIndex": 4}
    assert apply_tram_policy([behind], 6, 2, direction=1, forward_threshold=3, rear_threshold=1)[0]["alert"] is False

    missing = apply_tram_policy([person], None, 2, fail_safe_reason="TRAM_POSITION_MISSING")[0]
    assert missing["alert"] is True and missing["alertReason"] == "TRAM_POSITION_MISSING"

    no_position = apply_tram_policy([person], None, 2)[0]
    assert no_position["alert"] is False and no_position["alertReason"] == ""


def test_list_cameras_and_latest_jpeg(tmp_path):
    camera_dir = tmp_path / "analyzed" / "33_4" / "2026" / "08" / "19"
    camera_dir.mkdir(parents=True)
    older = camera_dir / "a.jpg"
    newer = camera_dir / "b.jpg"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    older.stat()
    newer.touch()
    zones = tmp_path / "zones"
    zones.mkdir()
    (zones / "7.json").write_text('{"width":1,"height":1,"polygons":[]}')
    cameras = list_cameras(tmp_path / "analyzed", zones)
    assert cameras == [{"id": 4, "name": "33"}, {"id": 7, "name": "7"}]
    assert latest_jpeg(tmp_path / "analyzed", 4).name == "b.jpg"


def test_zone_polygons_scale_points():
    zone = {"width": 1280, "height": 720, "polygons": [[[100, 100], [100, 600], [1100, 600]]]}
    scaled = zone_polygons(zone, 640, 360)
    assert scaled == [[(50.0, 50.0), (50.0, 300.0), (550.0, 300.0)]]


def test_legacy_lines_still_scale():
    zone = {
        "width": 1280,
        "height": 720,
        "lines": [[[100, 100], [100, 600]], [[1100, 600], [1100, 100]]],
    }
    assert zone_polygon(zone, 640, 360) == [(50, 50), (50, 300), (550, 300), (550, 50)]


def test_objects_outside_polygons_are_dropped():
    polygons = [[(100, 100), (100, 600), (1100, 600), (1100, 100)]]
    detections = [
        ("person", 0.91234, (120, 20, 220, 150)),
        ("bicycle", 0.8, (1200, 200, 1270, 300)),
        ("dog", 0.7, (300, 200, 500, 400)),
    ]
    assert objects_in_polygons(detections, polygons, {"person", "bicycle"}) == [
        {
            "class": "person",
            "confidence": 0.9123,
            "bbox": [120, 20, 220, 150],
            "zoneLevel": "danger",
            "zoneName": "danger-1",
        },
    ]
    assert objects_in_polygons(detections, [], {"person"}) == []
