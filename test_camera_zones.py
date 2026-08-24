from camera_zones import (
    latest_jpeg,
    list_cameras,
    load_zone_file,
    parse_camera_folder,
    polygons_from_payload,
    save_zone_file,
    validate_zone_payload,
    zone_file,
)
from frame_objects import objects_in_polygons, zone_polygon, zone_polygons


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
    assert loaded["polygons"][0][0] == [120, 180]


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
        {"class": "person", "confidence": 0.9123, "bbox": [120, 20, 220, 150]},
    ]
    assert objects_in_polygons(detections, [], {"person"}) == []
