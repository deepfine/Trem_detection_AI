from object_analysis_server import objects_in_zone, zone_polygon


def test_zone_scales_1280x720_coordinates_and_fixes_reversed_line():
    zone = {
        "width": 1280,
        "height": 720,
        "lines": [[[100, 100], [100, 600]], [[1100, 600], [1100, 100]]],
    }
    assert zone_polygon(zone, 640, 360) == [(50, 50), (50, 300), (550, 300), (550, 50)]


def test_objects_use_bottom_center_and_target_class():
    polygon = [(100, 100), (100, 600), (1100, 600), (1100, 100)]
    detections = [
        ("person", 0.91234, (120, 20, 220, 150)),
        ("bicycle", 0.8, (1200, 200, 1270, 300)),
        ("dog", 0.7, (300, 200, 500, 400)),
    ]
    assert objects_in_zone(detections, polygon, {"person", "bicycle"}) == [
        {"class": "person", "confidence": 0.9123, "bbox": [120, 20, 220, 150]},
    ]
