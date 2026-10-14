import numpy as np

from frame_objects import DEFAULT_TARGET_CLASSES, YoloDnnDetector, objects_in_zone, zone_polygon


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [type("Input", (), {"name": "images"})()]

    def run(self, *_args):
        return [self.output]


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
        {
            "class": "person",
            "confidence": 0.9123,
            "bbox": [120, 20, 220, 150],
            "zoneLevel": "danger",
            "zoneName": "danger-1",
        },
    ]


def test_detector_merges_objects365_and_mobility_names():
    detector = YoloDnnDetector("unused.onnx", confidence=0.35)
    detector.models = [
        (FakeSession(np.array([[[10, 20, 30, 40, 0.9, 112]]], np.float32)), {112: "trolley"}, True, None),
        (
            FakeSession(np.array([[[20], [20], [10], [10], [0.1], [0.1], [0.8], [0.1]]], np.float32)),
            {0: "Bike", 1: "Pedestrian", 2: "Scooter", 3: "Wheelchair"},
            False,
            {"scooter", "wheelchair"},
        ),
    ]
    frame = np.zeros((640, 640, 3), np.uint8)

    assert [item[0] for item in detector._detections(frame)] == ["cart", "scooter"]


def test_default_targets_include_animals_and_exclude_other_objects():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    included = ["person", "bicycle", "motorcycle", "scooter", "wheelchair", "cat", "dog", "bird", "rabbit"]
    excluded = ["car", "bus", "truck", "cart", "backpack", "suitcase"]
    detections = [(label, 0.9, (10, 10, 30, 30)) for label in included + excluded]
    expected = included[:5] + ["animal"] * 4
    assert [item["class"] for item in objects_in_zone(detections, polygon, DEFAULT_TARGET_CLASSES)] == expected
    # A request cannot re-enable objects outside the agreed classification scope.
    assert [item["class"] for item in objects_in_zone(detections, polygon, set(included + excluded))] == expected


def test_detector_normalizes_wild_bird_and_filters_vehicles():
    detector = YoloDnnDetector("unused.onnx")
    detector.models = [
        (FakeSession(np.array([[[10, 10, 30, 30, 0.9, 0], [40, 10, 60, 30, 0.9, 1]]], np.float32)),
         {0: "wild bird", 1: "car"}, True, None),
    ]
    frame = np.zeros((640, 640, 3), np.uint8)
    zone = {"width": 640, "height": 640, "zones": [
        {"name": "danger-1", "level": "danger", "points": [[0, 0], [640, 0], [640, 640], [0, 640]]},
    ]}
    objects, _ = detector.detect(frame, zone)
    assert [item["class"] for item in objects] == ["animal"]


def test_animal_target_returns_each_animal_without_merging_boxes():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    detections = [(label, 0.9, (10, 10, 30, 30)) for label in ("cat", "dog", "wild bird", "fish", "scooter", "person")]
    objects = objects_in_zone(detections, polygon, {"animal"})
    assert len(objects) == 4
    assert {item["class"] for item in objects} == {"animal"}
