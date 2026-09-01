import numpy as np

from frame_objects import YoloDnnDetector, objects_in_zone, zone_polygon


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
