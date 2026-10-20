import numpy as np
import pytest

from frame_objects import DEFAULT_TARGET_CLASSES, YoloDnnDetector, objects_in_zone, zone_polygon


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [type("Input", (), {"name": "images"})()]

    def run(self, _outputs, inputs):
        self.input = inputs["images"]
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


def test_default_targets_include_animals_and_vehicles_and_exclude_other_objects():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    included = ["person", "bicycle", "motorcycle", "scooter", "wheelchair", "car", "bus", "truck", "cat", "dog", "bird", "rabbit"]
    excluded = ["train", "cart", "backpack", "suitcase"]
    detections = [(label, 0.9, (10, 10, 30, 30)) for label in included + excluded]
    expected = included[:8] + ["animal"] * 4
    assert [item["class"] for item in objects_in_zone(detections, polygon, DEFAULT_TARGET_CLASSES)] == expected
    # A request cannot re-enable objects outside the agreed classification scope.
    assert [item["class"] for item in objects_in_zone(detections, polygon, set(included + excluded))] == expected


def test_detector_normalizes_wild_bird_and_filters_tram():
    detector = YoloDnnDetector("unused.onnx")
    detector.models = [
        (FakeSession(np.array([[[10, 10, 30, 30, 0.9, 0], [40, 10, 60, 30, 0.9, 1]]], np.float32)),
         {0: "wild bird", 1: "train"}, True, None),
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


@pytest.mark.parametrize("height,width,left,top", [(720, 1280, 0, 140), (1280, 720, 140, 0)])
def test_letterbox_restores_all_target_classes_and_discards_padding(height, width, left, top):
    names = {0: "person", 1: "cat", 2: "bicycle", 3: "motorcycle", 4: "scooter", 5: "wheelchair"}
    rows = [[left + 10, top + 20, left + 50, top + 100, .9, cid] for cid in names]
    # Entirely outside the camera content: never report a box in the padding.
    rows.append([0, 0, 10, 10, .9, 0])
    session = FakeSession(np.array([rows], np.float32))
    detector = YoloDnnDetector("unused")
    detector.models = [(session, names, True, None)]
    detections = list(detector._detections(np.full((height, width, 3), 255, np.uint8)))
    assert [label for label, _, _ in detections] == ["person", "animal", "bicycle", "motorcycle", "scooter", "wheelchair"]
    assert all(box == (20, 40, 100, 200) for _, _, box in detections)
    assert session.input.shape == (1, 3, 640, 640)
    assert np.allclose(session.input[0, :, 0, 0], 114 / 255)
    assert np.all(session.input[0, :, top + 20, left + 20] == 1)


def test_letterbox_clips_partial_boxes_and_handles_odd_padding():
    session = FakeSession(np.array([[[-10, 99, 650, 541, .9, 0]]], np.float32))
    detector = YoloDnnDetector("unused")
    detector.models = [(session, {0: "person"}, True, None)]
    # 1001x691 becomes 640x442 with 99px vertical padding.
    assert list(detector._detections(np.zeros((691, 1001, 3), np.uint8)))[0][2] == (0, 0, 1001, 691)


def test_mobility_keeps_stretch_input_and_overlapping_different_classes():
    # Same-class duplicates are removed; scooter and wheelchair both survive.
    output = np.array([[[30, 200, 40, 80, .9, .1], [30, 200, 40, 80, .8, .1],
                        [30, 200, 40, 80, .1, .85]]], np.float32)
    detector = YoloDnnDetector("unused")
    session = FakeSession(output)
    detector.models = [(session, {0: "Scooter", 1: "Wheelchair"}, False, {"scooter", "wheelchair"})]
    detections = list(detector._detections(np.full((720, 1280, 3), 255, np.uint8)))
    assert [label for label, _, _ in detections] == ["scooter", "wheelchair"]
    assert all(box == (20, 180, 100, 270) for _, _, box in detections)
    assert np.all(session.input == 1)


def test_mobility_rejects_frame_sized_boxes():
    # 640x640 stretch input of a 1280x720 frame: 456x483 source box covers 24% of the frame.
    output = np.array([[[521, 434, 228, 429, .1, .9], [30, 200, 40, 80, .1, .9]]], np.float32)
    detector = YoloDnnDetector("unused")
    detector.models = [(FakeSession(output), {0: "Scooter", 1: "Wheelchair"}, False, {"scooter", "wheelchair"})]
    detections = list(detector._detections(np.zeros((720, 1280, 3), np.uint8)))
    assert [box for _, _, box in detections] == [(20, 180, 100, 270)]


def test_raw_head_keeps_best_class_per_anchor_and_per_class_thresholds():
    # Rows are anchors here; the 84-row transposed layout is covered by the shaped-session tests.
    output = np.array([[[100, 100, 20, 40, .2, .05, .01], [300, 300, 20, 20, .05, .3, .01],
                        [500, 300, 20, 20, .01, .02, .9], [200, 200, 20, 20, .05, .4, .01]]], np.float32)
    detector = YoloDnnDetector("unused", confidence=.35)
    detector.models = [(FakeSession(output), {0: "person", 1: "bicycle", 2: "car"}, False, None)]
    detections = list(detector._detections(np.zeros((640, 640, 3), np.uint8)))
    # Weak person stays a proposal, weak bicycle is dropped, car passes the base threshold.
    assert [(label, round(score, 2), box) for label, score, box in detections] == [
        ("car", .9, (490, 290, 510, 310)), ("bicycle", .4, (190, 190, 210, 210)), ("person", .2, (90, 80, 110, 120))]


def test_class_confidence_lowers_animal_cut_without_moving_other_classes():
    output = np.array([[[100, 100, 20, 20, .4, .01, .01], [300, 300, 20, 20, .01, .4, .01], [500, 300, 20, 20, .01, .01, .4]]], np.float32)
    detector = YoloDnnDetector("unused", confidence=.55, class_confidence={"animal": .35})
    detector.models = [(FakeSession(output), {0: "dog", 1: "bicycle", 2: "car"}, False, None)]
    zone = {"width": 640, "height": 640, "zones": [{"name": "danger-1", "level": "danger", "points": [[0, 0], [640, 0], [640, 640], [0, 640]]}]}
    assert [item["class"] for item in detector.detect(np.zeros((640, 640, 3), np.uint8), zone)[0]] == ["animal"]


def test_class_confidence_parser():
    from frame_objects import parse_class_confidence
    assert parse_class_confidence("") == {}
    assert parse_class_confidence(" Animal=0.35, car=0.4 ") == {"animal": .35, "car": .4}
    with pytest.raises(ValueError):
        parse_class_confidence("animal=1.5")


@pytest.mark.parametrize("person_relabel,expected", [(False, "person"), (True, "bicycle")])
def test_person_to_bicycle_relabel_is_opt_in(monkeypatch, person_relabel, expected):
    detector = YoloDnnDetector("unused", person_relabel=person_relabel)
    detector.models = [(FakeSession(np.empty((1, 0, 6))), {0: "person"}, True, None)]
    monkeypatch.setattr(detector, "_detections", lambda frame: iter([("person", .8, (180, 170, 220, 230))]))
    monkeypatch.setattr(detector, "_refine_bicycles", lambda frame, values: [("bicycle", .8, values[0][2])])
    assert [item["class"] for item in detector.detect(np.zeros((720, 1280, 3), np.uint8))[0]] == [expected]


def test_main_confidence_does_not_raise_the_mobility_threshold():
    main = FakeSession(np.array([[[100, 100, 140, 200, .5, 0], [300, 100, 340, 200, .6, 0]]], np.float32))
    mobility = FakeSession(np.array([[[30], [200], [40], [80], [.1], [.48]]], np.float32))
    detector = YoloDnnDetector("unused", confidence=.55)
    detector.models = [(main, {0: "person"}, True, None), (mobility, {0: "Bike", 1: "Scooter"}, False, {"scooter", "wheelchair"})]
    objects, _ = detector.detect(np.zeros((640, 640, 3), np.uint8))
    assert sorted((item["class"], item["confidence"]) for item in objects) == [("person", .6), ("scooter", .48)]


def test_main_model_bicycle_on_mobility_scooter_is_dropped(monkeypatch):
    detector = YoloDnnDetector("unused")
    detector.models = [(FakeSession(np.empty((1, 0, 6))), {0: "person"}, True, None)]
    monkeypatch.setattr(detector, "_detections", lambda frame: iter([
        ("bicycle", .62, (420, 450, 745, 970)), ("scooter", .48, (418, 446, 748, 968)),
        ("bicycle", .7, (100, 100, 160, 180))]))
    objects, _ = detector.detect(np.zeros((988, 1280, 3), np.uint8))
    assert sorted((item["class"], item["bbox"][0]) for item in objects) == [("bicycle", 100), ("scooter", 418)]


@pytest.mark.parametrize('bike_score,person_score,expected', [(.8, .1, 'bicycle'), (.5, .1, 'person'), (.8, .8, 'person')])
def test_small_person_is_reclassified_only_with_strong_bicycle_evidence(bike_score, person_score, expected):
    # 256px crop: (72,72)-(328,328); source target is (180,170)-(220,230).
    output = np.array([[[320, 320, 100, 150, bike_score, .01],
                        [320, 320, 100, 150, .01, person_score]]], np.float32)
    session = FakeSession(output)
    detector = YoloDnnDetector('unused')
    detector.models = [(session, {0: 'Bike', 1: 'Pedestrian'}, False, {'scooter', 'wheelchair'})]
    results = detector._refine_bicycles(np.zeros((720, 1280, 3), np.uint8), [('person', .42, (180, 170, 220, 230))])
    assert results[0][0] == expected
    assert results[0][2] == (180, 170, 220, 230)


def test_weak_person_proposals_are_not_returned_as_people(monkeypatch):
    detector = YoloDnnDetector('unused')
    detector.models = [(FakeSession(np.empty((1, 0, 6))), {0: 'person'}, True, None)]
    monkeypatch.setattr(detector, '_detections', lambda frame: iter([('person', .15, (100, 100, 140, 160))]))
    assert detector.detect(np.zeros((720, 1280, 3), np.uint8))[0] == []


def test_bicycle_elsewhere_in_crop_does_not_relabel_person():
    output = np.array([[[550, 550, 80, 100, .95, .01]]], np.float32)
    detector = YoloDnnDetector('unused')
    detector.models = [(FakeSession(output), {0: 'Bike', 1: 'Pedestrian'}, False, {'scooter', 'wheelchair'})]
    source = [('person', .9, (180, 170, 220, 230))]
    assert detector._refine_bicycles(np.zeros((720, 1280, 3), np.uint8), source.copy()) == source


def test_bicycle_refinement_has_a_four_crop_budget():
    class CountingSession(FakeSession):
        calls = 0
        def run(self, *args):
            self.calls += 1
            return super().run(*args)
    session = CountingSession(np.empty((1, 6, 0), np.float32))
    detector = YoloDnnDetector('unused')
    detector.models = [(session, {0: 'Bike', 1: 'Pedestrian'}, False, {'scooter', 'wheelchair'})]
    detections = [('person', .42, (x, 170, x + 40, 230)) for x in range(0, 1200, 100)]
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert detector._refine_bicycles(frame, detections.copy()) == detections
    assert session.calls == 4


def test_refined_bicycle_duplicate_keeps_highest_confidence():
    detector = YoloDnnDetector('unused')
    detector.models = [(FakeSession(np.empty((1, 6, 0), np.float32)), {0: 'Bike', 1: 'Pedestrian'}, False, {'scooter'})]
    source = [('bicycle', .7, (100, 100, 150, 180)), ('bicycle', .9, (101, 101, 151, 181)),
              ('person', .9, (500, 100, 600, 400))]
    result = detector._refine_bicycles(np.zeros((720, 1280, 3), np.uint8), source)
    assert result == source[1:]


class ShapedSession(FakeSession):
    def __init__(self, output, shape, dtype='tensor(float)'):
        super().__init__(output)
        self.shape, self.dtype = shape, dtype

    def get_inputs(self):
        return [type('Input', (), {'name': 'images', 'shape': self.shape, 'type': self.dtype})()]


def test_each_model_uses_its_own_fixed_resolution():
    main = ShapedSession(np.array([[[100, 300, 200, 400, .9, 0]]], np.float32), [1, 3, 1280, 1280])
    mobility = ShapedSession(np.array([[[30], [200], [40], [80], [.9], [.1]]], np.float32), [1, 3, 640, 640])
    detector = YoloDnnDetector('unused')
    detector.models = [(main, {0: 'person'}, True, None),
                       (mobility, {0: 'Scooter', 1: 'Wheelchair'}, False, {'scooter', 'wheelchair'})]
    objects = list(detector._detections(np.zeros((720, 1280, 3), np.uint8)))
    assert [item[2] for item in objects] == [(100, 20, 200, 120), (20, 180, 100, 270)]
    assert main.input.shape == (1, 3, 1280, 1280)
    assert mobility.input.shape == (1, 3, 640, 640)


def test_raw_accuracy_head_keeps_weak_person_proposals_until_final_filter():
    session = ShapedSession(np.array([[[150], [350], [100], [100], [.2]]], np.float32), [1, 3, 1280, 1280])
    detector = YoloDnnDetector('unused')
    detector.models = [(session, {0: 'person'}, False, None)]
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert list(detector._detections(frame))[0][2] == (100, 20, 200, 120)
    assert detector.detect(frame)[0] == []


def test_fp16_rectangular_model_input_and_coordinates():
    session = ShapedSession(np.array([[[100, 100, 200, 200, .9, 0]]], np.float32), [1, 3, 736, 1280], 'tensor(float16)')
    detector = YoloDnnDetector('unused')
    detector.models = [(session, {0: 'bicycle'}, True, None)]
    objects = list(detector._detections(np.zeros((720, 1280, 3), np.uint8)))
    assert objects[0][2] == (100, 92, 200, 192)
    assert session.input.dtype == np.float16
    assert session.input.shape == (1, 3, 736, 1280)


def test_dynamic_spatial_dimensions_use_configured_fallback():
    session = ShapedSession(np.empty((1, 0, 6), np.float32), ['batch_size', 3, 'height', 'width'])
    detector = YoloDnnDetector('unused', size=960)
    assert detector._input_details(session)[1:3] == (960, 960)


@pytest.mark.parametrize('shape,dtype', [([2, 3, 640, 640], 'tensor(float)'), ([1, 640, 640, 3], 'tensor(float)'),
                                       ([1, 3, 641, 640], 'tensor(float)'), ([1, 3, 640, 640], 'tensor(uint8)')])
def test_rejects_incompatible_model_input(shape, dtype):
    session = ShapedSession(np.empty((1, 0, 6)), shape, dtype)
    with pytest.raises(ValueError):
        YoloDnnDetector('unused')._input_details(session)


@pytest.mark.parametrize('task,output_shape,accepted', [('detect', [1, 300, 6], True),
                                                      ('detect', [1, 5, 8400], True),
                                                      ('pose', [1, 300, 57], False),
                                                      ('detect', [1, 300, 57], False)])
def test_load_checks_task_output_and_warms_at_actual_input_shape(tmp_path, monkeypatch, task, output_shape, accepted):
    import frame_objects
    session = ShapedSession(np.empty((1, 0, 6), np.float32), [1, 3, 1280, 1280])
    session.get_providers = lambda: ['CUDAExecutionProvider']
    session.get_outputs = lambda: [type('Output', (), {'shape': output_shape})()]
    session.get_modelmeta = lambda: type('Metadata', (), {'custom_metadata_map': {
        'names': "{0: 'person'}", 'task': task, 'end2end': 'False'}})()
    monkeypatch.setattr(frame_objects.ort, 'InferenceSession', lambda *args, **kwargs: session)
    monkeypatch.setattr(frame_objects.ort, 'preload_dlls', lambda: None)
    model = tmp_path / 'model.onnx';model.write_bytes(b'fake model for adapter test')
    detector = YoloDnnDetector(model)
    if not accepted:
        with pytest.raises(ValueError):
            detector.load()
    else:
        detector.load()
        assert session.input.shape == (1, 3, 1280, 1280)
        assert detector.models[0][2] == (output_shape[-1] == 6)


def test_tensorrt_request_fails_if_provider_is_not_available(tmp_path, monkeypatch):
    import frame_objects
    model = tmp_path / 'model.onnx';model.write_bytes(b'fake')
    monkeypatch.setattr(frame_objects.ort, 'preload_dlls', lambda: None)
    monkeypatch.setattr(frame_objects.ort, 'get_available_providers', lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    with pytest.raises(RuntimeError, match='TensorRT provider is unavailable'):
        YoloDnnDetector(model, tensorrt=True).load()


@pytest.mark.parametrize('initialized', [True, False])
def test_tensorrt_fp16_is_explicit_and_initialization_is_checked(tmp_path, monkeypatch, initialized):
    import frame_objects
    session = ShapedSession(np.empty((1, 0, 6), np.float32), [1, 3, 640, 640])
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] if initialized else ['CUDAExecutionProvider']
    session.get_providers = lambda: providers
    session.get_outputs = lambda: [type('Output', (), {'shape': [1, 300, 6]})()]
    session.get_modelmeta = lambda: type('Metadata', (), {'custom_metadata_map': {'names': "{0: 'person'}", 'task': 'detect'}})()
    captured = {}
    def create_session(*args, **kwargs):
        captured.update(kwargs)
        return session
    monkeypatch.setattr(frame_objects.ort, 'preload_dlls', lambda: None)
    monkeypatch.setattr(frame_objects.ort, 'get_available_providers', lambda: ['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    monkeypatch.setattr(frame_objects.ort, 'InferenceSession', create_session)
    model = tmp_path / 'model.onnx';model.write_bytes(b'fake')
    detector = YoloDnnDetector(model, tensorrt=True)
    if initialized:
        detector.load()
    else:
        with pytest.raises(RuntimeError, match='failed to initialize'):
            detector.load()
    assert captured['providers'][0] == ('TensorrtExecutionProvider', {'device_id': 0, 'trt_fp16_enable': True})
    assert captured['providers'][1][0] == 'CUDAExecutionProvider'
