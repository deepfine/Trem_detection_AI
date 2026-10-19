import numpy as np

from crowd_analysis_protocol import api_callback_payload
from frame_objects import YoloDnnDetector, apply_object_events


BOX = (60, 50, 110, 110)
BIKE = ('bicycle', .8, BOX)


def setup_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('frame_objects.time.monotonic', lambda: clock[0])
    return clock


def image(seed=1):
    return np.random.default_rng(seed).integers(0, 256, (180, 240, 3), dtype=np.uint8)


def test_visual_match_bridges_but_never_refreshes_model_age(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector = YoloDnnDetector('unused', visual_recovery=True)
    state, frame = {}, image()
    detector._recover_bicycles(frame, [BIKE], state)
    for now in [.5, 1.5, 2.0]:
        clock[0] = now
        recovered, sources = detector._recover_bicycles(frame, [], state)
        assert recovered == [BIKE]
        assert sources[BOX]['detectionSource'] == 'VISUAL_TRACKER'
        assert sources[BOX]['detectionAgeSeconds'] == now
        assert state['bicycleRecovery']['tracks'][0]['seen'] == 0
    clock[0] = 2.01
    assert detector._recover_bicycles(frame, [], state)[0] == []


def test_disappeared_bicycle_is_not_held(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, state, frame = YoloDnnDetector('unused', visual_recovery=True), {}, image()
    detector._recover_bicycles(frame, [BIKE], state)
    frame[45:115, 55:115] = 127
    clock[0] = .5
    assert detector._recover_bicycles(frame, [], state)[0] == []
    assert state['bicycleRecovery']['tracks'] == []


def test_camera_change_or_occlusion_clears_recovery(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, state = YoloDnnDetector('unused', visual_recovery=True), {}
    detector._recover_bicycles(image(), [BIKE], state)
    clock[0] = .5
    assert detector._recover_bicycles(image(seed=2), [], state)[0] == []
    detector._recover_bicycles(image(), [BIKE], state)
    assert detector._recover_bicycles(image()[:120], [], state)[0] == []


def test_uniform_patch_does_not_create_a_tracking_template(monkeypatch):
    setup_clock(monkeypatch)
    detector, state = YoloDnnDetector('unused', visual_recovery=True), {}
    detector._recover_bicycles(np.zeros((180, 240, 3), np.uint8), [BIKE], state)
    assert state['bicycleRecovery']['tracks'] == []


def test_roi_redetection_is_distinct_from_visual_hold(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, state, frame = YoloDnnDetector('unused', visual_recovery=True), {}, image()
    detector._recover_bicycles(frame, [BIKE], state)
    monkeypatch.setattr(detector, '_refine_bicycles', lambda frame, proposals: [('bicycle', .75, BOX)])
    clock[0] = 1
    recovered, sources = detector._recover_bicycles(frame, [], state)
    assert recovered == [('bicycle', .75, BOX)]
    assert sources[BOX] == {'detectionSource': 'ROI_REDETECTION'}
    assert state['bicycleRecovery']['tracks'][0]['seen'] == 1


def test_recovery_uses_current_zone_and_preserves_track_id(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, state, frame = YoloDnnDetector('unused', visual_recovery=True), {}, image()
    detector.models = [None]
    detections = [BIKE]
    monkeypatch.setattr(detector, '_detections', lambda frame: iter(detections))
    monkeypatch.setattr(detector, '_refine_bicycles', lambda frame, values: values)
    zone = {'width': 240, 'height': 180, 'zones': [{'name': 'track', 'level': 'danger',
             'points': [[0, 0], [240, 0], [240, 180], [0, 180]]}]}
    first = apply_object_events(detector.detect(frame, zone, state=state)[0], state, frame, 100)
    assert first[0]['alertEvent'] == 'ENTER'
    detections.clear();clock[0] = .5
    second = apply_object_events(detector.detect(frame, zone, state=state)[0], state, frame, 100)
    assert second[0]['detectionSource'] == 'VISUAL_TRACKER'
    assert second[0]['trackId'] == first[0]['trackId']
    assert second[0]['alertEvent'] == 'STAY'
    # A changed zone must be applied before current tram policy and events.
    third = detector.detect(frame, state=state)[0]
    assert third[0]['zoneLevel'] == 'safe'
    detector.detect(frame, target_classes=('person',), state=state)
    assert 'bicycleRecovery' not in state


def test_recovery_metadata_reaches_backend_payload_only_when_enabled(monkeypatch):
    obj = {'class': 'bicycle', 'confidence': .8, 'bbox': list(BOX), 'detectionSource': 'VISUAL_TRACKER',
           'visualTrackingScore': .93, 'detectionAgeSeconds': .5}
    # The current backend rejects unknown object fields with 422, losing the whole frame.
    monkeypatch.delenv('OBJECT_DETECTION_SOURCE_FIELDS', raising=False)
    assert api_callback_payload({'objects': [obj]})['objects'] == [{'class': 'bicycle', 'confidence': .8, 'bbox': list(BOX)}]
    monkeypatch.setenv('OBJECT_DETECTION_SOURCE_FIELDS', '1')
    assert api_callback_payload({'objects': [obj]})['objects'] == [obj]


def test_current_model_can_redetect_when_visual_appearance_changes(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, state = YoloDnnDetector('unused', visual_recovery=True), {}
    detector._recover_bicycles(image(), [BIKE], state)
    monkeypatch.setattr(detector, '_refine_bicycles', lambda frame, proposals: [('bicycle', .9, BOX)])
    clock[0] = .5
    recovered, sources = detector._recover_bicycles(image(seed=2), [], state)
    assert recovered == [('bicycle', .9, BOX)]
    assert sources[BOX]['detectionSource'] == 'ROI_REDETECTION'


def test_recovery_does_not_bridge_people_or_mix_camera_states(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, camera_a, camera_b = YoloDnnDetector('unused', visual_recovery=True), {}, {}
    frame = image()
    detector._recover_bicycles(frame, [('person', .9, BOX)], camera_a)
    clock[0] = .5
    assert detector._recover_bicycles(frame, [], camera_a)[0] == []
    detector._recover_bicycles(frame, [BIKE], camera_a)
    assert detector._recover_bicycles(frame, [], camera_b)[0] == []


def test_additional_roi_inference_is_bounded_per_frame(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, state, frame = YoloDnnDetector('unused', visual_recovery=True), {}, image()
    bikes = [('bicycle', .8, (x, 50, x + 40, 110)) for x in [0, 45, 90, 135, 180]]
    detector._recover_bicycles(frame, bikes, state)
    calls = []
    def no_model_confirmation(frame, proposals):
        calls.append(proposals)
        return proposals
    monkeypatch.setattr(detector, '_refine_bicycles', no_model_confirmation)
    clock[0] = .5
    recovered, sources = detector._recover_bicycles(frame, [], state)
    assert len(calls) == 4
    assert len(recovered) == 5
    assert all(value['detectionSource'] == 'VISUAL_TRACKER' for value in sources.values())


def test_visual_only_results_are_disabled_by_default(monkeypatch):
    clock = setup_clock(monkeypatch)
    detector, state, frame = YoloDnnDetector('unused'), {}, image()
    detector._recover_bicycles(frame, [BIKE], state)
    clock[0] = .5
    assert detector._recover_bicycles(frame, [], state)[0] == []
