import numpy as np
import pytest

from crowd_analysis_protocol import alert_summary, api_callback_payload
from frame_falls import FALL_DWELL_SECONDS, FALL_GAP_SECONDS, update_fall_state
from frame_objects import LOST_TRACK_SECONDS, YoloDnnDetector, apply_object_events


WIDE = [10, 40, 150, 100]
TALL = [60, 10, 100, 100]
SHAPE = (200, 200, 3)


def sample_for(state, box, start=0, seconds=FALL_DWELL_SECONDS):
    result = None
    for now in np.arange(start, start + seconds + 0.25, 0.25):
        result = update_fall_state(box, SHAPE, state, float(now))
    return result


def test_fall_needs_long_dwell_and_horizontal_box():
    state = {}
    assert sample_for(state, WIDE, seconds=FALL_DWELL_SECONDS - 0.25)["fallEvent"] == "NONE"
    assert update_fall_state(WIDE, SHAPE, state, FALL_DWELL_SECONDS)["fallEvent"] == "ENTER"
    assert update_fall_state(WIDE, SHAPE, state, FALL_DWELL_SECONDS + 0.25)["fallEvent"] == "STAY"
    assert update_fall_state(TALL, SHAPE, state, FALL_DWELL_SECONDS + 0.5)["fallEvent"] == "EXIT"
    assert sample_for({}, TALL)["fallState"] == "NORMAL"


def test_long_dwell_then_wide_box_can_trigger_without_another_wait():
    state = {}
    sample_for(state, TALL)
    result = update_fall_state(WIDE, SHAPE, state, FALL_DWELL_SECONDS + 0.25)
    assert result["fallEvent"] == "ENTER"


def test_short_detection_dropouts_keep_the_dwell():
    # A lying person drops below the person threshold for a few frames at a time.
    state = {}
    for now in np.arange(0, FALL_DWELL_SECONDS + 2.5, 0.5):
        if int(now * 2) % 6 in (1, 2, 3):
            continue  # misses ~1.5 s out of every 3 s
        result = update_fall_state(WIDE, SHAPE, state, float(now))
    assert result["fallState"] == "FALL_SUSPECTED"


def test_motion_missing_frames_and_clipped_boxes_do_not_establish_dwell():
    state = {}
    sample_for(state, WIDE, seconds=FALL_DWELL_SECONDS - 0.25)
    assert update_fall_state([50, 40, 190, 100], SHAPE, state, FALL_DWELL_SECONDS)["fallEvent"] == "NONE"
    assert update_fall_state(WIDE, SHAPE, {}, 100)["fallEvent"] == "NONE"
    sample_for(state, WIDE, start=100, seconds=FALL_DWELL_SECONDS - 0.25)
    assert update_fall_state(WIDE, SHAPE, state, 100 + FALL_DWELL_SECONDS + FALL_GAP_SECONDS + 1)["fallEvent"] == "NONE"
    assert sample_for({}, [0, 40, 150, 100])["fallState"] == "UNKNOWN"
    assert update_fall_state([1, 1, 1, 1], SHAPE, {}, 0)["fallState"] == "UNKNOWN"


def test_person_is_kept_without_zone_for_independent_fall_analysis(monkeypatch):
    detector = YoloDnnDetector("unused")
    detector.models = [None]
    monkeypatch.setattr(detector, "_detections", lambda frame: iter([("person", 0.9, WIDE)]))
    monkeypatch.setattr(detector, "_refine_bicycles", lambda frame, detections: detections)
    objects, _ = detector.detect(np.zeros(SHAPE, np.uint8))
    assert len(objects) == 1
    assert objects[0]["class"] == "person" and objects[0]["zoneLevel"] == "safe"


def test_danger_fall_observation_reaches_callback(monkeypatch):
    state = {}
    frame = np.zeros(SHAPE, np.uint8)
    now = [0.0]
    monkeypatch.setattr("frame_objects.time.monotonic", lambda: now[0])
    def sample():
        return apply_object_events([{
            "class": "person", "confidence": 0.9, "bbox": WIDE,
            "zoneLevel": "danger", "alert": True, "alertReason": "TRAM_ZONE_PROXIMITY",
        }], state, frame, 100)[0]
    first = sample()
    for step in range(1, int(FALL_DWELL_SECONDS * 4) + 1):
        now[0] = step / 4
        fall = sample()
    assert fall["trackId"] == first["trackId"]
    assert fall["alert"] is True and fall["alertReason"] == "PERSON_FALL_SUSPECTED"
    assert fall["fallEvent"] == "ENTER" and fall["alertNotify"] is True
    assert alert_summary([fall]) == ("danger", 1)
    assert api_callback_payload({"objects": [fall]})["objects"][0]["fallState"] == "FALL_SUSPECTED"
    now[0] += 0.25
    stay = sample()
    assert stay["fallEvent"] == "STAY" and stay["fallNotify"] is False


@pytest.mark.parametrize("box,expected", [
    ([44, 40, 116, 100], "ENTER"),  # width / height == 1.2
    ([45, 40, 115, 100], "NONE"),
    (TALL, "NONE"),
])
def test_width_ratio_boundary(box, expected):
    assert sample_for({}, box)["fallEvent"] == expected


@pytest.mark.parametrize("box", [
    [0, 40, 150, 100], [10, 40, 200, 100], [10, 0, 150, 100],
    [10, 40, 150, 200], [150, 40, 10, 100],
    [float("nan"), 40, 150, 100], [10, 40, float("inf"), 100],
])
def test_invalid_boxes_cannot_start_fall(box):
    result = sample_for({}, box)
    assert result == {"fallState": "UNKNOWN", "fallEvent": "NONE", "fallNotify": False}


def test_brief_clipping_is_not_evidence_of_recovery():
    state = {}
    assert sample_for(state, WIDE)["fallEvent"] == "ENTER"
    clipped = update_fall_state([0, 40, 150, 100], SHAPE, state, 10.25)
    assert clipped["fallEvent"] == "STAY"
    resumed = update_fall_state(WIDE, SHAPE, state, 10.5)
    assert resumed["fallEvent"] == "STAY"
    assert resumed["fallNotify"] is False
    assert update_fall_state(TALL, SHAPE, state, 10.75)["fallEvent"] == "EXIT"


def test_fall_recovery_and_repeated_incident_emit_once():
    state = {}
    assert sample_for(state, WIDE)["fallEvent"] == "ENTER"
    assert update_fall_state(TALL, SHAPE, state, 10.25)["fallEvent"] == "EXIT"
    assert update_fall_state(TALL, SHAPE, state, 10.5)["fallEvent"] == "NONE"
    assert update_fall_state(WIDE, SHAPE, state, 10.75)["fallEvent"] == "ENTER"
    assert update_fall_state(WIDE, SHAPE, state, 11)["fallEvent"] == "STAY"


def test_fall_state_is_isolated_between_people_and_retired_ids(monkeypatch):
    state = {}
    frame = np.zeros((400, 400, 3), np.uint8)
    now = [0.0]
    monkeypatch.setattr("frame_objects.time.monotonic", lambda: now[0])
    objects = [
        {"class": "person", "confidence": 0.9, "bbox": WIDE, "zoneLevel": "danger", "alert": True, "alertReason": "TRAM_ZONE_PROXIMITY"},
        {"class": "person", "confidence": 0.9, "bbox": [260, 100, 300, 220], "zoneLevel": "danger", "alert": True, "alertReason": "TRAM_ZONE_PROXIMITY"},
        {"class": "dog", "confidence": 0.9, "bbox": [210, 250, 350, 300], "zoneLevel": "safe", "alert": False},
    ]
    for step in range(int(FALL_DWELL_SECONDS * 4) + 1):
        now[0] = step / 4
        result = apply_object_events(objects, state, frame, 100, max_missed=2)
    assert [item.get("fallState") for item in result] == ["FALL_SUSPECTED", "NORMAL", None]
    old_id = result[0]["trackId"]
    for step in range(20):
        now[0] += 0.25
        apply_object_events([], state, frame, 100, max_missed=2)
    assert state["falls"] == {}
    # Reappearing at the same place within LOST_TRACK_SECONDS is the same person: no second fall alarm.
    result = apply_object_events(objects[:1], state, frame, 100, max_missed=2)[0]
    assert result["trackId"] == old_id
    assert result["fallEvent"] == "STAY" and result["alertEvent"] == "STAY"
    for step in range(int(LOST_TRACK_SECONDS * 4) + 4):
        now[0] += 0.25
        apply_object_events([], state, frame, 100, max_missed=2)
    result = apply_object_events(objects[:1], state, frame, 100, max_missed=2)[0]
    assert result["trackId"] != old_id
    assert result["fallEvent"] == "NONE"


@pytest.mark.parametrize("zone,expected", [("danger", True), ("warning", False), ("safe", False)])
def test_fall_observation_is_limited_to_danger_zone(zone, expected, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("frame_objects.time.monotonic", lambda: now[0])
    state = {}
    image = np.zeros(SHAPE, np.uint8)
    for step in range(int(FALL_DWELL_SECONDS * 4) + 1):
        now[0] = step / 4
        item = apply_object_events([{"class": "person", "confidence": 0.9, "bbox": WIDE, "zoneLevel": zone, "alert": False}], state, image, 100)[0]
    assert (item["fallEvent"] == "ENTER") is expected
    assert (item.get("alertReason") == "PERSON_FALL_SUSPECTED") is expected
