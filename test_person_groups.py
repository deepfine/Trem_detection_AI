from person_groups import assistive_for_person, classify_person, expected_adult_height


CONFIG = {
    "adult_height_reference": [[200, 100], [1000, 500]],
    "child_max_ratio": 0.55,
    "child_min_aspect_ratio": 1.65,
    "adult_min_ratio": 0.82,
    "min_person_height_px": 40,
}


def test_expected_adult_height_interpolates_perspective():
    assert expected_adult_height(600, CONFIG["adult_height_reference"]) == 300


def test_person_group_keeps_uncertain_band():
    shape = (1080, 1920, 3)
    assert classify_person((100, 300, 200, 600), shape, CONFIG) == "adult_estimated"
    assert classify_person((100, 435, 175, 600), shape, CONFIG) == "child_estimated"
    assert classify_person((100, 360, 200, 600), shape, CONFIG) == "person_uncertain"


def test_short_occluded_box_is_not_called_child():
    assert classify_person((1128, 358, 1223, 502), (1080, 1920, 3), CONFIG) == "person_uncertain"


def test_distant_short_adult_candidate_is_not_called_child():
    camera_config = {**CONFIG, "adult_height_reference": [[200, 90], [400, 160], [600, 280], [800, 390], [1000, 500]]}
    assert classify_person((1366, 249, 1419, 339), (1080, 1920, 3), camera_config) == "person_uncertain"


def test_clipped_person_is_uncertain():
    assert classify_person((0, 300, 100, 600), (1080, 1920, 3), CONFIG) == "person_uncertain"


def test_assistive_device_must_be_near_person():
    detections = [
        ("wheelchair", 0.8, (150, 250, 260, 500)),
        ("walker", 0.9, (800, 250, 900, 500)),
    ]
    assert assistive_for_person((100, 100, 220, 500), detections) == "wheelchair"
