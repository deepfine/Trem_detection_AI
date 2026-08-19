from blur_tracking import keep_recent_boxes


def test_keep_recent_boxes_reuses_only_missing_boxes():
    current = [(10, 10, 20, 20, 0.9)]
    held = [
        ((11, 11, 20, 20, 0.8), 2),
        ((100, 100, 20, 20, 0.7), 1),
        ((200, 200, 20, 20, 0.6), 0),
    ]

    assert keep_recent_boxes(current, held) == [
        (10, 10, 20, 20, 0.9),
        (100, 100, 20, 20, 0.7),
    ]
