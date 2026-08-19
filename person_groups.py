from __future__ import annotations

import json
from pathlib import Path


def load_group_config(path: Path):
    data = json.loads(path.read_text())
    points = sorted(data["adult_height_reference"], key=lambda point: point[0])
    if len(points) < 2:
        raise ValueError("adult_height_reference needs at least 2 points")
    data["adult_height_reference"] = points
    return data


def expected_adult_height(foot_y, points):
    if foot_y <= points[0][0]:
        return points[0][1] * foot_y / max(points[0][0], 1)
    for (y1, h1), (y2, h2) in zip(points, points[1:]):
        if foot_y <= y2:
            return h1 + (h2 - h1) * (foot_y - y1) / (y2 - y1)
    return points[-1][1]


def classify_person(box, frame_shape, config):
    x1, y1, x2, y2 = box
    frame_height, frame_width = frame_shape[:2]
    height = y2 - y1
    clipped = x1 <= 2 or y1 <= 2 or x2 >= frame_width - 2 or y2 >= frame_height - 2
    if clipped or height < config.get("min_person_height_px", 80):
        return "person_uncertain"
    expected = expected_adult_height(y2, config["adult_height_reference"])
    ratio = height / max(expected, 1)
    aspect_ratio = height / max(x2 - x1, 1)
    if ratio <= config.get("child_max_ratio", 0.55) and aspect_ratio >= config.get("child_min_aspect_ratio", 1.65):
        return "child_estimated"
    if ratio >= config.get("adult_min_ratio", 0.82):
        return "adult_estimated"
    return "person_uncertain"


def assistive_for_person(person_box, detections):
    x1, y1, x2, y2 = person_box
    width, height = x2 - x1, y2 - y1
    expanded = (x1 - width * 0.5, y1 - height * 0.2, x2 + width * 0.5, y2 + height * 0.25)
    matches = []
    for label, score, (ax1, ay1, ax2, ay2) in detections:
        cx, cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        if expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]:
            matches.append((score, label))
    return max(matches)[1] if matches else ""
