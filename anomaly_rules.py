from __future__ import annotations

import math


OBSTACLE_CLASSES = {"person", "bicycle", "motorcycle", "car", "truck", "bus", "cat", "dog", "horse"}


def sample_due(source_frame, source_fps, next_sample_seconds):
    return source_frame / source_fps + 0.5 / source_fps >= next_sample_seconds


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def direction_vector(name):
    vectors = {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
        "up-left": (-1, -1),
        "up-right": (1, -1),
        "down-left": (-1, 1),
        "down-right": (1, 1),
    }
    dx, dy = vectors[name]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def moving_toward(delta, direction, min_alignment):
    dx, dy = delta
    length = math.hypot(dx, dy)
    if length == 0:
        return False
    tx, ty = direction
    return (dx / length) * tx + (dy / length) * ty >= min_alignment


def average_delta(points):
    if len(points) < 2:
        return None
    return (points[-1][0] - points[0][0], points[-1][1] - points[0][1], len(points) - 1)


def entered_zone(points, zone):
    return len(points) >= 2 and not point_in_polygon(points[-2], zone) and point_in_polygon(points[-1], zone)


def classify_event(label, box, points, zone, direction, fps, min_speed, min_alignment, entry_only=False):
    current_center = center(box)
    in_zone = point_in_polygon(current_center, zone) if zone else False
    if in_zone and label in OBSTACLE_CLASSES:
        if entry_only:
            return "zone_entry_obstacle" if entered_zone(points or [], zone) else ""
        return "masked_zone_obstacle"
    movement = average_delta(points or [])
    if label in OBSTACLE_CLASSES and not in_zone and movement:
        dx, dy, frame_span = movement
        speed = math.hypot(dx, dy) * fps / frame_span
        delta = (dx, dy)
        if speed >= min_speed and moving_toward(delta, direction, min_alignment):
            return "fast_incoming"
    return ""
