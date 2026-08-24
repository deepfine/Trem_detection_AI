from __future__ import annotations

import time
from pathlib import Path

import cv2

from anomaly_rules import point_in_polygon
from detect_anomalies import yolo_detections


DEFAULT_TARGET_CLASSES = {
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "backpack",
    "suitcase",
}


def zone_polygon(zone: dict | None, frame_width: int, frame_height: int):
    polygons = zone_polygons(zone, frame_width, frame_height)
    return polygons[0] if polygons else None


def zone_polygons(zone: dict | None, frame_width: int, frame_height: int):
    if not zone:
        return []
    from camera_zones import payload_size, polygons_from_payload

    polygons = polygons_from_payload(zone)
    size = payload_size(zone)
    if not polygons:
        width, height, lines = zone.get("width"), zone.get("height"), zone.get("lines")
        if not isinstance(width, (int, float)) or width <= 0 or not isinstance(height, (int, float)) or height <= 0:
            return []
        if not isinstance(lines, list) or len(lines) != 2 or any(not isinstance(line, list) or len(line) != 2 for line in lines):
            return []
        try:
            a, b = [[[float(x), float(y)] for x, y in line] for line in lines]
        except (TypeError, ValueError):
            return []
        same = sum((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2 for i in range(2))
        crossed = sum((a[i][0] - b[1 - i][0]) ** 2 + (a[i][1] - b[1 - i][1]) ** 2 for i in range(2))
        if crossed < same:
            b.reverse()
        polygons = [[a[0], a[1], b[1], b[0]]]
        size = (int(width), int(height))
    if not polygons:
        return []
    src_w, src_h = size if size and size[0] > 0 and size[1] > 0 else (frame_width, frame_height)
    scale_x, scale_y = frame_width / src_w, frame_height / src_h
    return [[(float(x) * scale_x, float(y) * scale_y) for x, y in polygon] for polygon in polygons]


def objects_in_zone(detections, polygon, target_classes):
    return objects_in_polygons(detections, [polygon] if polygon else [], target_classes)


def objects_in_polygons(detections, polygons, target_classes):
    if not polygons:
        return []
    objects = []
    for label, confidence, (x1, y1, x2, y2) in detections:
        if label not in target_classes:
            continue
        foot = ((x1 + x2) / 2, y2)
        if not any(point_in_polygon(foot, polygon) for polygon in polygons):
            continue
        objects.append(
            {
                "class": label,
                "confidence": round(float(confidence), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
        )
    return objects


class YoloDnnDetector:
    def __init__(self, model: Path, size=640, confidence=0.35, nms=0.45):
        self.model = Path(model)
        self.size = size
        self.confidence = confidence
        self.nms = nms
        self.net = None

    def load(self):
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        self.net = cv2.dnn.readNetFromONNX(str(self.model))

    def detect(self, frame, zone=None, target_classes=()):
        if self.net is None:
            raise RuntimeError("object detector is not loaded")
        started = time.perf_counter()
        height, width = frame.shape[:2]
        detections = list(yolo_detections(self.net, frame, self.size, self.confidence, self.nms))
        classes = set(target_classes) or DEFAULT_TARGET_CLASSES
        polygons = zone_polygons(zone, width, height)
        objects = objects_in_polygons(detections, polygons, classes)
        return objects, (time.perf_counter() - started) * 1000
