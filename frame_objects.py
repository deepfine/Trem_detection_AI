from __future__ import annotations

import ast
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from anomaly_rules import point_in_polygon


DEFAULT_TARGET_CLASSES = {
    "person",
    "bicycle",
    "motorcycle",
    "scooter",
    "wheelchair",
    "cart",
    "car",
    "bus",
    "truck",
    "backpack",
    "suitcase",
}
ZONE_PRIORITY = {"safe": 0, "warning": 1, "danger": 2}
CLASS_ALIASES = {
    "bike": "bicycle",
    "pedestrian": "person",
    "trolley": "cart",
    "rickshaw": "cart",
    "carriage": "cart",
}


def zone_polygon(zone: dict | None, frame_width: int, frame_height: int):
    polygons = zone_polygons(zone, frame_width, frame_height)
    return polygons[0] if polygons else None


def zone_polygons(zone: dict | None, frame_width: int, frame_height: int):
    return [area["points"] for area in zone_areas(zone, frame_width, frame_height)]


def zone_areas(zone: dict | None, frame_width: int, frame_height: int):
    if not zone:
        return []
    from camera_zones import payload_size, zones_from_payload

    areas = zones_from_payload(zone)
    size = payload_size(zone)
    if not areas:
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
        areas = [{"name": "danger-1", "level": "danger", "points": [a[0], a[1], b[1], b[0]]}]
        size = (int(width), int(height))
    if not areas:
        return []
    src_w, src_h = size if size and size[0] > 0 and size[1] > 0 else (frame_width, frame_height)
    scale_x, scale_y = frame_width / src_w, frame_height / src_h
    return [
        {**area, "points": [(float(x) * scale_x, float(y) * scale_y) for x, y in area["points"]]}
        for area in areas
    ]


def objects_in_zone(detections, polygon, target_classes):
    return objects_in_polygons(detections, [polygon] if polygon else [], target_classes)


def objects_in_polygons(detections, polygons, target_classes):
    areas = [
        {"name": f"danger-{number}", "level": "danger", "points": polygon}
        for number, polygon in enumerate(polygons, 1)
    ]
    return objects_in_areas(detections, areas, target_classes)


def objects_in_areas(detections, areas, target_classes, include_outside=False):
    if not areas:
        return []
    objects = []
    for label, confidence, (x1, y1, x2, y2) in detections:
        if label not in target_classes:
            continue
        foot = ((x1 + x2) / 2, y2)
        matches = [area for area in areas if point_in_polygon(foot, area["points"])]
        if not matches:
            if not include_outside:
                continue
            area = {"name": "safe-default", "level": "safe"}
        else:
            area = max(matches, key=lambda item: ZONE_PRIORITY[item["level"]])
        item = {
            "class": label,
            "confidence": round(float(confidence), 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "zoneLevel": area["level"],
            "zoneName": area["name"],
        }
        if "index" in area:
            item["zoneIndex"] = area["index"]
        objects.append(item)
    return objects


def apply_tram_policy(
    objects,
    tram_zone: int | None,
    threshold: int,
    *,
    direction=0,
    uncertainty=0,
    forward_threshold=None,
    rear_threshold=None,
    fail_safe_reason="",
):
    if tram_zone is None and not fail_safe_reason:
        return objects
    candidates = list(range(max(0, tram_zone - uncertainty), tram_zone + uncertainty + 1)) if tram_zone is not None else []
    enriched = []
    for detected in objects:
        item = dict(detected)
        index = item.get("zoneIndex")
        alertable = item.get("class") == "person" and item.get("zoneLevel") in {"danger", "warning"}
        if fail_safe_reason:
            item.update(alert=alertable, alertReason=fail_safe_reason)
        elif isinstance(index, int) and not isinstance(index, bool):
            gap = min(abs(index - candidate) for candidate in candidates)
            signed_gap = (index - tram_zone) * direction
            limit = (forward_threshold if signed_gap >= 0 else rear_threshold) if direction else threshold
            limit = threshold if limit is None else limit
            alert = alertable and gap <= limit
            item.update(
                tramZone=tram_zone,
                tramZoneCandidates=candidates,
                tramDirection=direction,
                zoneGap=gap,
                zoneGapThreshold=limit,
                alert=alert,
                alertReason=("TRAM_POSITION_UNCERTAIN" if uncertainty else "TRAM_ZONE_PROXIMITY") if alert else "",
            )
        else:
            item["alert"] = False
        enriched.append(item)
    return enriched


class YoloDnnDetector:
    def __init__(self, model: Path, device_id=0, mobility_model: Path | None = None, size=640, confidence=0.35, nms=0.45):
        self.model = Path(model)
        self.mobility_model = Path(mobility_model) if mobility_model else None
        self.device_id = device_id
        self.size = size
        self.confidence = confidence
        self.nms = nms
        self.models = []

    def load(self):
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        ort.preload_dlls()
        paths = [(self.model, None)]
        if self.mobility_model:
            if not self.mobility_model.is_file():
                raise FileNotFoundError(self.mobility_model)
            paths.append((self.mobility_model, {"scooter", "wheelchair"}))
        for path, allowed in paths:
            session = ort.InferenceSession(
                str(path),
                providers=[("CUDAExecutionProvider", {"device_id": self.device_id}), "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError(f"CUDA provider unavailable: {session.get_providers()}")
            metadata = session.get_modelmeta().custom_metadata_map
            names = ast.literal_eval(metadata.get("names", "{}"))
            if not isinstance(names, dict) or not names:
                raise ValueError(f"missing class names in ONNX metadata: {path}")
            self.models.append((session, names, metadata.get("end2end") == "True", allowed))
            session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, self.size, self.size), np.float32)})

    def _detections(self, frame):
        height, width = frame.shape[:2]
        image = cv2.resize(frame, (self.size, self.size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        for session, names, end2end, allowed in self.models:
            output = session.run(None, {session.get_inputs()[0].name: image})[0][0]
            if end2end:
                for x1, y1, x2, y2, score, class_id in output:
                    label = CLASS_ALIASES.get(names[int(class_id)].lower(), names[int(class_id)].lower())
                    if score >= self.confidence and (allowed is None or label in allowed):
                        yield label, float(score), (
                            int(x1 * width / self.size), int(y1 * height / self.size),
                            int(x2 * width / self.size), int(y2 * height / self.size),
                        )
                continue
            boxes, scores, class_ids = [], [], []
            for row in output.T if output.shape[0] == 4 + len(names) else output:
                class_id = int(np.argmax(row[4:]))
                score = float(row[4 + class_id])
                label = CLASS_ALIASES.get(names[class_id].lower(), names[class_id].lower())
                if score < self.confidence or (allowed is not None and label not in allowed):
                    continue
                x, y, w, h = row[:4]
                x1, y1 = int((x - w / 2) * width / self.size), int((y - h / 2) * height / self.size)
                boxes.append([x1, y1, int(w * width / self.size), int(h * height / self.size)])
                scores.append(score)
                class_ids.append(class_id)
            keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.nms)
            for index in np.array(keep).reshape(-1) if len(keep) else []:
                x, y, w, h = boxes[index]
                label = CLASS_ALIASES.get(names[class_ids[index]].lower(), names[class_ids[index]].lower())
                yield label, scores[index], (x, y, x + w, y + h)

    def detect(self, frame, zone=None, target_classes=()):
        if not self.models:
            raise RuntimeError("object detector is not loaded")
        started = time.perf_counter()
        height, width = frame.shape[:2]
        detections = list(self._detections(frame))
        classes = set(target_classes) or DEFAULT_TARGET_CLASSES
        areas = zone_areas(zone, width, height)
        objects = objects_in_areas(detections, areas, classes, include_outside=bool(zone and "zones" in zone))
        return objects, (time.perf_counter() - started) * 1000
