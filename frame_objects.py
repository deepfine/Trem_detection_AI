from __future__ import annotations

import ast
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from norfair import Detection, Tracker

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
        return [{**item, "alert": False, "alertReason": ""} for item in objects]
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


def _appearance(frame, bbox):
    # ponytail: HSV avoids another GPU model; add learned ReID if site footage still shows ID switches.
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.1
    x1, x2 = max(0, int(x1 + dx)), min(width, int(x2 - dx))
    y1, y2 = max(0, int(y1 + dy)), min(height, int(y2 - dy))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(128, np.float32)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
    norm = np.linalg.norm(histogram)
    return histogram / norm if norm else histogram


def _appearance_distance(first, second):
    a, b = first.last_detection.embedding, second.last_detection.embedding
    return 1.0 - float(np.clip(np.dot(a, b), 0, 1))


def _tracking_distance(detection, tracked):
    center = detection.points.mean(axis=0)
    predicted = tracked.estimate.mean(axis=0)
    spatial = float(np.linalg.norm(center - predicted)) / detection.data["maxDistance"]
    if spatial > 1:
        return 1.0
    appearance = 1.0 - float(np.clip(np.dot(detection.embedding, tracked.last_detection.embedding), 0, 1))
    return 0.45 * spatial + 0.55 * appearance


def apply_object_events(objects, state, frame, max_distance, max_missed=30, reid_threshold=0.25):
    """Track objects with motion and appearance, then emit alert state changes once."""
    tracker = state.get("tracker")
    if tracker is None:
        tracker = state["tracker"] = Tracker(
            distance_function=_tracking_distance,
            distance_threshold=0.8,
            hit_counter_max=max_missed,
            initialization_delay=0,
            reid_distance_function=_appearance_distance,
            reid_distance_threshold=reid_threshold,
            reid_hit_counter_max=max_missed * 2,
        )
    previous = state.setdefault("previous", {})
    token = state["frame"] = state.get("frame", 0) + 1
    detections = [
        Detection(
            points=np.asarray([item["bbox"][:2], item["bbox"][2:]], np.float32),
            scores=np.asarray([item["confidence"]] * 2, np.float32),
            data={"frame": token, "index": index, "item": item, "maxDistance": max_distance},
            label=item["class"],
            embedding=_appearance(frame, item["bbox"]),
        )
        for index, item in enumerate(objects)
    ]
    current_tracks = [
        tracked for tracked in tracker.update(detections=detections)
        if tracked.last_detection.data["frame"] == token
    ]
    current_tracks.sort(key=lambda tracked: tracked.last_detection.data["index"])
    enriched = []
    for tracked in current_tracks:
        item = tracked.last_detection.data["item"]
        track_id = tracked.id
        old = previous.get(track_id)
        old_zone = old["zoneLevel"] if old else None
        zone = item.get("zoneLevel", "safe")
        if old_zone == zone:
            zone_transition = "NONE"
        elif old_zone is None:
            zone_transition = f"ENTER_{zone.upper()}" if zone in {"warning", "danger"} else "NONE"
        else:
            zone_transition = f"{old_zone.upper()}_TO_{zone.upper()}"

        was_alert = bool(old and old["alert"])
        is_alert = bool(item.get("alert"))
        if is_alert and not was_alert:
            alert_event = "ENTER"
        elif is_alert and zone_transition == "WARNING_TO_DANGER":
            alert_event = "ESCALATE"
        elif is_alert:
            alert_event = "STAY"
        elif was_alert:
            alert_event = "EXIT"
        else:
            alert_event = "NONE"

        current = {
            **item,
            "trackId": track_id,
            "previousZoneLevel": old_zone,
            "zoneTransition": zone_transition,
            "alertEvent": alert_event,
            "alertNotify": alert_event in {"ENTER", "ESCALATE", "EXIT"},
        }
        previous[track_id] = {"zoneLevel": zone, "alert": is_alert}
        enriched.append(current)

    active_ids = {tracked.id for tracked in tracker.tracked_objects}
    for track_id in set(previous) - active_ids:
        del previous[track_id]
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
