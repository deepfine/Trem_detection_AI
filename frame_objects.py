from __future__ import annotations

import math
import ast
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from norfair import Detection, Tracker

from anomaly_rules import OBSTACLE_CLASSES, normalize_class, point_in_polygon
from frame_falls import update_fall_state


DEFAULT_TARGET_CLASSES = OBSTACLE_CLASSES
ZONE_PRIORITY = {"safe": 0, "warning": 1, "danger": 2}
# A lower zone must persist this many frames before it replaces the current one; entering stays immediate.
# A parked bicycle on a zone boundary otherwise produced an EXIT/ENTER pair almost every frame.
ZONE_EXIT_FRAMES = 3
# A stationary object straddling a boundary flickers in and out for minutes; it leaves only after this long outside.
STATIONARY_EXIT_SECONDS = 60.0
# A lost track's id and zone state are handed to a new track of the same class at the same place,
# only when the lost object had stayed put; a walker's spot must not hide the next person's ENTER.
# Intermittently detected parked objects otherwise got a new id, and a new ENTER, every few frames.
LOST_TRACK_SECONDS = 60.0
LOST_TRACK_IOU = 0.5
# Mobility boxes covering a fifth of a CCTV frame were stretched-input false wheelchairs.
MOBILITY_MAX_AREA = 0.2
CLASS_ALIASES = {
    "bike": "bicycle",
    "pedestrian": "person",
    "trolley": "cart",
    "rickshaw": "cart",
    "carriage": "cart",
}


def bbox_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def same_place(a, b):
    """Overlapping boxes, or centres within half a box for small boxes whose IoU jitters."""
    reach = 0.5 * max(b[2] - b[0], b[3] - b[1])
    return bbox_iou(a, b) >= LOST_TRACK_IOU or math.dist(((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) <= reach


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
    target_classes = {normalize_class(label) for label in target_classes}
    objects = []
    for label, confidence, (x1, y1, x2, y2) in detections:
        label = normalize_class(label)
        if label not in DEFAULT_TARGET_CLASSES or label not in target_classes:
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


def zone_situation_event(old_zone, zone, zone_transition):
    """위험·주의 구역 체류 상황. 트램 접근 여부와 무관하게 API가 나중에 판단한다."""
    active = {"warning", "danger"}
    in_zone = zone in active
    was_in_zone = old_zone in active
    if in_zone and zone_transition == "WARNING_TO_DANGER":
        return "ESCALATE"
    if in_zone and not was_in_zone:
        return "ENTER"
    if in_zone:
        return "STAY"
    if was_in_zone:
        return "EXIT"
    return "NONE"


def apply_object_events(objects, state, frame, max_distance, max_missed=30, reid_threshold=0.25):
    """Track objects with motion and appearance, then emit zone-situation changes once."""
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
    falls = state.setdefault("falls", {})
    aliases = state.setdefault("aliases", {})
    lost = state.setdefault("lost", {})
    now = time.monotonic()
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
    # Ids of tracks already known this frame cannot be handed to a new track.
    seen_now = {aliases[tracked.id] for tracked in current_tracks if tracked.id in aliases}
    enriched = []
    for tracked in current_tracks:
        item = tracked.last_detection.data["item"]
        if tracked.id not in aliases:
            aliases[tracked.id] = tracked.id
            # Fading tracks stay in the tracker for a while, so unseen live ids are candidates too.
            stale = {key: {"class": entry["class"], "bbox": entry["bbox"], "time": entry["time"], "origin": entry["origin"]}
                     for key, entry in previous.items() if key not in seen_now and entry["time"] < now}
            match = max(((bbox_iou(item["bbox"], entry["bbox"]), key) for key, entry in {**lost, **stale}.items()
                         if entry["class"] == item["class"] and now - entry["time"] <= LOST_TRACK_SECONDS
                         and same_place(entry["bbox"], entry["origin"]) and same_place(item["bbox"], entry["bbox"])), default=(-1, None))
            if match[1] is not None:
                aliases[tracked.id] = match[1]
                if match[1] in lost:
                    entry = lost.pop(match[1])
                    previous[match[1]] = entry["previous"]
                    if entry["fall"] is not None:
                        falls[match[1]] = entry["fall"]
        track_id = aliases[tracked.id]
        seen_now.add(track_id)
        old = previous.get(track_id)
        old_zone = old["zoneLevel"] if old else None
        zone = item.get("zoneLevel", "safe")
        lower_frames, lower_since = 0, None
        if old_zone and ZONE_PRIORITY.get(zone, 0) < ZONE_PRIORITY.get(old_zone, 0):
            lower_frames = old.get("lowerFrames", 0) + 1
            lower_since = old.get("lowerSince") or now
            stationary = same_place(item["bbox"], old["origin"])
            if lower_frames < ZONE_EXIT_FRAMES or (stationary and now - lower_since < STATIONARY_EXIT_SECONDS):
                zone = old_zone
                item = {**item, "zoneLevel": old_zone, "zoneName": old.get("zoneName", item.get("zoneName"))}
            else:
                lower_frames, lower_since = 0, None
        if old_zone == zone:
            zone_transition = "NONE"
        elif old_zone is None:
            zone_transition = f"ENTER_{zone.upper()}" if zone in {"warning", "danger"} else "NONE"
        else:
            zone_transition = f"{old_zone.upper()}_TO_{zone.upper()}"

        alert_event = zone_situation_event(old_zone, zone, zone_transition)

        current = {
            **item,
            "trackId": track_id,
            "previousZoneLevel": old_zone,
            "zoneTransition": zone_transition,
            "alertEvent": alert_event,
            "alertNotify": alert_event in {"ENTER", "ESCALATE", "EXIT"},
        }
        if item["class"] == "person":
            fall = falls.setdefault(track_id, {})
            current.update(update_fall_state(item["bbox"], frame.shape, fall, now))
            eligible = zone == "danger"
            active = current["fallState"] == "FALL_SUSPECTED" and eligible
            was_active = fall.get("signaled", False)
            event = "ENTER" if active and not was_active else "EXIT" if was_active and not active else "STAY" if active else "NONE"
            fall["signaled"] = active
            current.update(fallEvent=event, fallNotify=event in {"ENTER", "EXIT"})
            if not eligible:
                current["fallState"] = "UNKNOWN"
            if active:
                current.update(alert=True, alertReason="PERSON_FALL_SUSPECTED")
            current["alertNotify"] = current["alertNotify"] or current["fallNotify"]
        previous[track_id] = {"zoneLevel": zone, "zoneName": item.get("zoneName"), "lowerFrames": lower_frames, "lowerSince": lower_since,
                              "class": item["class"], "bbox": item["bbox"], "time": now,
                              "origin": old["origin"] if old else item["bbox"]}
        enriched.append(current)

    active_ids = {aliases.get(tracked.id, tracked.id) for tracked in tracker.tracked_objects}
    for track_id in set(previous) - active_ids:
        last = previous.pop(track_id)
        lost[track_id] = {"class": last["class"], "bbox": last["bbox"], "time": last["time"], "origin": last["origin"],
                          "previous": last, "fall": falls.pop(track_id, None)}
    for track_id in [key for key, entry in lost.items() if now - entry["time"] > LOST_TRACK_SECONDS]:
        del lost[track_id]
    live = {tracked.id for tracked in tracker.tracked_objects}
    for tracker_id in [key for key in aliases if key not in live]:
        del aliases[tracker_id]
    return enriched


def _raw_candidates(output, names, thresholds):
    """Best class per raw anchor row, kept when its score reaches that class's threshold."""
    rows = output.T if output.shape[0] == 4 + len(names) else output
    class_scores = rows[:, 4:]
    class_ids = class_scores.argmax(1) if len(rows) else np.empty(0, int)
    scores = class_scores[np.arange(len(rows)), class_ids]
    keep = scores >= thresholds[class_ids]
    return class_ids[keep], scores[keep], rows[keep, :4]


class YoloDnnDetector:
    def __init__(self, model: Path, device_id=0, mobility_model: Path | None = None, size=640, confidence=0.35, nms=0.45, tensorrt=False, visual_recovery=False,
                 person_relabel=True, mobility_confidence=0.35):
        self.model = Path(model)
        self.mobility_model = Path(mobility_model) if mobility_model else None
        self.device_id = device_id
        self.size = size
        self.confidence = confidence
        # Scooters and wheelchairs come from the mobility model, whose calibration does not follow the main model.
        self.mobility_confidence = mobility_confidence
        self.nms = nms
        self.tensorrt = tensorrt
        self.visual_recovery = visual_recovery
        # Legacy nano-model fix kept as default; with COCO yolo26l every relabeled box was a real pedestrian or rider.
        self.person_relabel = person_relabel
        self.models = []

    def _input_details(self, session):
        model_input = session.get_inputs()[0]
        shape = getattr(model_input, "shape", [1, 3, self.size, self.size])
        if len(shape) != 4 or (isinstance(shape[0], int) and shape[0] != 1) or shape[1] != 3:
            raise ValueError(f"expected batch-one NCHW RGB input, got {shape}")
        height, width = [value if isinstance(value, int) else self.size for value in shape[2:]]
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(f"model input dimensions must be positive multiples of 32: {shape}")
        input_type = getattr(model_input, "type", "tensor(float)")
        if input_type not in {"tensor(float)", "tensor(float16)"}:
            raise ValueError(f"unsupported model input dtype: {input_type}")
        dtype = np.float16 if input_type == "tensor(float16)" else np.float32
        return model_input.name, height, width, dtype

    def load(self):
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        ort.preload_dlls()
        providers = [("CUDAExecutionProvider", {"device_id": self.device_id}), "CPUExecutionProvider"]
        if self.tensorrt:
            if "TensorrtExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("TensorRT requested but ONNX Runtime TensorRT provider is unavailable")
            providers.insert(0, ("TensorrtExecutionProvider", {"device_id": self.device_id, "trt_fp16_enable": True}))
        paths = [(self.model, None)]
        if self.mobility_model:
            if not self.mobility_model.is_file():
                raise FileNotFoundError(self.mobility_model)
            paths.append((self.mobility_model, {"scooter", "wheelchair"}))
        for path, allowed in paths:
            session = ort.InferenceSession(
                str(path),
                providers=providers,
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError(f"CUDA provider unavailable: {session.get_providers()}")
            if self.tensorrt and "TensorrtExecutionProvider" not in session.get_providers():
                raise RuntimeError(f"TensorRT provider failed to initialize: {session.get_providers()}")
            metadata = session.get_modelmeta().custom_metadata_map
            names = ast.literal_eval(metadata.get("names", "{}"))
            if not isinstance(names, dict) or not names:
                raise ValueError(f"missing class names in ONNX metadata: {path}")
            if metadata.get("task", "detect") != "detect":
                raise ValueError(f"object detection model required: {path}")
            input_name, height, width, dtype = self._input_details(session)
            output_shape = session.get_outputs()[0].shape
            if len(output_shape) != 3:
                raise ValueError(f"unsupported detection output: {output_shape}")
            # Built-in NMS also returns xyxy/score/class rows, even with end2end=False.
            processed = output_shape[-1] == 6
            if not processed and 4 + len(names) not in output_shape[1:]:
                raise ValueError(f"unsupported detection output: {output_shape}")
            self.models.append((session, names, processed, allowed))
            session.run(None, {input_name: np.zeros((1, 3, height, width), dtype)})

    def _detections(self, frame):
        height, width = frame.shape[:2]
        for session, names, end2end, allowed in self.models:
            input_name, input_height, input_width, dtype = self._input_details(session)
            # Keep mobility input: letterboxing introduced cyclist-to-wheelchair false positives.
            if allowed is not None:
                model_image = cv2.resize(frame, (input_width, input_height))
                offset_x, offset_y = 0, 0
                scale_x, scale_y = input_width / width, input_height / height
            else:
                scale_x = scale_y = min(input_width / width, input_height / height)
                resized_width, resized_height = max(1, round(width * scale_x)), max(1, round(height * scale_y))
                offset_x, offset_y = (input_width - resized_width) // 2, (input_height - resized_height) // 2
                model_image = np.full((input_height, input_width, 3), 114, np.uint8)
                model_image[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = cv2.resize(frame, (resized_width, resized_height))
            tensor = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(dtype) / 255.0

            def original_box(x1, y1, x2, y2):
                return (
                    int(np.clip((x1 - offset_x) / scale_x, 0, width)),
                    int(np.clip((y1 - offset_y) / scale_y, 0, height)),
                    int(np.clip((x2 - offset_x) / scale_x, 0, width)),
                    int(np.clip((y2 - offset_y) / scale_y, 0, height)),
                )

            output = session.run(None, {input_name: tensor})[0][0]
            if end2end:
                for x1, y1, x2, y2, score, class_id in output:
                    label = normalize_class(CLASS_ALIASES.get(names[int(class_id)].lower(), names[int(class_id)].lower()))
                    # Weak person boxes are refinement proposals, filtered before output.
                    base = self.mobility_confidence if allowed is not None else self.confidence
                    threshold = min(base, 0.1) if label == "person" else base
                    if score >= threshold and (allowed is None or label in allowed):
                        box = original_box(x1, y1, x2, y2)
                        if box[2] > box[0] and box[3] > box[1]:
                            yield label, float(score), box
                continue
            labels = {class_id: normalize_class(CLASS_ALIASES.get(name.lower(), name.lower())) for class_id, name in names.items()}
            # Per-class thresholds keep the weak person proposals; disallowed classes never pass.
            thresholds = np.full(max(names) + 1, np.inf, np.float32)
            for class_id, label in labels.items():
                if allowed is None or label in allowed:
                    base = self.mobility_confidence if allowed is not None else self.confidence
                    thresholds[class_id] = min(base, 0.1) if label == "person" else base
            # Vectorized: a per-anchor Python loop cost ~18 ms on 1280-wide inputs.
            raw_ids, raw_scores, raw_boxes = _raw_candidates(output, names, thresholds)
            x, y, w, h = raw_boxes.T
            x1 = np.clip((x - w / 2 - offset_x) / scale_x, 0, width).astype(int)
            y1 = np.clip((y - h / 2 - offset_y) / scale_y, 0, height).astype(int)
            x2 = np.clip((x + w / 2 - offset_x) / scale_x, 0, width).astype(int)
            y2 = np.clip((y + h / 2 - offset_y) / scale_y, 0, height).astype(int)
            valid = (x2 > x1) & (y2 > y1)
            if allowed is not None:
                valid &= (x2 - x1) * (y2 - y1) <= MOBILITY_MAX_AREA * width * height
            boxes = np.stack([x1, y1, x2 - x1, y2 - y1], 1)[valid].tolist()
            scores = raw_scores[valid].tolist()
            class_ids = raw_ids[valid].tolist()
            keep = []
            # Overlapping objects of different classes must not suppress each other.
            for class_id in sorted(set(class_ids)):
                indices = [i for i, value in enumerate(class_ids) if value == class_id]
                label = normalize_class(CLASS_ALIASES.get(names[class_id].lower(), names[class_id].lower()))
                base = self.mobility_confidence if allowed is not None else self.confidence
                threshold = min(base, 0.1) if label == "person" else base
                selected = cv2.dnn.NMSBoxes([boxes[i] for i in indices], [scores[i] for i in indices], threshold, self.nms)
                keep.extend(indices[int(i)] for i in np.asarray(selected).reshape(-1))
            for index in sorted(keep, key=lambda i: scores[i], reverse=True):
                x, y, w, h = boxes[index]
                label = normalize_class(CLASS_ALIASES.get(names[class_ids[index]].lower(), names[class_ids[index]].lower()))
                yield label, scores[index], (x, y, x + w, y + h)

    def _refine_bicycles(self, frame, detections):
        mobility = next(((session, names) for session, names, end2end, allowed in self.models
                         if allowed is not None and not end2end and "bike" in {name.lower() for name in names.values()}), None)
        if mobility is None:
            return detections
        session, names = mobility
        input_name, input_height, input_width, dtype = self._input_details(session)
        height, width = frame.shape[:2]
        # shortcut: an initial detector proposal is still required before track-based reacquisition.
        candidates = [i for i, (label, score, box) in enumerate(detections)
                      if label == "person" and height * 0.05 <= box[3] - box[1] <= height * 0.2
                      and not any(other == "bicycle" and confidence >= self.confidence and bbox_iou(box, bicycle) >= 0.4
                                  for other, confidence, bicycle in detections)]
        # Bound extra inference; crowded scenes need separate throughput validation.
        checked = []
        for index in sorted(candidates, key=lambda i: (detections[i][2][2] - detections[i][2][0]) /
                            (detections[i][2][3] - detections[i][2][1]), reverse=True):
            _, _, box = detections[index]
            if any(bbox_iou(box, previous) >= 0.5 for previous in checked):
                continue
            if len(checked) == 4:
                break
            checked.append(box)
            side = min(min(width, height), max(256, 4 * max(box[2] - box[0], box[3] - box[1])))
            left = int(np.clip((box[0] + box[2] - side) / 2, 0, width - side))
            top = int(np.clip((box[1] + box[3] - side) / 2, 0, height - side))
            crop = cv2.resize(frame[top:top + side, left:left + side], (input_width, input_height))
            tensor = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(dtype) / 255.0
            output = session.run(None, {input_name: tensor})[0][0]
            bicycles, people = [], []
            thresholds = np.asarray([self.confidence if names.get(i, "").lower() in {"bike", "pedestrian"} else np.inf
                                     for i in range(max(names) + 1)], np.float32)
            class_ids, scores, rows = _raw_candidates(output, names, thresholds)
            for class_id, score, row in zip(class_ids.tolist(), scores.tolist(), rows):
                label = names[class_id].lower()
                x, y, w, h = row * np.asarray([side / input_width, side / input_height] * 2)
                restored = (max(left, int(left + x - w / 2)), max(top, int(top + y - h / 2)),
                            min(left + side, int(left + x + w / 2)), min(top + side, int(top + y + h / 2)))
                if label == "pedestrian":
                    people.append(restored)
                elif score >= max(0.55, self.confidence) and bbox_iou(box, restored) >= 0.4:
                    bicycles.append(("bicycle", score, restored))
            if bicycles and not any(bbox_iou(box, person) >= 0.3 for person in people):
                detections[index] = max(bicycles, key=lambda item: item[1])
        kept_bicycles, duplicate_indices = [], set()
        for index in sorted(range(len(detections)), key=lambda i: detections[i][1], reverse=True):
            label, _, box = detections[index]
            if label != "bicycle":
                continue
            if any(bbox_iou(box, kept) >= self.nms for kept in kept_bicycles):
                duplicate_indices.add(index)
            else:
                kept_bicycles.append(box)
        return [item for index, item in enumerate(detections) if index not in duplicate_indices]

    def _recover_bicycles(self, frame, detections, state):
        now = time.monotonic()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        previous = state.get("bicycleRecovery", {})
        tracks = previous.get("tracks", []) if previous.get("shape") == gray.shape else []
        sources, held, attempts = {}, [], 0
        for track in tracks:
            box, age = track["bbox"], now - track["seen"]
            if not 0 <= age <= 2.0:
                continue
            if any(label == "bicycle" and bbox_iou(box, current) >= 0.3 for label, _, current in detections):
                continue
            # Re-run the model even when appearance changed; only visual-only holds need a match.
            recovered = []
            if attempts < 4:
                recovered = self._refine_bicycles(frame, [("person", 0.0, box)])
                attempts += 1
            bicycle = next((item for item in recovered if item[0] == "bicycle" and item[1] >= self.confidence), None)
            if bicycle is not None:
                detections.append(bicycle)
                sources[bicycle[2]] = {"detectionSource": "ROI_REDETECTION"}
                continue
            if not self.visual_recovery:
                continue
            x1, y1, x2, y2 = box
            search = gray[max(0, y1 - 3):min(height, y2 + 3), max(0, x1 - 3):min(width, x2 + 3)]
            template = track["template"]
            if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                continue
            _, similarity, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED))
            if not np.isfinite(similarity) or similarity < 0.85:
                continue
            detections.append(("bicycle", track["confidence"], box))
            sources[box] = {"detectionSource": "VISUAL_TRACKER", "visualTrackingScore": round(similarity, 4),
                            "detectionAgeSeconds": round(age, 3)}
            held.append(track)
        updated = []
        for label, confidence, box in detections:
            if label != "bicycle" or sources.get(box, {}).get("detectionSource") == "VISUAL_TRACKER":
                continue
            x1, y1, x2, y2 = box
            template = gray[y1:y2, x1:x2]
            if min(template.shape, default=0) < 8 or float(template.std()) < 8:
                continue
            if not any(bbox_iou(box, item["bbox"]) >= 0.3 for item in updated):
                updated.append({"bbox": box, "confidence": confidence, "template": template.copy(), "seen": now})
        # Visual-only matches never refresh the last model-observation timestamp.
        state["bicycleRecovery"] = {"shape": gray.shape, "tracks": (updated + held)[:20]}
        return detections, sources

    def detect(self, frame, zone=None, target_classes=(), state=None):
        if not self.models:
            raise RuntimeError("object detector is not loaded")
        started = time.perf_counter()
        height, width = frame.shape[:2]
        detections = list(self._detections(frame))
        if self.person_relabel:
            detections = self._refine_bicycles(frame, detections)
        detections = [item for item in detections
                      if item[1] >= (self.mobility_confidence if item[0] in {"scooter", "wheelchair"} else self.confidence)]
        # COCO main models have no scooter class and call kick scooters bicycles; the mobility model decides.
        scooters = [box for label, _, box in detections if label == "scooter"]
        detections = [item for item in detections
                      if item[0] not in {"bicycle", "motorcycle"} or not any(bbox_iou(item[2], box) >= 0.5 for box in scooters)]
        classes = {normalize_class(label) for label in target_classes} or DEFAULT_TARGET_CLASSES
        sources = {}
        if state is not None:
            if "bicycle" in classes:
                detections, sources = self._recover_bicycles(frame, detections, state)
            else:
                state.pop("bicycleRecovery", None)
        areas = zone_areas(zone, width, height)
        if not areas:
            areas = [{"name": "safe-default", "level": "safe", "points": [(0, 0), (width, 0), (width, height), (0, height)]}]
        objects = objects_in_areas(detections, areas, classes, include_outside=True)
        for item in objects:
            item.update(sources.get(tuple(item["bbox"]), {}))
        return objects, (time.perf_counter() - started) * 1000
