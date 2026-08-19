from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from anomaly_rules import OBSTACLE_CLASSES, average_delta, center, classify_event, direction_vector, point_in_polygon, sample_due
from blur_tracking import iou, keep_recent_boxes
from blur_video_scrfd_gpu import detect as detect_faces
from blur_video_scrfd_gpu import make_detector as make_face_detector
from blur_video_yunet import blur_boxes
from detect_anomalies import COCO, load_zone, track, yolo_detections
from person_groups import assistive_for_person, classify_person, load_group_config


def avg(total, frames):
    return total / frames if frames else 0.0


def make_yolo_session(model: Path, device_id: int):
    return ort.InferenceSession(
        str(model),
        providers=[("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
    )


def yolo_ort_detections(session, frame, size, conf_threshold, nms_threshold):
    height, width = frame.shape[:2]
    image = cv2.resize(frame, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    output = session.run(None, {session.get_inputs()[0].name: image})[0][0]
    boxes, scores, class_ids = [], [], []
    for row in output.T:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < conf_threshold:
            continue
        x, y, w, h = row[:4]
        scale_x, scale_y = (width, height) if max(x, y, w, h) <= 2 else (width / size, height / size)
        x1 = int((x - w / 2) * scale_x)
        y1 = int((y - h / 2) * scale_y)
        boxes.append([x1, y1, int(w * scale_x), int(h * scale_y)])
        scores.append(score)
        class_ids.append(class_id)
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    for i in np.array(keep).reshape(-1) if len(keep) else []:
        x, y, w, h = boxes[i]
        yield COCO[class_ids[i]], scores[i], (x, y, x + w, y + h)


def assistive_detections(model, frame, device_id, size, confidence):
    result = model.predict(frame, device=device_id, imgsz=size, conf=confidence, verbose=False)[0]
    if result.boxes is None:
        return []
    return [
        (result.names[int(box.cls)], float(box.conf), tuple(int(value) for value in box.xyxy[0].tolist()))
        for box in result.boxes
    ]


ZONE_COLORS = {
    "safe": (0, 180, 0),
    "warning": (0, 215, 255),
    "danger": (0, 0, 255),
}
ZONE_PRIORITY = {"safe": 0, "warning": 1, "danger": 2}


def load_areas(path: Path):
    import json

    data = json.loads(path.read_text())
    if isinstance(data, dict) and "zones" in data:
        areas = []
        for zone in data["zones"]:
            level = zone["level"]
            if level not in ZONE_PRIORITY:
                raise ValueError(f"unknown zone level: {level}")
            points = []
            for point in zone.get("points") or zone.get("mask") or []:
                point = [int(point[0]), int(point[1])]
                if not points or point != points[-1]:
                    points.append(point)
            if len(points) < 3:
                raise ValueError(f"zone needs at least 3 points: {zone.get('name', level)}")
            areas.append({"name": zone.get("name", level), "level": level, "points": points})
        return areas, None
    zone, entry_only, guides = load_zone(path)
    return None, (zone, entry_only, guides)


def area_for_point(point, areas):
    matches = [area for area in areas if point_in_polygon(point, area["points"])]
    if not matches:
        return None
    return max(matches, key=lambda item: ZONE_PRIORITY[item["level"]])


def area_at(box, areas):
    return area_for_point(center(box), areas)


def predicted_area(points, areas, fps, seconds):
    movement = average_delta(points)
    if not movement:
        return None
    dx, dy, span = movement
    future_frames = int(round(fps * seconds))
    for step in range(1, future_frames + 1):
        point = (points[-1][0] + dx * step / span, points[-1][1] + dy * step / span)
        area = area_for_point(point, areas)
        if area and area["level"] in {"danger", "warning"}:
            return area
    return None


def object_zone(label, box, areas):
    if label not in OBSTACLE_CLASSES:
        return None
    area = area_at(box, areas)
    return area if area and area["level"] in {"danger", "warning"} else None


def confirm_alert(alerts, key, frame, confirm_frames):
    state = alerts.setdefault(key, {"hits": 0, "last_frame": -10, "emitted": False})
    if frame - state["last_frame"] > 2:
        state["hits"] = 0
        state["emitted"] = False
    state["hits"] += 1
    state["last_frame"] = frame
    active = state["hits"] >= confirm_frames
    new_event = active and not state["emitted"]
    if new_event:
        state["emitted"] = True
    return active, new_event


def draw_corner_alert(frame, level, count, incoming_count=0, groups=None):
    color = ZONE_COLORS[level]
    height, width = frame.shape[:2]
    length = 130
    thickness = 14 if level == "danger" else 10
    for x1, y1, x2, y2 in [
        (0, 0, length, 0), (0, 0, 0, length),
        (width - length, 0, width, 0), (width - 1, 0, width - 1, length),
        (0, height - 1, length, height - 1), (0, height - length, 0, height - 1),
        (width - length, height - 1, width, height - 1), (width - 1, height - length, width - 1, height - 1),
    ]:
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{level.upper()} INCOMING x{incoming_count}" if incoming_count else f"{level.upper()} DETECTED x{count}"
    cv2.putText(frame, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    if groups:
        detail = f"ADULT {groups['adult_estimated']}  CHILD {groups['child_estimated']}  ASSISTIVE {groups['assistive_device_user_estimated']}  UNKNOWN {groups['person_uncertain']}"
        cv2.putText(frame, detail, (30, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_zone_overlay(frame, zone, guides, entry_only, direction):
    overlay = frame.copy()
    polygon = np.array(zone, np.int32)
    cv2.fillPoly(overlay, [polygon], (255, 0, 0))
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for guide in guides:
        cv2.polylines(frame, [np.array(guide, np.int32)], not entry_only, (255, 255, 0), 3)
    x, y = 80, 90
    dx, dy = direction
    cv2.arrowedLine(frame, (x, y), (int(x + dx * 90), int(y + dy * 90)), (0, 255, 255), 4, tipLength=0.25)
    cv2.putText(frame, "FAST DIRECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def draw_motion_arrow(frame, box, points):
    movement = average_delta(points)
    if not movement:
        return
    dx, dy, _ = movement
    cx, cy = center(box)
    cv2.arrowedLine(frame, (int(cx - dx), int(cy - dy)), (int(cx), int(cy)), (0, 255, 255), 3, tipLength=0.3)


def show_danger_box(event, active):
    return active and event == "danger_zone_object"


def main():
    parser = argparse.ArgumentParser(description="Apply face blur and anomaly warnings in one pass.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--zone", type=Path, default=Path("entry_lines.json"))
    parser.add_argument("--out-video", default="result/blur_anomalies_entry_lines.mp4")
    parser.add_argument("--out-dir", type=Path, default=Path("result/blur_anomalies_entry_lines"))
    parser.add_argument("--face-model", type=Path, default=Path("result/models/scrfd_det_10g.onnx"))
    parser.add_argument("--object-model", type=Path, default=Path("result/models/yolov8n.onnx"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--object-device-id", type=int, default=1)
    parser.add_argument("--object-backend", choices=["ort-cuda", "opencv"], default="ort-cuda")
    parser.add_argument("--det-size", default="960x544")
    parser.add_argument("--face-threshold", type=float, default=0.35)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--pad", type=float, default=0.25)
    parser.add_argument("--direction", default="down", choices=["left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right"])
    parser.add_argument("--min-speed", type=float, default=650)
    parser.add_argument("--min-alignment", type=float, default=0.75)
    parser.add_argument("--object-conf", type=float, default=0.35)
    parser.add_argument("--object-nms", type=float, default=0.45)
    parser.add_argument("--object-size", type=int, default=640)
    parser.add_argument("--group-config", type=Path, default=Path("person_groups.json"))
    parser.add_argument("--assistive-model", type=Path, default=Path("result/models/assistive_yolov8s_worldv2.pt"))
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--analysis-fps", type=float, help="sample the source at this rate; output video uses the same rate")
    parser.add_argument("--confirm-frames", type=int, default=3, help="consecutive detections required before an alert")
    parser.add_argument("--track-max-missed", type=int, default=10, help="frames to retain a missing track")
    parser.add_argument("--prediction-seconds", type=float, default=1.5, help="future path horizon for fast incoming alerts")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    cv2.setNumThreads(1)

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {ort.get_available_providers()}")
    ort.set_default_logger_severity(4)
    for path in [args.zone, args.face_model, args.object_model, args.group_config, args.assistive_model]:
        if not path.exists():
            raise FileNotFoundError(path)

    det_w, det_h = (int(v) for v in args.det_size.lower().split("x", 1))
    face_detector = make_face_detector(args.face_model, args.device_id, (det_w, det_h), args.face_threshold)
    object_net = make_yolo_session(args.object_model, args.object_device_id) if args.object_backend == "ort-cuda" else cv2.dnn.readNetFromONNX(str(args.object_model))
    from ultralytics import YOLO

    group_config = load_group_config(args.group_config)
    assistive_model = YOLO(str(args.assistive_model))
    areas, legacy_zone = load_areas(args.zone)
    zone, entry_only, guides = legacy_zone if legacy_zone else (None, False, [])
    target_direction = direction_vector(args.direction)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("source has no frames")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    analysis_fps = args.analysis_fps or fps
    if not 0 < analysis_fps <= fps:
        raise ValueError(f"analysis fps must be in (0, {fps}], got {analysis_fps}")
    height, width = frame.shape[:2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), analysis_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output: {args.out_video}")

    tracks = {}
    alerts = {}
    assistive_boxes = []
    group_events = Counter()
    held_faces = []
    frames = faces = objects = events = 0
    source_frame = 0
    next_sample_seconds = 0.0
    face_seconds = yolo_seconds = assistive_seconds = track_seconds = write_seconds = total_frame_seconds = 0.0
    assistive_runs = assistive_objects = 0
    started = time.perf_counter()
    with (args.out_dir / "frame_times.csv").open("w", newline="") as times_file, (args.out_dir / "events.csv").open("w", newline="") as events_file:
        time_rows = csv.writer(times_file)
        event_rows = csv.writer(events_file)
        time_rows.writerow(["frame", "faces", "objects", "assistive_objects", "events", "face_seconds", "yolo_seconds", "assistive_seconds", "track_seconds", "write_seconds", "total_seconds"])
        event_rows.writerow(["frame", "track_id", "label", "person_group", "assistive_device", "score", "event", "level", "zone", "x1", "y1", "x2", "y2"])
        try:
            while ok and (args.max_frames is None or frames < args.max_frames):
                if not sample_due(source_frame, fps, next_sample_seconds):
                    source_frame += 1
                    ok, frame = cap.read()
                    continue
                next_sample_seconds += 1 / analysis_fps
                frame_started = time.perf_counter()

                yolo_started = time.perf_counter()
                if args.object_backend == "ort-cuda":
                    detections = list(yolo_ort_detections(object_net, frame, args.object_size, args.object_conf, args.object_nms))
                else:
                    detections = list(yolo_detections(object_net, frame, args.object_size, args.object_conf, args.object_nms))
                yolo_elapsed = time.perf_counter() - yolo_started

                assistive_elapsed = 0.0
                if frames % group_config["assistive_interval"] == 0:
                    assistive_started = time.perf_counter()
                    assistive_boxes = assistive_detections(
                        assistive_model,
                        frame,
                        args.object_device_id,
                        args.object_size,
                        group_config["assistive_confidence"],
                    )
                    assistive_elapsed = time.perf_counter() - assistive_started
                    assistive_runs += 1
                    assistive_objects += len(assistive_boxes)

                face_started = time.perf_counter()
                current_faces = detect_faces(face_detector, frame)
                face_boxes = keep_recent_boxes(current_faces, held_faces) if args.hold_frames else current_faces
                held_faces = [(box, args.hold_frames) for box in current_faces] + [
                    (box, ttl - 1)
                    for box, ttl in held_faces
                    if ttl > 1 and all(iou(box, current) < 0.3 for current in current_faces)
                ]
                blur_boxes(frame, face_boxes, args.pad)
                face_elapsed = time.perf_counter() - face_started

                track_started = time.perf_counter()
                frame_events = 0
                if not areas:
                    draw_zone_overlay(frame, zone, guides, entry_only, target_direction)
                frame_level_counts = {"danger": 0, "warning": 0}
                incoming_level_counts = {"danger": 0, "warning": 0}
                frame_group_counts = {level: Counter() for level in frame_level_counts}
                for tid, label, score, box, points in track(detections, tracks, args.object_size / 2, args.history, args.track_max_missed):
                    person_group = assistive_device = ""
                    if label == "person":
                        assistive_device = assistive_for_person(box, assistive_boxes)
                        person_group = "assistive_device_user_estimated" if assistive_device else classify_person(box, frame.shape, group_config)
                    current_area = area_at(box, areas) if areas else None
                    area = object_zone(label, box, areas) if areas else None
                    if area:
                        event = f"{area['level']}_zone_object"
                        level = area["level"]
                        zone_name = area["name"]
                    else:
                        event = classify_event(label, box, points, None if areas else zone, target_direction, analysis_fps, args.min_speed, args.min_alignment, entry_only) if not current_area else ""
                        incoming_area = predicted_area(points, areas, analysis_fps, args.prediction_seconds) if event == "fast_incoming" and areas else None
                        if event == "fast_incoming" and areas and not incoming_area:
                            event = ""
                        level = incoming_area["level"] if incoming_area else ("danger" if event else "")
                        zone_name = incoming_area["name"] if incoming_area else ""
                    if event:
                        x1, y1, x2, y2 = box
                        if not areas:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"WARNING {label}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            draw_motion_arrow(frame, box, points)
                        active, new_event = confirm_alert(alerts, (tid, event, zone_name), frames, args.confirm_frames)
                        if areas and show_danger_box(event, active):
                            detail = person_group.replace("_estimated", "").replace("_", " ").upper()
                            text = f"DANGER {label.upper()}" + (f" / {detail}" if detail else "")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), ZONE_COLORS["danger"], 3)
                            cv2.putText(frame, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ZONE_COLORS["danger"], 2)
                        if active and level in frame_level_counts:
                            frame_level_counts[level] += 1
                            if person_group:
                                frame_group_counts[level][person_group] += 1
                            if event == "fast_incoming":
                                incoming_level_counts[level] += 1
                        if new_event:
                            event_rows.writerow([frames, tid, label, person_group, assistive_device, f"{score:.3f}", event, level, zone_name, x1, y1, x2, y2])
                            if person_group:
                                group_events[person_group] += 1
                            frame_events += 1
                alerts = {key: state for key, state in alerts.items() if frames - state["last_frame"] <= args.track_max_missed}
                if areas:
                    if frame_level_counts["danger"]:
                        draw_corner_alert(frame, "danger", frame_level_counts["danger"], incoming_level_counts["danger"], frame_group_counts["danger"])
                    elif frame_level_counts["warning"]:
                        draw_corner_alert(frame, "warning", frame_level_counts["warning"], incoming_level_counts["warning"], frame_group_counts["warning"])
                track_elapsed = time.perf_counter() - track_started

                write_started = time.perf_counter()
                writer.write(frame)
                write_elapsed = time.perf_counter() - write_started
                total_elapsed = time.perf_counter() - frame_started

                time_rows.writerow([frames, len(face_boxes), len(detections), len(assistive_boxes), frame_events, f"{face_elapsed:.6f}", f"{yolo_elapsed:.6f}", f"{assistive_elapsed:.6f}", f"{track_elapsed:.6f}", f"{write_elapsed:.6f}", f"{total_elapsed:.6f}"])
                frames += 1
                source_frame += 1
                faces += len(face_boxes)
                objects += len(detections)
                events += frame_events
                face_seconds += face_elapsed
                yolo_seconds += yolo_elapsed
                assistive_seconds += assistive_elapsed
                track_seconds += track_elapsed
                write_seconds += write_elapsed
                total_frame_seconds += total_elapsed
                ok, frame = cap.read()
        finally:
            cap.release()
            writer.release()

    wall_seconds = time.perf_counter() - started
    summary = (
        f"source={args.source}\n"
        f"source_fps={fps:.1f}\n"
        f"analysis_fps={analysis_fps:.1f}\n"
        f"zone={args.zone}\n"
        f"out_video={args.out_video}\n"
        f"face_device_id={args.device_id}\n"
        f"object_backend={args.object_backend}\n"
        f"object_device_id={args.object_device_id}\n"
        f"group_config={args.group_config}\n"
        f"assistive_model={args.assistive_model}\n"
        f"prediction_seconds={args.prediction_seconds}\n"
        f"frames={frames}\n"
        f"faces={faces}\n"
        f"objects={objects}\n"
        f"events={events}\n"
        f"total_seconds={wall_seconds:.3f}\n"
        f"seconds_per_frame={avg(wall_seconds, frames):.6f}\n"
        f"fps={frames / wall_seconds:.1f}\n"
        f"face_blur_seconds_per_frame={avg(face_seconds, frames):.6f}\n"
        f"face_blur_fps={frames / face_seconds:.1f}\n"
        f"yolo_seconds_per_frame={avg(yolo_seconds, frames):.6f}\n"
        f"yolo_fps={frames / yolo_seconds:.1f}\n"
        f"assistive_runs={assistive_runs}\n"
        f"assistive_objects={assistive_objects}\n"
        f"assistive_seconds_per_run={avg(assistive_seconds, assistive_runs):.6f}\n"
        f"adult_estimated_events={group_events['adult_estimated']}\n"
        f"child_estimated_events={group_events['child_estimated']}\n"
        f"assistive_device_user_estimated_events={group_events['assistive_device_user_estimated']}\n"
        f"person_uncertain_events={group_events['person_uncertain']}\n"
        f"track_event_seconds_per_frame={avg(track_seconds, frames):.6f}\n"
        f"track_event_fps={frames / track_seconds:.1f}\n"
        f"write_seconds_per_frame={avg(write_seconds, frames):.6f}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
