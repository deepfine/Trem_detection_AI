from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from anomaly_rules import classify_event, direction_vector, point_in_polygon


COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_zone(path: Path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return data, False, [data]
    if "lines" in data:
        a, b = data["lines"]
        return [a[0], a[1], b[1], b[0]], True, data["lines"]
    return data["mask"], False, [data["mask"]]


def yolo_detections(net, frame, size, conf_threshold, nms_threshold):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()[0]
    boxes, scores, class_ids = [], [], []
    for row in output.T if output.shape[0] < output.shape[1] else output:
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


def track(detections, tracks, max_distance, history_size, max_missed=10):
    assigned = {}
    next_tracks = {
        tid: {**old, "missed": old.get("missed", 0) + 1}
        for tid, old in tracks.items()
        if old.get("missed", 0) < max_missed
    }
    next_id = max(tracks, default=0) + 1
    for label, score, box in detections:
        cx, cy = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        candidates = [
            (tid, (cx - old["points"][-1][0]) ** 2 + (cy - old["points"][-1][1]) ** 2)
            for tid, old in tracks.items()
            if old["label"] == label and tid not in assigned
        ]
        tid, distance = min(candidates, key=lambda item: item[1], default=(None, None))
        if tid is None or distance > max_distance**2:
            tid = next_id
            next_id += 1
        old = tracks.get(tid)
        points = (old["points"] if old else []) + [(cx, cy)]
        assigned[tid] = True
        next_tracks[tid] = {"points": points[-history_size:], "label": label, "missed": 0}
        yield tid, label, score, box, next_tracks[tid]["points"]
    tracks.clear()
    tracks.update(next_tracks)


def main():
    parser = argparse.ArgumentParser(description="Detect zone obstacles and fast incoming objects.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--model", type=Path, default=Path("result/models/yolov8n.onnx"))
    parser.add_argument("--zone", type=Path, required=True, help='JSON polygon, {"mask": [[x,y],...]}, or {"lines": [[[x,y],[x,y]], [[x,y],[x,y]]]}')
    parser.add_argument("--out-dir", type=Path, default=Path("result/anomalies"))
    parser.add_argument("--out-video", default="result/anomalies.mp4")
    parser.add_argument("--direction", default="down", choices=["left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right"])
    parser.add_argument("--min-speed", type=float, default=650, help="pixels per second")
    parser.add_argument("--min-alignment", type=float, default=0.75)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--history", type=int, default=5, help="centers used for average movement vector")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"object detector model is missing: {args.model}")

    zone, entry_only, guides = load_zone(args.zone)
    target_direction = direction_vector(args.direction)
    net = cv2.dnn.readNetFromONNX(str(args.model))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output: {args.out_video}")

    tracks = {}
    frames = detections_count = events = 0
    started = time.perf_counter()
    with (args.out_dir / "events.csv").open("w", newline="") as events_file, (args.out_dir / "detections.csv").open("w", newline="") as detections_file:
        event_rows = csv.writer(events_file)
        detection_rows = csv.writer(detections_file)
        event_rows.writerow(["frame", "track_id", "label", "score", "event", "x1", "y1", "x2", "y2"])
        detection_rows.writerow(["frame", "track_id", "label", "score", "in_zone", "event", "x1", "y1", "x2", "y2"])
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (args.max_frames and frames >= args.max_frames):
                    break
                for guide in guides:
                    cv2.polylines(frame, [np.array(guide, np.int32)], not entry_only, (255, 0, 0), 2)
                detections = yolo_detections(net, frame, args.size, args.conf, args.nms)
                for tid, label, score, box, points in track(detections, tracks, args.size / 2, args.history):
                    event = classify_event(label, box, points, zone, target_direction, fps, args.min_speed, args.min_alignment, entry_only)
                    in_zone = point_in_polygon(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), zone)
                    x1, y1, x2, y2 = box
                    detection_rows.writerow([frames, tid, label, f"{score:.3f}", int(in_zone), event, x1, y1, x2, y2])
                    detections_count += 1
                    if event:
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, f"WARNING {label}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        event_rows.writerow([frames, tid, label, f"{score:.3f}", event, x1, y1, x2, y2])
                        events += 1
                writer.write(frame)
                frames += 1
        finally:
            cap.release()
            writer.release()

    (args.out_dir / "summary.txt").write_text(
        f"source={args.source}\n"
        f"model={args.model}\n"
        f"frames={frames}\n"
        f"detections={detections_count}\n"
        f"events={events}\n"
        f"fps={frames / (time.perf_counter() - started):.1f}\n"
        f"direction={args.direction}\n"
        f"min_speed={args.min_speed}\n"
        f"entry_only={entry_only}\n"
    )
    print((args.out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
