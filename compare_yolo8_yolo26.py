from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort

from anomaly_rules import OBSTACLE_CLASSES
from detect_anomalies import COCO
from process_blur_anomalies import yolo_ort_detections


def preload_cuda_libs():
    root = Path(torch.__file__).resolve().parent.parent / "nvidia"
    for pattern in ["**/lib/libcublasLt.so.12", "**/lib/libcublas.so.12", "**/lib/libcudnn.so.9", "**/lib/libcudart.so.12"]:
        for path in glob.glob(str(root / pattern), recursive=True):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def session(model: Path, device_id: int):
    return ort.InferenceSession(str(model), providers=[("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"])


def yolo26_detections(session, frame, size, conf_threshold, nms_threshold):
    height, width = frame.shape[:2]
    image = cv2.resize(frame, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    output = session.run(None, {session.get_inputs()[0].name: image})[0][0]
    boxes, scores, class_ids = [], [], []
    for x1, y1, x2, y2, score, class_id in output:
        score = float(score)
        if score < conf_threshold:
            continue
        boxes.append([int(x1 * width / size), int(y1 * height / size), int((x2 - x1) * width / size), int((y2 - y1) * height / size)])
        scores.append(score)
        class_ids.append(int(class_id))
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    for i in np.array(keep).reshape(-1) if len(keep) else []:
        x, y, w, h = boxes[i]
        yield COCO[class_ids[i]], scores[i], (x, y, x + w, y + h)


def target_only(detections):
    return [item for item in detections if item[0] in OBSTACLE_CLASSES]


def draw(frame, title, detections, seconds):
    cv2.putText(frame, f"{title}  {len(detections)} boxes  {seconds * 1000:.1f} ms", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    for label, score, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Save side-by-side YOLOv8n vs YOLO26n bbox comparison video.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--yolo8", type=Path, default=Path("result/models/yolov8n.onnx"))
    parser.add_argument("--yolo26", type=Path, default=Path("result/models/yolo26n.onnx"))
    parser.add_argument("--out-video", default="result/yolo8_vs_yolo26.avi")
    parser.add_argument("--out-dir", type=Path, default=Path("result/yolo8_vs_yolo26"))
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    preload_cuda_libs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    y8 = session(args.yolo8, args.device_id)
    y26 = session(args.yolo26, args.device_id)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (width, height // 2)
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output: {args.out_video}")

    frames = y8_boxes = y26_boxes = 0
    y8_seconds = y26_seconds = write_seconds = 0.0
    started = time.perf_counter()
    with (args.out_dir / "frame_times.csv").open("w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["frame", "yolo8_boxes", "yolo26_boxes", "yolo8_seconds", "yolo26_seconds"])
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (args.max_frames and frames >= args.max_frames):
                    break
                left = frame.copy()
                right = frame.copy()

                t = time.perf_counter()
                det8 = target_only(list(yolo_ort_detections(y8, frame, args.size, args.conf, args.nms)))
                s8 = time.perf_counter() - t

                t = time.perf_counter()
                det26 = target_only(list(yolo26_detections(y26, frame, args.size, args.conf, args.nms)))
                s26 = time.perf_counter() - t

                left = draw(left, "YOLOv8n", det8, s8)
                right = draw(right, "YOLO26n", det26, s26)
                combined = np.hstack([cv2.resize(left, (width // 2, height // 2)), cv2.resize(right, (width // 2, height // 2))])

                t = time.perf_counter()
                writer.write(combined)
                write_seconds += time.perf_counter() - t

                rows.writerow([frames, len(det8), len(det26), f"{s8:.6f}", f"{s26:.6f}"])
                frames += 1
                y8_boxes += len(det8)
                y26_boxes += len(det26)
                y8_seconds += s8
                y26_seconds += s26
        finally:
            cap.release()
            writer.release()

    total = time.perf_counter() - started
    summary = (
        f"source={args.source}\n"
        f"out_video={args.out_video}\n"
        f"frames={frames}\n"
        f"total_seconds={total:.3f}\n"
        f"seconds_per_frame={total / frames:.6f}\n"
        f"fps={frames / total:.1f}\n"
        f"yolo8_boxes={y8_boxes}\n"
        f"yolo8_seconds_per_frame={y8_seconds / frames:.6f}\n"
        f"yolo8_fps={frames / y8_seconds:.1f}\n"
        f"yolo26_boxes={y26_boxes}\n"
        f"yolo26_seconds_per_frame={y26_seconds / frames:.6f}\n"
        f"yolo26_fps={frames / y26_seconds:.1f}\n"
        f"write_seconds_per_frame={write_seconds / frames:.6f}\n"
        f"providers_yolo8={y8.get_providers()}\n"
        f"providers_yolo26={y26.get_providers()}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
