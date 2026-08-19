from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from save_face_bboxes import detect, draw, make_detector


def blur_boxes(frame, boxes, pad: float):
    height, width = frame.shape[:2]
    for x, y, w, h, _ in boxes:
        px, py = int(w * pad), int(h * pad)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(width, x + w + px), min(height, y + h + py)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = max(15, (min(x2 - x1, y2 - y1) // 2) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main():
    parser = argparse.ArgumentParser(description="Blur YuNet face bboxes and write a video.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--model", type=Path, default=Path("result/models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--out-video", default="result/test_video_yunet_blurred.mp4")
    parser.add_argument("--out-dir", type=Path, default=Path("result/blur_yunet"))
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--detect-width", type=int, default=0, help="resize width for detection; 0 keeps original")
    parser.add_argument("--detect-every", type=int, default=1, help="run detector every N frames and reuse boxes")
    parser.add_argument("--pad", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-debug-every", type=int, default=120)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("source has no frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output: {args.out_video}")

    detector = make_detector(args.model, width, height, args.score_threshold)
    frame_times = args.out_dir / "frame_times.csv"
    frames = detections = detector_runs = 0
    boxes = []
    started = time.perf_counter()
    with frame_times.open("w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["frame", "detections", "seconds"])
        try:
            while ok and (args.max_frames is None or frames < args.max_frames):
                frame_started = time.perf_counter()
                if frames % args.detect_every == 0:
                    boxes = detect(detector, frame, args.detect_width)
                    detector_runs += 1
                blur_boxes(frame, boxes, args.pad)
                writer.write(frame)
                seconds = time.perf_counter() - frame_started
                rows.writerow([frames, len(boxes), f"{seconds:.6f}"])
                detections += len(boxes)
                if boxes and args.save_debug_every and frames % args.save_debug_every == 0:
                    debug = frame.copy()
                    draw(debug, boxes)
                    cv2.imwrite(str(args.out_dir / f"frame_{frames:06d}.jpg"), debug)
                frames += 1
                ok, frame = cap.read()
        finally:
            cap.release()
            writer.release()

    total = time.perf_counter() - started
    summary = (
        f"source={args.source}\n"
        f"out_video={args.out_video}\n"
        f"frames={frames}\n"
        f"detections={detections}\n"
        f"detector_runs={detector_runs}\n"
        f"total_seconds={total:.3f}\n"
        f"seconds_per_frame={total / frames:.6f}\n"
        f"fps={frames / total:.1f}\n"
        f"score_threshold={args.score_threshold}\n"
        f"detect_width={args.detect_width}\n"
        f"detect_every={args.detect_every}\n"
        f"pad={args.pad}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
