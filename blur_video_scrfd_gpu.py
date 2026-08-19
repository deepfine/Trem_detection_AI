from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import onnxruntime as ort
from insightface.model_zoo import get_model

from blur_tracking import iou, keep_recent_boxes
from blur_video_yunet import blur_boxes


def make_detector(model: Path, device_id: int, det_size: tuple[int, int], threshold: float):
    providers = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]
    detector = get_model(str(model), providers=providers)
    detector.prepare(ctx_id=device_id, input_size=det_size, det_thresh=threshold)
    return detector


def detect(detector, frame):
    bboxes, _ = detector.detect(frame, max_num=0, metric="default")
    return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(score)) for x1, y1, x2, y2, score in bboxes]


def draw(frame, boxes):
    for x, y, w, h, score in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Blur faces with SCRFD on ONNX Runtime CUDA.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--model", type=Path, default=Path("result/models/scrfd_det_10g.onnx"))
    parser.add_argument("--out-video", default="result/test_video_scrfd_gpu_blurred.mp4")
    parser.add_argument("--out-dir", type=Path, default=Path("result/blur_scrfd_gpu"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--det-size", default="960x544", help="SCRFD input size, e.g. 960x544")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--pad", type=float, default=0.25)
    parser.add_argument("--hold-frames", type=int, default=2, help="reuse recent boxes for N frames when detection flickers")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-debug-every", type=int, default=120)
    args = parser.parse_args()

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {ort.get_available_providers()}")
    ort.set_default_logger_severity(3)

    det_w, det_h = (int(v) for v in args.det_size.lower().split("x", 1))
    if det_w % 32 or det_h % 32:
        raise ValueError("--det-size width and height must be multiples of 32")
    detector = make_detector(args.model, args.device_id, (det_w, det_h), args.threshold)

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

    frame_times = args.out_dir / "frame_times.csv"
    frames = detections = 0
    held_boxes = []
    started = time.perf_counter()
    with frame_times.open("w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["frame", "detections", "seconds"])
        try:
            while ok and (args.max_frames is None or frames < args.max_frames):
                frame_started = time.perf_counter()
                current_boxes = detect(detector, frame)
                boxes = keep_recent_boxes(current_boxes, held_boxes) if args.hold_frames else current_boxes
                held_boxes = [(box, args.hold_frames) for box in current_boxes] + [
                    (box, ttl - 1)
                    for box, ttl in held_boxes
                    if ttl > 1 and all(iou(box, current) < 0.3 for current in current_boxes)
                ]
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
        f"model={args.model}\n"
        f"device_id={args.device_id}\n"
        f"frames={frames}\n"
        f"detections={detections}\n"
        f"total_seconds={total:.3f}\n"
        f"seconds_per_frame={total / frames:.6f}\n"
        f"fps={frames / total:.1f}\n"
        f"det_size={args.det_size}\n"
        f"threshold={args.threshold}\n"
        f"pad={args.pad}\n"
        f"hold_frames={args.hold_frames}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
