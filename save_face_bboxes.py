from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2


def make_detector(model: Path, width: int, height: int, score_threshold: float):
    return cv2.FaceDetectorYN_create(str(model), "", (width, height), score_threshold, 0.3, 5000)


def detect(detector, frame, detect_width: int = 0):
    height, width = frame.shape[:2]
    source = frame
    ratio = 1.0
    if detect_width and width > detect_width:
        ratio = width / detect_width
        source = cv2.resize(frame, (detect_width, int(height / ratio)))
    source_height, source_width = source.shape[:2]
    detector.setInputSize((source_width, source_height))
    _, faces = detector.detect(source)
    if faces is None:
        return []
    return [
        (int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio), float(score))
        for x, y, w, h, *_, score in faces
    ]


def draw(frame, boxes):
    for x, y, w, h, score in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Save face bbox detections from a video.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--model", type=Path, default=Path("result/models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--out-dir", type=Path, default=Path("result/detections"))
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--detect-width", type=int, default=0, help="resize width for detection; 0 keeps original")
    parser.add_argument("--every", type=int, default=24, help="save annotated frame every N frames")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("source has no frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = make_detector(args.model, frame.shape[1], frame.shape[0], args.score_threshold)
    csv_path = args.out_dir / "bboxes.csv"

    frames = detections = saved = 0
    start = time.perf_counter()
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "w", "h", "score"])
        while ok and (args.max_frames is None or frames < args.max_frames):
            boxes = detect(detector, frame, args.detect_width)
            detections += len(boxes)
            for box in boxes:
                writer.writerow([frames, *box])
            if boxes and frames % args.every == 0:
                annotated = frame.copy()
                draw(annotated, boxes)
                cv2.imwrite(str(args.out_dir / f"frame_{frames:06d}.jpg"), annotated)
                saved += 1
            frames += 1
            ok, frame = cap.read()

    cap.release()
    (args.out_dir / "summary.txt").write_text(
        f"source={args.source}\n"
        f"model={args.model}\n"
        f"frames={frames}\n"
        f"detections={detections}\n"
        f"saved_frames={saved}\n"
        f"fps={frames / (time.perf_counter() - start):.1f}\n"
        f"score_threshold={args.score_threshold}\n"
        f"detect_width={args.detect_width}\n"
    )
    print((args.out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
