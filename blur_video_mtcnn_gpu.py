from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from blur_video_yunet import blur_boxes


def detect(mtcnn: MTCNN, frame, detect_width: int):
    height, width = frame.shape[:2]
    ratio = 1.0
    source = frame
    if detect_width and width > detect_width:
        ratio = width / detect_width
        source = cv2.resize(frame, (detect_width, int(height / ratio)))

    image = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(image)
    if boxes is None:
        return []
    return [
        (
            int(x1 * ratio),
            int(y1 * ratio),
            int((x2 - x1) * ratio),
            int((y2 - y1) * ratio),
            float(score),
        )
        for (x1, y1, x2, y2), score in zip(boxes, probs)
        if score is not None
    ]


def main():
    parser = argparse.ArgumentParser(description="Blur faces with PyTorch MTCNN on CUDA.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--out-video", default="result/test_video_mtcnn_gpu_blurred.mp4")
    parser.add_argument("--out-dir", type=Path, default=Path("result/blur_mtcnn_gpu"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detect-width", type=int, default=960)
    parser.add_argument("--min-face-size", type=int, default=20)
    parser.add_argument("--pad", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-debug-every", type=int, default=120)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available to PyTorch")

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

    mtcnn = MTCNN(keep_all=True, min_face_size=args.min_face_size, device=args.device)
    frame_times = args.out_dir / "frame_times.csv"
    frames = detections = 0
    started = time.perf_counter()
    with frame_times.open("w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["frame", "detections", "seconds"])
        try:
            while ok and (args.max_frames is None or frames < args.max_frames):
                frame_started = time.perf_counter()
                boxes = detect(mtcnn, frame, args.detect_width)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                blur_boxes(frame, boxes, args.pad)
                writer.write(frame)
                seconds = time.perf_counter() - frame_started
                rows.writerow([frames, len(boxes), f"{seconds:.6f}"])
                detections += len(boxes)
                if boxes and args.save_debug_every and frames % args.save_debug_every == 0:
                    debug = frame.copy()
                    for x, y, w, h, score in boxes:
                        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(debug, f"{score:.2f}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        f"device={args.device}\n"
        f"frames={frames}\n"
        f"detections={detections}\n"
        f"total_seconds={total:.3f}\n"
        f"seconds_per_frame={total / frames:.6f}\n"
        f"fps={frames / total:.1f}\n"
        f"detect_width={args.detect_width}\n"
        f"min_face_size={args.min_face_size}\n"
        f"pad={args.pad}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
