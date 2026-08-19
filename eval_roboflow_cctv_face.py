from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from cctv_face_blur import blur_faces, detect_faces

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def detector():
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(str(cascade))
    if det.empty():
        raise RuntimeError(f"failed to load face detector: {cascade}")
    return det


def yolo_boxes(label: Path, width: int, height: int):
    if not label.exists():
        return []
    boxes = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)
        boxes.append((x1, y1, x2, y2))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def score(preds, truths, threshold: float):
    matched = set()
    tp = 0
    for pred in preds:
        best = max(((iou(pred, gt), i) for i, gt in enumerate(truths) if i not in matched), default=(0, -1))
        if best[0] >= threshold:
            tp += 1
            matched.add(best[1])
    return tp, len(preds) - tp, len(truths) - tp


def label_for(image: Path, dataset: Path):
    parts = list(image.relative_to(dataset).parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return dataset.joinpath(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def eval_image(path: Path, dataset: Path, det, out: Path | None, threshold: float, detect_width: int):
    frame = cv2.imread(str(path))
    if frame is None:
        return 0, 0, 0, 0, 0, 0
    h, w = frame.shape[:2]
    preds = [(x, y, x + ww, y + hh) for x, y, ww, hh in detect_faces(frame, det, detect_width=detect_width)]
    truths = yolo_boxes(label_for(path, dataset), w, h)
    if out:
        out_file = out / path.relative_to(dataset)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), blur_faces(frame, det, detect_width=detect_width))
    tp, fp, fn = score(preds, truths, threshold) if truths else (0, len(preds), 0)
    return 1, len(preds), tp, fp, fn, len(truths)


def eval_video(path: Path, det, max_frames: int, detect_width: int):
    cap = cv2.VideoCapture(str(path))
    frames = faces = 0
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        faces += len(detect_faces(frame, det, detect_width=detect_width))
        frames += 1
    cap.release()
    return frames, faces


def main():
    parser = argparse.ArgumentParser(description="Evaluate current face blur detector on a Roboflow export.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, help="write blurred images here")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--video-frames", type=int, default=240)
    parser.add_argument("--detect-width", type=int, default=960)
    args = parser.parse_args()

    det = detector()
    started = time.perf_counter()
    image_rows = [p for p in args.dataset.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    video_rows = [p for p in args.dataset.rglob("*") if p.suffix.lower() in VIDEO_EXTS]

    images = preds = tp = fp = fn = truths = 0
    for path in image_rows:
        row = eval_image(path, args.dataset, det, args.out, args.iou, args.detect_width)
        images += row[0]
        preds += row[1]
        tp += row[2]
        fp += row[3]
        fn += row[4]
        truths += row[5]

    video_frames = video_faces = 0
    for path in video_rows:
        frames, faces = eval_video(path, det, args.video_frames, args.detect_width)
        video_frames += frames
        video_faces += faces

    seconds = time.perf_counter() - started
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"images={images} videos={len(video_rows)} video_frames={video_frames}")
    print(f"detections={preds + video_faces} labels={truths} tp={tp} fp={fp} fn={fn}")
    print(f"precision={precision:.3f} recall={recall:.3f} fps={(images + video_frames) / seconds:.1f}")


if __name__ == "__main__":
    main()
