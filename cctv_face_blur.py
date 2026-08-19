from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import requests


def detect_faces(frame, detector, scale_factor: float = 1.1, min_neighbors: int = 5, detect_width: int = 0):
    source = frame
    ratio = 1.0
    if detect_width and frame.shape[1] > detect_width:
        ratio = frame.shape[1] / detect_width
        source = cv2.resize(frame, (detect_width, int(frame.shape[0] / ratio)))
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scale_factor, min_neighbors)
    if ratio == 1.0:
        return faces
    return [(int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)) for x, y, w, h in faces]


def blur_faces(frame, detector, scale_factor: float = 1.1, min_neighbors: int = 5, detect_width: int = 0):
    for x, y, w, h in detect_faces(frame, detector, scale_factor, min_neighbors, detect_width):
        face = frame[y : y + h, x : x + w]
        k = max(15, (min(w, h) // 3) | 1)
        frame[y : y + h, x : x + w] = cv2.GaussianBlur(face, (k, k), 0)
    return frame


def jpeg_bytes(frame, quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode frame")
    return encoded.tobytes()


def sender(url: str, frames: queue.Queue[tuple[int, bytes]], stop: threading.Event, timeout: float):
    session = requests.Session()
    while not stop.is_set() or not frames.empty():
        try:
            frame_id, image = frames.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            session.post(
                url,
                files={"frame": (f"{frame_id}.jpg", image, "image/jpeg")},
                data={"frame_id": str(frame_id), "ts": str(time.time())},
                timeout=timeout,
            ).raise_for_status()
        except requests.RequestException as exc:
            print(f"send failed for frame {frame_id}: {exc}", flush=True)


def put_latest(frames: queue.Queue[tuple[int, bytes]], item: tuple[int, bytes]):
    try:
        frames.put_nowait(item)
    except queue.Full:
        try:
            frames.get_nowait()
        except queue.Empty:
            pass
        frames.put_nowait(item)


def blur_video_file(source: str, output: str, detect_width: int, max_frames: int | None):
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        raise RuntimeError(f"failed to load face detector: {cascade}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output: {output}")

    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and frames >= max_frames):
                break
            writer.write(blur_faces(frame, detector, detect_width=detect_width))
            frames += 1
    finally:
        cap.release()
        writer.release()
    return frames


def stream(
    source: str,
    server_url: str,
    fps: float,
    quality: int,
    timeout: float,
    max_frames: int | None,
    detect_width: int,
):
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        raise RuntimeError(f"failed to load face detector: {cascade}")

    cap = cv2.VideoCapture(0 if source == "0" else source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")

    frames: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=1)
    stop = threading.Event()
    worker = threading.Thread(target=sender, args=(server_url, frames, stop, timeout), daemon=True)
    worker.start()

    frame_id = 0
    next_frame = time.monotonic()
    period = 1.0 / fps
    try:
        while True:
            now = time.monotonic()
            if now < next_frame:
                time.sleep(min(0.01, next_frame - now))
                continue
            next_frame = now + period

            ok, frame = cap.read()
            if not ok:
                if max_frames:
                    break
                time.sleep(0.01)
                continue

            put_latest(frames, (frame_id, jpeg_bytes(blur_faces(frame, detector, detect_width=detect_width), quality)))
            frame_id += 1
            if max_frames and frame_id >= max_frames:
                break
    finally:
        stop.set()
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Blur CCTV faces and POST latest frames.")
    parser.add_argument("--source", default="0", help="camera index, video file, or RTSP/HTTP URL")
    parser.add_argument("--server-url", help="destination POST endpoint")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, help="stop after N processed frames")
    parser.add_argument("--detect-width", type=int, default=960, help="resize width for detection; 0 keeps original")
    parser.add_argument("--output", help="write blurred video instead of POST streaming")
    args = parser.parse_args()
    if args.output:
        print(blur_video_file(args.source, args.output, args.detect_width, args.max_frames))
        return
    if not args.server_url:
        parser.error("--server-url is required unless --output is set")
    stream(args.source, args.server_url, args.fps, args.quality, args.timeout, args.max_frames, args.detect_width)


if __name__ == "__main__":
    main()
