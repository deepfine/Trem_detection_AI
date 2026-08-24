from __future__ import annotations

import argparse
import time
import threading
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import onnxruntime as ort

from crowd_analysis_protocol import AnalysisJob, completed_object_payload, failed_payload
from crowd_analysis_server import env_path, env_str, make_handler, post_result_with_retry
from detect_anomalies import COCO
from frame_objects import DEFAULT_TARGET_CLASSES, objects_in_polygons, zone_polygons


class ObjectDetector:
    def __init__(self, model: Path, device_id=0, size=640, confidence=0.35, nms=0.45):
        self.model = model
        self.device_id = device_id
        self.size = size
        self.confidence = confidence
        self.nms = nms
        self.session = None

    def load(self):
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        ort.preload_dlls()
        self.session = ort.InferenceSession(
            str(self.model),
            providers=[("CUDAExecutionProvider", {"device_id": self.device_id}), "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(f"CUDA provider unavailable: {self.session.get_providers()}")
        self.session.run(None, {self.session.get_inputs()[0].name: np.zeros((1, 3, self.size, self.size), np.float32)})

    def detect(self, frame, zone, target_classes=()):
        height, width = frame.shape[:2]
        image = cv2.resize(frame, (self.size, self.size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.session.get_inputs()[0].name: image})[0][0]
        boxes, scores, class_ids = [], [], []
        for row in output.T:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence:
                continue
            x, y, w, h = row[:4]
            scale_x, scale_y = (width, height) if max(x, y, w, h) <= 2 else (width / self.size, height / self.size)
            x1, y1 = max(0, int((x - w / 2) * scale_x)), max(0, int((y - h / 2) * scale_y))
            x2, y2 = min(width, int((x + w / 2) * scale_x)), min(height, int((y + h / 2) * scale_y))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)
        keep = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, self.confidence, self.nms)
        detections = []
        for index in np.array(keep).reshape(-1) if len(keep) else []:
            x, y, w, h = boxes[index]
            detections.append((COCO[class_ids[index]], scores[index], (x, y, x + w, y + h)))
        polygons = zone_polygons(zone, width, height)
        if not polygons:
            raise ValueError("missing zone JSON")
        return objects_in_polygons(detections, polygons, set(target_classes) or DEFAULT_TARGET_CLASSES)


class ObjectAnalysisRuntime:
    def __init__(self, detector, analyzed_dir: Path, allowed_roots: list[Path], result_url: str, api_key: str, result_sink=None):
        self.detector = detector
        self.analyzed_dir = analyzed_dir.resolve()
        self.allowed_roots = [root.resolve() for root in allowed_roots]
        self.result_url = result_url
        self.api_key = api_key
        self.result_sink = result_sink
        self.ready = False
        self.jobs: Queue[AnalysisJob | None] = Queue()

    def start(self):
        self.detector.load()
        threading.Thread(target=self._worker, name="object-worker", daemon=True).start()
        self.ready = True

    def enqueue(self, job: AnalysisJob):
        self.jobs.put(job)

    def _worker(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                frame = cv2.imread(str(job.absolute_path))
                if frame is None:
                    raise FileNotFoundError(f"failed to read frame: {job.absolute_path}")
                started = time.perf_counter()
                objects = self.detector.detect(frame, job.zone, job.target_classes)
                payload = completed_object_payload(job, objects, (time.perf_counter() - started) * 1000)
            except Exception as error:
                payload = failed_payload(job, str(error))
            try:
                if self.result_sink is not None:
                    self.result_sink(payload)
                elif self.result_url:
                    post_result_with_retry(self.result_url, payload, self.api_key)
            except Exception as error:
                print(f"object callback failed id={job.id} reason={error}", flush=True)
            finally:
                self.jobs.task_done()


def main():
    parser = argparse.ArgumentParser(description="Detect configured objects between two lines.")
    parser.add_argument("--host", default=env_str("ANALYSIS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(env_str("ANALYSIS_PORT", "8080")))
    parser.add_argument("--model", type=Path, default=env_path("OBJECT_MODEL_PATH", "/app/result/models/yolov8n.onnx"))
    parser.add_argument("--device-id", type=int, default=int(env_str("OBJECT_DEVICE_ID", "0")))
    parser.add_argument("--analyzed-dir", type=Path, default=env_path("FRAME_ANALYZED_DIR", "/upload/visit_servant/analyzed"))
    parser.add_argument("--result-url", default=env_str("OBJECT_RESULT_URL", "http://api:3535/object/results"))
    parser.add_argument("--api-key", default=env_str("ANALYSIS_API_KEY", ""))
    args = parser.parse_args()

    analyzed_dir = args.analyzed_dir.resolve()
    runtime = ObjectAnalysisRuntime(
        ObjectDetector(args.model, args.device_id),
        analyzed_dir,
        [analyzed_dir, Path("/upload").resolve()],
        args.result_url,
        args.api_key,
    )
    runtime.start()
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime, "/object/analysis"))
    print(f"object analysis listening http://{args.host}:{args.port}/object/analysis", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
