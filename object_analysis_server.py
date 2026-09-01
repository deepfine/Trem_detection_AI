from __future__ import annotations

import argparse
import time
import threading
from pathlib import Path
from queue import Queue

import cv2

from crowd_analysis_protocol import AnalysisJob, completed_object_payload, failed_payload
from crowd_analysis_server import env_path, env_str, make_handler, post_result_with_retry
from frame_objects import YoloDnnDetector, apply_tram_policy


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
                objects, _ = self.detector.detect(frame, job.zone, job.target_classes)
                threshold = job.zone.get("zoneGapThreshold", 2) if job.zone else 2
                if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
                    threshold = 2
                objects = apply_tram_policy(
                    objects,
                    job.tram_zone,
                    threshold,
                    **job.tram_policy_options(),
                )
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
    parser.add_argument("--model", type=Path, default=env_path("OBJECT_MODEL_PATH", "/app/result/models/objects365_yolo26n.onnx"))
    parser.add_argument("--mobility-model", type=Path, default=env_path("MOBILITY_MODEL_PATH", "/app/result/models/mobility_yolov8s.onnx"))
    parser.add_argument("--device-id", type=int, default=int(env_str("OBJECT_DEVICE_ID", "0")))
    parser.add_argument("--analyzed-dir", type=Path, default=env_path("FRAME_ANALYZED_DIR", "/upload/visit_servant/analyzed"))
    parser.add_argument("--result-url", default=env_str("OBJECT_RESULT_URL", "http://api:3535/object/results"))
    parser.add_argument("--api-key", default=env_str("ANALYSIS_API_KEY", ""))
    args = parser.parse_args()

    analyzed_dir = args.analyzed_dir.resolve()
    runtime = ObjectAnalysisRuntime(
        YoloDnnDetector(args.model, args.device_id, args.mobility_model),
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
