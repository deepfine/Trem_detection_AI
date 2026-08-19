from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2

from crowd_analysis_protocol import (
    AnalysisJob,
    completed_payload,
    failed_payload,
    parse_analysis_body,
)
def env_path(name, default) -> Path:
    return Path(os.environ.get(name, default))


def env_str(name, default) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def count_people(path: Path, counter: CrowdCounter) -> tuple[int, float]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"failed to read frame: {path}")
    started = time.perf_counter()
    people = counter.count(frame)
    return people, (time.perf_counter() - started) * 1000


def post_json(url: str, payload: dict, api_key: str, timeout_s: float) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Analysis-Key"] = api_key
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=timeout_s) as response:
        response.read()


def post_result_with_retry(url: str, payload: dict, api_key: str, attempts=5) -> None:
    last_error = None
    for attempt in range(attempts):
        try:
            post_json(url, payload, api_key, timeout_s=10)
            return
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to post results after {attempts} attempts: {last_error}")


class AnalysisRuntime:
    def __init__(
        self,
        counter: CrowdCounter,
        analyzed_dir: Path,
        allowed_roots: list[Path],
        result_url: str,
        api_key: str,
        result_sink=None,
    ):
        self.counter = counter
        self.analyzed_dir = analyzed_dir.resolve()
        self.allowed_roots = [root.resolve() for root in allowed_roots]
        self.result_url = result_url
        self.api_key = api_key
        self.result_sink = result_sink
        self.ready = False
        self.jobs: Queue[AnalysisJob | None] = Queue()

    def start(self):
        self.counter.load()
        threading.Thread(target=self._worker, name="crowd-worker", daemon=True).start()
        self.ready = True

    def enqueue(self, job: AnalysisJob):
        self.jobs.put(job)

    def _worker(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self._run_job(job)
            except Exception as error:
                print(f"analysis worker error id={job.id} reason={error}", flush=True)
            finally:
                self.jobs.task_done()

    def _run_job(self, job: AnalysisJob):
        try:
            people, inference_ms = count_people(job.absolute_path, self.counter)
            payload = completed_payload(job, people, inference_ms)
            print(
                f"analysis done id={job.id} people={people} ms={inference_ms:.1f} path={job.absolute_path}",
                flush=True,
            )
        except Exception as error:
            payload = failed_payload(job, str(error))
            print(f"analysis failed id={job.id} reason={error}", flush=True)
        try:
            if self.result_sink is not None:
                self.result_sink(payload)
            elif self.result_url:
                post_result_with_retry(self.result_url, payload, self.api_key)
                print(
                    f"analysis posted id={job.id} status={payload.get('status')} url={self.result_url}",
                    flush=True,
                )
            else:
                print(f"analysis skipped callback id={job.id} reason=missing result url", flush=True)
        except Exception as error:
            print(f"analysis callback failed id={job.id} reason={error}", flush=True)


def make_handler(runtime: AnalysisRuntime, analysis_path="/crowd/analysis"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            print(f"analysis http {self.address_string()} {format % args}", flush=True)

        def _send(self, code: int, body: bytes = b"", content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                if not runtime.ready:
                    self._send(503, json.dumps({"ok": False, "reason": "model not loaded"}).encode())
                    return
                self._send(200, json.dumps({"ok": True, "model_loaded": True}).encode())
                return
            self._send(404, json.dumps({"ok": False, "reason": "not found"}).encode())

        def do_POST(self):
            if self.path.rstrip("/") != analysis_path:
                self._send(404, json.dumps({"ok": False, "reason": "not found"}).encode())
                return
            if not runtime.ready:
                self._send(503, json.dumps({"ok": False, "reason": "model not loaded"}).encode())
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                job = parse_analysis_body(payload, runtime.analyzed_dir, runtime.allowed_roots)
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                self._send(400, json.dumps({"ok": False, "reason": str(error)}).encode())
                return
            if not job.absolute_path.is_file():
                self._send(400, json.dumps({"ok": False, "reason": f"frame not found: {job.absolute_path}"}).encode())
                return
            runtime.enqueue(job)
            self._send(202, json.dumps({"ok": True, "id": job.id}).encode())

    return Handler


def main():
    from crowd_counter import CrowdCounter

    parser = argparse.ArgumentParser(description="Accept visit_servant_api crowd analysis handoff requests.")
    parser.add_argument("--host", default=env_str("ANALYSIS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(env_str("ANALYSIS_PORT", "8080")))
    parser.add_argument("--model", type=Path, default=env_path("CROWD_MODEL_PATH", "result/models/dm_count_qnrf.pth"))
    parser.add_argument("--device", default=env_str("CROWD_DEVICE", "cuda:0"))
    parser.add_argument("--analyzed-dir", type=Path, default=env_path("FRAME_ANALYZED_DIR", "/upload/visit_servant/analyzed"))
    parser.add_argument("--result-url", default=env_str("VISIT_SERVANT_RESULT_URL", "http://api:3535/crowd/results"))
    parser.add_argument("--api-key", default=env_str("ANALYSIS_API_KEY", ""))
    args = parser.parse_args()

    analyzed_dir = args.analyzed_dir.resolve()
    allowed_roots = [analyzed_dir, Path("/upload").resolve()]
    runtime = AnalysisRuntime(
        CrowdCounter(args.model, args.device),
        analyzed_dir,
        allowed_roots,
        args.result_url,
        args.api_key,
    )
    print(f"analysis loading model={args.model} device={args.device}", flush=True)
    runtime.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"analysis listening http://{args.host}:{args.port}/crowd/analysis", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
