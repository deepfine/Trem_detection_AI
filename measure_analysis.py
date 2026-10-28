"""Replay exported real frames at recorded arrival times through the full worker.

Measures local arrival-to-result latency, not camera/network/display E2E or accuracy.
Production callbacks are disabled. Run in an isolated GPU container.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from evaluate_kpi import parse_time, read_xlsx, summarize


def exported_frames(export: Path):
    records = []
    for row in read_xlsx(export)["분석로그"]:
        if not row.get("사진"):
            continue
        path = (export.parent / row["사진"]).resolve()
        if export.parent.resolve() not in path.parents:
            raise ValueError(f"frame outside export: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append((parse_time(row["촬영시각"]), int(row["로그ID"]),
                        int(row["카메라ID"]), path))
    if not records:
        raise ValueError("export has no available frames")
    return sorted(records)


def measure(args):
    import cv2
    import onnxruntime as ort
    import torch
    from crowd_analysis_protocol import AnalysisJob, api_callback_payload
    from crowd_analysis_server import AnalysisRuntime
    from crowd_counter import CrowdCounter
    from frame_faces import ScrfdFaceDetector
    from frame_objects import YoloDnnDetector, parse_class_confidence

    records = exported_frames(args.export)
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    detector = YoloDnnDetector(args.model, mobility_model=args.mobility_model,
                              person_relabel=False, confidence=0.55,
                              class_confidence=parse_class_confidence(
                                  "car=0.45,bus=0.45,truck=0.45,animal=0.45,person=0.65,bicycle=0.5"))
    samples, results, pending = [], [], {}
    completed_ids = set()

    def receive(payload):
        body = api_callback_payload(payload)
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False)
        ended = time.perf_counter()
        scheduled, submitted, started = pending[payload["id"]]
        samples.append({
            "id": payload["id"], "status": payload["status"],
            "sender_delay_ms": (submitted - scheduled) * 1000,
            "queue_ms": (started - submitted) * 1000,
            "worker_ms": (ended - started) * 1000,
            "arrival_to_result_ms": (ended - scheduled) * 1000,
            "reported_inference_ms": body.get("raw", {}).get("inference_ms"),
        })
        results.append(encoded)
        completed_ids.add(payload["id"])

    runtime = AnalysisRuntime(
        CrowdCounter(args.crowd_model, "cuda:0"), args.export.parent,
        [args.export.parent], "", "", result_sink=receive,
        object_detector=detector, face_detector=ScrfdFaceDetector(args.face_model),
        zone_dir=args.zone_dir,
    )
    original_run = runtime._run_job

    def run_job(job):
        pending[job.id].append(time.perf_counter())
        original_run(job)

    runtime._run_job = run_job
    runtime.start()
    # Exclude one real-frame warmup from steady-state measurements.
    first_time, first_id, first_camera, first_path = records[0]
    pending[-1] = [time.perf_counter(), time.perf_counter()]
    run_job(AnalysisJob(-1, first_camera, str(first_path), first_path, {}))
    if samples[-1]["status"] != "COMPLETED":
        raise RuntimeError("real-frame warmup failed")
    samples.clear()
    results.clear()
    completed_ids.clear()
    pending.clear()
    runtime.object_states.clear()
    start = time.perf_counter()
    maximum_queue = 0
    for captured, log_id, camera, path in records:
        scheduled = start + (captured - first_time).total_seconds()
        delay = scheduled - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        pending[log_id] = [scheduled, time.perf_counter()]
        runtime.enqueue(AnalysisJob(log_id, camera, str(path), path, {}))
        maximum_queue = max(maximum_queue, runtime.jobs.qsize())
    input_ended = time.perf_counter()
    outstanding_at_input_end = len(records) - len(completed_ids)
    deadline = input_ended + 120
    while len(completed_ids) < len(records) and time.perf_counter() < deadline:
        time.sleep(0.05)
    end = time.perf_counter()
    runtime.jobs.put(None)
    with (args.output / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)
    (args.output / "results.jsonl").write_text("\n".join(results) + "\n", encoding="utf-8")
    models = [args.model, args.mobility_model, args.crowd_model, args.face_model]
    report = {
        "mode": "paced replay of real exported frames through full worker; no production callbacks",
        "scope": "file available on local disk -> result serialization; excludes capture/network/display",
        "export": str(args.export), "export_sha256": hashlib.sha256(args.export.read_bytes()).hexdigest(),
        "models_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in models},
        "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in ("measure_analysis.py", "crowd_analysis_server.py", "frame_objects.py")},
        "onnxruntime": ort.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "camera_ids": sorted({r[2] for r in records}),
        "input_frames": len(records), "completed_frames": len(samples),
        "failed_frames": sum(r["status"] != "COMPLETED" for r in samples),
        "missing_results": len(records) - len(completed_ids),
        "maximum_queue": maximum_queue,
        "outstanding_at_input_end": outstanding_at_input_end,
        "input_span_s": (records[-1][0] - first_time).total_seconds(),
        "wall_s": end - start, "drain_s": end - input_ended,
        "throughput_fps": len(samples) / (end - start),
        "timings_ms": {key: summarize(r[key] for r in samples)
                       for key in ("sender_delay_ms", "queue_ms", "worker_ms", "arrival_to_result_ms")},
        "accuracy": None, "bev_latency": None, "display_e2e": None,
        "tram_position": "not replayed; zone candidates only",
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["failed_frames"] or report["missing_results"]:
        raise RuntimeError("replay has failures; inspect saved results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("result/models/ft3_yolo26l_736x1280_20261020.onnx"))
    parser.add_argument("--mobility-model", type=Path, default=Path("result/models/mobility_yolov8s.onnx"))
    parser.add_argument("--crowd-model", type=Path, default=Path("result/models/dm_count_qnrf.pth"))
    parser.add_argument("--face-model", type=Path, default=Path("result/models/scrfd_det_10g.onnx"))
    parser.add_argument("--zone-dir", type=Path, default=Path("/upload/visit_servant/zones"))
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
