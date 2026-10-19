"""Offline evaluation of a fixed object in a folder of images; no full-scene GT."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from frame_objects import YoloDnnDetector, bbox_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mobility-model", type=Path)
    parser.add_argument("--reference", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    files = sorted(p for p in args.images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files or not args.model.is_file() or (args.mobility_model and not args.mobility_model.is_file()):
        parser.error("image folder and local ONNX weights are required")
    x1, y1, x2, y2 = args.reference
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        parser.error("reference must be a nonempty xyxy box")
    if args.output.exists():
        parser.error("output must be a new directory")
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        parser.error(f"provider unavailable: {provider}")
    if args.device == "cuda":
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    detector = YoloDnnDetector(args.model)
    paths = [(args.model, None)]
    if args.mobility_model:
        paths.append((args.mobility_model, {"scooter", "wheelchair"}))
    for path, allowed in paths:
        session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
        if provider not in session.get_providers():
            raise RuntimeError(f"provider failed to initialize: {provider}")
        metadata = session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(metadata["names"])
        if not isinstance(names, dict) or not names or metadata.get("task", "detect") != "detect":
            raise ValueError(f"object detection model with class names required: {path}")
        shape = session.get_outputs()[0].shape
        if len(shape) != 3 or (shape[-1] != 6 and 4 + len(names) not in shape[1:]):
            raise ValueError(f"unsupported detection output: {shape}")
        detector.models.append((session, names, shape[-1] == 6, allowed))
        name, height, width, dtype = detector._input_details(session)
        session.run(None, {name: np.zeros((1, 3, height, width), dtype)})
    args.output.mkdir(parents=True)
    counts, timings = Counter(), []
    with (args.output / "detections.jsonl").open("w") as output:
        for index, path in enumerate(files):
            frame = cv2.imread(str(path))
            if frame is None or x2 > frame.shape[1] or y2 > frame.shape[0]:
                raise ValueError(f"unreadable image or reference outside image: {path}")
            objects, elapsed = detector.detect(frame)
            targets = [item for item in objects if bbox_iou(args.reference, item["bbox"]) >= 0.3]
            best = max(targets, key=lambda item: item["confidence"], default=None)
            label = best["class"] if best else "MISS"
            counts[label] += 1
            timings.append(elapsed)
            output.write(json.dumps({"file": path.name, "top_target_class": label,
                                     "objects": objects, "detector_ms": elapsed}, ensure_ascii=False) + "\n")
            if index % 50 == 0:
                print(f"Completed {index + 1}/{len(files)}", flush=True)
    summary = {"images": len(files), "model": str(args.model), "mobility_model": str(args.mobility_model) if args.mobility_model else None,
               "provider": provider, "reference_xyxy": args.reference, "association_iou": 0.3,
               "top_target_class_counts": dict(counts), "mean_detector_ms": float(np.mean(timings)),
               "p95_detector_ms": float(np.percentile(timings, 95)),
               "limitation": "Fixed target only; no full-scene ground truth. Offline detector time excludes image reading and HTTP/callbacks."}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
