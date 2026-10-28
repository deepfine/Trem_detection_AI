"""Join preserved raw frames to read-only production results for event review.

Creates separate candidate/issued-alert CSVs and empty GT/coverage templates.
No model output is promoted to ground truth or reviewed non-risk time.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from camera_zones import parse_camera_folder
from evaluate_kpi import parse_time


def stored_time(value):
    # This DB stores KST wall clocks in TIMESTAMP WITHOUT TIME ZONE; Prisma appends Z.
    return parse_time(value[:-1] if value.endswith("Z") else value)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(raw_root: Path, results_path: Path, output: Path):
    raw_root = raw_root.resolve()
    manifest = [json.loads(line) for line in (raw_root / "manifest.jsonl").read_text().splitlines()]
    production = json.loads(results_path.read_text())
    if not manifest or not isinstance(production, list):
        raise ValueError("nonempty raw manifest and production result array required")
    by_file = {}
    for row in production:
        key = (int(row["congestionSensorDeviceId"]), row["analyzedFileName"])
        if key in by_file:
            raise ValueError(f"ambiguous analysis rows for frame: {key}")
        by_file[key] = row

    frames, zone, falls, issued = [], [], [], []
    seen, per_camera = set(), {}
    for record in manifest:
        relative = Path(record["file"])
        path = (raw_root / relative).resolve()
        if raw_root not in path.parents or record.get("stage") != "before_analysis_overlay":
            raise ValueError("only preserved pre-overlay frames inside raw_root are accepted")
        camera = parse_camera_folder(relative.parts[0])
        if camera is None or camera[0] != record["camera_id"]:
            raise ValueError("camera ID does not match frame folder")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"raw input integrity check failed: {relative}")
        stamp = re.search(r"_(\d{8}-\d{6}-\d{3})_", path.name)
        if stamp is None:
            raise ValueError(f"frame has no capture timestamp: {relative}")
        captured = datetime.strptime(stamp[1], "%Y%m%d-%H%M%S-%f")
        key = (camera[0], path.name)
        if key in seen:
            raise ValueError(f"duplicate captured frame: {relative}")
        seen.add(key)
        row = by_file.get(key)
        result = ((row or {}).get("congestionData") or {}).get("result") or {}
        if row and stored_time(row["capturedAt"]) != captured:
            raise ValueError(f"capture time does not match analysis row: {relative}")
        status = row["status"] if row else "NO_MATCHING_ANALYSIS"
        errors = result.get("component_errors") or {}
        object_failed = "objects" in errors
        error = errors.get("objects") or ("object component failed" if object_failed else "")
        object_status = status
        if status == "COMPLETED":
            object_status = "FAILED" if object_failed else "COMPLETED" if result.get("object_ms") is not None else "UNKNOWN"
        elif row:
            error = row.get("errorMessage") or ""
        frames.append({"camera_id": camera[0], "captured_at": captured, "file": str(relative),
                       "sha256": record["sha256"], "log_id": row["id"] if row else "",
                       "analysis_status": status, "object_status": object_status,
                       "object_error": error, "object_count": len(result.get("objects", []))})
        cam = per_camera.setdefault(str(camera[0]), {"frames": 0, "linked": 0, "object_completed": 0})
        cam["frames"] += 1
        cam["linked"] += bool(row)
        cam["object_completed"] += object_status == "COMPLETED"
        completed = stored_time(row["completedAt"]) if row and row.get("completedAt") else ""
        if object_status == "COMPLETED":
            for obj in result.get("objects", []):
                onset = {"camera_id": camera[0], "time": captured, "track_id": obj.get("trackId") or "UNKNOWN",
                         "class": obj["class"], "completed_at": completed}
                if obj.get("alertEvent") in {"ENTER", "ESCALATE"}:
                    zone.append(onset)
                if obj.get("fallEvent") == "ENTER":
                    falls.append(onset)
        for alarm in row.get("riskAlarms", []) if row else []:
            issued.append({"camera_id": camera[0], "time": captured, "track_id": alarm.get("trackId") or "UNKNOWN",
                           "class": alarm["class"], "completed_at": completed,
                           "alert_id": alarm["id"], "issued_at": stored_time(alarm["riskDate"])})

    summary = {"raw_frames": len(frames), "linked_frames": sum(bool(f["log_id"]) for f in frames),
               "frames_without_matching_analysis": sum(not f["log_id"] for f in frames),
               "object_completed_frames": sum(f["object_status"] == "COMPLETED" for f in frames),
               "object_failed_frames": sum(f["object_status"] == "FAILED" for f in frames),
               "object_status_counts": dict(sorted(Counter(f["object_status"] for f in frames).items())),
               "zone_candidates": len(zone), "fall_candidates": len(falls), "issued_alerts": len(issued),
               "per_camera": per_camera, "ground_truth_status": "unreviewed", "metrics_ready": False,
               "precision": None, "recall": None, "f1": None, "detection_accuracy": None, "fpr": None,
               "time_basis": "stored KST wall clock; capture timestamps come from frame filenames",
               "source_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
               "note": "No matching analysis is not an object miss; candidates are not issued system alerts."}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "frames.csv", list(frames[0]), frames)
    columns = ["camera_id", "time", "track_id", "class", "completed_at"]
    write_csv(output / "zone_candidates.csv", columns, zone)
    write_csv(output / "fall_candidates.csv", columns, falls)
    write_csv(output / "issued_alerts.csv", columns + ["alert_id", "issued_at"], issued)
    write_csv(output / "gt.template.csv", ["camera_id", "start", "end", "scenario", "class", "track_id"], [])
    write_csv(output / "coverage.template.csv", ["camera_id", "start", "end"], [])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--results", type=Path, required=True,
                        help="read-only DB snapshot: camera/id/filename/times/status/result/riskAlarms")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = build_bundle(args.raw_root, args.results, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
