from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisJob:
    id: int
    congestion_sensor_device_id: int | None
    relative_path: str
    absolute_path: Path
    camera: dict
    zone: dict | None = None
    target_classes: tuple[str, ...] = ()


def _as_record(value) -> dict | None:
    return value if isinstance(value, dict) else None


def _positive_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _non_empty_str(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_frame_path(payload: dict, analyzed_dir: Path, allowed_roots: list[Path]) -> Path:
    absolute = _non_empty_str(payload.get("frame_abs_path") or payload.get("analyzedAbsolutePath"))
    relative = _non_empty_str(payload.get("frame_path") or payload.get("analyzedRelativePath"))
    if absolute:
        path = Path(absolute).resolve()
    elif relative:
        path = (analyzed_dir / relative).resolve()
    else:
        raise ValueError("missing frame path")
    roots = [root.resolve() for root in allowed_roots]
    if not any(is_relative_to(path, root) for root in roots):
        raise ValueError("frame path is outside allowed roots")
    return path


def parse_analysis_body(payload, analyzed_dir: Path, allowed_roots: list[Path]) -> AnalysisJob:
    data = _as_record(payload)
    if data is None:
        raise ValueError("invalid json object")
    job_id = _positive_int(data.get("id"))
    if job_id is None:
        raise ValueError("id must be a positive integer")
    relative = _non_empty_str(data.get("frame_path") or data.get("analyzedRelativePath")) or ""
    absolute = resolve_frame_path(data, analyzed_dir, allowed_roots)
    camera = _as_record(data.get("camera")) or {}
    zone = _as_record(data.get("zone"))
    if zone is None and "lines" in data:
        zone = {key: data[key] for key in ("width", "height", "lines") if key in data}
    target_classes = data.get("targetClasses") or []
    if not isinstance(target_classes, list) or not all(isinstance(item, str) and item.strip() for item in target_classes):
        raise ValueError("targetClasses must be a string array")
    return AnalysisJob(
        id=job_id,
        congestion_sensor_device_id=_positive_int(data.get("congestionSensorDeviceId") or data.get("camera_id")),
        relative_path=relative or str(absolute),
        absolute_path=absolute,
        camera=camera,
        zone=zone,
        target_classes=tuple(item.strip() for item in target_classes),
    )


def completed_payload(
    job: AnalysisJob,
    counted_people: int,
    inference_ms: float,
    objects=None,
    crowd_ms: float | None = None,
    object_ms: float | None = None,
    faces=None,
    face_ms: float | None = None,
) -> dict:
    detected = list(objects or [])
    detected_faces = list(faces or [])
    methods = ["dm_count"]
    if object_ms is not None:
        methods.append("yolov8n")
    if face_ms is not None:
        methods.append("scrfd")
    raw = {
        "countedPeople": counted_people,
        "detectedObjectCount": len(detected),
        "objects": detected,
        "detectedFaceCount": len(detected_faces),
        "faces": detected_faces,
        "method": "+".join(methods),
        "inference_ms": round(inference_ms, 2),
        "frame_path": job.relative_path,
        "frame_abs_path": str(job.absolute_path),
        "congestionSensorDeviceId": job.congestion_sensor_device_id,
        "camera": job.camera,
    }
    if crowd_ms is not None:
        raw["crowd_ms"] = round(crowd_ms, 2)
    if object_ms is not None:
        raw["object_ms"] = round(object_ms, 2)
    if face_ms is not None:
        raw["face_ms"] = round(face_ms, 2)
    return {
        "id": job.id,
        "status": "COMPLETED",
        "countedPeople": counted_people,
        "detectedObjectCount": len(detected),
        "objects": detected,
        "detectedFaceCount": len(detected_faces),
        "faces": detected_faces,
        "raw": raw,
    }


def failed_payload(job: AnalysisJob, error_message: str) -> dict:
    return {
        "id": job.id,
        "status": "FAILED",
        "errorMessage": error_message[:255],
        "raw": {
            "error": error_message[:1000],
            "frame_path": job.relative_path,
            "frame_abs_path": str(job.absolute_path),
            "congestionSensorDeviceId": job.congestion_sensor_device_id,
        },
    }


def completed_object_payload(job: AnalysisJob, objects: list[dict], inference_ms: float) -> dict:
    return {
        "id": job.id,
        "status": "COMPLETED",
        "detectedObjectCount": len(objects),
        "objects": objects,
        "raw": {
            "method": "yolov8n_roi",
            "inference_ms": round(inference_ms, 2),
            "frame_path": job.relative_path,
            "frame_abs_path": str(job.absolute_path),
            "congestionSensorDeviceId": job.congestion_sensor_device_id,
            "camera": job.camera,
        },
    }
