from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    tram_zone: int | None = None
    tram_direction: int = 0
    tram_zone_uncertainty: int = 0
    tram_observed_at: datetime | None = None
    tram_position_max_age_seconds: float | None = None
    forward_zone_gap_threshold: int | None = None
    rear_zone_gap_threshold: int | None = None
    fail_safe_on_missing_tram_position: bool = False

    def tram_position_stale(self) -> bool:
        if self.tram_observed_at is None or self.tram_position_max_age_seconds is None:
            return False
        age = datetime.now(timezone.utc) - self.tram_observed_at.astimezone(timezone.utc)
        return age.total_seconds() > self.tram_position_max_age_seconds

    def tram_policy_options(self) -> dict:
        reason = (
            "TRAM_POSITION_STALE" if self.tram_position_stale()
            else "TRAM_POSITION_MISSING" if self.tram_zone is None and self.fail_safe_on_missing_tram_position
            else ""
        )
        return {
            "direction": self.tram_direction,
            "uncertainty": self.tram_zone_uncertainty,
            "forward_threshold": self.forward_zone_gap_threshold,
            "rear_threshold": self.rear_zone_gap_threshold,
            "fail_safe_reason": reason,
        }


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


def _optional_number(data: dict, key: str, *, integer=True, minimum=0, maximum=None):
    if key not in data:
        return None
    value = data[key]
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int,) if integer else (int, float))
        or value < minimum
        or maximum is not None and value > maximum
    ):
        kind = "integer" if integer else "number"
        limit = f" and less than or equal to {maximum}" if maximum is not None else ""
        raise ValueError(f"{key} must be a {kind} greater than or equal to {minimum}{limit}")
    return int(value) if integer else float(value)


def _optional_datetime(data: dict, key: str) -> datetime | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{key} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{key} must include a timezone")
    return parsed


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
    direction = data.get("tramDirection", 0)
    if isinstance(direction, bool) or direction not in {-1, 0, 1}:
        raise ValueError("tramDirection must be -1, 0, or 1")
    fail_safe = data.get("failSafeOnMissingTramPosition", False)
    if not isinstance(fail_safe, bool):
        raise ValueError("failSafeOnMissingTramPosition must be a boolean")
    return AnalysisJob(
        id=job_id,
        congestion_sensor_device_id=_positive_int(data.get("congestionSensorDeviceId") or data.get("camera_id")),
        relative_path=relative or str(absolute),
        absolute_path=absolute,
        camera=camera,
        zone=zone,
        target_classes=tuple(item.strip() for item in target_classes),
        tram_zone=_optional_number(data, "tramZone") if "tramZone" in data else _optional_number(data, "tram_zone"),
        tram_direction=direction,
        tram_zone_uncertainty=_optional_number(data, "tramZoneUncertainty", maximum=100) or 0,
        tram_observed_at=_optional_datetime(data, "tramObservedAt"),
        tram_position_max_age_seconds=_optional_number(data, "tramPositionMaxAgeSeconds", integer=False),
        forward_zone_gap_threshold=_optional_number(data, "forwardZoneGapThreshold"),
        rear_zone_gap_threshold=_optional_number(data, "rearZoneGapThreshold"),
        fail_safe_on_missing_tram_position=fail_safe,
    )


def alert_summary(objects):
    alerted = [item for item in objects if item.get("alert")]
    level = "danger" if any(item.get("zoneLevel") == "danger" for item in alerted) else "warning" if alerted else None
    return level, len(alerted)


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
    alert_level, alert_count = alert_summary(detected)
    methods = ["dm_count"]
    if object_ms is not None:
        methods.append("objects365_yolo26n+mobility_yolov8s")
    if face_ms is not None:
        methods.append("scrfd")
    raw = {
        "countedPeople": counted_people,
        "detectedObjectCount": len(detected),
        "objects": detected,
        "detectedFaceCount": len(detected_faces),
        "faces": detected_faces,
        "alertLevel": alert_level,
        "alertObjectCount": alert_count,
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
        "alertLevel": alert_level,
        "alertObjectCount": alert_count,
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
    alert_level, alert_count = alert_summary(objects)
    return {
        "id": job.id,
        "status": "COMPLETED",
        "detectedObjectCount": len(objects),
        "objects": objects,
        "alertLevel": alert_level,
        "alertObjectCount": alert_count,
        "raw": {
            "method": "objects365_yolo26n+mobility_yolov8s_roi",
            "inference_ms": round(inference_ms, 2),
            "frame_path": job.relative_path,
            "frame_abs_path": str(job.absolute_path),
            "congestionSensorDeviceId": job.congestion_sensor_device_id,
            "camera": job.camera,
        },
    }
