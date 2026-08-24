from __future__ import annotations

import json
from pathlib import Path


def zone_file(zones_dir: Path, device_id: int) -> Path:
    return zones_dir / f"{int(device_id)}.json"


def parse_camera_folder(name: str) -> tuple[int, str] | None:
    if "_" not in name:
        return None
    memo, _, suffix = name.rpartition("_")
    if not suffix.isdigit():
        return None
    device_id = int(suffix)
    return device_id, memo or str(device_id)


def list_cameras(analyzed_dir: Path, zones_dir: Path | None = None) -> list[dict]:
    cameras: dict[int, str] = {}
    if analyzed_dir.is_dir():
        for entry in analyzed_dir.iterdir():
            if not entry.is_dir():
                continue
            parsed = parse_camera_folder(entry.name)
            if parsed is None:
                continue
            device_id, memo = parsed
            cameras[device_id] = memo
    if zones_dir is not None and zones_dir.is_dir():
        for entry in zones_dir.glob("*.json"):
            if not entry.stem.isdigit():
                continue
            cameras.setdefault(int(entry.stem), entry.stem)
    return [{"id": device_id, "name": cameras[device_id]} for device_id in sorted(cameras)]


def latest_jpeg(analyzed_dir: Path, device_id: int) -> Path | None:
    if not analyzed_dir.is_dir():
        return None
    matches: list[Path] = []
    for entry in analyzed_dir.iterdir():
        if not entry.is_dir():
            continue
        parsed = parse_camera_folder(entry.name)
        if parsed is None or parsed[0] != int(device_id):
            continue
        matches.extend(path for path in entry.rglob("*.jpg") if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def _as_point(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None
    return [x, y]


def _as_polygon(value) -> list[list[int]] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    points = [_as_point(item) for item in value]
    if any(point is None for point in points):
        return None
    return points


def polygons_from_payload(payload: dict | None) -> list[list[list[int]]]:
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("polygons"), list):
        polygons = [_as_polygon(item) for item in payload["polygons"]]
        return [item for item in polygons if item is not None]
    single = _as_polygon(payload.get("points"))
    if single is not None:
        return [single]
    zones = payload.get("zones")
    if isinstance(zones, list):
        polygons = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            polygon = _as_polygon(zone.get("points") or zone.get("mask"))
            if polygon is not None:
                polygons.append(polygon)
        return polygons
    return []


def payload_size(payload: dict | None) -> tuple[int, int] | None:
    if not isinstance(payload, dict):
        return None
    width, height = payload.get("width"), payload.get("height")
    if isinstance(width, (int, float)) and width > 0 and isinstance(height, (int, float)) and height > 0:
        return int(width), int(height)
    return None


def validate_zone_payload(payload, image_width: int | None = None, image_height: int | None = None) -> dict:
    data = payload if isinstance(payload, dict) else None
    if data is None:
        raise ValueError("zone JSON must be an object")
    polygons = polygons_from_payload(data)
    if "polygons" in data and not isinstance(data.get("polygons"), list):
        raise ValueError("polygons must be a list")
    if isinstance(data.get("polygons"), list) and any(_as_polygon(item) is None for item in data["polygons"]):
        raise ValueError("each polygon needs at least 3 integer [x, y] points")
    size = payload_size(data)
    if size is None:
        if image_width and image_height:
            size = (int(image_width), int(image_height))
        else:
            raise ValueError("zone width and height must be positive numbers")
    return {"width": size[0], "height": size[1], "polygons": polygons}


def load_zone_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return validate_zone_payload(payload)
    except ValueError:
        polygons = polygons_from_payload(payload if isinstance(payload, dict) else None)
        size = payload_size(payload if isinstance(payload, dict) else None)
        if not polygons:
            return None
        return {"width": size[0] if size else 0, "height": size[1] if size else 0, "polygons": polygons}


def save_zone_file(path: Path, payload: dict) -> dict:
    validated = validate_zone_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, ensure_ascii=False, indent=2) + "\n")
    return validated
