from __future__ import annotations

import json
from pathlib import Path


ZONE_LEVELS = {"danger", "warning", "safe"}


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


def zones_from_payload(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    zones = payload.get("zones")
    if isinstance(zones, list):
        parsed = []
        for number, zone in enumerate(zones, 1):
            if not isinstance(zone, dict):
                continue
            points = _as_polygon(zone.get("points") or zone.get("mask"))
            level = zone.get("level")
            if points is None or level not in ZONE_LEVELS:
                continue
            item = {"name": str(zone.get("name") or f"{level}-{number}"), "level": level, "points": points}
            index = zone.get("index", zone.get("zoneIndex"))
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                item["index"] = index
            parsed.append(item)
        return parsed
    polygons = payload.get("polygons")
    if isinstance(polygons, list):
        return [
            {"name": f"danger-{number}", "level": "danger", "points": points}
            for number, value in enumerate(polygons, 1)
            if (points := _as_polygon(value)) is not None
        ]
    single = _as_polygon(payload.get("points"))
    return [{"name": "danger-1", "level": "danger", "points": single}] if single is not None else []


def polygons_from_payload(payload: dict | None) -> list[list[list[int]]]:
    return [zone["points"] for zone in zones_from_payload(payload)]


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
    zones = zones_from_payload(data)
    if "polygons" in data and not isinstance(data.get("polygons"), list):
        raise ValueError("polygons must be a list")
    if isinstance(data.get("polygons"), list) and any(_as_polygon(item) is None for item in data["polygons"]):
        raise ValueError("each polygon needs at least 3 integer [x, y] points")
    if "zones" in data and not isinstance(data.get("zones"), list):
        raise ValueError("zones must be a list")
    if isinstance(data.get("zones"), list) and len(zones) != len(data["zones"]):
        raise ValueError("each zone needs level danger|warning|safe and at least 3 integer [x, y] points")
    if isinstance(data.get("zones"), list):
        for zone in data["zones"]:
            index = zone.get("index", zone.get("zoneIndex")) if isinstance(zone, dict) else None
            if index is not None and (not isinstance(index, int) or isinstance(index, bool) or index < 0):
                raise ValueError("zone index must be a non-negative integer")
    size = payload_size(data)
    if size is None:
        if image_width and image_height:
            size = (int(image_width), int(image_height))
        else:
            raise ValueError("zone width and height must be positive numbers")
    threshold = data.get("zoneGapThreshold", 2)
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("zoneGapThreshold must be a non-negative integer")
    return {"width": size[0], "height": size[1], "zoneGapThreshold": threshold, "zones": zones}


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
        zones = zones_from_payload(payload if isinstance(payload, dict) else None)
        size = payload_size(payload if isinstance(payload, dict) else None)
        if not zones:
            return None
        return {"width": size[0] if size else 0, "height": size[1] if size else 0, "zoneGapThreshold": 2, "zones": zones}


def save_zone_file(path: Path, payload: dict) -> dict:
    validated = validate_zone_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, ensure_ascii=False, indent=2) + "\n")
    return validated
