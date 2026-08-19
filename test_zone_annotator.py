import json

import pytest

from zone_annotator import load_cameras, validate_zones


def test_load_five_cameras(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"cameras": [
        {"id": f"camera_{i}", "name": f"{i}번 카메라", "url": f"rtsp://secret/{i}"}
        for i in range(1, 6)
    ]}))
    assert len(load_cameras(path)) == 5


def test_rejects_duplicate_camera_ids(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text('{"cameras":[{"id":"a","name":"A","url":"1"},{"id":"a","name":"B","url":"2"}]}')
    with pytest.raises(ValueError, match="duplicate"):
        load_cameras(path)


def test_validates_zone_payload():
    payload = {"zones": [{"name": "track", "level": "danger", "points": [[0, 0], [10, 0], [10, 10]]}]}
    assert validate_zones(payload) == payload
    with pytest.raises(ValueError, match="level"):
        validate_zones({"zones": [{"name": "x", "level": "unknown", "points": [[0, 0], [1, 0], [1, 1]]}]})
