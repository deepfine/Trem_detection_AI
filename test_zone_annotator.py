import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

from camera_zones import validate_zone_payload, zone_file
from zone_annotator import make_handler


def test_validates_polygon_payload():
    payload = {"width": 100, "height": 80, "polygons": [[[0, 0], [10, 0], [10, 10]]]}
    validated = validate_zone_payload(payload)
    assert validated["zones"][0]["level"] == "danger"
    assert validated["zones"][0]["points"] == payload["polygons"][0]
    try:
        validate_zone_payload({"width": 1, "height": 1, "polygons": [[[0, 0], [1, 0]]]})
    except ValueError as error:
        assert "3" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_editor_lists_cameras_and_saves_device_json(tmp_path):
    analyzed = tmp_path / "analyzed"
    frame_dir = analyzed / "33_4"
    frame_dir.mkdir(parents=True)
    (frame_dir / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    zones = tmp_path / "zones"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(analyzed, zones))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        conn = HTTPConnection(host, port, timeout=2)
        conn.request("GET", "/cameras")
        cameras = json.loads(conn.getresponse().read())
        assert cameras == [{"id": 4, "name": "33"}]
        body = json.dumps({
            "width": 32,
            "height": 24,
            "zoneGapThreshold": 3,
            "zones": [{"name": "track-z8", "level": "danger", "index": 8, "points": [[1, 1], [8, 1], [8, 8]]}],
        }).encode()
        conn.request("POST", "/zones/4", body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 200
        response.read()
        saved = json.loads(zone_file(zones, 4).read_text())
        assert saved["zones"][0]["index"] == 8
        assert saved["zones"][0]["points"][2] == [8, 8]
        assert saved["zoneGapThreshold"] == 3
    finally:
        server.shutdown()
        server.server_close()
