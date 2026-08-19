import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

from crowd_analysis_protocol import completed_payload, failed_payload, parse_analysis_body
from crowd_analysis_server import AnalysisRuntime, make_handler


class FakeCounter:
    def load(self):
        return None

    def count(self, frame):
        assert frame is not None
        return 12


def write_jpeg(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((32, 32, 3), np.uint8))


def test_parse_prefers_absolute_path(tmp_path):
    frame = tmp_path / "27_1" / "a.jpg"
    write_jpeg(frame)
    job = parse_analysis_body(
        {
            "id": 41,
            "congestionSensorDeviceId": 1,
            "frame_path": "27_1/a.jpg",
            "frame_abs_path": str(frame),
            "camera": {"id": 1, "memo": "27"},
        },
        tmp_path,
        [tmp_path],
    )
    assert job.id == 41
    assert job.congestion_sensor_device_id == 1
    assert job.absolute_path == frame.resolve()
    assert job.camera["memo"] == "27"


def test_parse_joins_relative_path(tmp_path):
    frame = tmp_path / "27_1" / "a.jpg"
    write_jpeg(frame)
    job = parse_analysis_body(
        {"id": 7, "analyzedRelativePath": "27_1/a.jpg"},
        tmp_path,
        [tmp_path],
    )
    assert job.absolute_path == frame.resolve()


def test_parse_rejects_path_outside_roots(tmp_path):
    outside = Path("/etc/passwd")
    try:
        parse_analysis_body({"id": 1, "frame_abs_path": str(outside)}, tmp_path, [tmp_path])
    except ValueError as error:
        assert "outside allowed roots" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_completed_payload_matches_api_contract():
    job = parse_analysis_body(
        {"id": 41, "congestionSensorDeviceId": 1, "frame_path": "a.jpg", "analyzedAbsolutePath": "/upload/a.jpg"},
        Path("/upload"),
        [Path("/upload")],
    )
    payload = completed_payload(job, 12, 40.29)
    assert payload["id"] == 41
    assert payload["status"] == "COMPLETED"
    assert payload["countedPeople"] == 12
    assert payload["raw"]["method"] == "dm_count"


def test_failed_payload_keeps_request_id():
    job = parse_analysis_body(
        {"id": 9, "frame_abs_path": "/upload/missing.jpg"},
        Path("/upload"),
        [Path("/upload")],
    )
    payload = failed_payload(job, "failed to read frame")
    assert payload["id"] == 9
    assert payload["status"] == "FAILED"
    assert "failed to read frame" in payload["errorMessage"]


def test_http_accepts_analysis_and_posts_result(tmp_path):
    frame = tmp_path / "27_1" / "a.jpg"
    write_jpeg(frame)
    posted = []
    runtime = AnalysisRuntime(FakeCounter(), tmp_path, [tmp_path], "", "", result_sink=posted.append)
    runtime.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        conn = HTTPConnection(host, port, timeout=2)
        conn.request("GET", "/health")
        health = conn.getresponse()
        assert health.status == 200
        health.read()
        body = json.dumps({
            "id": 41,
            "congestionSensorDeviceId": 1,
            "frame_path": "27_1/a.jpg",
            "frame_abs_path": str(frame),
        }).encode()
        conn.request("POST", "/crowd/analysis", body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 202
        accepted = json.loads(response.read())
        assert accepted["ok"] is True
        assert accepted["id"] == 41
        runtime.jobs.join()
        assert posted[0]["id"] == 41
        assert posted[0]["countedPeople"] == 12
    finally:
        server.shutdown()
        server.server_close()
