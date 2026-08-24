import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

from camera_zones import save_zone_file
from crowd_analysis_protocol import completed_payload, failed_payload, parse_analysis_body
from crowd_analysis_server import AnalysisRuntime, make_handler


class FakeCounter:
    def load(self):
        return None

    def count(self, frame):
        assert frame is not None
        return 12


class FakeObjectDetector:
    def load(self):
        return None

    def detect(self, frame, zone=None, target_classes=()):
        assert frame is not None
        return [{"class": "person", "confidence": 0.91, "bbox": [1, 2, 3, 4]}], 8.5


class FakeFaceDetector:
    def load(self):
        return None

    def detect(self, frame):
        assert frame is not None
        return [{"confidence": 0.94, "bbox": [5, 6, 15, 18]}]


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
    assert payload["detectedObjectCount"] == 0
    assert payload["objects"] == []
    assert payload["detectedFaceCount"] == 0
    assert payload["faces"] == []
    assert payload["raw"]["method"] == "dm_count"


def test_completed_payload_includes_objects():
    job = parse_analysis_body(
        {"id": 41, "congestionSensorDeviceId": 1, "frame_path": "a.jpg", "analyzedAbsolutePath": "/upload/a.jpg"},
        Path("/upload"),
        [Path("/upload")],
    )
    objects = [{"class": "person", "confidence": 0.9123, "bbox": [120, 80, 260, 430]}]
    payload = completed_payload(job, 12, 52.4, objects=objects, crowd_ms=40.29, object_ms=12.11)
    assert payload["detectedObjectCount"] == 1
    assert payload["objects"] == objects
    assert payload["raw"]["objects"] == objects
    assert payload["raw"]["method"] == "dm_count+yolov8n"
    assert payload["raw"]["crowd_ms"] == 40.29
    assert payload["raw"]["object_ms"] == 12.11


def test_completed_payload_includes_face_coordinates():
    job = parse_analysis_body(
        {"id": 41, "congestionSensorDeviceId": 1, "frame_path": "a.jpg", "analyzedAbsolutePath": "/upload/a.jpg"},
        Path("/upload"),
        [Path("/upload")],
    )
    faces = [{"confidence": 0.94, "bbox": [120, 80, 260, 230]}]
    payload = completed_payload(job, 12, 46.2, faces=faces, crowd_ms=40.29, face_ms=5.91)
    assert payload["detectedFaceCount"] == 1
    assert payload["faces"] == faces
    assert payload["raw"]["faces"] == faces
    assert payload["raw"]["method"] == "dm_count+scrfd"
    assert payload["raw"]["face_ms"] == 5.91


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
        assert posted[0]["detectedObjectCount"] == 0
        assert posted[0]["objects"] == []
    finally:
        server.shutdown()
        server.server_close()


def test_http_includes_object_detections(tmp_path):
    frame = tmp_path / "27_1" / "a.jpg"
    write_jpeg(frame)
    posted = []
    zones = tmp_path / "zones"
    save_zone_file(
        zones / "1.json",
        {"width": 32, "height": 32, "polygons": [[[0, 0], [31, 0], [31, 31], [0, 31]]]},
    )
    runtime = AnalysisRuntime(
        FakeCounter(),
        tmp_path,
        [tmp_path],
        "",
        "",
        result_sink=posted.append,
        object_detector=FakeObjectDetector(),
        zone_dir=zones,
    )
    runtime.start()
    runtime.enqueue(
        parse_analysis_body(
            {
                "id": 41,
                "congestionSensorDeviceId": 1,
                "frame_path": "27_1/a.jpg",
                "frame_abs_path": str(frame),
            },
            tmp_path,
            [tmp_path],
        )
    )
    runtime.jobs.join()
    assert posted[0]["detectedObjectCount"] == 1
    assert posted[0]["objects"][0]["class"] == "person"
    assert posted[0]["raw"]["object_ms"] == 8.5


def test_http_includes_face_coordinates(tmp_path):
    frame = tmp_path / "27_1" / "a.jpg"
    write_jpeg(frame)
    posted = []
    runtime = AnalysisRuntime(
        FakeCounter(),
        tmp_path,
        [tmp_path],
        "",
        "",
        result_sink=posted.append,
        face_detector=FakeFaceDetector(),
    )
    runtime.start()
    runtime.enqueue(
        parse_analysis_body(
            {"id": 41, "frame_abs_path": str(frame)},
            tmp_path,
            [tmp_path],
        )
    )
    runtime.jobs.join()
    assert posted[0]["detectedFaceCount"] == 1
    assert posted[0]["faces"] == [{"confidence": 0.94, "bbox": [5, 6, 15, 18]}]
    assert posted[0]["raw"]["face_ms"] >= 0


def test_http_skips_objects_when_zone_file_missing(tmp_path):
    frame = tmp_path / "27_1" / "a.jpg"
    write_jpeg(frame)
    posted = []
    runtime = AnalysisRuntime(
        FakeCounter(),
        tmp_path,
        [tmp_path],
        "",
        "",
        result_sink=posted.append,
        object_detector=FakeObjectDetector(),
        zone_dir=tmp_path / "zones",
    )
    runtime.start()
    runtime.enqueue(
        parse_analysis_body(
            {
                "id": 41,
                "congestionSensorDeviceId": 1,
                "frame_path": "27_1/a.jpg",
                "frame_abs_path": str(frame),
            },
            tmp_path,
            [tmp_path],
        )
    )
    runtime.jobs.join()
    assert posted[0]["detectedObjectCount"] == 0
    assert posted[0]["objects"] == []
