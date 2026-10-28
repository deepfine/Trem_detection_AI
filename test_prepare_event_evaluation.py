import csv
import hashlib
import json

import pytest

from prepare_event_evaluation import build_bundle


def inputs(tmp_path):
    root = tmp_path / "raw"
    name = "33_3/2026/10/27/33_3_20261027-134931-248_000001.jpg"
    path = root / name
    path.parent.mkdir(parents=True)
    data = b"\xff\xd8raw-frame\xff\xd9"
    path.write_bytes(data)
    record = {"camera_id": 3, "file": name, "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest(), "stage": "before_analysis_overlay"}
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    row = {"id": 1, "congestionSensorDeviceId": 3, "analyzedFileName": path.name,
           "capturedAt": "2026-10-27T13:49:31.248Z", "completedAt": "2026-10-27T13:49:31.800Z",
           "status": "COMPLETED", "congestionData": {"result": {"object_ms": 12.5, "objects": [
               {"class": "person", "trackId": 7, "alertEvent": "STAY", "fallEvent": "ENTER"},
           ]}}, "riskAlarms": []}
    results = tmp_path / "production.json"
    return root, record, row, results


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_fall_and_issued_alerts_are_separate_from_zone_candidates(tmp_path):
    root, _, row, results = inputs(tmp_path)
    row["riskAlarms"] = [{"id": "alarm-1", "class": "person", "trackId": 7,
                           "alertEvent": "STAY", "riskDate": "2026-10-27T13:49:32.000Z"}]
    results.write_text(json.dumps([row]))
    out = tmp_path / "bundle"
    report = build_bundle(root, results, out)
    assert (report["linked_frames"], report["zone_candidates"], report["fall_candidates"], report["issued_alerts"]) == (1, 0, 1, 1)
    assert csv_rows(out / "issued_alerts.csv")[0]["time"] == "2026-10-27 13:49:31.248000"
    assert csv_rows(out / "issued_alerts.csv")[0]["issued_at"] == "2026-10-27 13:49:32"
    assert not report["metrics_ready"]
    assert report["precision"] is report["recall"] is report["f1"] is report["fpr"] is None
    assert csv_rows(out / "gt.template.csv") == csv_rows(out / "coverage.template.csv") == []


def test_unprocessed_frame_is_not_an_object_miss(tmp_path):
    root, _, _, results = inputs(tmp_path)
    results.write_text("[]")
    out = tmp_path / "bundle"
    report = build_bundle(root, results, out)
    assert report["frames_without_matching_analysis"] == 1
    assert csv_rows(out / "frames.csv")[0]["object_status"] == "NO_MATCHING_ANALYSIS"
    assert report["recall"] is None


@pytest.mark.parametrize("message", ["tracker failed", ""])
def test_partial_object_failure_does_not_become_a_valid_empty_detection(tmp_path, message):
    root, _, row, results = inputs(tmp_path)
    result = row["congestionData"]["result"]
    result["component_errors"] = {"objects": message}
    results.write_text(json.dumps([row]))
    out = tmp_path / "bundle"
    report = build_bundle(root, results, out)
    assert report["object_failed_frames"] == 1
    assert report["object_completed_frames"] == report["fall_candidates"] == 0
    assert csv_rows(out / "frames.csv")[0]["object_error"] == (message or "object component failed")


def test_legacy_result_without_object_timing_is_unknown(tmp_path):
    root, _, row, results = inputs(tmp_path)
    row["congestionData"]["result"].pop("object_ms")
    results.write_text(json.dumps([row]))
    out = tmp_path / "bundle"
    build_bundle(root, results, out)
    assert csv_rows(out / "frames.csv")[0]["object_status"] == "UNKNOWN"
    assert csv_rows(out / "fall_candidates.csv") == []


@pytest.mark.parametrize("change", ["hash", "camera", "stage", "duplicate", "timestamp"])
def test_rejects_unreliable_input_joins(tmp_path, change):
    root, record, row, results = inputs(tmp_path)
    if change == "hash":
        record["sha256"] = "bad"
    elif change == "camera":
        record["camera_id"] = 5
    elif change == "stage":
        record["stage"] = "analyzed"
    elif change == "timestamp":
        row["capturedAt"] = "2026-10-27T13:49:30.248Z"
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    results.write_text(json.dumps([row, row] if change == "duplicate" else [row]))
    with pytest.raises(ValueError):
        build_bundle(root, results, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
