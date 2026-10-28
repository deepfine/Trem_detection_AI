import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from pathlib import Path

import pytest

import evaluate_kpi as kpi


def at(seconds):
    return datetime(2026, 10, 21, 12) + timedelta(seconds=seconds)


def event(start, end, scenario="normal", cls="", track=""):
    return kpi.GtEvent("1", at(start), at(end), scenario, cls, track)


def onset(seconds, cls="person", track="1"):
    return kpi.Onset("1", at(seconds), track, cls, at(seconds + 0.2))


def match(gt, alerts, start=0, end=60):
    return kpi.match_events(gt, alerts, {"1": [(at(start), at(end))]},
                            timedelta(0), timedelta(seconds=5), timedelta(seconds=10))


def test_event_metrics_and_non_risk_windows():
    m = match([event(10, 20), event(30, 40)],
              [onset(10), onset(11), onset(45), onset(46)])
    assert (m["tp"], m["fp"], m["fn"], m["duplicate_onsets"]) == (1, 2, 1, 1)
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == m["detection_accuracy"] == 0.5
    assert m["f1"] == pytest.approx(0.4)
    assert m["non_risk_windows"] == 4
    assert m["fpr"] == 0.25
    assert m["fp_per_hour"] == 120


def test_overlapping_events_do_not_lose_a_match():
    m = match([event(5, 20), event(10, 12)], [onset(10), onset(18)])
    assert m["tp"] == 2
    assert m["matched"][1].time == at(10)


def test_ambiguous_class_matches_can_be_reassigned():
    m = match([event(10, 20), event(10, 20, cls="person")],
              [onset(11), onset(12, cls="bicycle")])
    assert m["tp"] == 2
    assert m["matched"][1].cls == "person"


def test_wrong_class_or_track_cannot_satisfy_event():
    m = match([event(10, 20, cls="person", track="a")],
              [onset(11, cls="bicycle", track="a"), onset(12, track="b")])
    assert (m["tp"], m["fp"], m["fn"]) == (0, 2, 1)


def test_gt_surrounding_coverage_is_excluded_without_creating_non_risk_time():
    m = match([event(-10, 100)], [onset(15)])
    assert m["gt_events"] == 0
    assert m["gt_events_outside_coverage"] == 1
    assert m["non_risk_windows"] == 0
    assert m["fp"] == 0


def test_occluded_capture_is_before_entry_and_lead_is_not_non_risk():
    m = match([event(20, 30, "occluded")], [onset(16)])
    assert m["occluded_capture_rate"] == 1
    assert m["non_risk_windows"] == 5
    late = match([event(20, 30, "occluded")], [onset(21)])
    assert late["occluded_capture_rate"] == 0


def test_no_alerts_and_empty_denominators():
    m = match([event(10, 20)], [])
    assert m["recall"] == m["f1"] == 0
    assert m["precision"] is None
    empty = match([], [])
    assert empty["recall"] is empty["f1"] is None
    assert empty["fpr"] == 0


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError, match="window"):
        kpi.match_events([], [], {}, timedelta(0), timedelta(0), timedelta(0))


def test_risk_boundary_alert_of_wrong_class_does_not_pollute_fpr():
    m = match([event(10, 20, cls="person")], [onset(10, cls="bicycle")])
    assert m["fp"] == 1
    assert m["fpr"] == 0


def test_latency_does_not_bridge_disjoint_exports():
    logs = [{"camera": "1", "captured": at(t), "completed": at(t + 0.2), "export": source}
            for t, source in [(0, "a"), (1, "a"), (500, "b"), (501, "b")]]
    result = kpi.latency_metrics(logs)
    assert result["update_interval_ms"]["n"] == 2
    assert result["update_interval_ms"]["mean"] == 1000
    assert result["analysis_latency_ms"]["mean"] == 200


def test_coverage_is_merged_and_reversed_span_rejected(tmp_path):
    p = tmp_path / "coverage.csv"
    p.write_text("camera_id,start,end\n1,2026-10-21 12:00:00,2026-10-21 12:00:20\n"
                 "1,2026-10-21 12:00:10,2026-10-21 12:00:30\n")
    assert kpi.load_coverage(p, timedelta(0)) == {"1": [(at(0), at(30))]}
    p.write_text("camera_id,start,end\n1,2026-10-21 12:00:20,2026-10-21 12:00:00\n")
    with pytest.raises(ValueError):
        kpi.load_coverage(p, timedelta(0))


@pytest.mark.parametrize("contents", ["wrong,columns\n", "camera_id,start,end\n,2026-10-21 12:00:00,2026-10-21 12:00:10\n",
                                      "camera_id,start, start,end\n"])
def test_invalid_csv_cannot_silently_become_an_empty_evaluation(tmp_path, contents):
    p = tmp_path / "gt.csv"
    p.write_text(contents)
    with pytest.raises(ValueError):
        kpi.load_gt(p, timedelta(0))


def test_e2e_reports_missing_channels_and_does_not_reuse_next_event_display():
    gt = [event(10, 15), event(20, 25)]
    displays = [{"camera_id": "1", "track_id": "1", "channel": "control",
                 "displayed_at": str(at(20.5))}]
    e2e = kpi.e2e_metrics(gt, {0: onset(10), 1: onset(20)}, displays, timedelta(0))
    assert e2e["control"]["n"] == 1
    assert e2e["control"]["mean"] == 0.5
    assert e2e["control"]["matched_without_display"] == 1
    assert not e2e["control"]["complete"]
    assert e2e["glasses"]["n"] == 0


def test_coordinate_rmse_and_p95():
    rows = [{"point_id": "a", "gt_x_cm": "0", "gt_y_cm": "0",
             "meas_x_cm": "3", "meas_y_cm": "4"},
            {"point_id": "a", "gt_x_cm": "0", "gt_y_cm": "0",
             "meas_x_cm": "0", "meas_y_cm": "0"}]
    c = kpi.coordinate_metrics(rows)
    assert c["rmse_cm"] == pytest.approx((25 / 2) ** 0.5)
    assert c["p95_cm"] == 4.75
    rows[0]["meas_x_cm"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        kpi.coordinate_metrics(rows)


def test_overlapping_exports_are_deduplicated(monkeypatch):
    sheets = {"분석로그": [{"상태": "COMPLETED", "카메라ID": "1", "로그ID": "10",
                           "촬영시각": str(at(10)), "완료시각": str(at(11))}],
              "검출객체": [{"행동": "ENTER", "카메라ID": "1", "로그ID": "10", "촬영시각": str(at(10)),
                           "트랙ID": "a", "클래스": "person"}]}
    monkeypatch.setattr(kpi, "read_xlsx", lambda p: sheets)
    logs, alerts, statuses = kpi.load_export(["a", "a"], None)
    assert statuses == {"COMPLETED": 1}
    assert len(logs) == len(alerts) == 1
    assert alerts[0].completed == at(11)


def test_timeout_records_are_visible_in_export_health(monkeypatch):
    sheets = {"분석로그": [
        {"카메라ID": "1", "로그ID": "1", "상태": "TIMEOUT", "완료시각": str(at(1))},
        {"카메라ID": "1", "로그ID": "2", "상태": "COMPLETED",
         "촬영시각": str(at(2)), "완료시각": str(at(3))},
    ], "검출객체": []}
    monkeypatch.setattr(kpi, "read_xlsx", lambda p: sheets)
    logs, _, statuses = kpi.load_export(["a", "a"], None)
    assert len(logs) == 1
    assert statuses == {"COMPLETED": 1, "TIMEOUT": 1}


def test_fall_enter_is_evaluated_even_while_zone_event_is_stay(monkeypatch):
    rows = [{"행동": "STAY", "쓰러짐이벤트": action, "카메라ID": "1", "로그ID": str(i),
             "촬영시각": str(at(10 + i)), "트랙ID": "7", "클래스": "person"}
            for i, action in enumerate(("ENTER", "STAY", "EXIT"))]
    monkeypatch.setattr(kpi, "read_xlsx", lambda p: {"분석로그": [], "검출객체": rows})
    assert not kpi.load_export(["a"], None)[1]
    falls = kpi.load_export(["a", "a"], None, "fall")[1]
    assert len(falls) == 1
    assert falls[0].time == at(10)


def test_fall_evaluation_rejects_legacy_exports_without_fall_events(monkeypatch):
    monkeypatch.setattr(kpi, "read_xlsx", lambda p: {"분석로그": [], "검출객체": [{"행동": "ENTER"}]})
    with pytest.raises(ValueError, match="쓰러짐이벤트"):
        kpi.load_export(["a"], None, "fall")


def test_replay_uses_recorded_times_and_camera_ids(tmp_path, monkeypatch):
    import measure_analysis
    (tmp_path / "a.jpg").write_bytes(b"frame")
    monkeypatch.setattr(measure_analysis, "read_xlsx", lambda p: {"분석로그": [
        {"사진": "a.jpg", "촬영시각": str(at(3)), "로그ID": "15", "카메라ID": "7"},
    ]})
    rows = measure_analysis.exported_frames(tmp_path / "export.xlsx")
    assert rows == [(at(3), 15, 7, tmp_path / "a.jpg")]


@pytest.mark.parametrize("photo", ["../outside.jpg", "missing.jpg", ""])
def test_replay_rejects_unusable_frame_inputs(tmp_path, monkeypatch, photo):
    import measure_analysis
    monkeypatch.setattr(measure_analysis, "read_xlsx", lambda p: {"분석로그": [
        {"사진": photo, "촬영시각": str(at(3)), "로그ID": "15", "카메라ID": "7"},
    ]})
    with pytest.raises((ValueError, FileNotFoundError)):
        measure_analysis.exported_frames(tmp_path / "export.xlsx")


def test_full_report_uses_separate_bev_and_real_alert_inputs(tmp_path):
    root = Path(__file__).parent / "examples" / "kpi"
    args = SimpleNamespace(export=[], alerts=root / "alerts.csv", candidates=None, gt=root / "gt.csv",
                           coverage=root / "coverage.csv", display=root / "display.csv",
                           coords=root / "coords.csv", bev=root / "bev.csv", classes=None,
                           tolerance=0, occluded_lead=5, window=10, time_offset=0, event_type="zone")
    report, lines = kpi.build_report(args)
    assert report["events"]["tp"] == 2
    assert report["events"]["fp"] == 1
    assert report["events"]["occluded_capture_rate"] == 1
    assert report["latency"]["analysis_latency_ms"]["n"] == 0
    assert report["bev_latency_ms"]["mean"] == 250
    assert report["e2e"]["control"]["mean"] == pytest.approx(0.3)
    assert report["settings"]["evaluation_scope"] == "system_alerts"
    json.dumps(report, allow_nan=False)
    assert "BEV" in "\n".join(lines)
    args.candidates, args.alerts = args.alerts, None
    candidate_report, candidate_lines = kpi.build_report(args)
    assert candidate_report["settings"]["evaluation_scope"] == "zone_candidates_only"
    assert candidate_report["events"]["tp"] == report["events"]["tp"]
    assert any("후보 평가" in line for line in candidate_lines)
