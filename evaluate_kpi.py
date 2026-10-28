"""Compute the core performance KPIs from an analysis export and labeled ground truth.

Definitions follow the evaluation table (event-window basis) and are documented in
docs/kpi-evaluation.md. GT-dependent metrics are reported only when GT is given.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

ONSET_ACTIONS = {"ENTER", "ESCALATE"}
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")
TARGETS = {
    "precision": (">=", 0.90),
    "recall": (">=", 0.95),
    "f1": (">=", 0.92),
    "detection_accuracy": (">=", 0.95),
    "fpr": ("<=", 0.08),
    "occluded_capture_rate": (">=", 0.80),
    "bev_mean_ms": ("<=", 500),
    "bev_p95_ms": ("<=", 500),
    "e2e_mean_s": ("<=", 0.7),
    "e2e_p95_s": ("<=", 0.7),
    "coordinate_rmse_cm": ("<=", 15),
    "coordinate_p95_cm": ("<=", 15),
}


@dataclass(frozen=True)
class Onset:
    camera: str
    time: datetime
    track_id: str
    cls: str
    completed: datetime | None = None


@dataclass(frozen=True)
class GtEvent:
    camera: str
    start: datetime
    end: datetime
    scenario: str = "normal"
    cls: str = ""
    track_id: str = ""


def parse_time(value: str) -> datetime:
    value = value.strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognized time: {value!r}")


def percentile(values, q: float) -> float:
    """Linear-interpolated percentile (numpy default)."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(values) -> dict:
    values = list(values)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def read_xlsx(path: Path) -> dict[str, list[dict[str, str]]]:
    """Read every sheet as header-keyed rows without openpyxl."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        rels = {
            r.get("Id"): r.get("Target")
            for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        }
        sheets = {}
        for sheet in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", ns):
            target = rels[sheet.get(rel_ns)].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            rows = []
            for row in ET.fromstring(z.read(target)).iter(f"{{{ns['m']}}}row"):
                cells = {}
                for c in row.findall("m:c", ns):
                    col = re.match(r"[A-Z]+", c.get("r")).group()
                    kind = c.get("t")
                    if kind == "inlineStr":
                        text = "".join(t.text or "" for t in c.iter(f"{{{ns['m']}}}t"))
                    else:
                        v = c.find("m:v", ns)
                        text = "" if v is None else v.text or ""
                        if kind == "s" and text:
                            text = shared[int(text)]
                    cells[col] = text
                rows.append(cells)
            if rows:
                header = rows[0]
                sheets[sheet.get("name")] = [
                    {header[col]: r.get(col, "") for col in header} for r in rows[1:]
                ]
            else:
                sheets[sheet.get("name")] = []
        return sheets


def load_export(paths: list[Path], classes: set[str] | None, event_type="zone"):
    """Return completed logs, onsets and deduplicated status counts."""
    logs, onsets = [], []
    statuses = {}
    seen_logs, seen_onsets = set(), set()
    for path in paths:
        sheets = read_xlsx(path)
        completed = {}
        for row in sheets["분석로그"]:
            status_key = (row["카메라ID"], row["로그ID"])
            if statuses.get(status_key) != "COMPLETED":
                statuses[status_key] = row["상태"]
            if row["상태"] != "COMPLETED" or not row["완료시각"]:
                continue
            item = {
                "camera": row["카메라ID"],
                "export": str(path),
                "captured": parse_time(row["촬영시각"]),
                "completed": parse_time(row["완료시각"]),
            }
            key = (item["camera"], row["로그ID"])
            if key not in seen_logs:
                if item["completed"] < item["captured"]:
                    raise ValueError(f"completion before capture: {row}")
                logs.append(item)
                seen_logs.add(key)
            completed[key] = item["completed"]
        for row in sheets["검출객체"]:
            if event_type == "fall" and "쓰러짐이벤트" not in row:
                raise ValueError("fall evaluation requires the 쓰러짐이벤트 export column")
            action = row["쓰러짐이벤트"] if event_type == "fall" else row["행동"]
            if action not in ({"ENTER"} if event_type == "fall" else ONSET_ACTIONS):
                continue
            if classes and row["클래스"] not in classes:
                continue
            key = (row["카메라ID"], row["로그ID"], row["트랙ID"], row["클래스"], action)
            if key in seen_onsets:
                continue
            seen_onsets.add(key)
            onsets.append(Onset(
                row["카메라ID"], parse_time(row["촬영시각"]), row["트랙ID"], row["클래스"],
                completed.get((row["카메라ID"], row["로그ID"])),
            ))
    onsets.sort(key=lambda o: (o.camera, o.time))
    return logs, onsets, dict(sorted(_count(statuses.values()).items()))


def read_csv(path: Path, required=()) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"missing or duplicate CSV headers: {path}")
        if not set(required) <= set(reader.fieldnames):
            raise ValueError(f"missing required CSV columns {required}: {path}")
        rows = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"CSV row does not match headers: {path}")
            row = {k: v.strip() for k, v in row.items()}
            if any(not row[k] for k in required):
                raise ValueError(f"empty required CSV field: {path}")
            rows.append(row)
        return rows


def load_alerts(path: Path, classes: set[str] | None) -> list[Onset]:
    onsets = set()
    for row in read_csv(path, ("camera_id", "time", "track_id", "class")):
        if classes and row["class"] not in classes:
            continue
        time = parse_time(row["time"])
        completed = parse_time(row["completed_at"]) if row.get("completed_at") else None
        if completed and completed < time:
            raise ValueError("alert completion before event time")
        onsets.add(Onset(row["camera_id"], time, row["track_id"], row["class"], completed))
    return sorted(onsets, key=lambda o: (o.camera, o.time, o.track_id, o.cls))



def load_gt(path: Path, offset: timedelta) -> list[GtEvent]:
    events = []
    for row in read_csv(path, ("camera_id", "start", "end")):
        start, end = parse_time(row["start"]) + offset, parse_time(row["end"]) + offset
        if end < start:
            raise ValueError(f"GT end before start: {row}")
        scenario = row.get("scenario") or "normal"
        if scenario not in {"normal", "occluded"}:
            raise ValueError(f"scenario must be normal or occluded: {row}")
        events.append(GtEvent(row["camera_id"], start, end, scenario, row.get("class", ""), row.get("track_id", "")))
    return events


def load_coverage(path: Path, offset: timedelta) -> dict[str, list[tuple[datetime, datetime]]]:
    coverage = defaultdict(list)
    for row in read_csv(path, ("camera_id", "start", "end")):
        start, end = parse_time(row["start"]) + offset, parse_time(row["end"]) + offset
        if end <= start:
            raise ValueError(f"coverage end must be after start: {row}")
        coverage[row["camera_id"]].append((start, end))
    merged = {}
    for cam, spans in coverage.items():
        merged[cam] = []
        for a, b in sorted(spans):
            if merged[cam] and a <= merged[cam][-1][1]:
                merged[cam][-1] = (merged[cam][-1][0], max(b, merged[cam][-1][1]))
            else:
                merged[cam].append((a, b))
    return merged


def _inside(t: datetime, spans) -> bool:
    return any(a <= t <= b for a, b in spans)


def _subtract(span, holes):
    """Return parts of span not covered by holes."""
    parts = [span]
    for h0, h1 in sorted(holes):
        next_parts = []
        for a, b in parts:
            if h1 <= a or h0 >= b:
                next_parts.append((a, b))
                continue
            if a < h0:
                next_parts.append((a, h0))
            if h1 < b:
                next_parts.append((h1, b))
        parts = next_parts
    return parts


def match_events(gt: list[GtEvent], onsets: list[Onset], coverage, tolerance: timedelta,
                 occluded_lead: timedelta, window: timedelta) -> dict:
    """Event-window matching of alert onsets against labeled risk events."""
    if window <= timedelta(0) or min(tolerance, occluded_lead) < timedelta(0):
        raise ValueError("window must be positive; tolerance and lead must be nonnegative")
    # Only completely reviewed events enter the denominator.
    in_scope = [i for i, e in enumerate(gt)
                if any(a <= e.start and e.end <= b for a, b in coverage.get(e.camera, []))]
    gt_by_cam = defaultdict(list)
    for i, e in enumerate(gt):
        if i in in_scope:
            gt_by_cam[e.camera].append(i)
    first_onset: dict[int, Onset] = {}
    candidates = []
    false_positives, duplicates, outside = [], [], 0
    for onset in sorted(onsets, key=lambda o: (o.camera, o.time)):
        spans = coverage.get(onset.camera, [])
        if not _inside(onset.time, spans):
            outside += 1
            continue
        hits = [
            i for i in gt_by_cam[onset.camera]
            if gt[i].start - (occluded_lead if gt[i].scenario == "occluded" else tolerance)
            <= onset.time <= (gt[i].start if gt[i].scenario == "occluded" else gt[i].end + tolerance)
            and (not gt[i].cls or gt[i].cls == onset.cls)
            and (not gt[i].track_id or gt[i].track_id == onset.track_id)
        ]
        if not hits:
            if any(e.camera == onset.camera and i not in in_scope
                   and e.start - max(tolerance, occluded_lead) <= onset.time <= e.end + tolerance
                   for i, e in enumerate(gt)):
                outside += 1
                continue
            false_positives.append(onset)
            continue
        candidates.append((onset, sorted(hits, key=lambda i: (gt[i].end, gt[i].start, i))))

    owners = {}

    def assign(j, visited):
        for i in candidates[j][1]:
            if i not in visited and i not in owners:
                owners[i] = j
                return True
        for i in candidates[j][1]:
            if i in visited:
                continue
            visited.add(i)
            if i not in owners or assign(owners[i], visited):
                owners[i] = j
                return True
        return False

    # Reassign ambiguous overlaps so a broad window cannot consume the only
    # alert available to a narrower event.
    for j in range(len(candidates)):
        if not assign(j, set()):
            duplicates.append(candidates[j][0])
    first_onset = {i: candidates[j][0] for i, j in owners.items()}

    tp = sum(1 for i in in_scope if i in first_onset)
    fn = len(in_scope) - tp
    fp = len(false_positives)

    # Non-risk windows: coverage minus each GT span widened by the tolerance.
    negative_windows = positive_windows = 0
    reviewed_seconds = sum((b - a).total_seconds() for spans in coverage.values() for a, b in spans)
    fp_times = defaultdict(list)
    for o in false_positives:
        fp_times[o.camera].append(o.time)
    for cam, spans in coverage.items():
        # Partial events still exclude risk time from the non-risk denominator.
        holes = [(e.start - (occluded_lead if e.scenario == "occluded" else tolerance),
                  e.end + tolerance) for e in gt if e.camera == cam]
        non_risk_fp = [t for t in fp_times[cam] if not _inside(t, holes)]
        for span in spans:
            for a, b in _subtract(span, holes):
                t = a
                while t < b:
                    u = min(t + window, b)
                    negative_windows += 1
                    if any(t <= x < u or (u == b and x == b) for x in non_risk_fp):
                        positive_windows += 1
                    t = u

    occluded = [i for i in in_scope if gt[i].scenario == "occluded"]
    occluded_hit = sum(1 for i in occluded if i in first_onset
                       and first_onset[i].time <= gt[i].start)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    response = [
        (first_onset[i].completed - gt[i].start).total_seconds()
        for i in in_scope if i in first_onset and first_onset[i].completed
    ]
    return {
        "gt_events": len(in_scope),
        "gt_events_outside_coverage": len(gt) - len(in_scope),
        "tp": tp, "fp": fp, "fn": fn,
        "duplicate_onsets": len(duplicates),
        "onsets_outside_coverage": outside,
        "precision": precision, "recall": recall, "f1": f1,
        "non_risk_windows": negative_windows,
        "non_risk_windows_with_alert": positive_windows,
        "fpr": positive_windows / negative_windows if negative_windows else None,
        "fp_per_hour": fp / (reviewed_seconds / 3600) if reviewed_seconds else None,
        "reviewed_seconds": reviewed_seconds,
        "detection_accuracy": recall,
        "occluded_events": len(occluded),
        "occluded_captured": occluded_hit,
        "occluded_capture_rate": occluded_hit / len(occluded) if occluded else None,
        "gt_start_to_analysis_complete_s": summarize(response),
        "fp_by_class": dict(sorted(_count(o.cls for o in false_positives).items())),
        "matched": {i: first_onset[i] for i in in_scope if i in first_onset},
    }


def _count(items):
    counts = defaultdict(int)
    for x in items:
        counts[x] += 1
    return counts


def latency_metrics(logs) -> dict:
    by_cam = defaultdict(list)
    for item in logs:
        by_cam[item["camera"]].append(item)
    intervals, per_camera = [], {}
    for cam, items in sorted(by_cam.items()):
        items.sort(key=lambda x: x["captured"])
        gaps = [(b["captured"] - a["captured"]).total_seconds() * 1000 for a, b in zip(items, items[1:])
                if a.get("export") == b.get("export")]
        intervals += gaps
        per_camera[cam] = {
            "analysis_latency_ms": summarize((x["completed"] - x["captured"]).total_seconds() * 1000 for x in items),
            "update_interval_ms": summarize(gaps),
        }
    return {
        "analysis_latency_ms": summarize((x["completed"] - x["captured"]).total_seconds() * 1000 for x in logs),
        "update_interval_ms": summarize(intervals),
        "per_camera": per_camera,
    }


def e2e_metrics(gt: list[GtEvent], matched: dict[int, Onset], displays: list[dict], offset: timedelta) -> dict:
    """GT event start -> first display of the matched track on each channel."""
    shown = defaultdict(list)
    for row in displays:
        shown[(row["camera_id"], row["track_id"], row.get("channel") or "display")].append(
            parse_time(row["displayed_at"]) + offset)
    channels = sorted({"control", "glasses"} | {key[2] for key in shown})
    result = {}
    for channel in channels:
        values, missing = [], 0
        for i, onset in matched.items():
            next_times = [other.time for j, other in matched.items()
                          if j != i and (other.camera, other.track_id) == (onset.camera, onset.track_id)
                          and other.time > onset.time]
            upper = min(next_times) if next_times else datetime.max
            times = [t for t in shown.get((onset.camera, onset.track_id, channel), [])
                     if onset.time <= t < upper]
            if times:
                values.append(max(0.0, (min(times) - gt[i].start).total_seconds()))
            else:
                missing += 1
        result[channel] = {**summarize(values), "matched_without_display": missing,
                           "unmatched_gt_events": len(gt) - len(matched),
                           "complete": bool(matched) and missing == 0 and len(matched) == len(gt)}
    return result


def coordinate_metrics(rows: list[dict[str, str]]) -> dict:
    """Euclidean error per repeated measurement, in cm."""
    errors = [
        math.hypot(float(r["meas_x_cm"]) - float(r["gt_x_cm"]), float(r["meas_y_cm"]) - float(r["gt_y_cm"]))
        for r in rows
    ]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError("coordinates must be finite")
    return {
        "n": len(errors),
        "points": len({r["point_id"] for r in rows}),
        "rmse_cm": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
        "p95_cm": percentile(errors, 95) if errors else None,
        "max_cm": max(errors) if errors else None,
    }


def _judge(key: str, value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "미측정"
    op, target = TARGETS[key]
    ok = value >= target if op == ">=" else value <= target
    return f"{'충족' if ok else '미달'} (목표 {op} {target})"


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def build_report(args) -> tuple[dict, list[str]]:
    if not any((args.export, args.alerts, args.candidates, args.coords, args.bev)):
        raise ValueError("provide an export, --alerts, --candidates, --coords or --bev")
    if not all(math.isfinite(v) for v in (args.time_offset, args.tolerance, args.occluded_lead, args.window)):
        raise ValueError("time parameters must be finite")
    if args.window <= 0 or min(args.tolerance, args.occluded_lead) < 0:
        raise ValueError("window must be positive; tolerance and lead must be nonnegative")
    if args.display and not args.gt:
        raise ValueError("--display requires --gt and --coverage")
    classes = {v.strip() for v in args.classes.split(",") if v.strip()} if args.classes else None
    logs, onsets, statuses = load_export(args.export, classes, args.event_type)
    onset_csv = args.alerts or args.candidates
    if onset_csv:
        onsets = load_alerts(onset_csv, classes)
    offset = timedelta(seconds=args.time_offset)
    tolerance = timedelta(seconds=args.tolerance)
    report = {
        "settings": {
            "export": [str(p) for p in args.export],
            "tolerance_s": args.tolerance,
            "occluded_lead_s": args.occluded_lead,
            "non_risk_window_s": args.window,
            "time_offset_s": args.time_offset,
            "classes": sorted(classes) if classes else "all",
            "event_type": args.event_type,
            "onset_actions": ["ENTER"] if args.event_type == "fall" else sorted(ONSET_ACTIONS),
            "alert_source": str(onset_csv) if onset_csv else ("fall_candidates" if args.event_type == "fall" else "zone_transition_candidates"),
            "accuracy_definition": "matched risk scenarios / reviewed risk scenarios",
            "bev": str(args.bev) if args.bev else None,
            "gt": str(args.gt) if args.gt else None,
            "coverage": str(args.coverage) if args.coverage else None,
            "display": str(args.display) if args.display else None,
            "coords": str(args.coords) if args.coords else None,
            "evaluation_scope": "system_alerts" if args.alerts else f"{args.event_type}_candidates_only",
            "judgment_scope": "input_sample_only; not independent certification",
        },
        "export_health": {"total_logs": sum(statuses.values()), "status_counts": statuses,
                          "excluded_from_latency": sum(statuses.values()) - len(logs)},
        "latency": latency_metrics(logs),
        "alert_onsets": len(onsets),
    }
    lat = report["latency"]["analysis_latency_ms"]
    lines = [
        f"분석 로그 전체 {sum(statuses.values())}건, 상태별 {statuses}, 지연 표본 {len(logs)}건",
        f"평가 대상 경보/구역 전이 {len(onsets)}건",
        "목표 판정은 입력 표본 기준이며 공식 시험의 달성 판정과 구분한다.",
        "",
        "[분석 지연] 촬영→분석완료의 보조 지표이며 BEV 지연과 구분한다.",
        f"  분석 지연 ms: 평균 {_fmt(lat['mean'], 1)}, P95 {_fmt(lat['p95'], 1)}",
    ]
    gap = report["latency"]["update_interval_ms"]
    lines.append(f"  카메라별 촬영 간격 ms(파일 간 공백 제외): 평균 {_fmt(gap['mean'], 1)}, P95 {_fmt(gap['p95'], 1)}")

    if not args.alerts:
        candidate = "쓰러짐이벤트 ENTER" if args.event_type == "fall" else "검출객체 ENTER/ESCALATE"
        lines.append(f"  주의: {candidate} 후보 평가이며 실제 시스템 경보 성능은 --alerts로 평가한다.")
    if args.bev:
        rows = read_csv(args.bev, ("camera_id", "frame_id", "captured_at", "updated_at"))
        delays = [(parse_time(r["updated_at"]) - parse_time(r["captured_at"])).total_seconds() * 1000
                  for r in rows]
        if any(v < 0 for v in delays):
            raise ValueError("BEV update before capture")
        bev = report["bev_latency_ms"] = summarize(delays)
        lines.append(f"[BEV 갱신 지연] 평균 {_fmt(bev['mean'], 1)}ms → {_judge('bev_mean_ms', bev['mean'])}, "
                     f"P95 {_fmt(bev['p95'], 1)}ms → {_judge('bev_p95_ms', bev['p95'])}")
    else:
        lines.append("[BEV 갱신 지연] 미측정: BEV 시각 CSV(--bev) 없음")

    if args.gt:
        if not args.coverage:
            raise SystemExit("--gt requires --coverage: unlabeled time cannot count as non-risk")
        gt = load_gt(args.gt, offset)
        if classes:
            if any(not e.cls for e in gt):
                raise ValueError("--classes requires class on every GT event")
            gt = [e for e in gt if e.cls in classes]
        coverage = load_coverage(args.coverage, offset)
        m = match_events(gt, onsets, coverage, tolerance, timedelta(seconds=args.occluded_lead),
                         timedelta(seconds=args.window))
        matched = m.pop("matched")
        report["events"] = m
        lines += [
            "",
            f"[이벤트 매칭] GT {m['gt_events']}건 (평가구간 밖 {m['gt_events_outside_coverage']}건 제외), "
            f"TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, 중복 경보 {m['duplicate_onsets']}건, "
            f"평가구간 밖 경보 {m['onsets_outside_coverage']}건",
            f"  위험 시나리오 경보 일치율(TP/GT) {_fmt(m['detection_accuracy'])} → {_judge('detection_accuracy', m['detection_accuracy'])}",
            f"  Precision {_fmt(m['precision'])} → {_judge('precision', m['precision'])}",
            f"  Recall    {_fmt(m['recall'])} → {_judge('recall', m['recall'])}",
            f"  F1        {_fmt(m['f1'])} → {_judge('f1', m['f1'])}",
            f"  FPR       {_fmt(m['fpr'])} ({m['non_risk_windows_with_alert']}/{m['non_risk_windows']} 비위험 창) "
            f"→ {_judge('fpr', m['fpr'])}, 시간당 오경보 {_fmt(m['fp_per_hour'], 2)}건",
            f"  FP 클래스별: {m['fp_by_class'] or '-'}",
            f"  비가시권 포착률 {_fmt(m['occluded_capture_rate'])} ({m['occluded_captured']}/{m['occluded_events']}) "
            f"→ {_judge('occluded_capture_rate', m['occluded_capture_rate'])}",
            f"  GT 시작→분석완료 s: 평균 {_fmt(m['gt_start_to_analysis_complete_s']['mean'])}, "
            f"P95 {_fmt(m['gt_start_to_analysis_complete_s']['p95'])} (E2E 중 검출 구간)",
        ]
        if args.display:
            # Reindex reviewed events for E2E so excluded GT cannot enter its denominator.
            indices = [i for i, e in enumerate(gt)
                       if any(a <= e.start and e.end <= b for a, b in coverage.get(e.camera, []))]
            e2e = e2e_metrics([gt[i] for i in indices],
                              {j: matched[i] for j, i in enumerate(indices) if i in matched},
                              read_csv(args.display, ("camera_id", "track_id", "displayed_at", "channel")), offset)
            report["e2e"] = e2e
            for channel, s in e2e.items():
                lines.append(
                    f"  [E2E {channel}] GT 시작→표출 s: 평균 {_fmt(s['mean'])}, P95 {_fmt(s['p95'])}, "
                    f"표출 기록 없음 {s['matched_without_display']}건 → 평균 {_judge('e2e_mean_s', s['mean'] if s['complete'] else None)}, "
                    f"P95 {_judge('e2e_p95_s', s['p95'] if s['complete'] else None)}; 전체 GT 표출 {s['complete']}")
        else:
            lines.append("  [E2E 반응성] 미측정: 표출시각 기록(--display) 없음")
    else:
        lines += ["", "[이벤트 지표] 미측정: GT 없음 (--gt, --coverage 필요)"]

    if args.coords:
        c = coordinate_metrics(read_csv(args.coords, ("point_id", "trial", "gt_x_cm", "gt_y_cm", "meas_x_cm", "meas_y_cm")))
        report["coordinates"] = c
        lines += ["", f"[좌표 정확도] {c['points']}개 지점 {c['n']}회 측정, RMSE {_fmt(c['rmse_cm'], 1)}cm "
                      f"→ {_judge('coordinate_rmse_cm', c['rmse_cm'])}, P95 {_fmt(c['p95_cm'], 1)}cm "
                      f"→ {_judge('coordinate_p95_cm', c['p95_cm'])}"]
    else:
        lines += ["", "[좌표 정확도] 미측정: 좌표 측정 CSV(--coords) 없음"]
    return report, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, nargs="*", help="analysis export xlsx (분석로그/검출객체 sheets)")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--alerts", type=Path, help="actual issued alerts CSV: camera_id,time,track_id,class,completed_at")
    sources.add_argument("--candidates", type=Path, help="candidate-only onset CSV; does not certify final system alerts")
    parser.add_argument("--bev", type=Path, help="BEV timestamps CSV: camera_id,frame_id,captured_at,updated_at")
    parser.add_argument("--gt", type=Path, help="risk events CSV: camera_id,start,end,scenario")
    parser.add_argument("--coverage", type=Path, help="labeled spans CSV: camera_id,start,end")
    parser.add_argument("--display", type=Path, help="display log CSV: camera_id,track_id,displayed_at,channel")
    parser.add_argument("--coords", type=Path, help="coordinate CSV: point_id,trial,gt_x_cm,gt_y_cm,meas_x_cm,meas_y_cm")
    parser.add_argument("--tolerance", type=float, default=2.0, help="event window margin in seconds")
    parser.add_argument("--occluded-lead", type=float, default=5.0,
                        help="seconds before an occluded event's start in which an alert counts as a capture")
    parser.add_argument("--window", type=float, default=10.0, help="non-risk window length in seconds for FPR")
    parser.add_argument("--time-offset", type=float, default=0.0, help="seconds added to GT/coverage/display times")
    parser.add_argument("--classes", help="comma-separated alert classes to count (default: all)")
    parser.add_argument("--event-type", choices=("zone", "fall"), default="zone",
                        help="evaluate zone transitions or independent fall ENTER events; GT/alerts must describe the selected type")
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    args = parser.parse_args()
    try:
        report, lines = build_report(args)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print("\n".join(lines))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
