"""Offline source-time replay. No production callbacks or DB writes."""
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import frame_objects
from crowd_analysis_protocol import api_callback_payload
from frame_objects import YoloDnnDetector, apply_object_events, apply_tram_policy

import tempfile
Path("result").mkdir(exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="fall-validation-", dir="result"))
source = Path('test_video.mkv')
cap = cv2.VideoCapture(str(source))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = (int(cap.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
assert fps > 0 and total > 0
writer = cv2.VideoWriter(str(out / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 4, (width, height))
assert writer.isOpened()
detector = YoloDnnDetector(Path('result/models/objects365_yolo26n.onnx'), mobility_model=Path('result/models/mobility_yolov8s.onnx'))
detector.load()
state = {}
now = [0.0]
original_clock = frame_objects.time.monotonic
frame_objects.time.monotonic = lambda: now[0]
latencies, events, postures, ids = [], Counter(), Counter(), set()
frame_records, object_records = [], []
next_source_frame = 0
try:
    for sample in range(int((total - 1) / fps * 4) + 1):
        started = time.perf_counter()
        target = int(round(sample * fps / 4))
        if target >= total:
            break
        ok = True
        while next_source_frame <= target:
            ok = cap.grab()
            next_source_frame += 1
            if not ok:
                break
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        now[0] = target / fps
        objects, _ = detector.detect(frame)
        objects = apply_tram_policy(objects, None, 2)
        objects = apply_object_events(objects, state, frame, max(frame.shape[:2]) / 4)
        body = api_callback_payload({'id': sample + 1, 'status': 'COMPLETED', 'objects': objects})
        json.dumps(body, allow_nan=False)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        frame_records.append({'sample': sample, 'source_frame': target, 'source_seconds': now[0], 'objects': len(objects), 'offline_pipeline_ms': elapsed})
        for item in body['objects']:
            record = {'sample': sample, 'source_frame': target, 'source_seconds': now[0], **item}
            object_records.append(record)
            if item['class'] == 'person':
                ids.add(item['trackId'])
                postures[item['fallState']] += 1
                events[item['fallEvent']] += 1
            box = item['bbox']
            color = (0, 0, 255) if item.get('fallState') == 'FALL_SUSPECTED' else (0, 220, 0)
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
            cv2.putText(frame, f"{item['class']} #{item['trackId']} {item.get('fallState', '')}", (box[0], max(18, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, f"SOURCE {now[0]:.2f}s | OFFLINE 4 FPS SAMPLING | NO FALL GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        writer.write(frame)
        if sample == 120:
            cv2.imwrite(str(out / 'preview.jpg'), frame)
        if sample % 100 == 0:
            print('sample', sample, 'source_seconds', now[0], 'objects', len(objects), flush=True)
finally:
    frame_objects.time.monotonic = original_clock
    writer.release()
    cap.release()
with (out / 'frames.csv').open('w') as f:
    writer_csv = csv.DictWriter(f, fieldnames=frame_records[0].keys())
    writer_csv.writeheader(); writer_csv.writerows(frame_records)
with (out / 'objects.jsonl').open('w') as f:
    for record in object_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
summary = {
    'mode': 'offline replay using source timestamps, NOT a live FPS benchmark',
    'source': str(source.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_fps': fps, 'source_frames': total, 'source_size': [width, height],
    'replay_sample_fps': 4, 'processed_samples': len(frame_records),
    'last_source_seconds': frame_records[-1]['source_seconds'],
    'object_records': len(object_records), 'distinct_person_track_ids': len(ids),
    'person_fall_state_counts': dict(postures), 'person_fall_event_counts': dict(events),
    'offline_decode_detection_tracking_serialization_ms': {'mean': float(np.mean(latencies)), 'p95': float(np.percentile(latencies, 95))},
    'fall_ground_truth': None, 'accuracy': 'not measured; no reviewed fall ground truth',
    'video_records': 'annotated.mp4 is rendered from the same objects written to objects.jsonl',
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print('output_dir:', out.resolve(), flush=True)
print(json.dumps(summary, ensure_ascii=False), flush=True)
