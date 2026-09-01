from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from camera_zones import (
    latest_jpeg,
    list_cameras,
    load_zone_file,
    save_zone_file,
    zone_file,
)


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>위험구역 설정</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; background: #101216; color: #eee; }
.bar { min-height: 58px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 14px; background: #1b1f26; }
button, select, input { height: 36px; border: 1px solid #505762; background: #252b34; color: #eee; border-radius: 6px; padding: 0 10px; }
button:hover { background: #323945; }
#status { margin-left: auto; color: #bbc2cc; font-size: 14px; }
.wrap { height: calc(100vh - 58px); display: grid; place-items: center; overflow: auto; padding: 12px; }
canvas { max-width: 100%; max-height: calc(100vh - 82px); background: #000; cursor: crosshair; }
</style>
</head>
<body>
<div class="bar">
  <select id="camera" aria-label="카메라"></select>
  <select id="level" aria-label="위험 단계">
    <option value="danger">위험</option>
    <option value="warning">주의</option>
    <option value="safe">안전</option>
  </select>
  <input id="zoneIndex" type="number" min="0" placeholder="Zone 번호" aria-label="Zone 번호">
  <label>거리 기준 N <input id="gap" type="number" min="0" value="2" style="width:70px" aria-label="거리 기준"></label>
  <button id="add">구역 추가</button>
  <button id="undo">점 취소</button>
  <button id="remove">마지막 구역 삭제</button>
  <button id="refresh">화면 새로 받기</button>
  <button id="save">저장</button>
  <span id="status"></span>
</div>
<div class="wrap"><canvas id="canvas"></canvas></div>
<script>
const camera = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const level = document.getElementById("level");
const zoneIndex = document.getElementById("zoneIndex");
const gap = document.getElementById("gap");
const img = new Image();
let points = [];
let zones = [];
const colors = {danger: "#ff3030", warning: "#ffb000", safe: "#20a050"};

function draw() {
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0);
  ctx.lineWidth = 3;
  for (const zone of zones) {
    const polygon = zone.points;
    if (polygon.length < 2) continue;
    ctx.strokeStyle = colors[zone.level];
    ctx.fillStyle = `${colors[zone.level]}55`;
    ctx.beginPath(); ctx.moveTo(...polygon[0]);
    polygon.slice(1).forEach(p => ctx.lineTo(...p));
    ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = colors[zone.level];
    ctx.font = "bold 16px system-ui";
    ctx.fillText(`${zone.level}${zone.index === undefined ? "" : ` Z${zone.index}`}`, polygon[0][0] + 6, polygon[0][1] - 6);
  }
  ctx.strokeStyle = colors[level.value];
  ctx.fillStyle = colors[level.value];
  for (const [i, p] of points.entries()) {
    ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI * 2); ctx.fill();
    if (i) { ctx.beginPath(); ctx.moveTo(...points[i - 1]); ctx.lineTo(...p); ctx.stroke(); }
  }
}

async function selectCamera() {
  points = [];
  status.textContent = "불러오는 중...";
  const res = await fetch(`/zones/${encodeURIComponent(camera.value)}`);
  const data = res.ok ? await res.json() : {zones: []};
  zones = data.zones || (data.polygons || []).map((points, i) => ({name: `danger-${i + 1}`, level: "danger", points}));
  gap.value = data.zoneGapThreshold ?? 2;
  img.src = `/frame/${encodeURIComponent(camera.value)}.jpg?t=${Date.now()}`;
}

img.onload = () => {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  status.textContent = "단계와 Zone 번호를 선택한 뒤 꼭짓점을 지정하세요.";
  draw();
};
img.onerror = () => status.textContent = "분석 JPEG를 불러오지 못했습니다. 카메라 프레임이 쌓인 뒤 다시 시도하세요.";
canvas.onclick = event => {
  const rect = canvas.getBoundingClientRect();
  points.push([
    Math.round((event.clientX - rect.left) * canvas.width / rect.width),
    Math.round((event.clientY - rect.top) * canvas.height / rect.height),
  ]);
  draw();
};
camera.onchange = selectCamera;
document.getElementById("add").onclick = () => {
  if (points.length < 3) { status.textContent = "구역은 점 3개 이상이 필요합니다."; return; }
  const index = zoneIndex.value === "" ? undefined : Number(zoneIndex.value);
  const item = {name: `${level.value}-${zones.length + 1}`, level: level.value, points};
  if (index !== undefined) item.index = index;
  zones.push(item);
  points = [];
  draw();
};
document.getElementById("undo").onclick = () => { points.pop(); draw(); };
document.getElementById("remove").onclick = () => { zones.pop(); draw(); };
document.getElementById("refresh").onclick = () => {
  img.src = `/frame/${encodeURIComponent(camera.value)}.jpg?t=${Date.now()}`;
};
document.getElementById("save").onclick = async () => {
  if (points.length) { status.textContent = "그리는 중인 구역을 먼저 추가하거나 취소하세요."; return; }
  if (!img.naturalWidth) { status.textContent = "화면이 없어 저장할 수 없습니다."; return; }
  const res = await fetch(`/zones/${encodeURIComponent(camera.value)}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({width: canvas.width, height: canvas.height, zoneGapThreshold: Number(gap.value), zones}),
  });
  status.textContent = await res.text();
};

fetch("/cameras").then(r => r.json()).then(items => {
  for (const item of items) camera.add(new Option(`${item.name} (${item.id})`, String(item.id)));
  if (items.length) selectCamera();
  else status.textContent = "분석된 카메라 폴더가 없습니다.";
});
</script>
</body>
</html>
"""


def make_handler(analyzed_dir: Path, zones_dir: Path):
    analyzed_dir = analyzed_dir.resolve()
    zones_dir = zones_dir.resolve()
    zones_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="text/plain; charset=utf-8"):
            body = body if isinstance(body, bytes) else body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def camera_id(self, prefix, suffix=""):
            path = unquote(urlparse(self.path).path)
            if not path.startswith(prefix) or suffix and not path.endswith(suffix):
                return None
            end = -len(suffix) if suffix else None
            raw = path[len(prefix) : end]
            return int(raw) if raw.isdigit() else None

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send(200, HTML, "text/html; charset=utf-8")
                return
            if path == "/cameras":
                self.send(
                    200,
                    json.dumps(list_cameras(analyzed_dir, zones_dir), ensure_ascii=False),
                    "application/json",
                )
                return
            device_id = self.camera_id("/frame/", ".jpg")
            if device_id is not None:
                frame = latest_jpeg(analyzed_dir, device_id)
                if frame is None or not frame.is_file():
                    self.send(404, "분석 JPEG가 없습니다.")
                    return
                self.send(200, frame.read_bytes(), "image/jpeg")
                return
            device_id = self.camera_id("/zones/")
            if device_id is not None:
                payload = load_zone_file(zone_file(zones_dir, device_id)) or {
                    "width": 0,
                    "height": 0,
                    "zoneGapThreshold": 2,
                    "zones": [],
                }
                self.send(200, json.dumps(payload, ensure_ascii=False), "application/json")
                return
            self.send_error(404)

        def do_POST(self):
            device_id = self.camera_id("/zones/")
            if device_id is None:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    raise ValueError("request is too large")
                payload = json.loads(self.rfile.read(length))
                frame = latest_jpeg(analyzed_dir, device_id)
                image_w = image_h = None
                if frame is not None:
                    import cv2

                    image = cv2.imread(str(frame))
                    if image is not None:
                        image_h, image_w = image.shape[:2]
                saved = save_zone_file(
                    zone_file(zones_dir, device_id),
                    {
                        **payload,
                        "width": payload.get("width") or image_w,
                        "height": payload.get("height") or image_h,
                    },
                )
            except (ValueError, json.JSONDecodeError, TypeError) as error:
                self.send(400, str(error))
                return
            self.send(200, f"카메라 {device_id} 구역 {len(saved['zones'])}개를 저장했습니다.")

        def log_message(self, format, *args):
            print(f"zone editor {self.address_string()} {format % args}", flush=True)

    return Handler


def serve_annotator(analyzed_dir: Path, zones_dir: Path, host="0.0.0.0", port=8765):
    server = ThreadingHTTPServer((host, port), make_handler(analyzed_dir, zones_dir))
    print(f"zone editor http://{host}:{port}", flush=True)
    print(f"zones={zones_dir}", flush=True)
    server.serve_forever()


def start_annotator_thread(analyzed_dir: Path, zones_dir: Path, host="0.0.0.0", port=8765):
    try:
        server = ThreadingHTTPServer((host, port), make_handler(analyzed_dir, zones_dir))
    except OSError as error:
        print(f"zone editor failed to bind {host}:{port} reason={error}", flush=True)
        return None
    print(f"zone editor http://{host}:{port}", flush=True)
    print(f"zones={zones_dir}", flush=True)
    thread = threading.Thread(target=server.serve_forever, name="zone-annotator", daemon=True)
    thread.start()
    return thread


def main():
    parser = argparse.ArgumentParser(description="Mark danger, warning, and safe polygons on the latest analyzed JPEG.")
    parser.add_argument("--analyzed-dir", type=Path, default=Path("/upload/visit_servant/analyzed"))
    parser.add_argument("--zones-dir", type=Path, default=Path("/upload/visit_servant/zones"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve_annotator(args.analyzed_dir, args.zones_dir, args.host, args.port)


if __name__ == "__main__":
    main()
