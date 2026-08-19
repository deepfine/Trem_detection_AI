from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CCTV 구역 설정</title>
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
  <select id="level" aria-label="구역 등급">
    <option value="danger">위험</option>
    <option value="warning">경고</option>
    <option value="safe">안전</option>
  </select>
  <input id="name" value="zone_1" aria-label="구역 이름">
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
const level = document.getElementById("level");
const nameInput = document.getElementById("name");
const status = document.getElementById("status");
const colors = {danger: "#ff3030", warning: "#ffd21f", safe: "#30c060"};
const img = new Image();
let points = [];
let zones = [];

function draw() {
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0);
  ctx.lineWidth = 3;
  ctx.font = "18px system-ui";
  for (const zone of zones) {
    if (!zone.points.length) continue;
    ctx.beginPath(); ctx.moveTo(...zone.points[0]);
    zone.points.slice(1).forEach(p => ctx.lineTo(...p));
    ctx.closePath();
    ctx.fillStyle = colors[zone.level] + "55";
    ctx.strokeStyle = colors[zone.level];
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = colors[zone.level];
    ctx.fillText(`${zone.name} (${zone.level})`, zone.points[0][0], Math.max(20, zone.points[0][1] - 8));
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
  zones = res.ok ? (await res.json()).zones : [];
  img.src = `/frame/${encodeURIComponent(camera.value)}.jpg?t=${Date.now()}`;
}

img.onload = () => {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  status.textContent = "화면을 클릭해 구역 꼭짓점을 지정하세요.";
  draw();
};
img.onerror = () => status.textContent = "카메라 화면을 불러오지 못했습니다.";
canvas.onclick = event => {
  const rect = canvas.getBoundingClientRect();
  points.push([
    Math.round((event.clientX - rect.left) * canvas.width / rect.width),
    Math.round((event.clientY - rect.top) * canvas.height / rect.height),
  ]);
  draw();
};
camera.onchange = selectCamera;
level.onchange = draw;
document.getElementById("add").onclick = () => {
  if (points.length < 3) { status.textContent = "구역은 점 3개 이상이 필요합니다."; return; }
  zones.push({name: nameInput.value.trim() || `zone_${zones.length + 1}`, level: level.value, points});
  points = [];
  nameInput.value = `zone_${zones.length + 1}`;
  draw();
};
document.getElementById("undo").onclick = () => { points.pop(); draw(); };
document.getElementById("remove").onclick = () => { zones.pop(); draw(); };
document.getElementById("refresh").onclick = async () => {
  status.textContent = "카메라 화면 갱신 중...";
  const res = await fetch(`/refresh/${encodeURIComponent(camera.value)}`, {method: "POST"});
  if (!res.ok) { status.textContent = await res.text(); return; }
  img.src = `/frame/${encodeURIComponent(camera.value)}.jpg?t=${Date.now()}`;
};
document.getElementById("save").onclick = async () => {
  if (points.length) { status.textContent = "그리는 중인 구역을 먼저 추가하거나 취소하세요."; return; }
  const res = await fetch(`/zones/${encodeURIComponent(camera.value)}`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({zones}),
  });
  status.textContent = await res.text();
};

fetch("/cameras").then(r => r.json()).then(items => {
  for (const item of items) camera.add(new Option(item.name, item.id));
  if (items.length) selectCamera();
});
</script>
</body>
</html>
"""


def load_cameras(path: Path):
    data = json.loads(path.read_text())
    cameras = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 5:
        raise ValueError("cameras must contain 1 to 5 items")
    ids = set()
    for camera in cameras:
        if not isinstance(camera, dict) or not all(camera.get(key) for key in ("id", "name", "url")):
            raise ValueError("each camera needs id, name, and url")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", camera["id"]) or camera["id"] in ids:
            raise ValueError(f"invalid or duplicate camera id: {camera['id']}")
        ids.add(camera["id"])
    return cameras


def validate_zones(payload):
    zones = payload.get("zones") if isinstance(payload, dict) else None
    if not isinstance(zones, list):
        raise ValueError("zones must be a list")
    for zone in zones:
        if not isinstance(zone, dict) or zone.get("level") not in {"danger", "warning", "safe"}:
            raise ValueError("zone level must be danger, warning, or safe")
        if not isinstance(zone.get("name"), str) or not zone["name"].strip():
            raise ValueError("zone name is required")
        points = zone.get("points")
        if not isinstance(points, list) or len(points) < 3 or any(
            not isinstance(point, list) or len(point) != 2 or not all(isinstance(value, int) for value in point)
            for point in points
        ):
            raise ValueError("zone needs at least 3 integer [x, y] points")
    return {"zones": zones}


def extract_frame(source: str, output: Path):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("카메라에 연결할 수 없습니다.")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("카메라 프레임을 읽을 수 없습니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError("카메라 프레임을 저장할 수 없습니다.")


def main():
    parser = argparse.ArgumentParser(description="Configure danger, warning, and safe zones for up to five CCTV cameras.")
    parser.add_argument("--cameras", type=Path, required=True, help="camera JSON file")
    parser.add_argument("--out-dir", type=Path, default=Path("result/camera_zones"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    cameras = load_cameras(args.cameras)
    by_id = {camera["id"]: camera for camera in cameras}
    frames_dir = args.out_dir / "frames"
    zones_dir = args.out_dir / "zones"
    frames_dir.mkdir(parents=True, exist_ok=True)
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
            camera_id = path[len(prefix):end]
            return camera_id if camera_id in by_id else None

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send(200, HTML, "text/html; charset=utf-8")
                return
            if path == "/cameras":
                self.send(200, json.dumps([{"id": c["id"], "name": c["name"]} for c in cameras], ensure_ascii=False), "application/json")
                return
            camera_id = self.camera_id("/frame/", ".jpg")
            if camera_id:
                frame = frames_dir / f"{camera_id}.jpg"
                if not frame.exists():
                    try:
                        extract_frame(by_id[camera_id]["url"], frame)
                    except RuntimeError as error:
                        self.send(502, str(error))
                        return
                self.send(200, frame.read_bytes(), "image/jpeg")
                return
            camera_id = self.camera_id("/zones/")
            if camera_id:
                zone_file = zones_dir / f"{camera_id}.json"
                self.send(200, zone_file.read_text() if zone_file.exists() else '{"zones": []}', "application/json")
                return
            self.send_error(404)

        def do_POST(self):
            camera_id = self.camera_id("/refresh/")
            if camera_id:
                try:
                    extract_frame(by_id[camera_id]["url"], frames_dir / f"{camera_id}.jpg")
                except RuntimeError as error:
                    self.send(502, str(error))
                    return
                self.send(200, "화면을 갱신했습니다.")
                return
            camera_id = self.camera_id("/zones/")
            if not camera_id:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    raise ValueError("request is too large")
                payload = validate_zones(json.loads(self.rfile.read(length)))
            except (ValueError, json.JSONDecodeError) as error:
                self.send(400, str(error))
                return
            output = zones_dir / f"{camera_id}.json"
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            self.send(200, f"{by_id[camera_id]['name']} 구역을 저장했습니다.")

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"url=http://{args.host}:{args.port}")
    print(f"cameras={len(cameras)}")
    print(f"zones={zones_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
