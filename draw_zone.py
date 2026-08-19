from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np


def middle_frame(source: str, frame_no: int | None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = frame_no if frame_no is not None else max(0, count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame: {target}")
    return target, frame


def zone_data(mode: str, points):
    if mode == "lines":
        if len(points) != 4:
            raise ValueError("lines mode needs exactly 4 points")
        return {"lines": [[points[0], points[1]], [points[2], points[3]]]}
    if len(points) < 3:
        raise ValueError("mask mode needs at least 3 points")
    return {"mask": points}


def zones_data(zones):
    return {"zones": zones}


def draw_overlay(frame, points, mode):
    image = frame.copy()
    for i, point in enumerate(points):
        cv2.circle(image, point, 5, (0, 0, 255), -1)
        cv2.putText(image, str(i + 1), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if mode == "lines":
        for a, b in [(0, 1), (2, 3)]:
            if len(points) > b:
                cv2.line(image, points[a], points[b], (255, 0, 0), 2)
        if len(points) == 4:
            cv2.polylines(image, [np.array([points[0], points[1], points[3], points[2]], np.int32)], True, (0, 255, 255), 1)
    elif len(points) > 1:
        cv2.polylines(image, [np.array(points, np.int32)], len(points) > 2, (255, 0, 0), 2)
    cv2.putText(image, "click points | Enter save | u undo | r reset | q quit", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return image


def html_editor(frame, frame_no: int, mode: str):
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode frame")
    image = base64.b64encode(encoded).decode("ascii")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Zone editor</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #111; color: #eee; }}
    main {{ display: grid; grid-template-columns: 1fr 360px; gap: 16px; padding: 16px; }}
    canvas {{ max-width: 100%; background: #000; cursor: crosshair; }}
    aside {{ display: grid; gap: 10px; align-content: start; }}
    button, input, select, textarea {{ font: inherit; }}
    button, input, select {{ padding: 8px; }}
    textarea {{ width: 100%; height: 220px; box-sizing: border-box; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  </style>
</head>
<body>
<main>
  <canvas id="canvas"></canvas>
  <aside>
    <select id="mode">
      <option value="zones" {"selected" if mode == "zones" else ""}>zones: polygons</option>
      <option value="lines" {"selected" if mode == "lines" else ""}>lines: click 4 points</option>
      <option value="mask" {"selected" if mode == "mask" else ""}>mask: click 3+ points</option>
    </select>
    <div class="row">
      <select id="level">
        <option value="danger">danger</option>
        <option value="warning">warning</option>
        <option value="safe">safe</option>
      </select>
      <input id="name" value="zone_1">
    </div>
    <button id="addZone">Add zone</button>
    <button id="undo">Undo</button>
    <button id="reset">Reset</button>
    <button id="download">Download JSON</button>
    <textarea id="json" readonly></textarea>
    <div>Frame {frame_no}. Click points on the image. Use the JSON as <code>--zone</code>.</div>
  </aside>
</main>
<script>
const img = new Image();
img.src = "data:image/jpeg;base64,{image}";
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const mode = document.getElementById("mode");
const level = document.getElementById("level");
const nameInput = document.getElementById("name");
const out = document.getElementById("json");
let points = [];
let zones = [];
const colors = {{danger: "#ff3030", warning: "#ffd21f", safe: "#30c060"}};

function data() {{
  const pts = points.map(p => [Math.round(p.x), Math.round(p.y)]);
  if (mode.value === "zones") return {{zones}};
  if (mode.value === "lines") return pts.length === 4 ? {{lines: [[pts[0], pts[1]], [pts[2], pts[3]]]}} : {{lines: []}};
  return {{mask: pts}};
}}

function draw() {{
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = 3;
  zones.forEach(z => {{
    ctx.beginPath(); ctx.moveTo(z.points[0][0], z.points[0][1]);
    z.points.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = colors[z.level] + "66";
    ctx.strokeStyle = colors[z.level];
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = colors[z.level];
    ctx.fillText(`${{z.level}} ${{z.name}}`, z.points[0][0], Math.max(20, z.points[0][1] - 8));
  }});
  ctx.strokeStyle = mode.value === "zones" ? colors[level.value] : "#00a2ff";
  ctx.fillStyle = "#ff3333";
  points.forEach((p, i) => {{
    ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillText(String(i + 1), p.x + 8, p.y - 8);
  }});
  if (mode.value === "lines") {{
    [[0, 1], [2, 3]].forEach(pair => {{
      if (points[pair[1]]) {{
        ctx.beginPath(); ctx.moveTo(points[pair[0]].x, points[pair[0]].y); ctx.lineTo(points[pair[1]].x, points[pair[1]].y); ctx.stroke();
      }}
    }});
  }} else if (points.length > 1) {{
    ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
    if (points.length > 2) ctx.closePath();
    ctx.stroke();
  }}
  out.value = JSON.stringify(data(), null, 2);
}}

img.onload = () => {{ canvas.width = img.width; canvas.height = img.height; draw(); }};
canvas.onclick = e => {{
  const r = canvas.getBoundingClientRect();
  if (mode.value === "mask" || mode.value === "zones" || points.length < 4) points.push({{x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height}});
  draw();
}};
mode.onchange = () => {{ points = []; draw(); }};
document.getElementById("addZone").onclick = () => {{
  if (mode.value !== "zones" || points.length < 3) return;
  zones.push({{name: nameInput.value || `${{level.value}}_${{zones.length + 1}}`, level: level.value, points: points.map(p => [Math.round(p.x), Math.round(p.y)])}});
  points = [];
  nameInput.value = `zone_${{zones.length + 1}}`;
  draw();
}};
document.getElementById("undo").onclick = () => {{ points.pop(); draw(); }};
document.getElementById("reset").onclick = () => {{ points = []; if (mode.value === "zones") zones = []; draw(); }};
document.getElementById("download").onclick = () => {{
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out.value + "\\n"], {{type: "application/json"}}));
  a.download = mode.value === "zones" ? "zones.json" : mode.value === "lines" ? "entry_lines.json" : "mask_zone.json";
  a.click();
}};
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Draw mask polygon or two entry lines on the middle video frame.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--out", type=Path, default=Path("zone.json"))
    parser.add_argument("--mode", choices=["zones", "mask", "lines"], default="zones")
    parser.add_argument("--frame", type=int, help="frame number to show; default is video middle")
    parser.add_argument("--html", type=Path, help="write a browser editor instead of opening an OpenCV window")
    args = parser.parse_args()

    frame_no, frame = middle_frame(args.source, args.frame)
    if args.html:
        args.html.write_text(html_editor(frame, frame_no, args.mode))
        print(f"saved {args.html}")
        return

    points = []
    window = f"draw {args.mode} on frame {frame_no}"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if args.mode == "mask" or len(points) < 4:
                points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, draw_overlay(frame, points, args.mode))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            args.out.write_text(json.dumps(zone_data(args.mode, points), indent=2) + "\n")
            print(f"saved {args.out}")
            break
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
