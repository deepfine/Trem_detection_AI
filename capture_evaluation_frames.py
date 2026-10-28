"""Preserve newly captured JPEGs before analysis overlays; never modify source frames."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from camera_zones import parse_camera_folder


def preserve_frame(source: Path, source_root: Path, output: Path):
    relative = source.relative_to(source_root)
    camera = parse_camera_folder(relative.parts[0])
    if camera is None:
        return None
    before = source.stat()
    data = source.read_bytes()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return None
    if len(data) != before.st_size or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        return None
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as f:
        f.write(data)
    return {"camera_id": camera[0], "file": str(relative), "source": str(source),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "source_mtime": before.st_mtime, "copied_at": datetime.now().isoformat(),
            "stage": "before_analysis_overlay"}


def capture(source_root: Path, output: Path, seconds: float):
    source_root, output = source_root.resolve(), output.resolve()
    if not source_root.is_dir():
        raise ValueError("frames root must be an existing directory")
    if output == source_root or source_root in output.parents:
        raise ValueError("output must be outside the live frames root")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    start_wall, start = time.time(), time.monotonic()
    seen, records = set(), []
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        while time.monotonic() - start < seconds:
            for source in sorted(source_root.glob("*/*/*/*/*.jpg")):
                if source in seen:
                    continue
                try:
                    if source.stat().st_mtime < start_wall:
                        continue
                    record = preserve_frame(source, source_root, output)
                except FileNotFoundError:
                    continue  # Live retention can remove a frame before it is read.
                if record:
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                    records.append(record)
                    seen.add(source)
            time.sleep(0.1)
    summary = {"source_root": str(source_root), "seconds": seconds,
               "frames": len(records), "bytes": sum(r["bytes"] for r in records),
               "per_camera": {str(c): sum(r["camera_id"] == c for r in records)
                              for c in sorted({r["camera_id"] for r in records})},
               "labels": None, "accuracy": None,
               "purpose": "raw inputs for model comparison; no GT inferred from production predictions"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not records:
        raise RuntimeError("no new raw frames captured")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    capture(args.source_root, args.output, args.seconds)


if __name__ == "__main__":
    main()
