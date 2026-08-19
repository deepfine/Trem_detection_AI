from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run multiple GPU face-blur jobs across CUDA devices.")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("result/multi_gpu"))
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--detect-width", type=int, default=960)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    procs = []
    for i, source in enumerate(args.sources):
        name = Path(source).stem or f"stream_{i}"
        device = devices[i % len(devices)]
        stream_dir = args.out_dir / f"{i:02d}_{name}"
        cmd = [
            sys.executable,
            "blur_video_mtcnn_gpu.py",
            "--source",
            source,
            "--out-video",
            str(stream_dir.with_suffix(".mp4")),
            "--out-dir",
            str(stream_dir),
            "--device",
            device,
            "--detect-width",
            str(args.detect_width),
        ]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        procs.append(subprocess.Popen(cmd))
        print(f"started source={source} device={device} pid={procs[-1].pid}")

    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(max(codes))


if __name__ == "__main__":
    main()
