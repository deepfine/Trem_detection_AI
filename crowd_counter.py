from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VGG19_CONFIG = [
    64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
    512, 512, 512, 512, "M", 512, 512, 512, 512,
]


# DM-Count architecture, MIT License: CVLab@StonyBrook (2020).
class DMCount(nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for value in VGG19_CONFIG:
            if value == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend((nn.Conv2d(channels, value, 3, padding=1), nn.ReLU(inplace=True)))
                channels = value
        self.features = nn.Sequential(*layers)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, image):
        features = F.interpolate(self.features(image), scale_factor=2, mode="bilinear", align_corners=False)
        return self.density_layer(self.reg_layer(features))


def _resize_for_inference(frame, max_side):
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16))
    return frame if size == (width, height) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _normalized_tensor(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _infer_tiled(model, tensor, tile_size):
    density = np.zeros((tensor.shape[2] // 8, tensor.shape[3] // 8), dtype=np.float32)
    for top in range(0, tensor.shape[2], tile_size):
        for left in range(0, tensor.shape[3], tile_size):
            result = model(tensor[:, :, top:top + tile_size, left:left + tile_size])[0, 0].float().cpu().numpy()
            y, x = top // 8, left // 8
            density[y:y + result.shape[0], x:x + result.shape[1]] = result
    return density


class CrowdCounter:
    def __init__(self, model_path: Path, device="cuda:0", max_side=2048, tile_size=768):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.max_side = max_side
        self.tile_size = tile_size
        self.model = None

    def load(self):
        if self.model is not None:
            return
        model = DMCount()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model = model.to(self.device, dtype=self.dtype).eval()

    def count(self, frame):
        self.load()
        tensor = _normalized_tensor(_resize_for_inference(frame, self.max_side)).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            return max(0, round(float(_infer_tiled(self.model, tensor, self.tile_size).sum())))


def evaluate_video(source, counter, sample_fps, out_dir, max_frames=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if not 0 < sample_fps <= source_fps:
        raise ValueError(f"sample fps must be in (0, {source_fps}], got {sample_fps}")
    out_dir.mkdir(parents=True, exist_ok=True)
    counter.load()
    source_frame = samples = 0
    next_sample_seconds = 0.0
    counts, inference_times = [], []
    with (out_dir / "frame_counts.csv").open("w", newline="") as output:
        rows = csv.writer(output)
        rows.writerow(["source_frame", "source_seconds", "estimated_people", "inference_seconds"])
        while max_frames is None or samples < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_seconds = source_frame / source_fps
            source_frame += 1
            if frame_seconds + 0.5 / source_fps < next_sample_seconds:
                continue
            next_sample_seconds += 1 / sample_fps
            started = time.perf_counter()
            count = counter.count(frame)
            elapsed = time.perf_counter() - started
            counts.append(count)
            inference_times.append(elapsed)
            rows.writerow([source_frame - 1, f"{frame_seconds:.3f}", count, f"{elapsed:.6f}"])
            samples += 1
    cap.release()
    if not counts:
        raise RuntimeError("source has no frames")
    ordered_times = sorted(inference_times)
    p95 = ordered_times[max(0, round(len(ordered_times) * 0.95) - 1)]
    summary = (
        f"source={source}\n"
        f"model={counter.model_path}\n"
        f"method=dm_count_density_estimation\n"
        f"source_fps={source_fps:.1f}\n"
        f"sample_fps={sample_fps:.1f}\n"
        f"samples={samples}\n"
        f"average_people={sum(counts) / len(counts):.2f}\n"
        f"min_people={min(counts)}\n"
        f"max_people={max(counts)}\n"
        f"average_inference_ms={sum(inference_times) / len(inference_times) * 1000:.2f}\n"
        f"p95_inference_ms={p95 * 1000:.2f}\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate DM-Count on sampled video frames without YOLO person counting.")
    parser.add_argument("--source", default="test_video.mkv")
    parser.add_argument("--model", type=Path, default=Path("result/models/dm_count_qnrf.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, default=Path("result/crowd_evaluation"))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    print(evaluate_video(args.source, CrowdCounter(args.model, args.device, args.max_side), args.sample_fps, args.out_dir, args.max_frames))


if __name__ == "__main__":
    main()
