from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# DM-Count architecture, MIT License: CVLab@StonyBrook (2020).
class DMCount(nn.Module):
    def __init__(self):
        super().__init__()
        config = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
        layers, channels = [], 3
        for value in config:
            if value == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend((nn.Conv2d(channels, value, 3, padding=1), nn.ReLU(inplace=True)))
                channels = value
        self.features = nn.Sequential(*layers)
        self.reg_layer = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, image):
        return self.density_layer(self.reg_layer(F.interpolate(self.features(image), scale_factor=2, mode="bilinear", align_corners=False)))


class CrowdCounter:
    def __init__(self, model_path: Path, device="cuda:0", max_side=2048, tile_size=768):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.max_side = max_side
        self.tile_size = tile_size
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is None:
            model = DMCount()
            model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model = model.to(self.device, dtype=self.dtype).eval()

    def count(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        size = (max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16))
        image = frame if size == (width, height) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)) / torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        self.load()
        with self.lock, torch.inference_mode():
            tensor = tensor.to(self.device, dtype=self.dtype)
            density = np.zeros((tensor.shape[2] // 8, tensor.shape[3] // 8), dtype=np.float32)
            for top in range(0, tensor.shape[2], self.tile_size):
                for left in range(0, tensor.shape[3], self.tile_size):
                    tile = tensor[:, :, top:top + self.tile_size, left:left + self.tile_size]
                    result = self.model(tile)[0, 0].float().cpu().numpy()
                    y, x = top // 8, left // 8
                    density[y:y + result.shape[0], x:x + result.shape[1]] = result
        return round(float(density.sum()))
