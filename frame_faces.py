from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ScrfdFaceDetector:
    def __init__(self, model: Path, device_id=0, input_size=(960, 544), threshold=0.35, nms=0.4):
        self.model = Path(model)
        self.device_id = device_id
        self.input_size = input_size
        self.threshold = threshold
        self.nms = nms
        self.session = None

    def load(self):
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": self.device_id}), "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(f"CUDA provider unavailable: {self.session.get_providers()}")
        self.detect(np.zeros((self.input_size[1], self.input_size[0], 3), np.uint8))

    def detect(self, frame):
        if self.session is None:
            raise RuntimeError("face detector is not loaded")
        input_width, input_height = self.input_size
        height, width = frame.shape[:2]
        scale = min(input_width / width, input_height / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        image = np.zeros((input_height, input_width, 3), np.uint8)
        image[: resized.shape[0], : resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(image, 1 / 128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})

        boxes, scores = [], []
        for index, stride in enumerate((8, 16, 32)):
            score = outputs[index].reshape(-1)
            distance = outputs[index + 3] * stride
            grid = np.stack(np.mgrid[: input_height // stride, : input_width // stride][::-1], axis=-1).astype(np.float32)
            centers = np.repeat((grid * stride).reshape(-1, 2), 2, axis=0)
            for row in np.where(score >= self.threshold)[0]:
                x, y = centers[row]
                left, top, right, bottom = distance[row]
                x1, y1 = max(0, int((x - left) / scale)), max(0, int((y - top) / scale))
                x2, y2 = min(width, int((x + right) / scale)), min(height, int((y + bottom) / scale))
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(score[row]))

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.nms)
        return [
            {
                "confidence": round(scores[index], 4),
                "bbox": [boxes[index][0], boxes[index][1], boxes[index][0] + boxes[index][2], boxes[index][1] + boxes[index][3]],
            }
            for index in np.array(keep).reshape(-1) if len(keep)
        ]
