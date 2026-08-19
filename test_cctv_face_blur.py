import queue

import cv2
import numpy as np

from cctv_face_blur import blur_faces, jpeg_bytes, put_latest
from eval_roboflow_cctv_face import score


class FakeDetector:
    def detectMultiScale(self, gray, scale_factor, min_neighbors):
        return [(10, 10, 40, 40)]


def test_blur_and_encode_and_drop_old_frame():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (50, 50), (255, 255, 255), -1)
    cv2.line(frame, (10, 10), (50, 50), (0, 0, 0), 2)

    blurred = blur_faces(frame.copy(), FakeDetector())
    assert not np.array_equal(frame[10:50, 10:50], blurred[10:50, 10:50])
    assert jpeg_bytes(blurred, 80).startswith(b"\xff\xd8")

    frames = queue.Queue(maxsize=1)
    put_latest(frames, (1, b"old"))
    put_latest(frames, (2, b"new"))
    assert frames.get_nowait() == (2, b"new")


def test_score_matches_once_by_iou():
    preds = [(0, 0, 10, 10), (50, 50, 60, 60)]
    truths = [(1, 1, 11, 11)]
    assert score(preds, truths, 0.5) == (1, 1, 0)
