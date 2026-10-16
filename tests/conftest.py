import numpy as np
import pytest

from nexus.config import default_config
from nexus.engine import Detections


class FakeDetector:
    """Replays scripted detections frame by frame (no YOLO needed)."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = 0
        self.resets = 0

    def track(self, frame):
        i, self.calls = self.calls, self.calls + 1
        return self.script[i] if i < len(self.script) else Detections()

    def reset(self):
        self.resets += 1


def det(*people):
    """det((id, x_center, y_feet), ...) -> Detections with 40x150 boxes standing at those points."""
    boxes, ids = [], []
    for tid, x, y in people:
        boxes.append([x - 20, y - 150, x + 20, y])
        ids.append(tid)
    return Detections(np.array(boxes, dtype=float).reshape(-1, 4), np.array(ids, dtype=int),
                      np.full(len(ids), 0.9))


def walk(tid, x, y0, y1, step=10):
    """Detections for one person walking straight down/up from y0 to y1."""
    ys = range(y0, y1 + (1 if y1 >= y0 else -1), step if y1 >= y0 else -step)
    return [det((tid, x, y)) for y in ys]


@pytest.fixture
def frame():
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[:] = (90, 120, 160)
    return f


@pytest.fixture
def cfg(tmp_path):
    c = default_config()
    c.storage.db_path = str(tmp_path / "data" / "nexus.db")
    c.detection.min_frames = 1
    c.camera_id = "test-cam"
    return c.validate()
