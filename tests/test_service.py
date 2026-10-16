import json
import time

from nexus.service import Service
from nexus.sources import FrameSource, SourceError
from nexus.storage import EventStore

from .conftest import FakeDetector, walk


class ListSource:
    """Scripted source. None entries simulate dropouts on a live camera."""

    def __init__(self, items, live=False, reopen_ok=True, fail_open=False):
        self.items, self.live, self.reopen_ok, self.fail_open = list(items), live, reopen_ok, fail_open
        self.finished = False
        self.reopens = 0
        self.released = False
        self.is_live = live

    def open(self):
        if self.fail_open:
            raise SourceError("boom")
        return self

    def read(self):
        if not self.items:
            self.finished = not self.live
            return None
        return self.items.pop(0)

    def reopen(self):
        self.reopens += 1
        return self.reopen_ok

    def consume_wrap(self):
        return False

    def release(self):
        self.released = True


def make(cfg, frame, script, items=None, **kw):
    from nexus.engine import CountingEngine
    cfg.service.heartbeat_s = 0.0
    cfg.service.reconnect_delay_s = 0.0
    store = EventStore(cfg.storage.db_path)
    engine = CountingEngine(cfg, FakeDetector(script))
    source = ListSource(items if items is not None else [frame] * len(script), **kw)
    return Service(cfg, engine=engine, source=source, store=store), store, source


def test_service_counts_persists_and_writes_status(cfg, frame):
    svc, store, source = make(cfg, frame, walk(1, 320, 200, 440) + walk(2, 320, 440, 200))
    assert svc.run() == 0
    now = time.time()
    assert store.counts("test-cam", now - 60, now + 60) == (1, 1)
    assert svc.occupancy.value == 0
    status = json.loads(cfg.status_path.read_text())
    assert status["state"] == "stopped" and status["camera_id"] == "test-cam"
    assert store.heat_sum("test-cam", 0, now + 60) is not None      # final heat flush on shutdown
    assert source.released


def test_occupancy_tracks_entries(cfg, frame):
    svc, _, _ = make(cfg, frame, walk(1, 320, 200, 440))
    svc.run()
    assert svc.occupancy.value == 1


def test_open_failure_returns_error_and_status(cfg, frame):
    svc, _, _ = make(cfg, frame, [], fail_open=True)
    assert svc.run() == 1
    assert json.loads(cfg.status_path.read_text())["state"] == "error"


def test_live_dropout_triggers_reconnect_and_tracking_reset(cfg, frame):
    script = walk(1, 320, 200, 440)
    items = [frame] * 5 + [None, None] + [frame] * (len(script) - 5) + ["STOP"]
    svc, _, source = make(cfg, frame, script, items=items, live=True)
    original_read = source.read

    def read():
        item = original_read()
        if isinstance(item, str):               # sentinel: end the (otherwise endless) live run
            svc.stop()
            return None
        return item

    source.read = read
    svc.run()
    assert source.reopens == 2
    assert svc.engine.detector.resets == 2


def test_crash_in_loop_is_reported_not_swallowed(cfg, frame):
    svc, _, _ = make(cfg, frame, [], items=[frame])
    svc.engine.process = lambda f: (_ for _ in ()).throw(RuntimeError("model exploded"))
    assert svc.run() == 1
    assert json.loads(cfg.status_path.read_text())["state"] == "error"


def test_real_file_source_end_to_end(cfg, tmp_path):
    import cv2
    import numpy as np

    from nexus.engine import CountingEngine
    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for _ in range(30):
        vw.write(np.full((480, 640, 3), 120, np.uint8))
    vw.release()
    cfg.source = str(video)
    cfg.service.heartbeat_s = 0.0
    store = EventStore(cfg.storage.db_path)
    svc = Service(cfg, engine=CountingEngine(cfg, FakeDetector(walk(1, 320, 200, 440))),
                  source=FrameSource(str(video)), store=store)
    assert svc.run() == 0
    assert svc.engine.entries == 1 and svc.frames == 30
