import datetime as dt
import json

import cv2
import numpy as np
import pytest

from nexus import analytics as an
from nexus import demo_data
from nexus.calibrate import guess_kind, zones_with_polygons
from nexus.config import default_config, sample_zones
from nexus.engine import CountingEngine
from nexus.service import Service
from nexus.storage import EventStore

from .conftest import FakeDetector, det
from .test_service import ListSource


@pytest.fixture
def zcfg(cfg):
    cfg.zones = sample_zones()
    cfg.analytics.session_timeout_s = 1.0
    return cfg.validate()


def test_demo_data_is_deterministic_labelled_and_plausible(tmp_path, zcfg):
    store = EventStore(tmp_path / "d.db")
    n1 = demo_data.generate(store, zcfg, days=3, seed=11, end_day=dt.date(2026, 3, 4))
    t0, t1 = an.period_bounds("2026-03-04", 3)
    rep = an.build_report(store, zcfg, t0, t1)
    assert rep["demo_data"] is True and rep["overview"]["visitors"] == n1["sessions"] > 100
    areas = [z for z in rep["zones"] if z["kind"] == "area"]
    assert len(areas) == 3 and areas[0]["visitors"] > areas[-1]["visitors"]        # a clear winner and a laggard
    assert 0.05 < rep["overview"]["bounce_rate"] < 0.35
    assert max(rep["hourly"]["arrivals"][:9]) == 0                                 # nobody before opening
    assert rep["heat_share"] and sum(rep["heat_share"].values()) <= 1.0001
    other = EventStore(tmp_path / "e.db")
    demo_data.generate(other, zcfg, days=3, seed=11, end_day=dt.date(2026, 3, 4))
    assert an.build_report(other, zcfg, t0, t1)["overview"]["visitors"] == n1["sessions"]


def test_demo_data_needs_zones(tmp_path, cfg):
    with pytest.raises(ValueError, match="zones"):
        demo_data.generate(EventStore(tmp_path / "x.db"), cfg, days=1)


def floor_to_cam(engine, fx, fy):
    inv = np.linalg.inv(engine.mapper.matrix)
    x, y = cv2.perspectiveTransform(np.array([[[fx, fy]]], np.float32), inv)[0, 0]
    return float(x), float(y)


def test_service_turns_standing_in_a_zone_into_a_stored_session(zcfg, frame):
    engine = CountingEngine(zcfg, FakeDetector())
    cam_a = floor_to_cam(engine, 110, 120)                       # inside "Display A"
    cam_c = floor_to_cam(engine, 420, 340)                       # inside "Checkout"
    script = [det((1, *cam_a))] * 40 + [det((1, *cam_c))] * 30
    engine.detector.script = script
    ticks = iter(range(10_000))
    zcfg.service.heartbeat_s = 1e9
    svc = Service(zcfg, engine=engine, source=ListSource([frame] * len(script)),
                  store=EventStore(zcfg.storage.db_path), clock=lambda: next(ticks) * 0.1)
    assert svc.run() == 0
    (rec,) = svc.store.sessions(zcfg.camera_id, 0, 1e9)
    assert rec.path == ["Display A", "Checkout"]
    assert rec.visits[0].dwell_s > 2.0 and rec.dwell_s > 5.0
    rep = an.build_report(svc.store, zcfg, 0, 1e9)
    assert rep["overview"]["checkout_rate"] == 1.0
    status = json.loads(zcfg.status_path.read_text())
    assert set(status["zone_counts"]) == {z.name for z in sample_zones()}


def test_dashboard_analytics_endpoint_and_zone_heatmap(tmp_path, zcfg):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from nexus.dashboard import make_handler
    store = EventStore(zcfg.storage.db_path)
    demo_data.generate(store, zcfg, days=2, seed=3)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(zcfg, store))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        data = json.loads(urllib.request.urlopen(base + "/api/analytics?days=2", timeout=10).read())
        assert data["overview"]["visitors"] > 50 and data["days"] == 2 and data["demo_data"] is True
        assert {z["name"] for z in data["zone_polygons"]} == {z.name for z in sample_zones()}
        assert data["insights"] and data["funnel"][0]["step"] == "Visited the store"
        png = urllib.request.urlopen(base + "/api/heatmap.png?days=2", timeout=10).read()
        plain = urllib.request.urlopen(base + "/api/heatmap.png?days=2&zones=0", timeout=10).read()
        assert png[:4] == b"\x89PNG" and png != plain                       # zone outlines were drawn
        page = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        assert "store analytics" in page.lower() and "api/analytics" in page
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_zone_helpers():
    assert guess_kind("Main Entrance") == "entrance" and guess_kind("Cash desk") == "checkout"
    assert guess_kind("Shoes") == "area"
    sq = [(0, 0), (50, 0), (50, 50), (0, 50)]
    cfg = zones_with_polygons(default_config(), [("Door", sq), ("Shoes", sq[:3])])
    assert [(z.name, z.kind) for z in cfg.zones] == [("Door", "entrance"), ("Shoes", "area")]
