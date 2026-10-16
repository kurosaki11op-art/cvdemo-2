import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from nexus.dashboard import make_handler
from nexus.storage import EventStore


@pytest.fixture
def server(cfg, monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    store = EventStore(cfg.storage.db_path)
    now = time.time()
    for _ in range(3):
        store.add_event(now - 5, cfg.camera_id, "IN")
    store.add_event(now - 4, cfg.camera_id, "OUT")
    grid = np.zeros((500, 500), np.float32)
    grid[250, 250] = 50
    store.add_heat(cfg.camera_id, now - 100, now - 1, grid)

    def start(token=None):
        cfg.dashboard.api_token = token
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cfg, store))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"

    servers = []

    def factory(token=None):
        s = start(token)
        servers.append(s[0])
        return s[1]

    yield factory
    for s in servers:
        s.shutdown()
        s.server_close()
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_summary_and_hourly(server):
    base = server()
    code, ctype, body = get(base + "/api/summary")
    data = json.loads(body)
    assert code == 200 and ctype == "application/json"
    assert (data["entries"], data["exits"]) == (3, 1)
    assert data["camera_id"] == "test-cam" and data["healthy"] is False     # no service running
    hours = json.loads(get(base + "/api/hourly")[2])["hours"]
    assert len(hours) == 24 and sum(h["ins"] for h in hours) == 3


def test_page_and_heatmap_png(server):
    base = server()
    code, ctype, body = get(base + "/")
    assert code == 200 and "text/html" in ctype and b"NEXUS" in body
    code, ctype, body = get(base + "/api/heatmap.png?h0=0&h1=24")
    assert code == 200 and ctype == "image/png" and body[:8] == b"\x89PNG\r\n\x1a\n"


def test_heatmap_for_empty_period_still_returns_an_image(server):
    code, _, body = get(server() + "/api/heatmap.png?date=2001-01-01")
    assert code == 200 and body[:4] == b"\x89PNG"


def test_healthz_reflects_service_state(server, cfg):
    base = server()
    assert get(base + "/healthz")[0] == 503
    cfg.status_path.write_text(json.dumps({"state": "running", "ts": time.time(), "occupancy": 2}))
    assert get(base + "/healthz")[0] == 200
    cfg.status_path.write_text(json.dumps({"state": "running", "ts": time.time() - 3600}))
    assert get(base + "/healthz")[0] == 503                                  # stale heartbeat


def test_bad_input_and_unknown_routes(server):
    base = server()
    assert get(base + "/api/summary?date=not-a-date")[0] == 400
    assert get(base + "/api/heatmap.png?h0=abc")[0] == 400
    assert get(base + "/nope")[0] == 404


def test_token_protects_everything_except_healthz(server):
    base = server(token="s3cret")
    assert get(base + "/api/summary")[0] == 401
    assert get(base + "/api/summary?token=wrong")[0] == 401
    assert get(base + "/api/summary?token=s3cret")[0] == 200
    req = urllib.request.Request(base + "/api/summary", headers={"Authorization": "Bearer s3cret"})
    assert urllib.request.urlopen(req, timeout=5).status == 200
    assert get(base + "/healthz")[0] == 503                                  # reachable without token
