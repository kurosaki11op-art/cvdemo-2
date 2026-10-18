import pytest

pytest.importorskip("esda"); pytest.importorskip("h3"); pytest.importorskip("shap"); pytest.importorskip("osmnx")


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    import os
    os.environ["NEXUS_ALLOW_SYNTHETIC"] = "1"; os.environ["NEXUS_DATA"] = "/nonexistent"
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def test_endpoints(client):
    for u in ["health", "moran", "ablation", "shap", "buildings", "flood", "shap/1", "route/1"]:
        assert client.get("/api/v1/" + u).status_code == 200, u
    assert client.get("/api/v1/route/99999").status_code == 404


def test_moran_and_block_cv(client):
    m = client.get("/api/v1/moran").json(); assert m["p"] < .05
    for s in client.get("/api/v1/ablation").json(): assert s["r2_spatial_block_cv"] <= s["r2_random_cv"] + .05


def test_cors(client):
    r = client.options("/api/v1/moran", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.headers["access-control-allow-origin"] in ("*", "http://localhost:5173")


def test_route_penalty(client):
    from src.api.main import pipe
    assert all(d["weight"] == d["length"] * (1000 if d["blocked"] else 1) for *_, d in pipe().G.edges(keys=True, data=True))
