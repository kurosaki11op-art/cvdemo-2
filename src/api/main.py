"""FastAPI backend. Run: uvicorn src.api.main:app --port 8000 (from repo root)."""
import os
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.datascience.pipeline_end_to_end import Pipeline, synthetic_demo

DATA = Path(os.environ.get("NEXUS_DATA", "data"))
app = FastAPI(title="Post-Disaster Triage API")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def pipe() -> Pipeline:
    bf = DATA / "buildings.geojson"
    if bf.exists():
        fl = DATA / "flood.geojson"
        return Pipeline(gpd.read_file(bf), gpd.read_file(fl) if fl.exists() else gpd.GeoDataFrame(geometry=[], crs=4326),
                        roads_path=str(DATA / "roads.graphml"))
    if os.environ.get("NEXUS_ALLOW_SYNTHETIC") == "1":
        return Pipeline(*synthetic_demo(), synthetic=True)
    raise HTTPException(503, "No data/buildings.geojson. Run src.geoai.download_stac_and_xbd first.")


@app.get("/api/v1/health")
def health():
    return {"ok": True, "data_present": (DATA / "buildings.geojson").exists(),
            "synthetic": os.environ.get("NEXUS_ALLOW_SYNTHETIC") == "1" and not (DATA / "buildings.geojson").exists()}


@app.get("/api/v1/buildings")
def buildings(): return pipe().buildings_geojson()


@app.get("/api/v1/flood")
def flood(): return pipe().flood_geojson()


@app.get("/api/v1/moran")
def moran(): return pipe().moran


@app.get("/api/v1/ablation")
def ablation(): return pipe().ablation


@app.get("/api/v1/shap")
def shap_global(): return pipe().shap_global


@app.get("/api/v1/shap/{building_id}")
def shap_building(building_id: int):
    p = pipe()
    if not 0 <= building_id < len(p.b): raise HTTPException(404, "unknown building")
    return p.shap_for(building_id)


@app.get("/api/v1/route/{building_id}")
def route(building_id: int):
    p = pipe()
    if not 0 <= building_id < len(p.b): raise HTTPException(404, "unknown building")
    return p.route(building_id)
