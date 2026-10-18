"""Fit the XGBoost triage model on data/buildings.geojson and save data/triage.ubj.
  python -m src.datascience.train_triage [--synthetic]"""
import argparse
import json
from pathlib import Path

import geopandas as gpd

from src.datascience.pipeline_end_to_end import Pipeline, synthetic_demo

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args(); d = Path(a.data); d.mkdir(exist_ok=True)
if a.synthetic:
    p = Pipeline(*synthetic_demo(), synthetic=True)
else:
    fl = d / "flood.geojson"
    p = Pipeline(gpd.read_file(d / "buildings.geojson"), gpd.read_file(fl) if fl.exists() else gpd.GeoDataFrame(geometry=[], crs=4326),
                 roads_path=str(d / "roads.graphml"))
p.save_model(str(d / "triage.ubj"))
(d / "metrics.json").write_text(json.dumps({"moran": p.moran, "ablation": p.ablation, "shap": p.shap_global}, indent=1))
print("saved", d / "triage.ubj", "| Moran I =", round(p.moran["I"], 3), "p =", p.moran["p"])
