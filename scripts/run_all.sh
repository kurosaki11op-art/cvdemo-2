#!/usr/bin/env bash
# Full pipeline. Usage: scripts/run_all.sh [--demo]   (--demo = offline synthetic smoke run, NOT real data)
set -euo pipefail; cd "$(dirname "$0")/.."
pip install -r requirements.txt
if [[ "${1:-}" == "--demo" ]]; then
  python -m src.datascience.train_triage --synthetic
  export NEXUS_ALLOW_SYNTHETIC=1
else
  # Dhaka floods Aug 2024 - edit bbox/dates for your event
  python -m src.geoai.download_stac_and_xbd --bbox 90.30 23.70 90.45 23.85 --pre 2024-07-01/2024-07-15 --post 2024-08-10/2024-08-25
  # buildings need a damage_score column (xBD labels / U-Net output). Without labels, proxy from flood overlap:
  python - <<'P'
import geopandas as gpd
g = gpd.read_file("data/buildings.geojson")
if "damage_score" not in g: g["damage_score"] = (4 * g.flood_frac).clip(0, 4); print("WARNING: damage_score is a flood-overlap PROXY, not ground truth")
g.to_file("data/buildings.geojson", driver="GeoJSON")
P
  python -m src.datascience.train_triage
  [[ -d data/xbd ]] && python -m src.geoai.train --root data/xbd --epochs 20 || echo "skip U-Net training: put xBD tiles in data/xbd"
fi
python -m pytest -q --noconftest tests/test_geoai_pipeline.py
(cd web/geoai && python -m http.server 5173 &) ; sleep 1
echo "Dashboard: http://localhost:5173/?api=http://localhost:8000/api/v1"
uvicorn src.api.main:app --port 8000
