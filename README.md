# NEXUS Vision Node: retail footfall counter + 2D spatial heatmap

Counts people entering and leaving through a doorway (YOLOv8 + ByteTrack), maps where they
stand onto a top-down floor plan (homography), builds a dwell-time heatmap, stores anonymous
events, and serves a dashboard. Runs on a video file, a webcam or an RTSP IP camera.

```
camera --> FrameSource --> CountingEngine --> EventStore (SQLite) --> Dashboard / API
                           detector + tracker   events, heatmap deltas,
                           identity filter      occupancy
                           tripwire + homography
                           heatmap
```

## Quick start (demo window)
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python footfall_heatmap.py --source 0 --webcam-demo --conf 0.6    # desk test: lean right = IN
python footfall_heatmap.py --source store.mp4 --loop              # video file
```
Keys: `q` quit, `r` reset. (`footfall_heatmap.py` is a launcher for `python -m nexus demo`.)

## Production workflow
```bash
nexus init-config --out configs/nexus.yaml --camera-id entrance-1 --source rtsp://user:pass@cam/stream
nexus calibrate   --config configs/nexus.yaml --source rtsp://user:pass@cam/stream --floor-size 6x3
nexus run         --config configs/nexus.yaml          # headless service (24/7)
nexus dashboard   --config configs/nexus.yaml          # http://127.0.0.1:8080
```
Install with `pip install .` to get the `nexus` command (or use `python -m nexus ...`).

| Command | Purpose |
|---|---|
| `demo` | interactive dual-pane window |
| `run` | headless service: counts, stores, heartbeat, auto-reconnect |
| `dashboard` | web UI + JSON API (`/api/summary`, `/api/hourly`, `/api/heatmap.png`, `/healthz`) |
| `calibrate` | click floor corners + tripwire; writes the YAML; `--floor-size 6x3` keeps the map to scale |
| `label` / `evaluate` | hand-label a clip, then measure accuracy (precision/recall/count error) |
| `export-model` | export weights to ONNX / TensorRT / OpenVINO |
| `healthcheck` | exit code for Docker/Kubernetes probes |
| `purge` | apply the retention policy now |

Everything is configured in one YAML file ([`configs/example.yaml`](configs/example.yaml) documents every key).

## Docs
- [docs/ACCURACY.md](docs/ACCURACY.md): how to measure real-world accuracy (do this before trusting any number)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): Docker, systemd, hardware, RTSP, TLS, backups
- [docs/PRIVACY.md](docs/PRIVACY.md): what is (not) stored, retention, signage template
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): modules and data flow

## Development
```bash
pip install -r requirements-dev.txt
ruff check . && python -m pytest -q        # no GPU/YOLO needed: tests use a scripted fake detector
```
`legacy/` holds the original single-file scripts for reference.

Current Occupancy = initial occupancy + entries - exits (never below zero), reset nightly.
