# Architecture

| Module | Responsibility |
|---|---|
| `nexus/config.py` | Typed YAML config, strict validation with actionable errors (typos, wrong corner order, out-of-frame line) |
| `nexus/sources.py` | `FrameSource`: file / webcam (Windows backend fallback, black-frame rejection) / RTSP (TCP), looping, reconnect |
| `nexus/engine.py` | `CountingEngine`: detector -> duplicate removal -> identity filter -> tripwire + homography -> heatmaps. No I/O. `Detector` is a protocol; `YoloDetector` wraps ultralytics (imported lazily) |
| `nexus/identity.py` | Confirmation (`min_frames`), reflection/twin suppression, ID-switch repair, bounded memory |
| `nexus/geometry.py` | `TripwireCounter` (cross-product, deadband, segment check, re-arm) and `FloorMapper` (homography) |
| `nexus/heatmap.py` | Gaussian accumulation + JET rendering |
| `nexus/storage.py` | SQLite (WAL): events, zlib-compressed heatmap deltas, key/value; retention purge |
| `nexus/occupancy.py` | Persisted occupancy with nightly reset |
| `nexus/service.py` | Headless loop: throttle, persist, snapshots, atomic status file, reconnect, signals, JSON logs |
| `nexus/dashboard.py` | Stdlib HTTP server: page, JSON API, heatmap PNG, `/healthz`, optional token |
| `nexus/evaluate.py`, `label.py` | Ground-truth labelling and accuracy metrics |
| `nexus/calibrate.py` | Interactive calibration, floor-aspect-correct canvas |
| `nexus/ui.py`, `cli.py` | Demo window and command line |

Design choices
- **Engine has no side effects**, so it is unit-tested with scripted detections (no model, no camera) and reused by the demo, the service and the evaluator.
- **The service and dashboard are separate processes** sharing a SQLite file (WAL) and a status JSON; the dashboard can restart without interrupting counting.
- **One config + one service per camera**; `camera_id` partitions the data.
- **Counting semantics**: a crossing is a side flip of the cross product, outside a deadband, along the segment; an ID is blocked from counting again until it has moved `rearm_px` away, and while blocked its side is frozen (wobbling on the line counts once).
