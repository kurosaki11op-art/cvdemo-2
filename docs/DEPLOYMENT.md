# Deployment

> Status: the headless service and dashboard are tested end to end on a recorded video in CI-style tests and a
> manual run. **The Dockerfile, compose file, systemd units and the hardware figures below are written to
> standard practice but have not been run on target hardware**; treat the first install as a pilot.

## Choosing hardware
| Option | When | Notes |
|---|---|---|
| Mini PC / NUC (CPU) | 1-2 cameras, `yolov8n` | The service throttles live sources to `max_fps` (default 15); 10-15 fps is enough for doorway counting |
| NVIDIA Jetson | Edge box per store | Export with `nexus export-model --format engine --half` (TensorRT); video never leaves the store |
| GPU server | Many cameras | One service per camera; keep `device: "0"` |

Measure your own throughput: the service reports `fps` in `/api/summary` and `status-<camera>.json`.

## Docker
```bash
cp configs/docker.yaml configs/nexus.yaml     # edit source, db_path stays /data/nexus.db, set api_token
docker compose up -d --build
docker compose logs -f counter
```
Weights download on first start into the `/data` volume. The counter container has a healthcheck
(`nexus healthcheck`); `docker compose ps` shows `healthy`/`unhealthy`.

## systemd (bare metal)
```bash
sudo useradd --system --home /opt/nexus nexus
python3 -m venv /opt/nexus/venv && /opt/nexus/venv/bin/pip install /path/to/repo
sudo install -d -o nexus /var/lib/nexus /etc/nexus      # set storage.db_path: /var/lib/nexus/nexus.db
sudo cp deploy/nexus-*.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now nexus-counter nexus-dashboard
```

## Cameras
- Use the camera's **sub-stream** (640x480-ish) where available: it is cheaper and enough for counting.
- RTSP uses TCP by default (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`). The service reconnects automatically after dropouts.
- Run `nexus calibrate` on a frame from the same stream the service will read.
- Several cameras: one config (unique `camera_id` and `db_path`, or a shared `db_path`) and one `nexus run` process each.

## Exposing the dashboard
Keep `dashboard.host: 127.0.0.1` and terminate TLS in a reverse proxy:
```nginx
server {
  listen 443 ssl;  server_name footfall.example.com;
  # ssl_certificate ... ;  add real authentication here (basic auth / SSO)
  location / { proxy_pass http://127.0.0.1:8080; proxy_set_header Host $host; }
}
```
`/healthz` returns 200 only while the counter's heartbeat is fresh: point your uptime monitor at it.

## Operations
- **Backups:** `sqlite3 data/nexus.db ".backup 'backup.db'"` (safe while running).
- **Retention:** automatic per `storage.retention_days`; `nexus purge` runs it on demand.
- **Logs:** `--log-json` emits one JSON object per line for Loki/ELK/CloudWatch.
- **Occupancy drift:** miscounts accumulate, so occupancy resets nightly (`service.occupancy_reset`).
- **Upgrades:** stop the counter, deploy, start; the SQLite schema is created idempotently.
