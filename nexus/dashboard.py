"""Small dependency-free web dashboard and JSON API (stdlib http.server).

Routes: /  (page) | /api/summary | /api/hourly | /api/heatmap.png | /healthz
Bind to 127.0.0.1 and put a reverse proxy (TLS, auth) in front for anything but a LAN demo;
`dashboard.api_token` adds a simple shared-secret check.
"""

from __future__ import annotations

import datetime as dt
import hmac
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .config import Config
from .heatmap import render_heat
from .storage import EventStore

log = logging.getLogger("nexus.dashboard")


def day_bounds(day: Optional[str]) -> Tuple[float, float]:
    """Local-midnight epoch bounds [start, end) for YYYY-MM-DD (default: today)."""
    d = dt.date.fromisoformat(day) if day else dt.date.today()
    start = dt.datetime.combine(d, dt.time.min)
    return start.timestamp(), (start + dt.timedelta(days=1)).timestamp()


def read_status(cfg: Config) -> Dict[str, Any]:
    try:
        return json.loads(cfg.status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "unknown"}


def is_healthy(cfg: Config, now: Optional[float] = None) -> bool:
    """Service heartbeat is fresh and the loop is running."""
    status = read_status(cfg)
    now = now if now is not None else time.time()
    limit = max(30.0, 4 * cfg.service.heartbeat_s)
    return status.get("state") == "running" and now - float(status.get("ts", 0)) < limit


def build_summary(cfg: Config, store: EventStore, day: Optional[str] = None) -> Dict[str, Any]:
    t0, t1 = day_bounds(day)
    ins, outs = store.counts(cfg.camera_id, t0, t1)
    hourly = store.hourly(cfg.camera_id, t0, t1)
    peak = max(hourly, key=lambda b: b["ins"], default=None)
    status = read_status(cfg)
    return {"camera_id": cfg.camera_id, "date": dt.date.fromtimestamp(t0).isoformat(),
            "entries": ins, "exits": outs, "occupancy": status.get("occupancy"),
            "peak_hour": peak["hour"][-5:] if peak and peak["ins"] else None,
            "peak_hour_entries": peak["ins"] if peak else 0,
            "service_state": status.get("state"), "healthy": is_healthy(cfg),
            "fps": status.get("fps")}


def build_hourly(cfg: Config, store: EventStore, day: Optional[str] = None) -> Dict[str, Any]:
    t0, t1 = day_bounds(day)
    by_hour = {b["hour"][-5:]: b for b in store.hourly(cfg.camera_id, t0, t1)}
    rows = []
    for h in range(24):
        key = f"{h:02d}:00"
        b = by_hour.get(key, {})
        rows.append({"hour": key, "ins": b.get("ins", 0), "outs": b.get("outs", 0)})
    return {"date": dt.date.fromtimestamp(t0).isoformat(), "hours": rows}


def build_heatmap_png(cfg: Config, store: EventStore, day: Optional[str], h0: int, h1: int) -> bytes:
    t0, _ = day_bounds(day)
    grid = store.heat_sum(cfg.camera_id, t0 + h0 * 3600, t0 + h1 * 3600)
    if grid is None:
        w, h = cfg.homography.canvas
        grid = np.zeros((h, w), dtype=np.float32)
        img = render_heat(grid)
        cv2.putText(img, "no data for this period", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        img = render_heat(grid, min_peak=1e-6)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS Footfall</title><style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#8b93a7;--g:#3ddc84;--r:#ff6b6b;--b:#4aa3ff}
@media (prefers-color-scheme:light){:root{--bg:#f4f5f8;--card:#fff;--fg:#14161c;--mut:#5b6377}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--mut);margin-bottom:16px}.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.card{background:var(--card);border-radius:12px;padding:14px}.k{color:var(--mut);font-size:13px}
.v{font-size:34px;font-weight:700;margin-top:4px}.row{display:grid;gap:12px;margin-top:12px;grid-template-columns:1.4fr 1fr}
@media(max-width:800px){.row{grid-template-columns:1fr}}canvas,img{width:100%;border-radius:8px;display:block}
select,input{background:var(--bg);color:var(--fg);border:1px solid var(--mut);border-radius:6px;padding:4px 6px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
</style></head><body><main>
<h1>NEXUS footfall dashboard</h1><div class="sub" id="sub">loading…</div>
<div class="grid">
<div class="card"><div class="k">Entries</div><div class="v" id="in" style="color:var(--g)">–</div></div>
<div class="card"><div class="k">Exits</div><div class="v" id="out" style="color:var(--r)">–</div></div>
<div class="card"><div class="k">Occupancy now</div><div class="v" id="occ">–</div></div>
<div class="card"><div class="k">Peak hour (entries)</div><div class="v" id="peak">–</div></div></div>
<div class="row"><div class="card"><div class="k">Entries / exits per hour</div><canvas id="chart" width="700" height="300"></canvas></div>
<div class="card"><div class="k">Floor heatmap
<input type="date" id="day"> <select id="h0"></select>–<select id="h1"></select></div>
<img id="heat" alt="heatmap"></div></div></main><script>
const q=new URLSearchParams(location.search),tok=q.get('token')?'&token='+encodeURIComponent(q.get('token')):'';
const $=id=>document.getElementById(id);const today=new Date(Date.now()-new Date().getTimezoneOffset()*6e4).toISOString().slice(0,10);
$('day').value=today;for(let h=0;h<=24;h++){for(const s of ['h0','h1']){const o=document.createElement('option');o.value=o.textContent=h;$(s).appendChild(o)}}
$('h0').value=0;$('h1').value=24;
async function get(u){const r=await fetch(u+(u.includes('?')?'&':'?')+'x=1'+tok);return r.json()}
function chart(rows){const c=$('chart'),x=c.getContext('2d'),w=c.width,h=c.height,p=28;x.clearRect(0,0,w,h);
const m=Math.max(1,...rows.flatMap(r=>[r.ins,r.outs])),bw=(w-p)/24,st=getComputedStyle(document.body);x.font='11px system-ui';
rows.forEach((r,i)=>{const X=p+i*bw;const a=(h-p-8)*r.ins/m,b=(h-p-8)*r.outs/m;x.fillStyle='#3ddc84';x.fillRect(X+2,h-p-a,bw/2-3,a);
x.fillStyle='#ff6b6b';x.fillRect(X+bw/2,h-p-b,bw/2-3,b);if(i%3==0){x.fillStyle=st.color;x.fillText(r.hour.slice(0,2),X+2,h-10)}});
x.fillStyle=st.color;x.fillText(String(m),2,12)}
async function refresh(){try{const d=$('day').value;const s=await get('api/summary?date='+d);
$('in').textContent=s.entries;$('out').textContent=s.exits;$('occ').textContent=s.occupancy??'–';
$('peak').textContent=s.peak_hour?s.peak_hour+' ('+s.peak_hour_entries+')':'–';
$('sub').innerHTML='<span class="dot" style="background:'+(s.healthy?'var(--g)':'var(--r)')+'"></span>'+s.camera_id+' · '+s.date+' · '+(s.service_state||'unknown')+(s.fps?' · '+s.fps+' fps':'');
chart((await get('api/hourly?date='+d)).hours);
$('heat').src='api/heatmap.png?date='+d+'&h0='+$('h0').value+'&h1='+$('h1').value+'&t='+Date.now()+tok}catch(e){$('sub').textContent='error: '+e}}
for(const id of ['day','h0','h1'])$(id).onchange=refresh;refresh();setInterval(refresh,10000);
</script></body></html>"""


def make_handler(cfg: Config, store: EventStore) -> type:
    token = cfg.dashboard.api_token

    class Handler(BaseHTTPRequestHandler):
        server_version = "NexusDashboard/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:   # route access logs to logging
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _authorized(self, query: Dict[str, list]) -> bool:
            if not token:
                return True
            supplied = (query.get("token") or [""])[0]
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                supplied = auth[7:]
            return hmac.compare_digest(supplied.encode(), token.encode())

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/healthz":
                    ok = is_healthy(cfg)
                    return self._json(200 if ok else 503, {"healthy": ok, **read_status(cfg)})
                if not self._authorized(query):
                    return self._json(401, {"error": "unauthorized"})
                day = (query.get("date") or [None])[0]
                if url.path == "/":
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if url.path == "/api/summary":
                    return self._json(200, build_summary(cfg, store, day))
                if url.path == "/api/hourly":
                    return self._json(200, build_hourly(cfg, store, day))
                if url.path == "/api/heatmap.png":
                    h0 = max(0, min(24, int((query.get("h0") or ["0"])[0])))
                    h1 = max(h0, min(24, int((query.get("h1") or ["24"])[0])))
                    return self._send(200, build_heatmap_png(cfg, store, day, h0, h1), "image/png")
                return self._json(404, {"error": "not found"})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception:                                  # never leak internals to clients
                log.exception("request failed: %s", self.path)
                return self._json(500, {"error": "internal error"})

    return Handler


def serve(cfg: Config, store: Optional[EventStore] = None) -> None:
    store = store or EventStore(cfg.storage.db_path)
    server = ThreadingHTTPServer((cfg.dashboard.host, cfg.dashboard.port), make_handler(cfg, store))
    log.info("dashboard on http://%s:%d/", cfg.dashboard.host, cfg.dashboard.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
