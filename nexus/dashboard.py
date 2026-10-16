"""Small dependency-free web dashboard and JSON API (stdlib http.server).

Routes: /  (page) | /api/analytics | /api/summary | /api/hourly | /api/heatmap.png | /healthz
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


def build_analytics(cfg: Config, store: EventStore, day: Optional[str] = None, days: int = 1) -> Dict[str, Any]:
    from .analytics import build_report, period_bounds
    t0, t1 = period_bounds(day, days)
    rep = build_report(store, cfg, t0, t1)
    status = read_status(cfg)
    rep.update({"date": dt.date.fromtimestamp(t0).isoformat(), "days": days, "occupancy": status.get("occupancy"),
                "zone_counts": status.get("zone_counts", {}), "service_state": status.get("state"),
                "healthy": is_healthy(cfg), "fps": status.get("fps"),
                "zone_polygons": [{"name": z.name, "kind": z.kind, "polygon": [list(p) for p in z.polygon]}
                                  for z in cfg.zones]})
    return rep


def build_heatmap_png(cfg: Config, store: EventStore, day: Optional[str], h0: int, h1: int,
                      days: int = 1, zones: bool = True) -> bytes:
    from .ui import draw_zones
    t0, _ = day_bounds(day)
    t0 -= (days - 1) * 86400
    grid = store.heat_sum(cfg.camera_id, t0 + h0 * 3600, t0 + (days - 1) * 86400 + h1 * 3600)
    if grid is None:
        w, h = cfg.homography.canvas
        grid = np.zeros((h, w), dtype=np.float32)
        img = render_heat(grid)
        cv2.putText(img, "no data for this period", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        img = render_heat(grid, min_peak=1e-6)
    if zones:
        draw_zones(img, cfg.zones)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS Store Analytics</title><style>
:root{--bg:#eceff2;--card:#fff;--fg:#131820;--mut:#566170;--line:#d3d9e0;--g:#11743f;--r:#bb3b27;--a:#9a6200;--af:#f0b429;--bar:#c9d3df}
@media (prefers-color-scheme:dark){:root{--bg:#0d1116;--card:#151a21;--fg:#e5e9ef;--mut:#8d97a6;--line:#2a323d;--g:#3ecf8e;--r:#ff7d68;--a:#f0b429;--bar:#33404f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;gap:14px}
header{display:flex;flex-wrap:wrap;gap:10px 18px;justify-content:space-between;align-items:end}
h1{font-size:24px;margin:0}h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.sub{color:var(--mut)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.banner{background:var(--af);color:#1b1400;border-radius:8px;padding:8px 12px;font-weight:600}
.kpis{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.07em}.v{font:600 32px/1.15 ui-monospace,Menlo,Consolas,monospace;margin-top:2px}
.two{display:grid;gap:14px;grid-template-columns:1.15fr 1fr;align-items:start}
@media(max-width:860px){.two{grid-template-columns:1fr}}
ul.ins{margin:0;padding-left:18px;display:grid;gap:6px}
.zrow{display:grid;grid-template-columns:minmax(90px,140px) 1fr 104px 64px;gap:10px;align-items:center;padding:6px 0;border-bottom:1px solid var(--line)}
.zrow.head{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.07em}
.bar{height:14px;background:var(--bar);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--af)}
.pill{font-size:10px;border-radius:99px;padding:1px 7px;margin-left:6px;font-weight:700;vertical-align:middle}
.pill.top{background:var(--g);color:#fff}.pill.low{background:var(--r);color:#fff}
.num{text-align:right;font-variant-numeric:tabular-nums}
.frow{display:grid;grid-template-columns:150px 1fr 90px;gap:10px;align-items:center;margin:6px 0}
.small{color:var(--mut);font-size:12.5px}img,canvas{width:100%;display:block;border-radius:8px}
select,input{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
.flow div{display:flex;justify-content:space-between;gap:8px;padding:3px 0;border-bottom:1px solid var(--line)}
.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
</style></head><body><main>
<header><div><h1>NEXUS store analytics</h1><div class="sub" id="sub">loading…</div></div>
<div class="ctl"><input type="date" id="day" aria-label="Last day"><select id="days" aria-label="Range"><option value="1">1 day</option><option value="7">7 days</option><option value="30">30 days</option></select></div></header>
<div class="banner" id="demo" hidden>Demo data: these visitors are generated, not real.</div>
<section class="kpis">
<div class="card"><div class="k">Visitors</div><div class="v" id="visitors">–</div></div>
<div class="card"><div class="k">Avg visit</div><div class="v" id="avg">–</div></div>
<div class="card"><div class="k">Bounce rate</div><div class="v" id="bounce">–</div></div>
<div class="card"><div class="k">Reached checkout</div><div class="v" id="conv">–</div></div>
<div class="card"><div class="k">Door in / out</div><div class="v" id="door">–</div></div>
<div class="card"><div class="k">Inside now</div><div class="v" id="occ">–</div></div>
</section>
<section class="card"><h2>What the data says</h2><ul class="ins" id="insights"></ul></section>
<section class="two">
<div class="card"><h2>Zones, most visited first</h2><div id="zones"></div><div class="small" style="margin-top:8px">Visitors = people who stopped at least a few seconds. Avg stop = time per stop.</div></div>
<div class="card"><h2>Floor heatmap</h2><img id="heat" alt="Floor heatmap with zones"><div class="ctl small" style="margin-top:8px">Hours <select id="h0"></select> to <select id="h1"></select></div></div>
</section>
<section class="two">
<div class="card"><h2>Funnel</h2><div id="funnel"></div><h2 style="margin-top:16px">Most common paths</h2><div id="paths" class="small"></div></div>
<div class="card"><h2>Where visitors go next</h2><div class="flow small" id="flow"></div></div>
</section>
<section class="card"><h2>Arrivals by hour</h2><canvas id="chart" width="1000" height="260"></canvas></section>
</main><script>
const q=new URLSearchParams(location.search),tok=q.get('token')?'&token='+encodeURIComponent(q.get('token')):'';
const $=id=>document.getElementById(id),pct=v=>v==null?'–':Math.round(v*100)+'%';
const today=new Date(Date.now()-new Date().getTimezoneOffset()*6e4).toISOString().slice(0,10);$('day').value=today;
for(let h=0;h<=24;h++)for(const s of['h0','h1']){const o=document.createElement('option');o.value=o.textContent=h;$(s).appendChild(o)}$('h0').value=0;$('h1').value=24;
const fmt=s=>s>=90?(s/60).toFixed(1)+' min':Math.round(s)+' s';
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e}
async function get(u){const r=await fetch(u+(u.includes('?')?'&':'?')+'x=1'+tok);return r.json()}
function zones(r){const box=$('zones');box.replaceChildren();const h=el('div','zrow head');for(const t of['Zone','Share of visitors','Visitors','Avg stop'])h.appendChild(el('span',t=='Zone'||t=='Share of visitors'?'':'num',t));box.appendChild(h);
const rows=r.zones.filter(z=>z.kind!=='entrance');const areas=rows.filter(z=>z.kind==='area');const top=areas[0],low=areas[areas.length-1];
for(const z of rows){const row=el('div','zrow'),n=el('span','',z.zone);if(areas.length>1&&z===top&&z.visitors)n.appendChild(el('span','pill top','MOST'));if(areas.length>1&&z===low&&z!==top)n.appendChild(el('span','pill low','LEAST'));
const b=el('div','bar'),i=document.createElement('i');i.style.width=Math.round(z.visitor_share*100)+'%';b.appendChild(i);row.append(n,b,el('span','num',z.visitors+' · '+pct(z.visitor_share)),el('span','num',z.visits?fmt(z.avg_dwell_s):'–'));box.appendChild(row)}}
function funnel(r){const box=$('funnel');box.replaceChildren();for(const f of r.funnel){const row=el('div','frow');const b=el('div','bar'),i=document.createElement('i');i.style.width=Math.round(f.pct_of_first*100)+'%';b.appendChild(i);row.append(el('span','',f.step),b,el('span','num',f.count+' · '+pct(f.pct_of_first)));box.appendChild(row)}}
function chart(a){const c=$('chart'),x=c.getContext('2d'),w=c.width,h=c.height,p=26,m=Math.max(1,...a),bw=(w-p)/24,st=getComputedStyle(document.body);x.clearRect(0,0,w,h);x.font='12px system-ui';
a.forEach((v,i)=>{const bh=(h-p-10)*v/m;x.fillStyle='#f0b429';x.fillRect(p+i*bw+3,h-p-bh,bw-6,bh);x.fillStyle=st.color;if(i%2==0)x.fillText(String(i).padStart(2,'0'),p+i*bw+3,h-8)});x.fillStyle=st.color;x.fillText(String(m),0,12)}
async function refresh(){try{const d=$('day').value,n=$('days').value,r=await get('api/analytics?date='+d+'&days='+n),o=r.overview;
$('visitors').textContent=o.visitors;$('avg').textContent=o.visitors?fmt(o.avg_visit_s):'–';$('bounce').textContent=pct(o.bounce_rate);$('conv').textContent=pct(o.checkout_rate);$('door').textContent=r.entries+' / '+r.exits;$('occ').textContent=r.occupancy??'–';
$('sub').replaceChildren();const dot=el('span','dot');dot.style.background=r.healthy?'var(--g)':'var(--r)';$('sub').append(dot,r.camera_id+' · '+r.date+(r.days>1?' (last '+r.days+' days)':'')+' · '+(r.service_state||'no live service'));
$('demo').hidden=!r.demo_data;const ul=$('insights');ul.replaceChildren(...r.insights.map(t=>el('li','',t)));
zones(r);funnel(r);$('paths').replaceChildren(...(r.paths.length?r.paths.map(p=>el('div','',p.path.join(' → ')+' · '+p.count+' ('+pct(p.share)+')')):[el('div','','No paths yet.')]));
$('flow').replaceChildren(...r.flow.slice(0,10).map(f=>{const d=el('div');d.append(el('span','',f.source+' → '+f.target),el('b','',String(f.count)));return d}));
chart(r.hourly.arrivals);$('heat').src='api/heatmap.png?date='+d+'&days='+n+'&h0='+$('h0').value+'&h1='+$('h1').value+'&t='+Date.now()+tok}catch(e){$('sub').textContent='error: '+e}}
for(const id of['day','days','h0','h1'])$(id).onchange=refresh;refresh();setInterval(refresh,15000);
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
                if url.path == "/api/analytics":
                    days = max(1, min(90, int((query.get("days") or ["1"])[0])))
                    return self._json(200, build_analytics(cfg, store, day, days))
                if url.path == "/api/heatmap.png":
                    h0 = max(0, min(24, int((query.get("h0") or ["0"])[0])))
                    h1 = max(h0, min(24, int((query.get("h1") or ["24"])[0])))
                    days = max(1, min(90, int((query.get("days") or ["1"])[0])))
                    zones = (query.get("zones") or ["1"])[0] != "0"
                    return self._send(200, build_heatmap_png(cfg, store, day, h0, h1, days, zones), "image/png")
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
