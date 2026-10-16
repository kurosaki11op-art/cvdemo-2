"""Generate clearly-labelled DEMO data (visitor sessions, door events, heatmaps) for a presentation.

The data is synthetic: zone popularity, dwell times and arrival patterns are random but plausible,
so the analytics dashboard has something to show before a real camera is connected. The database is
flagged (`demo_data:<camera>` = 1), the dashboard shows a banner and `nexus report` says so.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from typing import Dict, List, Tuple

from .config import Config
from .heatmap import HeatmapAccumulator
from .sessions import SessionRecord, ZoneVisit
from .storage import EventStore
from .zones import ZoneMap

HOURLY = {9: 6, 10: 10, 11: 14, 12: 22, 13: 20, 14: 13, 15: 12, 16: 15, 17: 24, 18: 26, 19: 18, 20: 8}
WEEKDAY = [0.8, 0.8, 0.85, 0.9, 1.1, 1.5, 1.3]            # Monday .. Sunday


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def zone_profile(cfg: Config, rng: random.Random) -> Dict[str, Tuple[float, float]]:
    """name -> (popularity weight, mean dwell seconds); at least one busy and one near-dead area."""
    areas = [z.name for z in cfg.zones if z.kind == "area"]
    weights = [rng.uniform(0.25, 0.9) for _ in areas]
    if areas:
        weights[rng.randrange(len(areas))] = 1.0
        if len(areas) > 2:
            weights[rng.choice([i for i, w in enumerate(weights) if w < 1.0])] = 0.08
    return {n: (w, rng.uniform(18, 60)) for n, w in zip(areas, weights, strict=True)}


def generate(store: EventStore, cfg: Config, days: int = 7, seed: int = 42,
             end_day: dt.date | None = None) -> Dict[str, int]:
    """Write `days` days of demo data ending on `end_day` (default today). Returns row counts."""
    if not cfg.zones:
        raise ValueError("define zones first (configs/example.yaml has a sample layout)")
    rng = random.Random(seed)
    zmap = ZoneMap(cfg.zones)
    profile = zone_profile(cfg, rng)
    areas = list(profile)
    checkout = [z.name for z in cfg.zones if z.kind == "checkout"]
    entrance = [z.name for z in cfg.zones if z.kind == "entrance"]
    end_day = end_day or dt.date.today()
    sessions: List[SessionRecord] = []
    events: List[Tuple[float, str]] = []
    heat_rows: List[Tuple[float, float, "HeatmapAccumulator"]] = []
    w, h = cfg.homography.canvas

    for back in range(days - 1, -1, -1):
        day = end_day - dt.timedelta(days=back)
        factor = WEEKDAY[day.weekday()]
        for hour, base in HOURLY.items():
            start = dt.datetime.combine(day, dt.time(hour)).timestamp()
            heat = HeatmapAccumulator((w, h))
            for _ in range(_poisson(rng, base * factor)):
                ts = start + rng.uniform(0, 3600)
                rec = _visitor(rng, ts, profile, areas, checkout, entrance)
                sessions.append(rec)
                events.append((rec.ts_start, "IN"))
                if rec.exited:
                    events.append((rec.ts_end, "OUT"))
                for v in rec.visits:
                    cx, cy = zmap.centroid(v.zone)
                    for _ in range(min(int(v.dwell_s), 40)):
                        heat.add(cx + rng.gauss(0, 18), cy + rng.gauss(0, 14))
            if heat.canvas.any():
                heat_rows.append((start, start + 3600, heat))

    store.add_sessions(cfg.camera_id, sessions)
    store.add_events(cfg.camera_id, events)
    for t0, t1, heat in heat_rows:
        store.add_heat(cfg.camera_id, t0, t1, heat.canvas)
    store.set_kv(f"demo_data:{cfg.camera_id}", "1")
    return {"sessions": len(sessions), "events": len(events), "heatmaps": len(heat_rows)}


def _visitor(rng: random.Random, ts: float, profile: Dict[str, Tuple[float, float]], areas: List[str],
             checkout: List[str], entrance: List[str]) -> SessionRecord:
    t = ts
    visits: List[ZoneVisit] = []
    if entrance:
        d = rng.uniform(2, 5)
        visits.append(ZoneVisit(entrance[0], t, d))
        t += d + rng.uniform(1, 3)
    if areas and rng.random() > 0.14:                              # 14% bounce: leave without stopping
        k = rng.choices([1, 2, 3, 4], [0.3, 0.35, 0.25, 0.1])[0]
        pool = list(areas)
        for _ in range(min(k, len(pool) - 1 if len(pool) > 2 else len(pool))):
            name = rng.choices(pool, [profile[n][0] for n in pool])[0]
            pool.remove(name)
            d = max(2.5, rng.lognormvariate(math.log(profile[name][1]), 0.5))
            visits.append(ZoneVisit(name, t, d))
            t += d + rng.uniform(3, 9)
        if checkout and rng.random() < 0.4:
            d = rng.uniform(25, 90)
            visits.append(ZoneVisit(checkout[0], t, d))
            t += d
    else:
        t += rng.uniform(3, 8)
    return SessionRecord(ts, t + rng.uniform(1, 4), True, rng.random() < 0.95, visits)
