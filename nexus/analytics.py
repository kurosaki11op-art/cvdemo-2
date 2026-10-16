"""Store analytics: the web-analytics questions, answered for a physical store.

website              | physical store
---------------------|------------------------------------------
page                 | zone (named floor area)
page views           | zone visits (a stop of at least `min_zone_dwell_s`)
unique visitors      | sessions (one anonymous track, first to last sighting)
time on page         | dwell time in a zone
bounce rate          | share of visitors who never stopped at any zone beyond the entrance
user flow            | zone-to-zone transitions and the most common paths
conversion funnel    | entrance -> browse -> checkout
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .sessions import SessionRecord
from .storage import EventStore
from .zones import ZoneMap

MIN_SESSION_S = 1.0          # ignore sub-second blips (tracker noise)
SMALL_SAMPLE = 30            # below this many visitors, rankings are only indicative


@dataclass
class ZoneStat:
    zone: str
    kind: str
    visitors: int            # distinct visitors who stopped here
    visits: int              # stops (a visitor can stop more than once)
    total_dwell_s: float
    avg_dwell_s: float       # mean per stop
    median_dwell_s: float
    visitor_share: float     # visitors / all visitors
    dwell_share: float       # this zone's share of all zone time


def usable(sessions: Iterable[SessionRecord]) -> List[SessionRecord]:
    return [s for s in sessions if s.dwell_s >= MIN_SESSION_S]


def engagement_zones(zone_kinds: Dict[str, str]) -> set:
    return {z for z, k in zone_kinds.items() if k != "entrance"}


def zone_stats(sessions: Sequence[SessionRecord], zone_kinds: Dict[str, str]) -> List[ZoneStat]:
    """Per-zone numbers, most visited first. Zones nobody visited are included (dead zones)."""
    n = len(sessions)
    dwell: Dict[str, List[float]] = {z: [] for z in zone_kinds}
    visitors: Dict[str, set] = {z: set() for z in zone_kinds}
    for i, s in enumerate(sessions):
        for v in s.visits:
            dwell.setdefault(v.zone, []).append(v.dwell_s)
            visitors.setdefault(v.zone, set()).add(i)
    total = sum(sum(d) for d in dwell.values()) or 1.0
    out = []
    for z, d in dwell.items():
        out.append(ZoneStat(
            zone=z, kind=zone_kinds.get(z, "area"), visitors=len(visitors[z]), visits=len(d),
            total_dwell_s=sum(d), avg_dwell_s=(sum(d) / len(d)) if d else 0.0,
            median_dwell_s=statistics.median(d) if d else 0.0,
            visitor_share=len(visitors[z]) / n if n else 0.0, dwell_share=sum(d) / total))
    out.sort(key=lambda r: (-r.visitors, -r.total_dwell_s, r.zone))
    return out


def overview(sessions: Sequence[SessionRecord], zone_kinds: Dict[str, str]) -> Dict[str, Any]:
    n = len(sessions)
    eng = engagement_zones(zone_kinds)
    durations = [s.dwell_s for s in sessions]
    engaged = [s for s in sessions if any(v.zone in eng for v in s.visits)]
    checkout = {z for z, k in zone_kinds.items() if k == "checkout"}
    converted = [s for s in sessions if any(v.zone in checkout for v in s.visits)] if checkout else []
    zones_per = [len({v.zone for v in s.visits if v.zone in eng}) for s in engaged]
    return {
        "visitors": n,
        "entered": sum(1 for s in sessions if s.entered),
        "avg_visit_s": statistics.fmean(durations) if n else 0.0,
        "median_visit_s": statistics.median(durations) if n else 0.0,
        "engaged": len(engaged),
        "bounce_rate": (1 - len(engaged) / n) if n else None,
        "avg_zones_per_engaged_visit": statistics.fmean(zones_per) if zones_per else 0.0,
        "checkout_rate": (len(converted) / n) if (n and checkout) else None,
    }


def _path(s: SessionRecord, ignore: Iterable[str]) -> List[str]:
    skip = set(ignore)
    return [z for z in s.path if z not in skip]


def flow(sessions: Sequence[SessionRecord], limit: int = 20, ignore: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Zone-to-zone transitions (like a website's user-flow report). `ignore` hides e.g. the entrance."""
    c: Counter = Counter()
    for s in sessions:
        p = _path(s, ignore)
        if not p:
            continue
        c[("(arrive)", p[0])] += 1
        for a, b in zip(p, p[1:], strict=False):
            c[(a, b)] += 1
        c[(p[-1], "(leave)")] += 1
    return [{"source": a, "target": b, "count": n} for (a, b), n in c.most_common(limit)]


def top_paths(sessions: Sequence[SessionRecord], limit: int = 5,
              ignore: Iterable[str] = ()) -> List[Dict[str, Any]]:
    c: Counter = Counter(tuple(_path(s, ignore)) for s in sessions if _path(s, ignore))
    n = len(sessions) or 1
    return [{"path": list(p), "count": k, "share": k / n} for p, k in c.most_common(limit)]


def funnel(sessions: Sequence[SessionRecord], zone_kinds: Dict[str, str],
           steps: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Conversion funnel. Default: visitors -> stopped at any zone -> reached checkout.

    With `steps` (zone names) each step must be visited, in order, after the previous one.
    """
    eng = engagement_zones(zone_kinds)
    checkout = {z for z, k in zone_kinds.items() if k == "checkout"}
    n = len(sessions)
    top = 1.0 if n else 0.0
    rows = [{"step": "Visited the store", "count": n, "pct_of_first": top, "pct_of_prev": top}]

    if steps:
        def progress(s: SessionRecord) -> int:
            k = 0
            for v in s.visits:
                if k < len(steps) and v.zone == steps[k]:
                    k += 1
            return k

        prog = [progress(s) for s in sessions]
        counts = [(name, sum(1 for p in prog if p > i)) for i, name in enumerate(steps)]
    else:
        counts = [("Stopped at a zone", sum(1 for s in sessions if any(v.zone in eng for v in s.visits)))]
        if checkout:
            counts.append(("Reached checkout", sum(1 for s in sessions if any(v.zone in checkout for v in s.visits))))
    prev = n
    for label, c in counts:
        rows.append({"step": label, "count": c, "pct_of_first": c / n if n else 0.0,
                     "pct_of_prev": c / prev if prev else 0.0})
        prev = c
    return rows


def hourly(sessions: Sequence[SessionRecord], zone_kinds: Dict[str, str]) -> Dict[str, Any]:
    """Visitors arriving per local hour, and zone stops per hour."""
    arrivals = [0] * 24
    zone_hours: Dict[str, List[int]] = {z: [0] * 24 for z in zone_kinds}
    for s in sessions:
        arrivals[dt.datetime.fromtimestamp(s.ts_start).hour] += 1
        for v in s.visits:
            zone_hours.setdefault(v.zone, [0] * 24)[dt.datetime.fromtimestamp(v.ts_enter).hour] += 1
    return {"arrivals": arrivals, "zone_stops": zone_hours}


def heat_share(store: EventStore, cfg: Config, t0: float, t1: float) -> Dict[str, float]:
    """Each zone's share of the accumulated heatmap (independent cross-check of dwell)."""
    grid = store.heat_sum(cfg.camera_id, t0, t1)
    if grid is None or not cfg.zones:
        return {}
    h, w = grid.shape
    total = float(grid.sum()) or 1.0
    return {name: float(grid[m].sum()) / total for name, m in ZoneMap(cfg.zones).masks((w, h)).items()}


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def insights(ov: Dict[str, Any], stats: Sequence[ZoneStat], paths: Sequence[Dict[str, Any]],
             fun: Sequence[Dict[str, Any]], hrs: Dict[str, Any]) -> List[str]:
    """Plain-language findings, computed from the numbers (never invented)."""
    out: List[str] = []
    n = ov["visitors"]
    if n == 0:
        return ["No visitors recorded in this period yet."]
    if n < SMALL_SAMPLE:
        out.append(f"Only {n} visitors in this period, so treat the rankings below as indicative.")
    area = [s for s in stats if s.kind == "area"]
    if area:
        top, low = area[0], area[-1]
        if top.visitors:
            out.append(f"Most visited zone: {top.zone}, with {_pct(top.visitor_share)} of visitors stopping there "
                       f"for {top.avg_dwell_s:.0f} s on average.")
        if len(area) > 1 and low is not top:
            if low.visitors == 0:
                out.append(f"Nobody stopped at {low.zone}: a dead zone. Check that it is visible and reachable, "
                           "or move something there.")
            else:
                out.append(f"Least visited zone: {low.zone}, only {_pct(low.visitor_share)} of visitors "
                           f"({low.avg_dwell_s:.0f} s average stop).")
        longest = max(area, key=lambda s: s.avg_dwell_s)
        if longest.visits:
            out.append(f"People stay longest at {longest.zone}: {longest.avg_dwell_s:.0f} s per stop.")
    if ov["bounce_rate"] is not None and ov["bounce_rate"] >= 0.15:
        out.append(f"Bounce rate is {_pct(ov['bounce_rate'])}: those visitors left without stopping at any zone.")
    elif ov["bounce_rate"] is not None:
        out.append(f"Bounce rate is low at {_pct(ov['bounce_rate'])}; most visitors stopped somewhere.")
    arr = hrs["arrivals"]
    if sum(arr):
        peak = int(np.argmax(arr))
        out.append(f"Busiest hour: {peak:02d}:00 with {arr[peak]} arrivals.")
    if paths:
        p = paths[0]
        out.append(f"Most common path: {' → '.join(p['path'])} ({_pct(p['share'])} of visitors).")
    if len(fun) >= 3:
        drops = [(fun[i]["pct_of_prev"], fun[i]["step"]) for i in range(1, len(fun)) if fun[i - 1]["count"]]
        if drops:
            worst = min(drops)
            if worst[0] < 0.6:
                out.append(f"Biggest funnel drop-off: only {_pct(worst[0])} of visitors at the previous step "
                           f"go on to '{worst[1]}'.")
    return out


def build_report(store: EventStore, cfg: Config, t0: float, t1: float) -> Dict[str, Any]:
    """Everything the dashboard, the API and `nexus report` need, for sessions starting in [t0, t1)."""
    kinds = {z.name: z.kind for z in cfg.zones}
    sess = usable(store.sessions(cfg.camera_id, t0, t1))
    for s in sess:
        for v in s.visits:
            kinds.setdefault(v.zone, "area")
    stats = zone_stats(sess, kinds)
    ov = overview(sess, kinds)
    entrance = [z for z, k in kinds.items() if k == "entrance"]
    paths = top_paths(sess, ignore=entrance)
    fun = funnel(sess, kinds, cfg.analytics.funnel)
    hrs = hourly(sess, kinds)
    entries, exits = store.counts(cfg.camera_id, t0, t1)
    return {
        "camera_id": cfg.camera_id, "from": t0, "to": t1, "overview": ov,
        "entries": entries, "exits": exits,
        "zones": [asdict(s) for s in stats], "flow": flow(sess, ignore=entrance), "paths": paths, "funnel": fun,
        "hourly": hrs, "heat_share": heat_share(store, cfg, t0, t1),
        "insights": insights(ov, stats, paths, fun, hrs),
        "demo_data": store.get_kv(f"demo_data:{cfg.camera_id}") == "1",
    }


def format_report(rep: Dict[str, Any]) -> str:
    """Human-readable text/markdown report (for `nexus report`)."""
    ov = rep["overview"]
    t0, t1 = dt.datetime.fromtimestamp(rep["from"]), dt.datetime.fromtimestamp(rep["to"])
    when = f"{t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M}"
    lines = [f"# Store analytics: {rep['camera_id']}", f"Period: {when}"]
    if rep.get("demo_data"):
        lines.append("**NOTE: this database contains generated demo data, not real visitors.**")
    br = "n/a" if ov["bounce_rate"] is None else _pct(ov["bounce_rate"])
    lines += ["", "## Overview",
              f"- Visitors: {ov['visitors']}   (door counter: {rep['entries']} in / {rep['exits']} out)",
              f"- Average visit: {ov['avg_visit_s']:.0f} s (median {ov['median_visit_s']:.0f} s)",
              f"- Bounce rate: {br}",
              f"- Zones stopped at per engaged visit: {ov['avg_zones_per_engaged_visit']:.1f}", "",
              "## Zones, most visited first",
              f"{'Zone':<16}{'Visitors':>9}{'Share':>8}{'Stops':>7}{'Avg stop':>10}{'Dwell share':>13}"]
    for z in (z for z in rep["zones"] if z["kind"] != "entrance"):
        lines.append(f"{z['zone']:<16}{z['visitors']:>9}{_pct(z['visitor_share']):>8}{z['visits']:>7}"
                     f"{z['avg_dwell_s']:>9.0f}s{_pct(z['dwell_share']):>13}")
    lines += ["", "## Funnel"]
    for f in rep["funnel"]:
        lines.append(f"- {f['step']}: {f['count']} ({_pct(f['pct_of_first'])} of visitors, "
                     f"{_pct(f['pct_of_prev'])} of previous step)")
    lines += ["", "## Most common paths"]
    for p in rep["paths"]:
        lines.append(f"- {' → '.join(p['path'])}: {p['count']} ({_pct(p['share'])})")
    lines += ["", "## Findings"] + [f"- {t}" for t in rep["insights"]]
    return "\n".join(lines)


def period_bounds(date: Optional[str], days: int = 1) -> Tuple[float, float]:
    """Local-midnight bounds for `days` days ending on `date` (default today)."""
    end_day = dt.date.fromisoformat(date) if date else dt.date.today()
    start = dt.datetime.combine(end_day - dt.timedelta(days=days - 1), dt.time.min)
    return start.timestamp(), (dt.datetime.combine(end_day, dt.time.min) + dt.timedelta(days=1)).timestamp()
