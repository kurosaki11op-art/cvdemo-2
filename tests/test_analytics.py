import datetime as dt

import pytest

from nexus import analytics as an
from nexus.sessions import SessionRecord, ZoneVisit
from nexus.storage import EventStore

KINDS = {"Entrance": "entrance", "A": "area", "B": "area", "C": "area", "Checkout": "checkout"}
T0 = dt.datetime(2026, 3, 4, 0, 0).timestamp()


def sess(start_h, visits, dur=120.0, entered=True):
    """visits: [(zone, dwell)]"""
    t = T0 + start_h * 3600
    vs, cur = [], t
    for z, d in visits:
        vs.append(ZoneVisit(z, cur, d))
        cur += d
    return SessionRecord(t, t + dur, entered, False, vs)


DATA = [
    sess(10, [("A", 40), ("Checkout", 60)]),
    sess(10, [("A", 30), ("B", 20), ("Checkout", 50)]),
    sess(11, [("A", 50)]),
    sess(11, [("B", 25)]),
    sess(17, [("A", 35), ("Checkout", 40)]),
    sess(17, [("Entrance", 5)], dur=8),                      # bounce: only the entrance
    sess(17, [], dur=4),                                      # bounce: no stop at all
    sess(18, [("A", 45), ("B", 30), ("Checkout", 55)]),
]


def test_zone_stats_rank_share_and_dead_zone():
    stats = {s.zone: s for s in an.zone_stats(DATA, KINDS)}
    assert [s.zone for s in an.zone_stats(DATA, KINDS)][:2] == ["A", "Checkout"]
    a = stats["A"]
    assert (a.visitors, a.visits, a.total_dwell_s) == (5, 5, 200)
    assert a.avg_dwell_s == 40 and a.visitor_share == pytest.approx(5 / 8)
    assert stats["C"].visitors == 0 and stats["C"].avg_dwell_s == 0          # unvisited zone is listed
    assert sum(s.dwell_share for s in stats.values()) == pytest.approx(1.0)


def test_overview_bounce_and_checkout_rate():
    ov = an.overview(DATA, KINDS)
    assert ov["visitors"] == 8 and ov["engaged"] == 6
    assert ov["bounce_rate"] == pytest.approx(2 / 8)
    assert ov["checkout_rate"] == pytest.approx(4 / 8)
    assert ov["avg_zones_per_engaged_visit"] == pytest.approx((2 + 3 + 1 + 1 + 2 + 3) / 6)


def test_overview_empty():
    ov = an.overview([], KINDS)
    assert ov["visitors"] == 0 and ov["bounce_rate"] is None


def test_flow_and_paths():
    f = {(r["source"], r["target"]): r["count"] for r in an.flow(DATA, limit=50)}
    assert f[("(arrive)", "A")] == 5 and f[("A", "Checkout")] == 2 and f[("A", "B")] == 2
    assert f[("Checkout", "(leave)")] == 4
    paths = an.top_paths(DATA)
    assert paths[0]["path"] == ["A", "Checkout"] and paths[0]["count"] == 2
    with_entrance = [sess(9, [("Entrance", 3), ("A", 30)]), sess(9, [("Entrance", 3), ("A", 30)])]
    assert an.top_paths(with_entrance)[0]["path"] == ["Entrance", "A"]
    assert an.top_paths(with_entrance, ignore=["Entrance"])[0]["path"] == ["A"]
    assert ("(arrive)", "A") in {(r["source"], r["target"]) for r in an.flow(with_entrance, ignore=["Entrance"])}


def test_default_funnel_and_custom_funnel_order():
    rows = an.funnel(DATA, KINDS)
    assert [(r["step"], r["count"]) for r in rows] == [("Visited the store", 8), ("Stopped at a zone", 6),
                                                       ("Reached checkout", 4)]
    assert rows[2]["pct_of_prev"] == pytest.approx(4 / 6)
    custom = an.funnel(DATA, KINDS, ["A", "B", "Checkout"])               # must be visited in this order
    assert [r["count"] for r in custom] == [8, 5, 2, 2]
    assert [r["count"] for r in an.funnel(DATA, KINDS, ["B", "A"])] == [8, 3, 0]


def test_hourly_arrivals():
    h = an.hourly(DATA, KINDS)
    assert h["arrivals"][10] == 2 and h["arrivals"][17] == 3 and sum(h["arrivals"]) == 8
    assert h["zone_stops"]["A"][10] == 2


def test_insights_are_computed_from_the_numbers():
    kinds = KINDS
    stats = an.zone_stats(DATA * 5, kinds)                                  # 40 visitors: no small-sample note
    ov = an.overview(DATA * 5, kinds)
    text = " ".join(an.insights(ov, stats, an.top_paths(DATA * 5), an.funnel(DATA * 5, kinds),
                                an.hourly(DATA * 5, kinds)))
    assert "Most visited zone: A" in text and "dead zone" in text and "Nobody stopped at C" in text
    assert "Least visited zone: Checkout" not in text          # checkout is a destination, not an aisle
    assert "Bounce rate is 25%" in text and "Busiest hour: 17:00" in text
    assert "A → Checkout" in text and "indicative" not in text
    small = " ".join(an.insights(an.overview(DATA, kinds), an.zone_stats(DATA, kinds), [], [], an.hourly(DATA, kinds)))
    assert "indicative" in small
    assert an.insights(an.overview([], kinds), [], [], [], {"arrivals": [0] * 24}) == [
        "No visitors recorded in this period yet."]


def test_usable_drops_sub_second_blips():
    blip = SessionRecord(T0, T0 + 0.4)
    assert an.usable([blip, DATA[0]]) == [DATA[0]]


def test_period_bounds():
    a, b = an.period_bounds("2026-03-04", 1)
    assert b - a == 86400 and dt.datetime.fromtimestamp(a).day == 4
    a7, b7 = an.period_bounds("2026-03-04", 7)
    assert round((b7 - a7) / 86400) == 7


def test_store_roundtrip_report_and_retention(tmp_path, cfg):
    from nexus.config import sample_zones
    cfg.zones = sample_zones()
    cfg.validate()
    store = EventStore(tmp_path / "a.db")
    for s in DATA:
        store.add_session(cfg.camera_id, s)
    store.add_event(T0 + 36000, cfg.camera_id, "IN")
    back = store.sessions(cfg.camera_id, T0, T0 + 86400)
    assert len(back) == 8 and back[0].path == ["A", "Checkout"] and back[1].visits[1].zone == "B"
    rep = an.build_report(store, cfg, T0, T0 + 86400)
    assert rep["overview"]["visitors"] == 8 and rep["entries"] == 1
    assert rep["zones"][0]["zone"] == "A"          # zones in the data but not configured are still reported
    txt = an.format_report(rep)
    assert "Zones, most visited first" in txt and "Findings" in txt
    assert store.sessions("other", T0, T0 + 86400) == []
    store.purge(30, now=T0 + 90 * 86400)
    assert store.sessions(cfg.camera_id, T0, T0 + 86400) == []


def test_funnel_counts_a_checkout_only_visitor_as_having_stopped():
    only_checkout = [sess(10, [("Checkout", 30)])]
    rows = an.funnel(only_checkout, KINDS)
    assert [r["count"] for r in rows] == [1, 1, 1]
