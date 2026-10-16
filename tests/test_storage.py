import os
import sqlite3
import time

import numpy as np
import pytest

from nexus.occupancy import OccupancyTracker
from nexus.storage import EventStore


@pytest.fixture(autouse=True)
def utc(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db" / "n.db")
    yield s
    s.close()


T0 = 1_700_000_000.0 - (1_700_000_000.0 % 86400)      # a UTC midnight


def test_counts_and_cameras_are_separate(store):
    for ts, d, cam in [(T0 + 10, "IN", "a"), (T0 + 20, "IN", "a"), (T0 + 30, "OUT", "a"), (T0 + 40, "IN", "b")]:
        store.add_event(ts, cam, d)
    assert store.counts("a", T0, T0 + 86400) == (2, 1)
    assert store.counts("b", T0, T0 + 86400) == (1, 0)
    assert store.counts("a", T0 + 25, T0 + 86400) == (0, 1)       # t1 exclusive, t0 inclusive


def test_invalid_direction_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_event(T0, "a", "SIDEWAYS")


def test_hourly_buckets(store):
    store.add_event(T0 + 9 * 3600 + 5, "a", "IN")
    store.add_event(T0 + 9 * 3600 + 50, "a", "IN")
    store.add_event(T0 + 9 * 3600 + 60, "a", "OUT")
    store.add_event(T0 + 14 * 3600, "a", "IN")
    rows = store.hourly("a", T0, T0 + 86400)
    assert [(r["hour"][-5:], r["ins"], r["outs"]) for r in rows] == [("09:00", 2, 1), ("14:00", 1, 0)]


def test_heat_roundtrip_and_sum(store):
    a = np.zeros((20, 30), np.float32)
    a[5, 7] = 3.0
    b = np.ones((20, 30), np.float32)
    store.add_heat("a", T0, T0 + 300, a)
    store.add_heat("a", T0 + 300, T0 + 600, b)
    total = store.heat_sum("a", T0, T0 + 600)
    assert total.shape == (20, 30) and total[5, 7] == 4.0 and total[0, 0] == 1.0
    only_first = store.heat_sum("a", T0, T0 + 100)
    assert only_first[5, 7] == 3.0 and only_first[0, 0] == 0.0
    assert store.heat_sum("a", T0 + 1000, T0 + 2000) is None
    assert store.heat_sum("other", T0, T0 + 600) is None


def test_kv(store):
    assert store.get_kv("x") is None and store.get_kv("x", "d") == "d"
    store.set_kv("x", "1")
    store.set_kv("x", "2")
    assert store.get_kv("x") == "2"


def test_retention_purge(store):
    now = T0 + 100 * 86400
    store.add_event(now - 95 * 86400, "a", "IN")
    store.add_event(now - 5 * 86400, "a", "IN")
    store.add_heat("a", now - 96 * 86400, now - 95 * 86400, np.ones((4, 4), np.float32))
    assert store.purge(0, now) == 0                               # 0 = keep forever
    assert store.purge(90, now) == 2
    assert store.counts("a", 0, now + 1) == (1, 0)


def test_persists_on_disk(tmp_path):
    path = tmp_path / "p.db"
    s = EventStore(path)
    s.add_event(T0, "a", "IN")
    s.close()
    assert EventStore(path).counts("a", T0, T0 + 1) == (1, 0)
    assert os.path.exists(path)


# ---------------------------------------------------------------- occupancy
def test_occupancy_clamps_and_persists(store):
    occ = OccupancyTracker(store, "a", initial=2, reset_time=None)
    assert occ.value == 2
    occ.apply("IN")
    occ.apply("OUT")
    occ.apply("OUT")
    occ.apply("OUT")
    assert occ.value == 0                                         # never negative
    occ.apply("IN")
    assert OccupancyTracker(store, "a", initial=2, reset_time=None).value == 1   # survives restart


def test_nightly_reset_happens_once_per_day(store):
    occ = OccupancyTracker(store, "a", initial=0, reset_time="03:00")
    day = T0 + 86400                                              # UTC midnight (TZ=UTC)
    occ.apply("IN")
    occ.apply("IN")
    assert occ.maybe_reset(day + 1 * 3600) is False              # first run just records the time
    assert occ.value == 2
    assert occ.maybe_reset(day + 2 * 3600) is False              # before 03:00
    assert occ.maybe_reset(day + 3 * 3600 + 1) is True           # passes 03:00 -> reset
    assert occ.value == 0
    occ.apply("IN")
    assert occ.maybe_reset(day + 5 * 3600) is False              # not again the same day
    assert occ.maybe_reset(day + 86400 + 3 * 3600 + 5) is True   # next night
    assert occ.value == 0


def test_reset_disabled(store):
    occ = OccupancyTracker(store, "a", reset_time=None)
    occ.apply("IN")
    assert occ.maybe_reset(T0 + 10 * 86400) is False and occ.value == 1
