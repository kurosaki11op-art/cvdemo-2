"""SQLite persistence: anonymous crossing events, heatmap deltas and small key/value state.

Privacy by design: no images, no track ids and no appearance data are ever written;
only (timestamp, camera, direction) and aggregated dwell grids.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .sessions import SessionRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('IN', 'OUT'))
);
CREATE INDEX IF NOT EXISTS idx_events_cam_ts ON events (camera_id, ts);
CREATE TABLE IF NOT EXISTS heat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL NOT NULL,
    ts_end REAL NOT NULL,
    camera_id TEXT NOT NULL,
    w INTEGER NOT NULL,
    h INTEGER NOT NULL,
    data BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_heat_cam_ts ON heat (camera_id, ts_start);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    ts_start REAL NOT NULL,
    ts_end REAL NOT NULL,
    entered INTEGER NOT NULL DEFAULT 0,
    exited INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_cam_ts ON sessions (camera_id, ts_start);
CREATE TABLE IF NOT EXISTS zone_visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    zone TEXT NOT NULL,
    ts_enter REAL NOT NULL,
    dwell_s REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_zv_session ON zone_visits (session_id);
"""


class EventStore:
    """Thread-safe SQLite store (WAL mode: the dashboard can read while the service writes)."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # -- events ---------------------------------------------------------------- #
    def add_event(self, ts: float, camera_id: str, direction: str) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO events (ts, camera_id, direction) VALUES (?, ?, ?)",
                              (ts, camera_id, direction))
            self.conn.commit()

    def add_events(self, camera_id: str, rows: "List[Tuple[float, str]]") -> None:
        with self._lock:
            self.conn.executemany("INSERT INTO events (ts, camera_id, direction) VALUES (?, ?, ?)",
                                  [(ts, camera_id, d) for ts, d in rows])
            self.conn.commit()

    def counts(self, camera_id: str, t0: float, t1: float) -> Tuple[int, int]:
        """(entries, exits) with t0 <= ts < t1."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT direction, COUNT(*) FROM events WHERE camera_id=? AND ts>=? AND ts<? "
                "GROUP BY direction", (camera_id, t0, t1)).fetchall()
        tally = dict(rows)
        return int(tally.get("IN", 0)), int(tally.get("OUT", 0))

    def hourly(self, camera_id: str, t0: float, t1: float) -> List[Dict[str, Union[str, int]]]:
        """Entries/exits per local clock hour between t0 and t1 (hours without events omitted)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT strftime('%Y-%m-%d %H:00', ts, 'unixepoch', 'localtime') AS hour, direction, "
                "COUNT(*) FROM events WHERE camera_id=? AND ts>=? AND ts<? GROUP BY hour, direction "
                "ORDER BY hour", (camera_id, t0, t1)).fetchall()
        buckets: Dict[str, Dict[str, Union[str, int]]] = {}
        for hour, direction, n in rows:
            b = buckets.setdefault(hour, {"hour": hour, "ins": 0, "outs": 0})
            b["ins" if direction == "IN" else "outs"] = int(n)
        return list(buckets.values())

    # -- visitor sessions -------------------------------------------------------- #
    def add_session(self, camera_id: str, rec: "SessionRecord") -> None:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO sessions (camera_id, ts_start, ts_end, entered, exited) VALUES (?,?,?,?,?)",
                (camera_id, rec.ts_start, rec.ts_end, int(rec.entered), int(rec.exited)))
            sid = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO zone_visits (session_id, seq, zone, ts_enter, dwell_s) VALUES (?,?,?,?,?)",
                [(sid, i, v.zone, v.ts_enter, v.dwell_s) for i, v in enumerate(rec.visits)])
            self.conn.commit()

    def add_sessions(self, camera_id: str, recs: "List[SessionRecord]") -> None:
        """Bulk insert in one transaction (demo data, imports)."""
        with self._lock:
            for rec in recs:
                cur = self.conn.execute(
                    "INSERT INTO sessions (camera_id, ts_start, ts_end, entered, exited) VALUES (?,?,?,?,?)",
                    (camera_id, rec.ts_start, rec.ts_end, int(rec.entered), int(rec.exited)))
                self.conn.executemany(
                    "INSERT INTO zone_visits (session_id, seq, zone, ts_enter, dwell_s) VALUES (?,?,?,?,?)",
                    [(cur.lastrowid, i, v.zone, v.ts_enter, v.dwell_s) for i, v in enumerate(rec.visits)])
            self.conn.commit()

    def sessions(self, camera_id: str, t0: float, t1: float) -> "List[SessionRecord]":
        """Sessions that started in [t0, t1), with their zone visits."""
        from .sessions import SessionRecord, ZoneVisit
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, ts_start, ts_end, entered, exited FROM sessions "
                "WHERE camera_id=? AND ts_start>=? AND ts_start<? ORDER BY ts_start", (camera_id, t0, t1)).fetchall()
            visits = self.conn.execute(
                "SELECT zv.session_id, zv.zone, zv.ts_enter, zv.dwell_s FROM zone_visits zv "
                "JOIN sessions s ON s.id = zv.session_id "
                "WHERE s.camera_id=? AND s.ts_start>=? AND s.ts_start<? ORDER BY zv.session_id, zv.seq",
                (camera_id, t0, t1)).fetchall()
        by_session: Dict[int, List] = {}
        for sid, zone, ts, dwell in visits:
            by_session.setdefault(sid, []).append(ZoneVisit(zone, ts, dwell))
        return [SessionRecord(a, b, bool(e), bool(x), by_session.get(sid, [])) for sid, a, b, e, x in rows]

    def delete_demo_data(self, camera_id: str) -> None:
        with self._lock:
            for table in ("events", "heat", "sessions"):
                self.conn.execute(f"DELETE FROM {table} WHERE camera_id=?", (camera_id,))
            self.conn.commit()

    # -- heatmap deltas ----------------------------------------------------------- #
    def add_heat(self, camera_id: str, ts_start: float, ts_end: float, grid: np.ndarray) -> None:
        h, w = grid.shape
        blob = zlib.compress(np.ascontiguousarray(grid, dtype=np.float32).tobytes(), 6)
        with self._lock:
            self.conn.execute(
                "INSERT INTO heat (ts_start, ts_end, camera_id, w, h, data) VALUES (?,?,?,?,?,?)",
                (ts_start, ts_end, camera_id, w, h, blob))
            self.conn.commit()

    def heat_sum(self, camera_id: str, t0: float, t1: float) -> Optional[np.ndarray]:
        """Sum of all heatmap deltas overlapping [t0, t1), or None if there are none."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT w, h, data FROM heat WHERE camera_id=? AND ts_end>? AND ts_start<?",
                (camera_id, t0, t1)).fetchall()
        total: Optional[np.ndarray] = None
        for w, h, blob in rows:
            grid = np.frombuffer(zlib.decompress(blob), dtype=np.float32).reshape(h, w)
            if total is None:
                total = grid.copy()
            elif total.shape == grid.shape:
                total += grid
        return total

    # -- key/value state ---------------------------------------------------------- #
    def get_kv(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_kv(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO kv (key, value) VALUES (?, ?) "
                              "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self.conn.commit()

    # -- retention ---------------------------------------------------------------- #
    def purge(self, retention_days: int, now: Optional[float] = None) -> int:
        """Delete data older than `retention_days` (0 = keep forever). Returns rows removed."""
        if retention_days <= 0:
            return 0
        cutoff = (now if now is not None else time.time()) - retention_days * 86400
        with self._lock:
            n = self.conn.execute("DELETE FROM events WHERE ts<?", (cutoff,)).rowcount
            n += self.conn.execute("DELETE FROM heat WHERE ts_end<?", (cutoff,)).rowcount
            n += self.conn.execute("DELETE FROM sessions WHERE ts_end<?", (cutoff,)).rowcount
            self.conn.commit()
        return n
