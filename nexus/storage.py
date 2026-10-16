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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
            self.conn.commit()
        return n
