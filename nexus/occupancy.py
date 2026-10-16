"""Live occupancy with a persisted value, an initial headcount and a nightly reset."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from .storage import EventStore


class OccupancyTracker:
    """People currently inside = initial + entries - exits (never below zero).

    The value is persisted so a service restart does not lose it, and is reset to
    `initial` once per day at `reset_time` (local "HH:MM"), because counting errors
    otherwise accumulate forever (e.g. a missed exit at closing time).
    """

    def __init__(self, store: EventStore, camera_id: str, initial: int = 0,
                 reset_time: Optional[str] = "03:00") -> None:
        self.store, self.camera_id, self.initial = store, camera_id, initial
        self.reset_time = reset_time
        self._key = f"occupancy:{camera_id}"
        self._reset_key = f"occupancy_last_reset:{camera_id}"
        saved = store.get_kv(self._key)
        self.value = int(saved) if saved is not None else initial

    def _save(self) -> None:
        self.store.set_kv(self._key, str(self.value))

    def apply(self, direction: str) -> int:
        self.value = self.value + 1 if direction == "IN" else max(0, self.value - 1)
        self._save()
        return self.value

    def set(self, value: int) -> None:
        self.value = max(0, int(value))
        self._save()

    def maybe_reset(self, now: float) -> bool:
        """Reset once per day when the local clock passes reset_time. True if it reset."""
        if not self.reset_time:
            return False
        local = dt.datetime.fromtimestamp(now)
        hh, mm = (int(x) for x in self.reset_time.split(":"))
        due = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if local < due:
            due -= dt.timedelta(days=1)          # most recent reset point in the past
        last = self.store.get_kv(self._reset_key)
        if last is not None and float(last) >= due.timestamp():
            return False
        first_run = last is None
        self.store.set_kv(self._reset_key, str(now))
        if first_run:
            return False                          # don't wipe a value restored on first start
        self.value = self.initial
        self._save()
        return True
