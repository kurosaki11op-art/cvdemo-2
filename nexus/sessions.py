"""Visitor sessions: who went where, and for how long (the 'user journey' of a physical store).

A session is one anonymous track from first to last sighting. It records the ordered zone
visits with dwell times. Track ids are used only in memory; stored sessions carry no identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .zones import ZoneMap


@dataclass
class ZoneVisit:
    zone: str
    ts_enter: float
    dwell_s: float


@dataclass
class SessionRecord:
    ts_start: float
    ts_end: float
    entered: bool = False          # crossed the tripwire inwards during the session
    exited: bool = False
    visits: List[ZoneVisit] = field(default_factory=list)

    @property
    def dwell_s(self) -> float:
        return max(0.0, self.ts_end - self.ts_start)

    @property
    def path(self) -> List[str]:
        """Zone names in order, consecutive repeats merged."""
        out: List[str] = []
        for v in self.visits:
            if not out or out[-1] != v.zone:
                out.append(v.zone)
        return out


@dataclass
class _Live:
    start: float
    last: float
    entered: bool = False
    exited: bool = False
    zone: Optional[str] = None
    zone_since: float = 0.0
    cand: Optional[str] = None
    has_cand: bool = False           # a candidate zone (possibly "no zone") is waiting to be confirmed
    cand_since: float = 0.0
    visits: List[ZoneVisit] = field(default_factory=list)


class SessionTracker:
    """Turns per-frame (id, floor position) observations into finished SessionRecords.

    * Zone changes need to hold for `switch_hold_s` (no flicker at borders).
    * A zone visit shorter than `min_zone_dwell_s` is dropped (just walking past).
    * A track unseen for `timeout_s` is finished.
    """

    def __init__(self, zone_map: ZoneMap, min_zone_dwell_s: float = 2.0, switch_hold_s: float = 0.6,
                 timeout_s: float = 6.0) -> None:
        self.zones = zone_map
        self.min_dwell, self.hold, self.timeout = min_zone_dwell_s, switch_hold_s, timeout_s
        self.live: Dict[int, _Live] = {}

    def _close_zone(self, s: _Live, until: float) -> None:
        if s.zone is not None and until - s.zone_since >= self.min_dwell:
            s.visits.append(ZoneVisit(s.zone, s.zone_since, until - s.zone_since))

    def update(self, now: float, observations: Iterable[Tuple[int, Tuple[float, float]]],
               events: Iterable[Tuple[int, str]] = ()) -> List[SessionRecord]:
        """Feed one frame. `observations` = (track_id, (X', Y')); `events` = (track_id, "IN"/"OUT")."""
        for tid, (x, y) in observations:
            s = self.live.get(tid)
            if s is None:
                s = self.live[tid] = _Live(start=now, last=now)
            s.last = now
            zone = self.zones.zone_at(x, y)
            if zone == s.zone:
                s.has_cand = False
                continue
            if s.start == now and s.zone is None:
                s.zone, s.zone_since = zone, now          # first sighting: take the zone it is already in
                continue
            if not s.has_cand or zone != s.cand:
                s.cand, s.has_cand, s.cand_since = zone, True, now
            elif now - s.cand_since >= self.hold:
                self._close_zone(s, s.cand_since)
                s.zone, s.zone_since, s.has_cand = zone, s.cand_since, False
        for tid, direction in events:
            s = self.live.get(tid)
            if s is not None:
                if direction == "IN":
                    s.entered = True
                else:
                    s.exited = True
        return self._expire(now)

    def _finish(self, tid: int) -> SessionRecord:
        s = self.live.pop(tid)
        self._close_zone(s, s.last)
        return SessionRecord(s.start, s.last, s.entered, s.exited, s.visits)

    def _expire(self, now: float) -> List[SessionRecord]:
        done = [tid for tid, s in self.live.items() if now - s.last > self.timeout]
        return [self._finish(t) for t in done]

    def flush(self) -> List[SessionRecord]:
        """Finish every open session (shutdown / end of video)."""
        return [self._finish(t) for t in list(self.live)]

    def current_zone_counts(self) -> Dict[str, int]:
        """People currently inside each zone (for the live view)."""
        counts = {n: 0 for n in self.zones.names}
        for s in self.live.values():
            if s.zone is not None:
                counts[s.zone] += 1
        return counts
