"""The headless production loop: read frames, count, persist, heartbeat, reconnect."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from .config import Config
from .engine import CountingEngine
from .heatmap import render_heat
from .occupancy import OccupancyTracker
from .sessions import SessionTracker
from .sources import FrameSource, SourceError
from .storage import EventStore
from .zones import ZoneMap

log = logging.getLogger("nexus.service")


class JsonFormatter(logging.Formatter):
    """One JSON object per log line (easy to ship to Loki/ELK/CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_logs
                         else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


class Service:
    """Runs one camera. Everything external is injectable so it can be tested end to end."""

    def __init__(self, cfg: Config, engine: Optional[CountingEngine] = None,
                 source: Optional[FrameSource] = None, store: Optional[EventStore] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.clock = clock
        self.store = store or EventStore(cfg.storage.db_path)
        self._engine = engine            # built lazily so a bad source fails before the model loads
        self.source = source or FrameSource(cfg.source, loop=cfg.service.loop)
        sv = cfg.service
        a = cfg.analytics
        self.sessions = SessionTracker(ZoneMap(cfg.zones), a.min_zone_dwell_s, a.switch_hold_s, a.session_timeout_s)
        self.occupancy = OccupancyTracker(self.store, cfg.camera_id, sv.initial_occupancy, sv.occupancy_reset)
        self.stop_event = threading.Event()
        self.state = "starting"
        self.frames = 0
        self.last_event_ts: Optional[float] = None
        self._fps = 0.0
        self._period_start = self.clock()

    @property
    def engine(self) -> CountingEngine:
        if self._engine is None:
            self._engine = CountingEngine(self.cfg)
        return self._engine

    # -- control --------------------------------------------------------------- #
    def stop(self, *_: object) -> None:
        log.info("stop requested")
        self.stop_event.set()

    def _install_signals(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, self.stop)

    # -- persistence helpers -------------------------------------------------------- #
    def _flush_heat(self, now: float) -> None:
        grid = self.engine.period_heat.canvas
        if grid.any():
            self.store.add_heat(self.cfg.camera_id, self._period_start, now, grid)
            self.engine.period_heat.reset()
        self._period_start = now

    def _save_sessions(self, records: list) -> None:
        for rec in records:
            self.store.add_session(self.cfg.camera_id, rec)

    def _write_status(self, now: float) -> None:
        status = {"zone_counts": self.sessions.current_zone_counts(),
                  "camera_id": self.cfg.camera_id, "state": self.state, "ts": now,
                  "fps": round(self._fps, 1), "frames": self.frames,
                  "session_entries": self.engine.entries, "session_exits": self.engine.exits,
                  "occupancy": self.occupancy.value, "last_event_ts": self.last_event_ts}
        path = self.cfg.status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status), encoding="utf-8")
        os.replace(tmp, path)                       # atomic: readers never see half a file

    def _write_live_heatmap(self) -> None:
        img = render_heat(self.engine.display_heat.canvas, min_peak=30.0)
        path = Path(self.cfg.live_heatmap_path)
        tmp = path.with_suffix(".tmp.png")
        cv2.imwrite(str(tmp), img)
        os.replace(tmp, path)

    # -- main loop --------------------------------------------------------------------- #
    def run(self) -> int:
        cfg, sv = self.cfg, self.cfg.service
        self._install_signals()
        try:
            self.source.open()
        except SourceError as exc:
            log.error("cannot start: %s", exc)
            self.state = "error"
            self._write_status(self.clock())
            return 1
        log.info("camera %s started (source=%r)", cfg.camera_id, cfg.source)
        self.state = "running"
        min_interval = 1.0 / sv.max_fps if (sv.max_fps > 0 and self.source.is_live) else 0.0
        last_proc = last_beat = last_snap = last_live = last_purge = 0.0
        last_fps_t, fps_frames = self.clock(), 0
        try:
            while not self.stop_event.is_set():
                frame = self.source.read()
                if frame is None:
                    if self.source.finished:
                        log.info("end of stream")
                        break
                    self.state = "reconnecting"
                    self._write_status(self.clock())
                    log.warning("no frames; reconnecting in %.0fs", sv.reconnect_delay_s)
                    if self.stop_event.wait(sv.reconnect_delay_s):
                        break
                    if self.source.reopen():
                        self._save_sessions(self.sessions.flush())
                        self.engine.reset_tracking()
                        self.state = "running"
                    continue
                if self.source.consume_wrap():
                    self._save_sessions(self.sessions.flush())
                    self.engine.reset_tracking()
                now = self.clock()
                if min_interval and now - last_proc < min_interval:
                    continue                          # throttle live sources to max_fps
                last_proc = now

                result = self.engine.process(frame)
                self.frames += 1
                fps_frames += 1
                obs = [(int(t), (float(p[0]), float(p[1])))
                       for t, p in zip(result.ids, result.floor_points, strict=True)]
                self._save_sessions(self.sessions.update(now, obs, [(e.track_id, e.direction) for e in result.events]))
                for ev in result.events:
                    self.store.add_event(now, cfg.camera_id, ev.direction)
                    self.occupancy.apply(ev.direction)
                    self.last_event_ts = now
                    log.info("%s (occupancy=%d)", ev.direction, self.occupancy.value)

                if now - last_fps_t >= 2.0:
                    self._fps = fps_frames / (now - last_fps_t)
                    last_fps_t, fps_frames = now, 0
                if now - last_beat >= sv.heartbeat_s:
                    last_beat = now
                    if self.occupancy.maybe_reset(now):
                        log.info("occupancy reset to %d", self.occupancy.initial)
                    self._write_status(now)
                if now - last_live >= 10.0:
                    last_live = now
                    self._write_live_heatmap()
                if now - last_snap >= cfg.storage.snapshot_interval_s:
                    last_snap = now
                    self._flush_heat(now)
                if now - last_purge >= 3600.0:
                    last_purge = now
                    removed = self.store.purge(cfg.storage.retention_days, now)
                    if removed:
                        log.info("retention purge removed %d rows", removed)
        except Exception:
            log.exception("fatal error in main loop")
            self.state = "error"
            return_code = 1
        else:
            return_code = 0
        finally:
            now = self.clock()
            self._save_sessions(self.sessions.flush())
            self._flush_heat(now)
            if self.state != "error":
                self.state = "stopped"
            self._write_status(now)
            self.source.release()
            log.info("camera %s stopped (%d frames)", cfg.camera_id, self.frames)
        return return_code
