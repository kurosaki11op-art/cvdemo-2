"""Typed, validated configuration (YAML) for one NEXUS camera node."""

from __future__ import annotations

import dataclasses
import json
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised for any invalid configuration value (message says which one)."""


@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"            # .pt, or an exported .onnx / .engine file
    device: Optional[str] = None         # "cpu", "0", "mps"... (None = auto)
    conf: float = 0.5                    # raise to suppress false people
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    anchor: str = "feet"                 # "feet" (full-body views) or "center" (close-up webcam)
    min_frames: int = 5                  # frames a track must persist before it counts
    twin_filter: bool = True             # merge look-alike simultaneous boxes (reflections)
    reid: bool = True                    # re-link a briefly lost person to their old ID


@dataclass
class TripwireConfig:
    a: Any = (120, 320)
    b: Any = (520, 320)
    in_direction: str = "left_to_right"  # crossing left->right of the directed line A->B = IN
    deadband_px: float = 6.0             # ignore jitter this close to the line
    rearm_px: float = 40.0               # an ID can be counted again after moving this far away


@dataclass
class HomographyConfig:
    # Order: top-left, top-right, bottom-right, bottom-left (camera pixels in frame_size).
    src: Any = ((190, 167), (450, 167), (590, 467), (50, 467))
    dst: Any = ((0, 0), (499, 0), (499, 499), (0, 499))
    canvas: Any = (500, 500)             # (width, height) of the top-down floor grid


@dataclass
class StorageConfig:
    db_path: str = "data/nexus.db"
    snapshot_interval_s: float = 300.0   # how often heatmap deltas are written
    retention_days: int = 90             # events/snapshots older than this are purged (0 = keep)


@dataclass
class ServiceConfig:
    occupancy_reset: Optional[str] = "03:00"  # local "HH:MM" nightly reset; null = never
    initial_occupancy: int = 0                # people already inside at (re)start / reset
    max_fps: float = 15.0                     # live sources are throttled to this (0 = no limit)
    reconnect_delay_s: float = 5.0
    heartbeat_s: float = 5.0
    loop: bool = False                        # loop video files (testing)


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    api_token: Optional[str] = None           # if set, required as ?token= or Bearer header


@dataclass
class Config:
    camera_id: str = "camera-1"
    source: Any = None                        # file path, webcam index, or rtsp:// URL
    frame_size: Any = (640, 480)              # processing size; calibration uses this space
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tripwire: TripwireConfig = field(default_factory=TripwireConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # ------------------------------------------------------------------ helpers
    @property
    def data_dir(self) -> Path:
        return Path(self.storage.db_path).parent

    @property
    def status_path(self) -> Path:
        return self.data_dir / f"status-{self.camera_id}.json"

    @property
    def live_heatmap_path(self) -> Path:
        return self.data_dir / f"live-heatmap-{self.camera_id}.png"

    # ------------------------------------------------------------------ validation
    def validate(self) -> "Config":
        """Coerce types in place and raise ConfigError on anything unusable. Returns self."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(self.camera_id)):
            raise ConfigError("camera_id: use 1-64 letters, digits, '_', '.', '-'")
        self.frame_size = _int_pair(self.frame_size, "frame_size")

        d = self.detection
        if not 0.0 < float(d.conf) <= 1.0:
            raise ConfigError("detection.conf must be in (0, 1]")
        if d.anchor not in ("feet", "center"):
            raise ConfigError("detection.anchor must be 'feet' or 'center'")
        if int(d.min_frames) < 1:
            raise ConfigError("detection.min_frames must be >= 1")
        if int(d.imgsz) < 32:
            raise ConfigError("detection.imgsz must be >= 32")

        t = self.tripwire
        t.a = _float_pair(t.a, "tripwire.a")
        t.b = _float_pair(t.b, "tripwire.b")
        if t.a == t.b:
            raise ConfigError("tripwire.a and tripwire.b must be different points")
        if t.in_direction not in ("left_to_right", "right_to_left"):
            raise ConfigError("tripwire.in_direction must be 'left_to_right' or 'right_to_left'")
        if float(t.deadband_px) < 0 or float(t.rearm_px) <= float(t.deadband_px):
            raise ConfigError("tripwire: need 0 <= deadband_px < rearm_px")
        w, h = self.frame_size
        for name, (x, y) in (("a", t.a), ("b", t.b)):
            if not (0 <= x <= w and 0 <= y <= h):
                raise ConfigError(f"tripwire.{name} {(x, y)} lies outside frame_size {w}x{h}")

        hm = self.homography
        hm.canvas = _int_pair(hm.canvas, "homography.canvas")
        hm.src = _points4(hm.src, "homography.src")
        hm.dst = _points4(hm.dst, "homography.dst")
        _check_quad(hm.src, "homography.src")
        _check_quad(hm.dst, "homography.dst")
        matrix = cv2.getPerspectiveTransform(np.array(hm.src, np.float32), np.array(hm.dst, np.float32))
        if not np.all(np.isfinite(matrix)):
            raise ConfigError("homography points produce a degenerate transform")

        s = self.storage
        if float(s.snapshot_interval_s) < 1:
            raise ConfigError("storage.snapshot_interval_s must be >= 1")
        if int(s.retention_days) < 0:
            raise ConfigError("storage.retention_days must be >= 0")

        sv = self.service
        if sv.occupancy_reset is not None and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(sv.occupancy_reset)):
            raise ConfigError("service.occupancy_reset must be 'HH:MM' (24h) or null")
        if int(sv.initial_occupancy) < 0:
            raise ConfigError("service.initial_occupancy must be >= 0")
        if float(sv.max_fps) < 0:
            raise ConfigError("service.max_fps must be >= 0")

        dash = self.dashboard
        if not 1 <= int(dash.port) <= 65535:
            raise ConfigError("dashboard.port must be 1-65535")
        return self


# ---------------------------------------------------------------------- loading
def _int_pair(v: Any, name: str) -> Tuple[int, int]:
    try:
        a, b = v
        a, b = int(a), int(b)
    except (TypeError, ValueError):
        raise ConfigError(f"{name}: expected two integers like [640, 480]") from None
    if a <= 0 or b <= 0:
        raise ConfigError(f"{name}: values must be positive")
    return a, b


def _float_pair(v: Any, name: str) -> Tuple[float, float]:
    try:
        a, b = v
        return float(a), float(b)
    except (TypeError, ValueError):
        raise ConfigError(f"{name}: expected [x, y]") from None


def _points4(v: Any, name: str) -> Tuple[Tuple[float, float], ...]:
    try:
        pts = tuple(_float_pair(p, name) for p in v)
    except TypeError:
        raise ConfigError(f"{name}: expected a list of 4 [x, y] points") from None
    if len(pts) != 4:
        raise ConfigError(f"{name}: need exactly 4 points, got {len(pts)}")
    return pts


def _check_quad(pts: Tuple[Tuple[float, float], ...], name: str) -> None:
    """Reject collinear/zero-area or self-intersecting quads (wrong corner order)."""
    p = np.array(pts, dtype=np.float64)
    cross = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if any(abs(c) < 1e-6 for c in cross):
        raise ConfigError(f"{name}: points are collinear or duplicated")
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        raise ConfigError(f"{name}: points must form a convex quad in order "
                          "top-left, top-right, bottom-right, bottom-left")


def _from_dict(cls: Any, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; valid keys: {sorted(valid)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        kwargs[key] = _from_dict(hint, value, f"{where}.{key}") if dataclasses.is_dataclass(hint) else value
    return cls(**kwargs)


def config_from_dict(data: Any) -> Config:
    return _from_dict(Config, data, "config").validate()


def load_config(path: Union[str, Path]) -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML: {exc}") from None
    return config_from_dict(data)


def default_config() -> Config:
    return Config().validate()


def config_to_dict(cfg: Config) -> dict:
    # json round-trip turns tuples/numpy scalars into plain lists/numbers for YAML
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=float))


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    cfg.validate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")


__all__: List[str] = [
    "Config", "ConfigError", "DetectionConfig", "TripwireConfig", "HomographyConfig",
    "StorageConfig", "ServiceConfig", "DashboardConfig", "load_config", "save_config",
    "default_config", "config_from_dict", "config_to_dict",
]
