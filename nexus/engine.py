"""The headless counting engine: detector -> identity filter -> tripwire + floor heatmap.

No windows, files or databases here, so it can be unit-tested with a fake detector and
reused by the demo UI, the production service and the evaluation tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .config import Config
from .geometry import FloorMapper, TripwireCounter
from .heatmap import HeatmapAccumulator
from .identity import IdentityFilter, contained_duplicates


@dataclass
class Detections:
    """People found in one frame, with persistent tracker ids."""
    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    confs: np.ndarray = field(default_factory=lambda: np.empty((0,)))


class Detector(Protocol):
    """Anything that turns a BGR frame into tracked person detections."""

    def track(self, frame: np.ndarray) -> Detections: ...

    def reset(self) -> None: ...


class YoloDetector:
    """YOLOv8 person detector with the built-in ByteTrack tracker (ultralytics)."""

    def __init__(self, model: str = "yolov8n.pt", device: Optional[str] = None, conf: float = 0.5,
                 imgsz: int = 640, tracker: str = "bytetrack.yaml") -> None:
        from ultralytics import YOLO  # imported lazily: heavy, and not needed by the tests
        self.model = YOLO(model)
        self.device, self.conf, self.imgsz, self.tracker = device, conf, imgsz, tracker

    def track(self, frame: np.ndarray) -> Detections:
        result = self.model.track(frame, persist=True, tracker=self.tracker, classes=[0],
                                  conf=self.conf, imgsz=self.imgsz, device=self.device,
                                  verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return Detections()
        return Detections(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy().astype(int),
                          boxes.conf.cpu().numpy())

    def reset(self) -> None:
        self.model.predictor = None          # rebuilt (fresh tracker) on the next call


@dataclass
class CrossingEvent:
    direction: str      # "IN" or "OUT"
    track_id: int


@dataclass
class FrameResult:
    frame: np.ndarray                 # the resized frame that was processed
    xyxy: np.ndarray                  # confirmed person boxes
    ids: np.ndarray                   # canonical track ids
    points: List[Tuple[float, float]]  # tracked anchor point per person (camera pixels)
    floor_points: np.ndarray          # the same points on the floor canvas
    events: List[CrossingEvent]
    hidden_ids: List[int] = field(default_factory=list)   # alive but suppressed this frame (reflections)


class CountingEngine:
    """Frame in -> confirmed people, crossing events and heatmap updates out."""

    def __init__(self, cfg: Config, detector: Optional[Detector] = None) -> None:
        self.cfg = cfg
        d = cfg.detection
        self.detector: Detector = detector or YoloDetector(d.model, d.device, d.conf, d.imgsz, d.tracker)
        self.identity = IdentityFilter(d.min_frames, twin=d.twin_filter, reid=d.reid)
        t = cfg.tripwire
        self.tripwire = TripwireCounter(t.a, t.b, t.in_direction, t.deadband_px, t.rearm_px)
        self.mapper = FloorMapper(cfg.homography.src, cfg.homography.dst)
        canvas = tuple(cfg.homography.canvas)
        self.display_heat = HeatmapAccumulator(canvas, decay=0.999)   # live view, fades slowly
        self.period_heat = HeatmapAccumulator(canvas, decay=1.0)      # pure dwell, for storage
        self.frame_count = 0

    # -- main step ---------------------------------------------------------- #
    def process(self, frame: np.ndarray) -> FrameResult:
        w, h = self.cfg.frame_size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        self.frame_count += 1

        det = self.detector.track(frame)
        xyxy, ids, confs = det.xyxy, np.asarray(det.ids, dtype=int), det.confs
        if len(xyxy):
            keep_idx = [k for k in range(len(ids)) if k not in contained_duplicates(xyxy, confs)]
            xyxy, ids, confs = xyxy[keep_idx], ids[keep_idx], confs[keep_idx]
        keep, canon_ids = self.identity.update(frame, xyxy, ids, confs)
        xyxy, ids = xyxy[keep], np.asarray(canon_ids, dtype=int)

        points = [self._anchor(b) for b in xyxy]
        floor_pts = self.mapper.transform(points)

        events: List[CrossingEvent] = []
        for tid, pt in zip(ids, points, strict=True):
            direction = self.tripwire.update(int(tid), pt)
            if direction:
                events.append(CrossingEvent(direction, int(tid)))

        self.display_heat.add_points(floor_pts)
        self.display_heat.step_decay()
        self.period_heat.add_points(floor_pts)

        if self.frame_count % 300 == 0:
            self.tripwire.forget(self.identity.active_ids())
        return FrameResult(frame, xyxy, ids, points, floor_pts, events, sorted(self.identity.suppressed))

    def _anchor(self, box: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        y = y2 if self.cfg.detection.anchor == "feet" else (y1 + y2) / 2.0
        return ((x1 + x2) / 2.0, float(y))

    # -- control ------------------------------------------------------------ #
    @property
    def entries(self) -> int:
        return self.tripwire.entries

    @property
    def exits(self) -> int:
        return self.tripwire.exits

    def reset_tracking(self) -> None:
        """New tracker + identities (e.g. a video looped, or the stream reconnected)."""
        self.detector.reset()
        self.identity.reset()
        self.tripwire.reset_tracks()

    def reset_counts(self) -> None:
        self.tripwire.reset()
        self.display_heat.reset()
        self.period_heat.reset()
