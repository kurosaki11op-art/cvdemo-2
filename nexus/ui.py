"""OpenCV demo window: 640x480 live feed + 640x480 analytics pane (floor heatmap + KPIs)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .engine import CountingEngine, FrameResult
from .heatmap import render_heat


def _fit(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Letterbox `img` into (w, h) keeping its aspect ratio."""
    w, h = size
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    out = np.full((h, w, 3), 20, dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def draw_zones(img: np.ndarray, zones, counts: Optional[Dict[str, int]] = None) -> None:
    """Zone outlines and names on a floor-map image (optionally with live headcount)."""
    for z in zones:
        pts = np.array(z.polygon, dtype=np.int32).reshape(-1, 1, 2)
        colour = (240, 180, 41) if z.kind != "entrance" else (160, 160, 160)
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        label = z.name + (f" ({counts[z.name]})" if counts and counts.get(z.name) else "")
        x, y = int(pts[:, 0, 0].min()) + 6, int(pts[:, 0, 1].min()) + 16
        cv2.putText(img, label, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class DemoView:
    """Composes the dual-pane demo frame from an engine and its latest FrameResult."""

    def __init__(self, engine: CountingEngine) -> None:
        self.engine = engine
        self.w, self.h = engine.cfg.frame_size

    def compose(self, result: FrameResult, initial_occupancy: int = 0,
                zone_counts: Optional[Dict[str, int]] = None) -> np.ndarray:
        flashes: Dict[int, str] = {e.track_id: e.direction for e in result.events}
        return cv2.hconcat([self._left(result, flashes), self._right(result, initial_occupancy, zone_counts)])

    # -- left pane ---------------------------------------------------------------- #
    def _left(self, result: FrameResult, flashes: Dict[int, str]) -> np.ndarray:
        pane = result.frame.copy()
        eng = self.engine
        poly = eng.mapper.src.astype(np.int32).reshape(-1, 1, 2)
        overlay = pane.copy()
        cv2.fillPoly(overlay, [poly], (255, 120, 0))
        cv2.addWeighted(overlay, 0.15, pane, 0.85, 0, pane)
        cv2.polylines(pane, [poly], True, (255, 120, 0), 2, cv2.LINE_AA)

        tw = eng.tripwire
        a, b = tuple(int(v) for v in tw.a), tuple(int(v) for v in tw.b)
        cv2.line(pane, a, b, (0, 255, 255), 3, cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(pane, p, 6, (0, 255, 255), -1)
        mid = (tw.a + tw.b) / 2
        tip = mid + tw.into_vector * 45
        cv2.arrowedLine(pane, tuple(int(v) for v in mid), tuple(int(v) for v in tip),
                        (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.4)
        cv2.putText(pane, "IN", tuple(int(v) for v in tip + tw.into_vector * 6 - np.array([10, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        for (x1, y1, x2, y2), tid, (px, py) in zip(result.xyxy, result.ids, result.points, strict=True):
            flash = flashes.get(int(tid))
            color = (0, 255, 0) if flash == "IN" else (60, 60, 255) if flash == "OUT" else (255, 200, 0)
            cv2.rectangle(pane, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID {tid}" + (f" {flash}" if flash else "")
            (tw_px, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(int(y1) - th - 8, 0)
            cv2.rectangle(pane, (int(x1), top), (int(x1) + tw_px + 6, top + th + 8), color, -1)
            cv2.putText(pane, label, (int(x1) + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(pane, (int(px), int(py)), 5, (0, 0, 255), -1)
        cv2.putText(pane, "LIVE FEED", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return pane

    # -- right pane --------------------------------------------------------------- #
    def _right(self, result: FrameResult, initial_occupancy: int,
               zone_counts: Optional[Dict[str, int]] = None) -> np.ndarray:
        eng = self.engine
        floor = render_heat(eng.display_heat.canvas, min_peak=30.0, markers=result.floor_points)
        draw_zones(floor, eng.cfg.zones, zone_counts)
        cv2.putText(floor, "TOP-DOWN FLOOR HEATMAP", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        floor = _fit(floor, (self.h, self.h))
        hud_w = self.w - self.h
        hud = np.full((self.h, hud_w, 3), 20, dtype=np.uint8)
        occupancy = max(0, initial_occupancy + eng.entries - eng.exits)
        rows = [("Total Entries", eng.entries, (0, 255, 0)),
                ("Total Exits", eng.exits, (60, 60, 255)),
                ("Current Occupancy", occupancy, (255, 255, 255)),
                ("In view", len(result.ids), (200, 200, 200))]
        row_h = self.h // len(rows)
        for i, (label, value, color) in enumerate(rows):
            y0 = i * row_h
            cv2.putText(hud, label, (8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(hud, str(value), (8, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3, cv2.LINE_AA)
            cv2.line(hud, (0, y0 + row_h - 1), (hud_w, y0 + row_h - 1), (70, 70, 70), 1)
        return cv2.hconcat([floor, hud])
