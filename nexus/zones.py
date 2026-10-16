"""Store zones: named floor polygons, the physical-store equivalent of website pages."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ZoneConfig

Point = Tuple[float, float]


def point_in_polygon(x: float, y: float, poly: Sequence[Point]) -> bool:
    """Ray casting; points exactly on an edge count as inside."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
            elif x == xin:
                return True
    return inside


class ZoneMap:
    """Look up which zone a floor point lies in (first match wins; list small zones first)."""

    def __init__(self, zones: Sequence[ZoneConfig]) -> None:
        self.zones: List[ZoneConfig] = list(zones)
        self.kinds: Dict[str, str] = {z.name: z.kind for z in self.zones}

    @property
    def names(self) -> List[str]:
        return [z.name for z in self.zones]

    def zone_at(self, x: float, y: float) -> Optional[str]:
        for z in self.zones:
            if point_in_polygon(x, y, z.polygon):
                return z.name
        return None

    def centroid(self, name: str) -> Point:
        z = next(z for z in self.zones if z.name == name)
        pts = np.array(z.polygon, dtype=float)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def masks(self, size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Boolean raster mask per zone on a (w, h) canvas (for heat share)."""
        import cv2
        w, h = size
        out: Dict[str, np.ndarray] = {}
        for z in self.zones:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [np.array(z.polygon, dtype=np.int32).reshape(-1, 1, 2)], 1)
            out[z.name] = m.astype(bool)
        return out
