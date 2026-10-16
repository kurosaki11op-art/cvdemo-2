"""Gaussian dwell-time heatmap accumulation and colour rendering."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


class HeatmapAccumulator:
    """Float canvas that accumulates a Gaussian splat per person per frame.

    ``decay`` < 1 makes old footprints fade (live display); ``decay == 1`` keeps a pure
    cumulative dwell count (what gets stored in snapshots).
    """

    def __init__(self, size: Tuple[int, int], radius: int = 18, sigma: float = 7.0,
                 intensity: float = 1.0, decay: float = 1.0) -> None:
        self.w, self.h = size
        self.canvas = np.zeros((self.h, self.w), dtype=np.float32)
        self.decay = decay
        self.radius = radius
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        self.kernel = (intensity * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))).astype(np.float32)

    def add(self, x: float, y: float) -> None:
        """Splat the kernel centred at (x, y); clipped at the borders, ignored if far outside."""
        r = self.radius
        cx, cy = int(round(x)), int(round(y))
        if cx < -r or cy < -r or cx >= self.w + r or cy >= self.h + r:
            return
        x0, y0 = max(cx - r, 0), max(cy - r, 0)
        x1, y1 = min(cx + r + 1, self.w), min(cy + r + 1, self.h)
        kx0, ky0 = x0 - (cx - r), y0 - (cy - r)
        self.canvas[y0:y1, x0:x1] += self.kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]

    def add_points(self, points: Iterable[Tuple[float, float]]) -> None:
        for x, y in points:
            self.add(x, y)

    def step_decay(self) -> None:
        if self.decay < 1.0:
            self.canvas *= self.decay

    def reset(self) -> None:
        self.canvas[:] = 0


def render_heat(heat: np.ndarray, min_peak: float = 30.0, blur_sigma: float = 0.0,
                markers: Optional[Iterable[Tuple[float, float]]] = None) -> np.ndarray:
    """Colour-map an accumulation matrix (JET: blue = quiet, red = busy) with a floor grid.

    Normalisation divides by max(peak, min_peak) so a few frames of data don't saturate.
    """
    h, w = heat.shape
    norm = heat / max(float(heat.max()), min_peak)
    heat8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    if blur_sigma > 0:
        heat8 = cv2.GaussianBlur(heat8, (0, 0), sigmaX=blur_sigma)
    img = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
    grid = np.zeros_like(img)
    for x in range(0, w, 50):
        cv2.line(grid, (x, 0), (x, h - 1), (60, 60, 60), 1)
    for y in range(0, h, 50):
        cv2.line(grid, (0, y), (w - 1, y), (60, 60, 60), 1)
    img = cv2.addWeighted(img, 0.85, grid, 0.15, 0)
    for px, py in (markers if markers is not None else ()):
        cv2.circle(img, (int(px), int(py)), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py)), 6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
    return img
