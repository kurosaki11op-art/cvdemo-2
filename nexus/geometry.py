"""Tripwire crossing logic and the camera-to-floor homography."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


class TripwireCounter:
    """Directional line-crossing detector based on the 2D cross product.

    For the directed line A->B and a point P, ``cross(B-A, P-A)`` is positive when P is on
    the screen-right of A->B (image y points down). A crossing is a flip of a track's last
    confirmed side, taken only when

    * the point is clear of the line by ``deadband_px`` (hysteresis: jitter cannot flip it),
    * the crossing happens along the finite segment, not the infinite extension, and
    * the track is not "armed out": after a count the ID is blocked from counting again
      until it has moved ``rearm_px`` away from the line. While blocked its side is not
      updated, so hovering or wobbling around the line is counted once, while a genuine
      return trip is counted again.
    """

    def __init__(self, a: Point, b: Point, in_direction: str = "left_to_right",
                 deadband_px: float = 6.0, rearm_px: float = 40.0) -> None:
        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.vec = self.b - self.a
        self.length = float(np.hypot(*self.vec))
        if self.length == 0:
            raise ValueError("tripwire endpoints must be different points")
        if in_direction not in ("left_to_right", "right_to_left"):
            raise ValueError("in_direction must be 'left_to_right' or 'right_to_left'")
        self.in_sign = 1 if in_direction == "left_to_right" else -1
        self.deadband_px = deadband_px
        self.rearm_px = rearm_px
        self.prev_side: Dict[int, int] = {}
        self.prev_pos: Dict[int, np.ndarray] = {}
        self.counted_ids: set = set()
        self.entries = 0
        self.exits = 0

    # -- geometry ----------------------------------------------------------- #
    def signed_distance(self, p: np.ndarray) -> float:
        v = p - self.a
        return float(self.vec[0] * v[1] - self.vec[1] * v[0]) / self.length

    def within_segment(self, p: np.ndarray) -> bool:
        t = float(np.dot(p - self.a, self.vec)) / (self.length ** 2)
        return 0.0 <= t <= 1.0

    @property
    def into_vector(self) -> np.ndarray:
        """Unit vector pointing to the IN side (for drawing the direction arrow)."""
        normal = np.array([-self.vec[1], self.vec[0]]) / self.length
        return normal * self.in_sign

    # -- state -------------------------------------------------------------- #
    def update(self, track_id: int, point: Point) -> Optional[str]:
        """Feed one tracked point. Returns "IN", "OUT" or None."""
        p = np.asarray(point, dtype=np.float64)
        dist = self.signed_distance(p)
        previous = self.prev_pos.get(track_id)
        self.prev_pos[track_id] = p

        if abs(dist) > self.rearm_px:
            self.counted_ids.discard(track_id)
        if abs(dist) < self.deadband_px:
            return None
        side = 1 if dist > 0 else -1
        old_side = self.prev_side.get(track_id)
        if old_side is None:
            self.prev_side[track_id] = side
            return None
        if side == old_side:
            return None
        if track_id in self.counted_ids:
            return None             # wobble right after a count: keep the counted side
        self.prev_side[track_id] = side
        if not (self.within_segment(p) or (previous is not None and self.within_segment(previous))):
            return None

        self.counted_ids.add(track_id)
        if side == self.in_sign:
            self.entries += 1
            return "IN"
        self.exits += 1
        return "OUT"

    def forget(self, keep_ids: Iterable[int]) -> None:
        """Drop state of tracks that no longer exist (bounds memory on long runs)."""
        keep = set(keep_ids)
        for store in (self.prev_side, self.prev_pos):
            for tid in [t for t in store if t not in keep]:
                del store[tid]
        self.counted_ids &= keep

    def reset_tracks(self) -> None:
        self.prev_side.clear()
        self.prev_pos.clear()
        self.counted_ids.clear()

    def reset(self) -> None:
        self.reset_tracks()
        self.entries = self.exits = 0


class FloorMapper:
    """Maps camera-frame ground points to top-down floor-canvas coordinates."""

    def __init__(self, src: Sequence[Point], dst: Sequence[Point]) -> None:
        self.src = np.asarray(src, dtype=np.float32)
        self.dst = np.asarray(dst, dtype=np.float32)
        if self.src.shape != (4, 2) or self.dst.shape != (4, 2):
            raise ValueError("exactly 4 source and 4 destination points are required")
        self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)

    def transform(self, points: Sequence[Point]) -> np.ndarray:
        """(N points) -> ndarray (N, 2) of floor coordinates (X', Y')."""
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)
