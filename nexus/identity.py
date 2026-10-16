"""Identity hygiene between the tracker and the counter.

Handles four real-world problems of raw detector/tracker output:
duplicate boxes on one person, reflections ("twins"), ID switches, and flickering
false positives.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np


def contained_duplicates(xyxy: np.ndarray, confs: np.ndarray, thresh: float = 0.8) -> Set[int]:
    """Indices of boxes lying (almost) inside a stronger box: two boxes on one person."""
    drop: Set[int] = set()
    if len(xyxy) < 2:
        return drop
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for i in range(len(xyxy)):
        for j in range(len(xyxy)):
            if i == j or j in drop:
                continue
            iw = min(xyxy[i, 2], xyxy[j, 2]) - max(xyxy[i, 0], xyxy[j, 0])
            ih = min(xyxy[i, 3], xyxy[j, 3]) - max(xyxy[i, 1], xyxy[j, 1])
            inter = max(iw, 0) * max(ih, 0)
            if inter / max(areas[i], 1e-6) > thresh and (confs[j], areas[j]) >= (confs[i], areas[i]):
                drop.add(i)
                break
    return drop


class IdentityFilter:
    """Makes track IDs trustworthy before they reach the counter.

    * Confirmation: a track only counts after ``min_frames`` sightings (drops flicker).
    * Twin suppression: two simultaneous boxes with near-identical clothing colours and
      similar size are treated as ONE person (typical for a mirror/glass reflection);
      the older / more confident one is kept. Disable if people wear identical uniforms.
    * Re-association: if the tracker loses a confirmed person and issues a new raw ID
      shortly after, nearby and with matching appearance, the new ID is mapped back to the
      old one so the person is not counted twice.

    Appearance is an HSV histogram of the upper body (flip-invariant, so mirror images
    match). It is used only within a session and is never stored.
    """

    def __init__(self, min_frames: int = 5, twin: bool = True, reid: bool = True,
                 twin_thresh: float = 0.85, reid_thresh: float = 0.7, reid_frames: int = 60,
                 reid_max_dist: float = 500.0, forget_frames: int = 900) -> None:
        self.min_frames, self.twin, self.reid = min_frames, twin, reid
        self.twin_thresh, self.reid_thresh = twin_thresh, reid_thresh
        self.reid_frames, self.reid_max_dist = reid_frames, reid_max_dist
        self.forget_frames = forget_frames
        self.reset()

    def reset(self) -> None:
        self.frame_no = 0
        self.canon: Dict[int, int] = {}      # raw tracker id -> canonical id
        self.hits: Dict[int, int] = {}       # canonical id -> frames seen
        self.last_seen: Dict[int, int] = {}  # canonical id -> frame number
        self.gallery: Dict[int, dict] = {}   # canonical id -> last appearance (confirmed only)

    def active_ids(self) -> Set[int]:
        """Canonical ids still remembered (anything else can be forgotten by the counter)."""
        return set(self.hits)

    @staticmethod
    def _hist(frame: np.ndarray, box: Sequence[float]):
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        w, h = x2 - x1, y2 - y1
        x1, x2 = x1 + int(0.15 * w), x2 - int(0.15 * w)
        y2 = y1 + int(0.6 * h)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_img), min(y2, h_img)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)

    @staticmethod
    def _sim(a, b) -> float:
        if a is None or b is None:
            return 0.0
        return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))

    @staticmethod
    def _area(box: Sequence[float]) -> float:
        return max(box[2] - box[0], 1.0) * max(box[3] - box[1], 1.0)

    def update(self, frame: np.ndarray, xyxy: np.ndarray, raw_ids: Sequence[int],
               confs: Sequence[float]) -> Tuple[List[int], List[int]]:
        """Returns (indices_to_keep, canonical_ids_for_those_indices)."""
        self.frame_no += 1
        n = len(xyxy)
        hists = [self._hist(frame, b) for b in xyxy]
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in xyxy]

        # 1) raw id -> canonical id (re-associate brand-new raw ids with recently lost people)
        canon_of: List[int] = [0] * n
        present = {self.canon[int(r)] for r in raw_ids if int(r) in self.canon}
        for k, raw in enumerate(int(r) for r in raw_ids):
            if raw in self.canon:
                canon_of[k] = self.canon[raw]
                continue
            best, best_sim = None, self.reid_thresh
            if self.reid:
                for cid, g in self.gallery.items():
                    if cid in present or self.frame_no - g["frame"] > self.reid_frames:
                        continue
                    dist = float(np.hypot(centers[k][0] - g["center"][0], centers[k][1] - g["center"][1]))
                    ratio = self._area(xyxy[k]) / g["area"]
                    sim = self._sim(hists[k], g["hist"])
                    if dist <= self.reid_max_dist and 0.4 <= ratio <= 2.5 and sim >= best_sim:
                        best, best_sim = cid, sim
            self.canon[raw] = best if best is not None else raw
            canon_of[k] = self.canon[raw]
            present.add(canon_of[k])

        # 2) twin suppression: strongest (oldest, then most confident) first
        order = sorted(range(n), key=lambda k: (self.hits.get(canon_of[k], 0), confs[k]), reverse=True)
        kept: List[int] = []
        for k in order:
            if self.twin and any(
                    self._sim(hists[k], hists[j]) >= self.twin_thresh
                    and 0.25 <= self._area(xyxy[k]) / self._area(xyxy[j]) <= 4.0
                    for j in kept):
                continue
            kept.append(k)

        # 3) confirmation + gallery update
        out: List[int] = []
        for k in sorted(kept):
            cid = canon_of[k]
            self.hits[cid] = self.hits.get(cid, 0) + 1
            self.last_seen[cid] = self.frame_no
            if self.hits[cid] >= self.min_frames:
                out.append(k)
                self.gallery[cid] = {"hist": hists[k], "center": centers[k],
                                     "area": self._area(xyxy[k]), "frame": self.frame_no}
        if self.frame_no % 300 == 0:
            self._prune()
        return out, [canon_of[k] for k in out]

    def _prune(self) -> None:
        """Forget people unseen for `forget_frames` so memory stays bounded on 24/7 runs."""
        stale = {c for c, f in self.last_seen.items() if self.frame_no - f > self.forget_frames}
        for c in stale:
            self.hits.pop(c, None)
            self.last_seen.pop(c, None)
            self.gallery.pop(c, None)
        if stale:
            self.canon = {raw: c for raw, c in self.canon.items() if c not in stale}
