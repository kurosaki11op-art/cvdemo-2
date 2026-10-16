#!/usr/bin/env python3
"""Retail Footfall Counter and 2D Spatial Heatmap.

YOLOv8 + ByteTrack person tracking, directional tripwire counting, homography
mapping to a top-down floor plan, and a live dwell-time heatmap, shown in one
dual-pane window.

Usage:
    python footfall_heatmap.py --source store.mp4 --loop
    python footfall_heatmap.py --source 0                  # webcam
    python footfall_heatmap.py --source 0 --webcam-demo    # desk test: lean right = IN, left = OUT
    python footfall_heatmap.py --source store.mp4 --pick-points   # calibration helper

Keys: 'q' quit | 'r' reset counts and heatmap

=============================  CALIBRATION GUIDE  =============================
All coordinates below live in the *processing frame* (FRAME_W x FRAME_H, every
frame is resized to it), so they stay valid regardless of the source resolution.

1. Run with --pick-points. Click on the image; coordinates are printed to the
   terminal (and drawn on screen). Press any key / 'q' to finish.
2. FLOOR_SRC_POINTS: click the 4 corners of a rectangular patch of floor
   (ideally a real-world rectangle, e.g. floor tiles or a marked area), in the
   order: top-left, top-right, bottom-right, bottom-left (as seen on screen).
3. FLOOR_DST_POINTS: the same 4 corners in the top-down grid, same order. For a
   full-grid mapping keep them as the grid corners:
   (0,0) (W-1,0) (W-1,H-1) (0,H-1). Use a non-square rectangle if the real
   floor patch is not square, to keep proportions right.
4. TRIPWIRE: click the two ends of the line people cross (e.g. the doorway).
   The line's direction defines IN/OUT: see TRIPWIRE_IN_SIDE below.
===============================================================================
"""

import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Configuration (edit these for your camera)
# --------------------------------------------------------------------------- #
FRAME_W, FRAME_H = 640, 480            # processing resolution = size of EACH pane

# Virtual tripwire (x1, y1) -> (x2, y2), in processing-frame pixels (DUMMY VALUES).
TRIPWIRE = ((120, 320), (520, 320))
# "left_to_right": crossing from the left-hand side of the directed line A->B to
# its right-hand side counts as IN (for a line drawn left->right on screen that
# means moving DOWN the screen). Use "right_to_left" to swap IN and OUT.
TRIPWIRE_IN_DIRECTION = "left_to_right"

# Homography calibration (DUMMY VALUES - recalibrate, see guide above).
# Order: top-left, top-right, bottom-right, bottom-left.
FLOOR_SRC_POINTS = [(190, 167), (450, 167), (590, 467), (50, 467)]
FLOOR_GRID_SIZE = (500, 500)          # (width, height) of the top-down grid
FLOOR_DST_POINTS = [(0, 0), (499, 0), (499, 499), (0, 499)]

# Heatmap tuning
KERNEL_RADIUS = 18                    # Gaussian footprint radius on the grid (px)
KERNEL_SIGMA = 7.0
KERNEL_INTENSITY = 1.0                # added per person per frame
HEAT_DECAY = 0.9995                   # per-frame multiplicative decay (1.0 = none)
HEAT_MIN_PEAK = 30.0                  # normalisation floor: avoids saturating early on
HEAT_BLUR_KSIZE = 0                   # extra display smoothing (odd int, 0 = off)

# Tripwire tuning
SIDE_DEADBAND_PX = 6.0                # ignore jitter within this distance of the line

# Detection
MODEL_WEIGHTS = "yolov8n.pt"
TRACKER_CFG = "bytetrack.yaml"
CONF_THRESHOLD = 0.5
WINDOW_NAME = "Retail Footfall Counter & Heatmap"


# --------------------------------------------------------------------------- #
# Tripwire counter
# --------------------------------------------------------------------------- #
class TripwireCounter:
    """Counts directional line crossings per track ID using a cross-product test.

    For the directed line A->B and point P, sign(cross(B-A, P-A)) tells which side
    P is on. A crossing is a sign flip of a track's last *confirmed* side, taken
    only when the point is clear of the line by SIDE_DEADBAND_PX (hysteresis, so
    jitter on the line cannot double count) and the movement passes through the
    finite segment (not the infinite extension of the line).
    """

    def __init__(self, p1, p2, in_direction="left_to_right", deadband=SIDE_DEADBAND_PX):
        self.a = np.array(p1, dtype=np.float64)
        self.b = np.array(p2, dtype=np.float64)
        self.d = self.b - self.a
        self.length = float(np.hypot(*self.d))
        if self.length == 0:
            raise ValueError("Tripwire endpoints must differ")
        if in_direction not in ("left_to_right", "right_to_left"):
            raise ValueError("in_direction must be 'left_to_right' or 'right_to_left'")
        # Image y points down, so cross(d, p-a) > 0 means P is on the screen-RIGHT
        # of the directed line A->B. in_sign is the side people move INTO for "IN".
        self.in_sign = 1 if in_direction == "left_to_right" else -1
        self.deadband = deadband
        self.last_side = {}      # track id -> last confirmed side (+1/-1)
        self.last_point = {}     # track id -> previous point
        self.count_in = 0
        self.count_out = 0

    def _signed_distance(self, p):
        v = p - self.a
        return (self.d[0] * v[1] - self.d[1] * v[0]) / self.length

    def _within_segment(self, p):
        t = float(np.dot(p - self.a, self.d) / (self.length ** 2))
        return 0.0 <= t <= 1.0

    def update(self, track_id, point):
        """Feed one tracked point; returns 'IN', 'OUT' or None."""
        p = np.asarray(point, dtype=np.float64)
        dist = self._signed_distance(p)
        prev_p = self.last_point.get(track_id)
        self.last_point[track_id] = p

        if abs(dist) < self.deadband:
            return None                          # too close to call; keep old side
        side = 1 if dist > 0 else -1
        prev_side = self.last_side.get(track_id)
        self.last_side[track_id] = side
        if prev_side is None or prev_side == side:
            return None

        # Side flipped: make sure the crossing happened along the finite segment.
        crossing_ok = self._within_segment(p) or (prev_p is not None and self._within_segment(prev_p))
        if not crossing_ok:
            return None

        if side == self.in_sign:
            self.count_in += 1
            return "IN"
        self.count_out += 1
        return "OUT"

    def forget(self, active_ids):
        """Drop state for tracks that no longer exist (keeps memory bounded)."""
        for store in (self.last_side, self.last_point):
            for tid in list(store):
                if tid not in active_ids:
                    del store[tid]

    def reset(self):
        self.last_side.clear()
        self.last_point.clear()
        self.count_in = self.count_out = 0

    @property
    def occupancy(self):
        return max(0, self.count_in - self.count_out)

    def draw(self, frame):
        a = tuple(int(v) for v in self.a)
        b = tuple(int(v) for v in self.b)
        cv2.line(frame, a, b, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, a, 6, (0, 255, 255), -1)
        cv2.circle(frame, b, 6, (0, 255, 255), -1)
        # Arrow at the midpoint pointing in the IN direction
        normal = np.array([-self.d[1], self.d[0]]) / self.length   # points to cross>0 side
        into = normal * self.in_sign
        mid = (self.a + self.b) / 2
        cv2.arrowedLine(frame, tuple(int(v) for v in mid),
                        tuple(int(v) for v in mid + into * 45), (0, 255, 0), 3,
                        cv2.LINE_AA, tipLength=0.4)
        cv2.putText(frame, "IN", tuple(int(v) for v in mid + into * 62 - np.array([14, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Homography
# --------------------------------------------------------------------------- #
class FloorMapper:
    """Maps camera-frame ground points to top-down floor-grid coordinates."""

    def __init__(self, src_pts, dst_pts):
        src = np.asarray(src_pts, dtype=np.float32)
        dst = np.asarray(dst_pts, dtype=np.float32)
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("Exactly 4 source and 4 destination points are required")
        self.src = src
        self.matrix = cv2.getPerspectiveTransform(src, dst)

    def transform(self, points):
        """points: iterable of (x, y) -> ndarray (N, 2) of (X', Y')."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if pts.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def draw_region(self, frame):
        poly = self.src.astype(np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], (255, 120, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [poly], True, (255, 120, 0), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #
class HeatmapAccumulator:
    """Float accumulation canvas with Gaussian splats, decay and JET rendering."""

    def __init__(self, size, radius=KERNEL_RADIUS, sigma=KERNEL_SIGMA,
                 intensity=KERNEL_INTENSITY, decay=HEAT_DECAY, min_peak=HEAT_MIN_PEAK):
        self.w, self.h = size
        self.canvas = np.zeros((self.h, self.w), dtype=np.float32)
        self.decay = decay
        self.min_peak = min_peak
        self.radius = radius
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        self.kernel = (intensity * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))).astype(np.float32)
        self.base = self._make_floor_grid()

    def _make_floor_grid(self, step=50):
        grid = np.full((self.h, self.w, 3), 25, dtype=np.uint8)
        for x in range(0, self.w, step):
            cv2.line(grid, (x, 0), (x, self.h - 1), (60, 60, 60), 1)
        for y in range(0, self.h, step):
            cv2.line(grid, (0, y), (self.w - 1, y), (60, 60, 60), 1)
        return grid

    def add(self, x, y):
        """Splat the Gaussian kernel centred at (x, y); clipped at the borders."""
        cx, cy, r = int(round(x)), int(round(y)), self.radius
        if cx < -r or cy < -r or cx >= self.w + r or cy >= self.h + r:
            return                                   # entirely off the grid
        x0, y0 = max(cx - r, 0), max(cy - r, 0)
        x1, y1 = min(cx + r + 1, self.w), min(cy + r + 1, self.h)
        kx0, ky0 = x0 - (cx - r), y0 - (cy - r)
        self.canvas[y0:y1, x0:x1] += self.kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]

    def step_decay(self):
        if self.decay < 1.0:
            self.canvas *= self.decay

    def reset(self):
        self.canvas[:] = 0

    def render(self, current_points=(), out_size=None):
        """Colourised heatmap over the floor grid, with live person markers."""
        norm = self.canvas / max(float(self.canvas.max()), self.min_peak)
        heat8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        if HEAT_BLUR_KSIZE:
            heat8 = cv2.GaussianBlur(heat8, (HEAT_BLUR_KSIZE, HEAT_BLUR_KSIZE), 0)
        colored = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)    # low = blue, high = red
        # Keep the floor grid lines visible on top of the colours
        colored = cv2.addWeighted(colored, 0.85, self.base, 0.15, 0)
        for px, py in current_points:
            cv2.circle(colored, (int(px), int(py)), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(colored, (int(px), int(py)), 6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(colored, (0, 0), (self.w - 1, self.h - 1), (255, 255, 255), 2)
        cv2.putText(colored, "TOP-DOWN FLOOR HEATMAP", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if out_size is not None:
            colored = cv2.resize(colored, out_size, interpolation=cv2.INTER_LINEAR)
        return colored


# --------------------------------------------------------------------------- #
# Identity filter: duplicates, reflections and ID switches
# --------------------------------------------------------------------------- #
class IdentityFilter:
    """Makes track IDs trustworthy before they reach the counter.

    * Confirmation: a track only counts after `min_frames` sightings (drops flicker).
    * Twin suppression: two simultaneous boxes with near-identical clothing colours
      and similar size are treated as ONE person (typical for a mirror/glass
      reflection); only the older / more confident one is kept.
    * Re-association: if the tracker loses a confirmed person and issues a new raw ID
      shortly after, nearby and with matching appearance, the new ID is mapped back to
      the old one so the person is never counted as someone new.

    Appearance = HSV histogram of the upper body (flip-invariant, so mirror images match).
    """

    def __init__(self, min_frames=5, twin=True, reid=True, twin_thresh=0.85,
                 reid_thresh=0.7, reid_frames=60, reid_max_dist=500.0):
        self.min_frames, self.twin, self.reid = min_frames, twin, reid
        self.twin_thresh, self.reid_thresh = twin_thresh, reid_thresh
        self.reid_frames, self.reid_max_dist = reid_frames, reid_max_dist
        self.reset()

    def reset(self):
        self.frame_no = 0
        self.canon = {}      # raw tracker id -> canonical id
        self.hits = {}       # canonical id -> frames seen
        self.gallery = {}    # canonical id -> last known appearance/position (confirmed only)

    @staticmethod
    def _hist(frame, box):
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        w, h = x2 - x1, y2 - y1
        x1, x2 = x1 + int(0.15 * w), x2 - int(0.15 * w)      # trim background at the sides
        y2 = y1 + int(0.6 * h)                                # upper body only
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_img), min(y2, h_img)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)

    @staticmethod
    def _sim(a, b):
        if a is None or b is None:
            return 0.0
        return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))

    @staticmethod
    def _area(box):
        return max(box[2] - box[0], 1.0) * max(box[3] - box[1], 1.0)

    def update(self, frame, xyxy, raw_ids, confs):
        """Returns (indices_to_keep, canonical_ids_for_those_indices)."""
        self.frame_no += 1
        n = len(xyxy)
        hists = [self._hist(frame, b) for b in xyxy]
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in xyxy]

        # 1) raw id -> canonical id (re-associate brand-new raw ids with lost people)
        canon_of = [None] * n
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
                    dist = np.hypot(centers[k][0] - g["center"][0], centers[k][1] - g["center"][1])
                    ratio = self._area(xyxy[k]) / g["area"]
                    sim = self._sim(hists[k], g["hist"])
                    if dist <= self.reid_max_dist and 0.4 <= ratio <= 2.5 and sim >= best_sim:
                        best, best_sim = cid, sim
            self.canon[raw] = best if best is not None else raw
            canon_of[k] = self.canon[raw]
            present.add(canon_of[k])

        # 2) twin suppression: strongest (oldest, then most confident) first
        order = sorted(range(n), key=lambda k: (self.hits.get(canon_of[k], 0), confs[k]), reverse=True)
        kept = []
        for k in order:
            if self.twin and any(
                    self._sim(hists[k], hists[j]) >= self.twin_thresh
                    and 0.25 <= self._area(xyxy[k]) / self._area(xyxy[j]) <= 4.0
                    for j in kept):
                continue
            kept.append(k)

        # 3) confirmation + gallery update
        out = []
        for k in sorted(kept):
            cid = canon_of[k]
            self.hits[cid] = self.hits.get(cid, 0) + 1
            if self.hits[cid] >= self.min_frames:
                out.append(k)
                self.gallery[cid] = {"hist": hists[k], "center": centers[k],
                                     "area": self._area(xyxy[k]), "frame": self.frame_no}
        return out, [canon_of[k] for k in out]


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
@dataclass
class FootfallApp:
    source: str
    loop: bool = False
    conf: float = CONF_THRESHOLD
    imgsz: int = 640
    device: str = None
    anchor: str = "feet"
    line: tuple = None               # ((x1, y1), (x2, y2)); defaults to TRIPWIRE
    in_direction: str = TRIPWIRE_IN_DIRECTION
    min_frames: int = 5              # a track must be seen this many frames to count
    twin_filter: bool = True         # merge look-alike simultaneous boxes (reflections)
    reid: bool = True                # re-link a lost person to their old ID
    scale: float = 1.0               # display scaling only (processing is unaffected)
    model: YOLO = field(init=False)
    counter: TripwireCounter = field(init=False)
    mapper: FloorMapper = field(init=False)
    heatmap: HeatmapAccumulator = field(init=False)

    def __post_init__(self):
        self.model = YOLO(MODEL_WEIGHTS)
        self.counter = TripwireCounter(*(self.line or TRIPWIRE), in_direction=self.in_direction)
        self.identity = IdentityFilter(self.min_frames, twin=self.twin_filter, reid=self.reid)
        self.mapper = FloorMapper(FLOOR_SRC_POINTS, FLOOR_DST_POINTS)
        self.heatmap = HeatmapAccumulator(FLOOR_GRID_SIZE)
        self.is_file = not str(self.source).isdigit()

    # -- video io ----------------------------------------------------------- #
    def _open(self):
        if self.is_file:
            cap = cv2.VideoCapture(self.source)
        else:
            cap = self._open_webcam(int(self.source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")
        return cap

    @staticmethod
    def _open_webcam(index):
        """Open a webcam, preferring DirectShow on Windows (MSMF often fails to grab frames)."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                for _ in range(10):          # warm-up: first frames are often empty
                    ok, _frame = cap.read()
                    if ok:
                        return cap
            cap.release()
        raise RuntimeError(
            f"Webcam {index} opened but delivered no frames. Close other apps using the "
            "camera (another copy of this script, Zoom/Teams/browser tabs) and try "
            "--source 1 if you have more than one camera.")

    def _reset_tracker(self):
        """Forget tracker state (new IDs) - used when a video loops."""
        self.model.predictor = None          # rebuilt on next .track() call
        self.counter.last_side.clear()
        self.counter.last_point.clear()
        self.identity.reset()

    # -- per-frame ---------------------------------------------------------- #
    def _track(self, frame):
        results = self.model.track(
            frame, persist=True, tracker=TRACKER_CFG, classes=[0],
            conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return np.empty((0, 4)), np.empty((0,), dtype=int)
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        return self._filter_tracks(frame, xyxy, ids, confs)

    @staticmethod
    def _contained_duplicates(xyxy, confs, thresh=0.8):
        """Indices of boxes that are (almost) inside a stronger box: same person, 2 boxes."""
        drop = set()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        for i in range(len(xyxy)):
            for j in range(len(xyxy)):
                if i == j or j in drop:
                    continue
                iw = min(xyxy[i, 2], xyxy[j, 2]) - max(xyxy[i, 0], xyxy[j, 0])
                ih = min(xyxy[i, 3], xyxy[j, 3]) - max(xyxy[i, 1], xyxy[j, 1])
                inter = max(iw, 0) * max(ih, 0)
                # i mostly inside j, and j is the more confident / larger box -> drop i
                if inter / max(areas[i], 1e-6) > thresh and (confs[j], areas[j]) >= (confs[i], areas[i]):
                    drop.add(i)
                    break
        return drop

    def _filter_tracks(self, frame, xyxy, ids, confs):
        """Dedupe boxes, merge reflections, repair ID switches, ignore flickering tracks."""
        drop = self._contained_duplicates(xyxy, confs)
        idx = [k for k in range(len(ids)) if k not in drop]
        xyxy, ids, confs = xyxy[idx], ids[idx], confs[idx]
        keep, canon_ids = self.identity.update(frame, xyxy, ids, confs)
        return xyxy[keep], np.asarray(canon_ids, dtype=int)

    def process_frame(self, frame):
        xyxy, ids = self._track(frame)

        if self.anchor == "center":     # close-up webcam demo: use the box centre
            feet = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in xyxy]
        else:                           # real CCTV view: feet contact point (centre-bottom)
            feet = [((x1 + x2) / 2.0, y2) for x1, y1, x2, y2 in xyxy]
        floor_pts = self.mapper.transform(feet)

        flashes = {}
        for tid, foot in zip(ids, feet):
            event = self.counter.update(int(tid), foot)
            if event:
                flashes[int(tid)] = event

        for fx, fy in floor_pts:
            self.heatmap.add(fx, fy)
        self.heatmap.step_decay()

        self.counter.forget(self._recent_ids(ids))

        return self._compose(frame, xyxy, ids, feet, floor_pts, flashes)

    def _recent_ids(self, ids):
        # Keep recently-seen IDs (ByteTrack may drop an ID for a few frames).
        self._seen = getattr(self, "_seen", {})
        for i in ids:
            self._seen[int(i)] = 0
        for k in list(self._seen):
            self._seen[k] += 1
            if self._seen[k] > 90:
                del self._seen[k]
        return set(self._seen)

    # -- drawing ------------------------------------------------------------ #
    def _compose(self, frame, xyxy, ids, feet, floor_pts, flashes):
        left = frame.copy()
        self.mapper.draw_region(left)
        self.counter.draw(left)

        for (x1, y1, x2, y2), tid, foot in zip(xyxy, ids, feet):
            flash = flashes.get(int(tid))
            color = (0, 255, 0) if flash == "IN" else (0, 0, 255) if flash == "OUT" else (255, 200, 0)
            cv2.rectangle(left, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID {tid}" + (f" {flash}" if flash else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(left, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), color, -1)
            cv2.putText(left, label, (int(x1) + 3, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(left, (int(foot[0]), int(foot[1])), 5, (0, 0, 255), -1)

        cv2.putText(left, "LIVE FEED", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        right = self._build_right_pane(floor_pts, len(ids))
        return cv2.hconcat([left, right])

    def _build_right_pane(self, floor_pts, in_view):
        """640x480 analytics pane: 480x480 floor-plan heatmap + a 160px HUD column."""
        floor = self.heatmap.render(floor_pts, out_size=(FRAME_H, FRAME_H))
        hud_w = FRAME_W - FRAME_H
        hud = np.full((FRAME_H, hud_w, 3), 20, dtype=np.uint8)
        rows = [
            ("Total Entries", self.counter.count_in, (0, 255, 0)),
            ("Total Exits", self.counter.count_out, (60, 60, 255)),
            ("Current Occupancy", self.counter.occupancy, (255, 255, 255)),
            ("In view", in_view, (200, 200, 200)),
        ]
        row_h = FRAME_H // len(rows)
        for i, (label, value, color) in enumerate(rows):
            y0 = i * row_h
            cv2.putText(hud, label, (8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(hud, str(value), (8, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        color, 3, cv2.LINE_AA)
            cv2.line(hud, (0, y0 + row_h - 1), (hud_w, y0 + row_h - 1), (70, 70, 70), 1)
        return cv2.hconcat([floor, hud])

    # -- main loop ---------------------------------------------------------- #
    def run(self):
        cap = self._open()
        failed_reads = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok and not self.is_file and failed_reads < 30:
                    failed_reads += 1        # live camera: tolerate brief dropouts
                    cv2.waitKey(10)
                    continue
                failed_reads = 0
                if not ok:
                    if self.is_file and self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._reset_tracker()
                        continue
                    print("End of stream.")
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                view = self.process_frame(frame)
                if self.scale != 1.0:
                    view = cv2.resize(view, None, fx=self.scale, fy=self.scale,
                                      interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW_NAME, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self.counter.reset()
                    self.heatmap.reset()
        finally:
            cap.release()
            cv2.destroyAllWindows()
        print(f"Final -> IN: {self.counter.count_in}  OUT: {self.counter.count_out}  "
              f"Occupancy: {self.counter.occupancy}")


# --------------------------------------------------------------------------- #
# Calibration helper
# --------------------------------------------------------------------------- #
def pick_points(source):
    """Show the first frame; print coordinates of mouse clicks (processing space)."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read a frame from {source!r}")
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    shown = frame.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x}, {y})")
            cv2.circle(shown, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(shown, f"({x},{y})", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("Pick points (press q to finish)")
    cv2.setMouseCallback("Pick points (press q to finish)", on_mouse)
    while True:
        cv2.imshow("Pick points (press q to finish)", shown)
        if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="Retail footfall counter + 2D heatmap")
    ap.add_argument("--source", default="0", help="video file path or webcam index (default 0)")
    ap.add_argument("--loop", action="store_true", help="loop video files instead of exiting")
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="detection confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0, mps")
    ap.add_argument("--anchor", choices=["feet", "center"], default="feet",
                    help="point tracked for counting/heatmap: 'feet' (default, for full-body "
                         "camera views) or 'center' (for close-up webcam testing)")
    ap.add_argument("--line", default=None, metavar="X1,Y1,X2,Y2",
                    help="tripwire endpoints in 640x480 frame coords (overrides the default)")
    ap.add_argument("--in-direction", choices=["left_to_right", "right_to_left"],
                    default=TRIPWIRE_IN_DIRECTION,
                    help="crossing from the left-hand to the right-hand side of the line "
                         "(looking from its first point to its second) counts as IN")
    ap.add_argument("--webcam-demo", action="store_true",
                    help="preset for testing at a desk: tracks the box centre and uses a "
                         "vertical line in the middle; move RIGHT = IN, LEFT = OUT")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="frames a track must persist before it is shown/counted")
    ap.add_argument("--no-twin-filter", action="store_true",
                    help="keep look-alike boxes (disable reflection/duplicate merging)")
    ap.add_argument("--no-reid", action="store_true",
                    help="disable re-linking of lost tracks to their previous ID")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="window size factor (default 1.0 = 1280x480 window)")
    ap.add_argument("--pick-points", action="store_true",
                    help="click on the first frame to print calibration coordinates")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.pick_points:
        pick_points(args.source)
        return 0
    line, anchor = None, args.anchor
    if args.webcam_demo:
        # Directed upwards, so its right-hand side is the screen's right: left->right = IN
        line, anchor = ((FRAME_W // 2, FRAME_H - 100), (FRAME_W // 2, 100)), "center"
    if args.line:
        try:
            x1, y1, x2, y2 = (int(v) for v in args.line.split(","))
        except ValueError:
            print("Error: --line must be X1,Y1,X2,Y2 (four integers)", file=sys.stderr)
            return 1
        line = ((x1, y1), (x2, y2))
    try:
        FootfallApp(args.source, loop=args.loop, conf=args.conf, imgsz=args.imgsz,
                    device=args.device, anchor=anchor, line=line,
                    in_direction=args.in_direction, min_frames=args.min_frames,
                    twin_filter=not args.no_twin_filter, reid=not args.no_reid,
                    scale=args.scale).run()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
