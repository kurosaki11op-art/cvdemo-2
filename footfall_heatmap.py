#!/usr/bin/env python3
"""Retail Footfall Counter and 2D Spatial Heatmap.

YOLOv8 + ByteTrack person tracking, directional tripwire counting, homography
mapping to a top-down floor plan, and a live dwell-time heatmap, shown in one
dual-pane window.

Usage:
    python footfall_heatmap.py --source store.mp4 --loop
    python footfall_heatmap.py --source 0                  # webcam
    python footfall_heatmap.py --source 0 --anchor center  # close-up webcam test
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
FRAME_W, FRAME_H = 1280, 720          # processing resolution

# Virtual tripwire (x1, y1) -> (x2, y2), in processing-frame pixels (DUMMY VALUES).
TRIPWIRE = ((240, 480), (1040, 480))
# "left_to_right": crossing from the left-hand side of the directed line A->B to
# its right-hand side counts as IN (for a line drawn left->right on screen that
# means moving DOWN the screen). Use "right_to_left" to swap IN and OUT.
TRIPWIRE_IN_DIRECTION = "left_to_right"

# Homography calibration (DUMMY VALUES - recalibrate, see guide above).
# Order: top-left, top-right, bottom-right, bottom-left.
FLOOR_SRC_POINTS = [(380, 250), (900, 250), (1180, 700), (100, 700)]
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
    model: YOLO = field(init=False)
    counter: TripwireCounter = field(init=False)
    mapper: FloorMapper = field(init=False)
    heatmap: HeatmapAccumulator = field(init=False)

    def __post_init__(self):
        self.model = YOLO(MODEL_WEIGHTS)
        self.counter = TripwireCounter(*TRIPWIRE, in_direction=TRIPWIRE_IN_DIRECTION)
        self.mapper = FloorMapper(FLOOR_SRC_POINTS, FLOOR_DST_POINTS)
        self.heatmap = HeatmapAccumulator(FLOOR_GRID_SIZE)
        self.is_file = not str(self.source).isdigit()

    # -- video io ----------------------------------------------------------- #
    def _open(self):
        src = self.source if self.is_file else int(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")
        return cap

    def _reset_tracker(self):
        """Forget tracker state (new IDs) - used when a video loops."""
        self.model.predictor = None          # rebuilt on next .track() call
        self.counter.last_side.clear()
        self.counter.last_point.clear()

    # -- per-frame ---------------------------------------------------------- #
    def _track(self, frame):
        results = self.model.track(
            frame, persist=True, tracker=TRACKER_CFG, classes=[0],
            conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return np.empty((0, 4)), np.empty((0,), dtype=int)
        return boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy().astype(int)

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

        self._draw_hud(left, len(ids))

        right = self.heatmap.render(floor_pts, out_size=(FRAME_H, FRAME_H))
        return cv2.hconcat([left, right])

    def _draw_hud(self, img, in_view):
        lines = [
            (f"IN: {self.counter.count_in}", (0, 255, 0)),
            (f"OUT: {self.counter.count_out}", (0, 0, 255)),
            (f"Current Occupancy: {self.counter.occupancy}", (255, 255, 255)),
            (f"In view: {in_view}", (200, 200, 200)),
        ]
        cv2.rectangle(img, (10, 10), (360, 20 + 34 * len(lines)), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (360, 20 + 34 * len(lines)), (255, 255, 255), 1)
        for i, (text, color) in enumerate(lines):
            cv2.putText(img, text, (20, 42 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2, cv2.LINE_AA)

    # -- main loop ---------------------------------------------------------- #
    def run(self):
        cap = self._open()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if self.is_file and self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._reset_tracker()
                        continue
                    print("End of stream.")
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                cv2.imshow(WINDOW_NAME, self.process_frame(frame))
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
    ap.add_argument("--pick-points", action="store_true",
                    help="click on the first frame to print calibration coordinates")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.pick_points:
        pick_points(args.source)
        return 0
    try:
        FootfallApp(args.source, loop=args.loop, conf=args.conf,
                    imgsz=args.imgsz, device=args.device, anchor=args.anchor).run()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
