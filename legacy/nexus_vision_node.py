#!/usr/bin/env python3
"""LEGACY single-file version, kept for reference (superseded by the `nexus` package).

Known limitation fixed in the package: after a counted crossing, a wobble back over the line
updated the stored side and the return to the original side could be counted a second time.

NEXUS Vision Node - real-time retail footfall counter and 2D spatial heatmap.

YOLOv8 + ByteTrack people tracking, directional tripwire counting (vector cross
product), homography mapping to a 500x500 top-down floor canvas, and a decaying
Gaussian heatmap, shown in one 1280x640 dual-pane window.

Run:
    python nexus_vision_node.py --source meeting_room.mp4 --loop
    python nexus_vision_node.py --source 0
    python nexus_vision_node.py --source meeting_room.mp4 --pick-points   # click to read coordinates

Keys: 'q' quit | 'r' reset counters and heatmap
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

# =============================================================================
# CALIBRATION - EDIT THESE FOR YOUR MEETING ROOM / CAMERA
# =============================================================================
# All camera-frame coordinates are in the 640x640 DASHBOARD FRAME (every camera frame is
# resized to 640x640 before processing). To read coordinates off your own footage run:
#     python nexus_vision_node.py --source <video or 0> --pick-points
# and click on the image; each click prints "(x, y)".
#
# --- 1) HOMOGRAPHY -----------------------------------------------------------
# SRC_POINTS: the four corners of a RECTANGULAR patch of floor as the camera sees it
# (e.g. the corners of the carpet / a marked area / the room's floor), clicked in this order:
#     top-left, top-right, bottom-right, bottom-left
# Far corners are closer together on screen than near ones (perspective), so they form a
# trapezoid. Use the real floor rectangle, not walls or furniture.
SRC_POINTS = np.array([[200, 230],     # top-left     (DUMMY VALUES - replace)
                       [440, 230],     # top-right
                       [610, 600],     # bottom-right
                       [30, 600]],     # bottom-left
                      dtype=np.float32)
# DST_POINTS: the same four corners on the flat 500x500 floor canvas, same order. Keep these
# as the canvas corners to stretch the patch over the whole canvas. If the real floor patch is
# not square (say 6 m x 3 m), use a matching rectangle, e.g. [[0,0],[499,0],[499,249],[0,249]].
CANVAS_SIZE = 500
DST_POINTS = np.array([[0, 0],
                       [CANVAS_SIZE - 1, 0],
                       [CANVAS_SIZE - 1, CANVAS_SIZE - 1],
                       [0, CANVAS_SIZE - 1]], dtype=np.float32)

# --- 2) TRIPWIRE -------------------------------------------------------------
# Line from A(x1, y1) to B(x2, y2) in dashboard-frame coordinates (put it across a doorway
# or the walkway). DEFAULT: a horizontal line across the middle of the frame.
LINE_A = (60, 380)
LINE_B = (580, 380)
# DIRECTION: looking from A towards B, crossing from the LEFT-hand side to the RIGHT-hand side
# counts as "IN" when True. For the default A->B line (drawn left to right, image y pointing
# down) that is walking DOWN the screen, towards the camera. Set False to swap IN and OUT.
IN_IS_LEFT_TO_RIGHT = True
# =============================================================================

PANE_SIZE = 640                      # each pane is 640x640 -> dashboard 1280x640
WINDOW_NAME = "NEXUS Vision Node"
Point = Tuple[float, float]


class NexusVisionNode:
    """Detects, tracks and counts people, and builds a live floor-plan heatmap.

    Pipeline per frame: resize -> YOLOv8n + ByteTrack (people only) -> feet point ->
    tripwire test + homography to the 500x500 canvas -> Gaussian heatmap accumulate/decay ->
    dual-pane dashboard.
    """

    def __init__(
        self,
        source: Union[str, int] = 0,
        line: Tuple[Point, Point] = (LINE_A, LINE_B),
        src_points: np.ndarray = SRC_POINTS,
        dst_points: np.ndarray = DST_POINTS,
        in_is_left_to_right: bool = IN_IS_LEFT_TO_RIGHT,
        conf: float = 0.5,
        loop: bool = False,
        anchor: str = "feet",
        min_hits: int = 5,
        decay: float = 0.99,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            source: path to an .mp4 file, or an integer webcam index (also accepts "0").
            line: ((x1, y1), (x2, y2)) tripwire in the 640x640 frame.
            src_points / dst_points: 4x2 homography correspondences (see CALIBRATION above).
            in_is_left_to_right: which crossing direction counts as IN.
            conf: YOLO confidence threshold (raise it to suppress false people).
            loop: restart a video file when it ends instead of exiting.
            anchor: "feet" (bottom-centre of box, normal for a full-body view) or "center"
                (middle of box, useful when testing with a close-up desk webcam).
            min_hits: frames a track must persist before it is shown/counted (filters flicker).
            decay: per-frame heatmap multiplier (0.99 = old footprints fade slowly).
            device: ultralytics device string ("cpu", "0", "mps"), or None for auto.
        """
        self.source = int(source) if str(source).isdigit() else source
        self.is_webcam = isinstance(self.source, int)
        self.loop = loop
        self.conf = conf
        self.anchor = anchor
        self.min_hits = min_hits
        self.decay = decay
        self.device = device

        self.model = YOLO("yolov8n.pt")

        # ---- homography ----
        self.src_points = np.asarray(src_points, dtype=np.float32)
        self.dst_points = np.asarray(dst_points, dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # ---- tripwire ----
        self.a = np.array(line[0], dtype=np.float64)
        self.b = np.array(line[1], dtype=np.float64)
        self.line_vec = self.b - self.a
        self.line_len = float(np.hypot(*self.line_vec))
        if self.line_len == 0:
            raise ValueError("Tripwire endpoints must be different points")
        # cross(B-A, P-A) > 0 means P is on the screen-RIGHT of A->B (image y points down).
        self.in_sign = 1 if in_is_left_to_right else -1
        self.deadband_px = 6.0       # ignore jitter this close to the line
        self.rearm_px = 40.0         # an ID may be counted again after moving this far away

        self.prev_side: Dict[int, int] = {}      # id -> last confirmed side (+1/-1)
        self.prev_pos: Dict[int, np.ndarray] = {}  # id -> previous point
        self.counted_ids: Set[int] = set()       # ids that crossed and have not re-armed yet
        self.hits: Dict[int, int] = {}           # id -> frames seen
        self.entries = 0
        self.exits = 0

        # ---- heatmap ----
        self.heat = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        radius, sigma = 18, 7.0
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        self.kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
        self.k_radius = radius
        self.heat_min_peak = 30.0    # normalisation floor so a few frames don't saturate the map

    # ------------------------------------------------------------------ video I/O
    def _open_source(self) -> cv2.VideoCapture:
        """Open the video file or webcam (webcams: try several Windows backends)."""
        if not self.is_webcam:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {self.source!r}")
            return cap
        if sys.platform == "win32":
            backends = [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("default", cv2.CAP_ANY)]
        else:
            backends = [("default", cv2.CAP_ANY)]
        for name, backend in backends:
            cap = cv2.VideoCapture(self.source, backend)
            if cap.isOpened():
                for _ in range(40):                  # warm-up: auto-exposure, empty first frames
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.mean() > 8.0:
                        print(f"Webcam {self.source} opened ({name} backend).")
                        return cap
                    cv2.waitKey(30)
            cap.release()
        raise RuntimeError(
            f"Webcam {self.source} gave no usable frames. Close other apps using the camera, "
            "check the privacy shutter / Windows camera privacy settings, or try --source 1.")

    # ------------------------------------------------------------------ tracking
    def _detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run YOLOv8 + ByteTrack (people only). Returns (xyxy boxes, track ids)."""
        result = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            conf=self.conf, device=self.device, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return np.empty((0, 4)), np.empty((0,), dtype=int)
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        keep = []
        for k, tid in enumerate(ids):                # confirm tracks (ignore 1-3 frame flickers)
            self.hits[int(tid)] = self.hits.get(int(tid), 0) + 1
            if self.hits[int(tid)] >= self.min_hits:
                keep.append(k)
        return xyxy[keep], ids[keep]

    def _anchor_point(self, box: np.ndarray) -> Point:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, y2 if self.anchor == "feet" else (y1 + y2) / 2.0)

    # ------------------------------------------------------------------ tripwire
    def _signed_distance(self, p: np.ndarray) -> float:
        """Signed perpendicular distance of p from the line (sign = which side)."""
        v = p - self.a
        return float(self.line_vec[0] * v[1] - self.line_vec[1] * v[0]) / self.line_len

    def _within_segment(self, p: np.ndarray) -> bool:
        t = float(np.dot(p - self.a, self.line_vec)) / (self.line_len ** 2)
        return 0.0 <= t <= 1.0

    def _update_tripwire(self, track_id: int, point: Point) -> Optional[str]:
        """Feed one tracked point. Returns "IN", "OUT" or None.

        A crossing is a flip of the point's side of the line (cross-product sign). The side
        is only updated once the point is clear of the line (deadband) so jitter cannot
        double-count, and the crossing must happen along the line segment itself.
        `counted_ids` blocks a second count for the same ID until it has moved well away
        from the line again (re-armed).
        """
        p = np.asarray(point, dtype=np.float64)
        dist = self._signed_distance(p)
        previous = self.prev_pos.get(track_id)
        self.prev_pos[track_id] = p

        if abs(dist) > self.rearm_px:
            self.counted_ids.discard(track_id)
        if abs(dist) < self.deadband_px:
            return None
        side = 1 if dist > 0 else -1
        old_side = self.prev_side.get(track_id)
        self.prev_side[track_id] = side
        if old_side is None or old_side == side:
            return None
        if track_id in self.counted_ids:
            return None
        if not (self._within_segment(p) or (previous is not None and self._within_segment(previous))):
            return None

        self.counted_ids.add(track_id)
        if side == self.in_sign:
            self.entries += 1
            return "IN"
        self.exits += 1
        return "OUT"

    # ------------------------------------------------------------------ heatmap
    def _to_canvas(self, points: List[Point]) -> np.ndarray:
        """Warp camera-frame points (x, y) to floor-canvas coordinates (X', Y')."""
        if not points:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def _accumulate(self, canvas_points: np.ndarray) -> None:
        """Decay the heatmap, then splat a Gaussian dot for every active person."""
        self.heat *= self.decay
        r = self.k_radius
        for x, y in canvas_points:
            cx, cy = int(round(x)), int(round(y))
            if cx < -r or cy < -r or cx >= CANVAS_SIZE + r or cy >= CANVAS_SIZE + r:
                continue                                   # mapped outside the floor canvas
            x0, y0 = max(cx - r, 0), max(cy - r, 0)
            x1, y1 = min(cx + r + 1, CANVAS_SIZE), min(cy + r + 1, CANVAS_SIZE)
            kx0, ky0 = x0 - (cx - r), y0 - (cy - r)
            self.heat[y0:y1, x0:x1] += self.kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]

    def _render_heatmap(self, canvas_points: np.ndarray) -> np.ndarray:
        """Normalise, blur and colour-map the accumulator into a 640x640 BGR pane."""
        norm = self.heat / max(float(self.heat.max()), self.heat_min_peak)
        heat8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        heat8 = cv2.GaussianBlur(heat8, (0, 0), sigmaX=6)
        colored = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)      # blue = quiet, red = busy
        for gx in range(0, CANVAS_SIZE, 50):                       # faint floor grid
            cv2.line(colored, (gx, 0), (gx, CANVAS_SIZE - 1), (255, 255, 255), 1)
            cv2.line(colored, (0, gx), (CANVAS_SIZE - 1, gx), (255, 255, 255), 1)
        colored = cv2.addWeighted(colored, 0.9, np.full_like(colored, 0), 0.1, 0)
        pane = cv2.resize(colored, (PANE_SIZE, PANE_SIZE), interpolation=cv2.INTER_LINEAR)
        scale = PANE_SIZE / CANVAS_SIZE
        for x, y in canvas_points:                                 # live markers
            cv2.circle(pane, (int(x * scale), int(y * scale)), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(pane, (int(x * scale), int(y * scale)), 7, (0, 0, 0), 1, cv2.LINE_AA)
        self._draw_hud(pane)
        return pane

    # ------------------------------------------------------------------ drawing
    def _draw_hud(self, pane: np.ndarray) -> None:
        """High-contrast KPI panel across the top of the heatmap pane."""
        overlay = pane.copy()
        cv2.rectangle(overlay, (0, 0), (PANE_SIZE, 92), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, pane, 0.22, 0, pane)
        occupancy = max(0, self.entries - self.exits)
        cells = [("Total Entries", self.entries, (0, 255, 0)),
                 ("Total Exits", self.exits, (70, 70, 255)),
                 ("Live Occupancy", occupancy, (255, 255, 255))]
        cell_w = PANE_SIZE // len(cells)
        for i, (label, value, color) in enumerate(cells):
            x0 = i * cell_w
            cv2.putText(pane, label, (x0 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (210, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(pane, str(value), (x0 + 12, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                        color, 3, cv2.LINE_AA)
            if i:
                cv2.line(pane, (x0, 8), (x0, 84), (120, 120, 120), 1)

    def _draw_camera_pane(self, frame: np.ndarray, xyxy: np.ndarray, ids: np.ndarray,
                          points: List[Point], events: Dict[int, str]) -> np.ndarray:
        """Left pane: boxes, track IDs, tripwire (with IN arrow) and homography corners."""
        pane = frame.copy()
        poly = self.src_points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(pane, [poly], True, (255, 140, 0), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(self.src_points):               # the 4 homography source points
            cv2.circle(pane, (int(x), int(y)), 7, (255, 140, 0), -1, cv2.LINE_AA)
            cv2.putText(pane, f"P{i + 1}", (int(x) + 9, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 200, 120), 2, cv2.LINE_AA)

        a, b = tuple(int(v) for v in self.a), tuple(int(v) for v in self.b)
        cv2.line(pane, a, b, (0, 255, 255), 3, cv2.LINE_AA)
        normal = np.array([-self.line_vec[1], self.line_vec[0]]) / self.line_len  # cross>0 side
        into = normal * self.in_sign
        mid = (self.a + self.b) / 2
        tip = mid + into * 40
        cv2.arrowedLine(pane, tuple(int(v) for v in mid), tuple(int(v) for v in tip),
                        (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.4)
        cv2.putText(pane, "IN", tuple(int(v) for v in tip + into * 8 - np.array([10, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        for (x1, y1, x2, y2), tid, (px, py) in zip(xyxy, ids, points):
            event = events.get(int(tid))
            color = (0, 255, 0) if event == "IN" else (60, 60, 255) if event == "OUT" else (255, 200, 0)
            cv2.rectangle(pane, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID {tid}" + (f" {event}" if event else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(int(y1) - th - 8, 0)
            cv2.rectangle(pane, (int(x1), top), (int(x1) + tw + 6, top + th + 8), color, -1)
            cv2.putText(pane, label, (int(x1) + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(pane, (int(px), int(py)), 5, (0, 0, 255), -1)
        cv2.putText(pane, "LIVE CAMERA", (10, PANE_SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return pane

    # ------------------------------------------------------------------ main loop
    def process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Process one camera frame and return the 1280x640 dashboard image."""
        frame = cv2.resize(raw_frame, (PANE_SIZE, PANE_SIZE))
        xyxy, ids = self._detect_and_track(frame)
        points = [self._anchor_point(b) for b in xyxy]
        canvas_pts = self._to_canvas(points)

        events: Dict[int, str] = {}
        for tid, pt in zip(ids, points):
            event = self._update_tripwire(int(tid), pt)
            if event:
                events[int(tid)] = event
        self._accumulate(canvas_pts)

        left = self._draw_camera_pane(frame, xyxy, ids, points, events)
        right = self._render_heatmap(canvas_pts)
        return cv2.hconcat([left, right])

    def reset(self) -> None:
        """Clear counters, tripwire memory and the heatmap."""
        self.entries = self.exits = 0
        self.prev_side.clear()
        self.prev_pos.clear()
        self.counted_ids.clear()
        self.hits.clear()
        self.heat[:] = 0

    def run(self) -> None:
        """Main loop: read, process, display. 'q' quits, 'r' resets."""
        cap = self._open_source()
        failed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok and self.is_webcam and failed < 30:
                    failed += 1                    # tolerate brief webcam dropouts
                    cv2.waitKey(10)
                    continue
                failed = 0
                if not ok:
                    if not self.is_webcam and self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.model.predictor = None        # fresh tracker for the new pass
                        self.prev_side.clear()
                        self.prev_pos.clear()
                        self.counted_ids.clear()
                        self.hits.clear()
                        continue
                    print("End of stream.")
                    break
                cv2.imshow(WINDOW_NAME, self.process_frame(frame))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self.reset()
        finally:
            cap.release()
            cv2.destroyAllWindows()
        print(f"Final -> Entries: {self.entries}  Exits: {self.exits}  "
              f"Occupancy: {max(0, self.entries - self.exits)}")


def pick_points(source: Union[str, int]) -> None:
    """Show one frame (640x640) and print the coordinates of every mouse click."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read a frame from {source!r}")
    shown = cv2.resize(frame, (PANE_SIZE, PANE_SIZE))
    title = "Pick points - click, then press q"

    def on_click(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x}, {y})")
            cv2.circle(shown, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(shown, f"({x},{y})", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    while (cv2.waitKey(20) & 0xFF) not in (ord("q"), 27):
        cv2.imshow(title, shown)
    cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description="NEXUS Vision Node - footfall counter + heatmap")
    ap.add_argument("--source", default="0", help="video file path or webcam index (default 0)")
    ap.add_argument("--loop", action="store_true", help="loop a video file")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    ap.add_argument("--anchor", choices=["feet", "center"], default="feet",
                    help="tracked point: box bottom-centre (default) or box centre (desk webcam)")
    ap.add_argument("--line", default=None, metavar="X1,Y1,X2,Y2",
                    help="tripwire endpoints in 640x640 frame coords")
    ap.add_argument("--swap", action="store_true", help="swap IN and OUT")
    ap.add_argument("--device", default=None, help="cpu, 0, mps ...")
    ap.add_argument("--pick-points", action="store_true", help="click to print coordinates")
    args = ap.parse_args()

    if args.pick_points:
        pick_points(args.source)
        return 0
    line = (LINE_A, LINE_B)
    if args.line:
        try:
            x1, y1, x2, y2 = (int(v) for v in args.line.split(","))
        except ValueError:
            print("Error: --line must be X1,Y1,X2,Y2", file=sys.stderr)
            return 1
        line = ((x1, y1), (x2, y2))
    try:
        NexusVisionNode(args.source, line=line, conf=args.conf, loop=args.loop,
                        anchor=args.anchor, device=args.device,
                        in_is_left_to_right=IN_IS_LEFT_TO_RIGHT != args.swap).run()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
