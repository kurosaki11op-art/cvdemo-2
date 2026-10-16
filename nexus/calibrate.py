"""Interactive calibration: click 4 floor corners + 2 tripwire points, write them to YAML."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .config import Config, default_config, load_config, save_config
from .sources import FrameSource

Point = Tuple[float, float]


def floor_canvas(floor_size: Optional[Tuple[float, float]], longest: int = 500) -> Tuple[int, int]:
    """Canvas (w, h) whose aspect matches the real floor patch (e.g. 6 m x 3 m -> 500x250)."""
    if not floor_size:
        return longest, longest
    wm, hm = floor_size
    if wm <= 0 or hm <= 0:
        raise ValueError("floor size must be positive")
    if wm >= hm:
        return longest, max(50, round(longest * hm / wm))
    return max(50, round(longest * wm / hm)), longest


def calibrated_config(base: Config, floor_pts: Sequence[Point], line_pts: Sequence[Point],
                      floor_size: Optional[Tuple[float, float]] = None) -> Config:
    """Return a copy of `base` with the new homography/tripwire (validated)."""
    if len(floor_pts) != 4 or len(line_pts) != 2:
        raise ValueError("need 4 floor corners and 2 tripwire points")
    cfg = copy.deepcopy(base)
    w, h = floor_canvas(floor_size)
    cfg.homography.src = [list(map(float, p)) for p in floor_pts]
    cfg.homography.dst = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    cfg.homography.canvas = [w, h]
    cfg.tripwire.a, cfg.tripwire.b = list(map(float, line_pts[0])), list(map(float, line_pts[1]))
    return cfg.validate()


def grab_frame(source: Union[str, int], size: Tuple[int, int]) -> np.ndarray:
    src = FrameSource(source).open()
    try:
        for _ in range(5):                      # skip a few frames so cameras settle
            frame = src.read()
        if frame is None:
            raise RuntimeError(f"cannot read a frame from {source!r}")
    finally:
        src.release()
    return cv2.resize(frame, size)


STEPS = ["floor corner 1: TOP-LEFT", "floor corner 2: TOP-RIGHT", "floor corner 3: BOTTOM-RIGHT",
         "floor corner 4: BOTTOM-LEFT", "tripwire point A", "tripwire point B (IN = left->right of A->B)"]


def run_calibration(source: Union[str, int], out_path: Union[str, Path], base_config: Optional[str] = None,
                    floor_size: Optional[Tuple[float, float]] = None) -> bool:
    """Click 6 points, preview the top-down warp, press 'y' to save or 'r' to redo."""
    base = load_config(base_config) if base_config else (
        load_config(out_path) if Path(out_path).is_file() else default_config())
    frame = grab_frame(source, tuple(base.frame_size))
    title = "NEXUS calibration (u=undo, r=restart, q=quit)"
    clicks: List[Point] = []

    def draw() -> np.ndarray:
        img = frame.copy()
        for i, p in enumerate(clicks):
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (int(p[0]) + 7, int(p[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)
        if len(clicks) >= 2:
            cv2.polylines(img, [np.array(clicks[:4], np.int32).reshape(-1, 1, 2)], len(clicks) >= 4,
                          (255, 140, 0), 2)
        if len(clicks) == 6:
            cv2.line(img, tuple(int(v) for v in clicks[4]), tuple(int(v) for v in clicks[5]), (0, 255, 255), 3)
        msg = STEPS[len(clicks)] if len(clicks) < 6 else "press 'y' to preview + save, 'u' to undo"
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, f"Click {msg}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return img

    def on_click(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 6:
            clicks.append((float(x), float(y)))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    saved = False
    try:
        while True:
            cv2.imshow(title, draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and clicks:
                clicks.pop()
            if key == ord("r"):
                clicks.clear()
            if key == ord("y") and len(clicks) == 6:
                try:
                    cfg = calibrated_config(base, clicks[:4], clicks[4:], floor_size)
                except Exception as exc:        # ConfigError: e.g. corners clicked in wrong order
                    print(f"Calibration rejected: {exc}")
                    clicks.clear()
                    continue
                m = cv2.getPerspectiveTransform(np.array(cfg.homography.src, np.float32),
                                                np.array(cfg.homography.dst, np.float32))
                w, h = cfg.homography.canvas
                preview = cv2.warpPerspective(frame, m, (w, h))
                cv2.imshow("Top-down preview (y=save, any other key=back)", preview)
                if cv2.waitKey(0) & 0xFF == ord("y"):
                    save_config(cfg, out_path)
                    print(f"Saved calibration to {out_path}")
                    saved = True
                    break
                cv2.destroyWindow("Top-down preview (y=save, any other key=back)")
    finally:
        cv2.destroyAllWindows()
    return saved


def pick_points(source: Union[str, int], size: Tuple[int, int] = (640, 480)) -> None:
    """Show a frame and print the coordinates of every click (for manual YAML editing)."""
    frame = grab_frame(source, size)
    title = "Pick points - click, then press q"
    shown = frame.copy()

    def on_click(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x}, {y})")
            cv2.circle(shown, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(shown, f"({x},{y})", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    while True:
        cv2.imshow(title, shown)
        if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
