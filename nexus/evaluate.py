"""Accuracy evaluation: compare counted crossings with a hand-labelled ground truth.

Ground truth is a CSV with a header ``time_s,direction`` - one row per real crossing, the
time in seconds from the start of the clip, direction IN or OUT (create it with
``nexus label``). Predicted and true events match when the direction is equal and the times
differ by at most ``tolerance_s``; each event is matched at most once.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from .config import Config
from .engine import CountingEngine
from .sources import FrameSource

Event = Tuple[float, str]


def load_truth_csv(path: Union[str, Path]) -> List[Event]:
    events: List[Event] = []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].strip().startswith("#")]
    if not rows:
        return events
    start = 1 if rows[0][0].strip().lower() in ("time_s", "time", "t") else 0
    for n, row in enumerate(rows[start:], start=start + 1):
        try:
            t, direction = float(row[0]), row[1].strip().upper()
        except (ValueError, IndexError):
            raise ValueError(f"{path}: row {n} must be 'time_s,direction', got {row!r}") from None
        if direction not in ("IN", "OUT"):
            raise ValueError(f"{path}: row {n}: direction must be IN or OUT, got {row[1]!r}")
        events.append((t, direction))
    return sorted(events)


def save_truth_csv(events: List[Event], path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "direction"])
        for t, d in sorted(events):
            w.writerow([f"{t:.2f}", d])


@dataclass
class DirectionStats:
    truth: int = 0
    predicted: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    count_error_pct: Optional[float] = None     # (predicted - truth) / truth * 100


@dataclass
class EvalReport:
    tolerance_s: float
    directions: Dict[str, DirectionStats] = field(default_factory=dict)
    overall: DirectionStats = field(default_factory=DirectionStats)
    f1: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def format_text(self) -> str:
        def pct(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v * 100:.1f}%"

        lines = [f"Accuracy report (match tolerance {self.tolerance_s:.1f} s)",
                 f"{'':8}{'truth':>7}{'counted':>9}{'TP':>5}{'FP':>5}{'FN':>5}"
                 f"{'precision':>11}{'recall':>9}{'count err':>11}"]
        for name, s in (*self.directions.items(), ("TOTAL", self.overall)):
            err = "n/a" if s.count_error_pct is None else f"{s.count_error_pct:+.1f}%"
            lines.append(f"{name:8}{s.truth:>7}{s.predicted:>9}{s.true_positives:>5}{s.false_positives:>5}"
                         f"{s.false_negatives:>5}{pct(s.precision):>11}{pct(s.recall):>9}{err:>11}")
        lines.append(f"F1 (overall): {pct(self.f1)}")
        return "\n".join(lines)


def match_events(pred: List[Event], truth: List[Event], tolerance_s: float) -> Tuple[int, int, int]:
    """Greedy one-to-one matching per direction. Returns (TP, FP, FN)."""
    tp = 0
    for direction in ("IN", "OUT"):
        p = sorted(t for t, d in pred if d == direction)
        used = [False] * len(p)
        for t_true in sorted(t for t, d in truth if d == direction):
            best, best_dt = None, tolerance_s + 1e-9
            for i, t_pred in enumerate(p):
                dt = abs(t_pred - t_true)
                if not used[i] and dt <= best_dt:
                    best, best_dt = i, dt
            if best is not None:
                used[best] = True
                tp += 1
    return tp, len(pred) - tp, len(truth) - tp


def _stats(pred: List[Event], truth: List[Event], tol: float) -> DirectionStats:
    tp, fp, fn = match_events(pred, truth, tol)
    return DirectionStats(
        truth=len(truth), predicted=len(pred), true_positives=tp, false_positives=fp,
        false_negatives=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        count_error_pct=(len(pred) - len(truth)) / len(truth) * 100 if truth else None)


def evaluate(pred: List[Event], truth: List[Event], tolerance_s: float = 1.5) -> EvalReport:
    report = EvalReport(tolerance_s=tolerance_s)
    for direction in ("IN", "OUT"):
        report.directions[direction] = _stats([e for e in pred if e[1] == direction],
                                              [e for e in truth if e[1] == direction], tolerance_s)
    report.overall = _stats(pred, truth, tolerance_s)
    p, r = report.overall.precision, report.overall.recall
    report.f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return report


def run_engine_on_video(cfg: Config, video: Union[str, Path], stride: int = 1,
                        engine: Optional[CountingEngine] = None,
                        progress: Optional[Callable[[int, int], None]] = None) -> List[Event]:
    """Run the engine over a clip and return predicted (time_s, direction) events.

    Event time is the video timestamp (frame index / fps), so results do not depend on how
    fast this machine processes. ``stride`` > 1 skips frames for speed (at some accuracy cost).
    """
    engine = engine or CountingEngine(cfg)
    src = FrameSource(str(video), loop=False).open()
    total = int(src.cap.get(7)) if src.cap is not None else 0
    fps = src.fps
    events: List[Event] = []
    try:
        index = 0
        while True:
            frame = src.read()
            if frame is None:
                break
            index += 1
            if stride > 1 and index % stride:
                continue
            for ev in engine.process(frame).events:
                events.append((index / fps, ev.direction))
            if progress and index % 100 == 0:
                progress(index, total)
    finally:
        src.release()
    return events


def write_report(report: EvalReport, path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
