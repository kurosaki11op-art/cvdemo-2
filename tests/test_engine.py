import numpy as np

from nexus.engine import CountingEngine, Detections

from .conftest import FakeDetector, det, walk


def run(cfg, frame, script):
    engine = CountingEngine(cfg, FakeDetector(script))
    results = [engine.process(frame) for _ in script]
    return engine, results


def test_person_walking_down_counts_in(cfg, frame):
    engine, results = run(cfg, frame, walk(1, 320, 200, 440))
    assert (engine.entries, engine.exits) == (1, 0)
    assert sum(len(r.events) for r in results) == 1


def test_person_walking_up_counts_out(cfg, frame):
    engine, _ = run(cfg, frame, walk(1, 320, 440, 200))
    assert (engine.entries, engine.exits) == (0, 1)


def test_two_people_opposite_directions(cfg, frame):
    cfg.detection.twin_filter = False        # the flat test frame makes both people look identical
    down, up = walk(1, 250, 200, 440), walk(2, 400, 440, 200)
    script = [Detections(np.vstack([a.xyxy, b.xyxy]), np.concatenate([a.ids, b.ids]),
                         np.concatenate([a.confs, b.confs])) for a, b in zip(down, up, strict=True)]
    engine, _ = run(cfg, frame, script)
    assert (engine.entries, engine.exits) == (1, 1)


def test_person_walking_beside_the_line_is_not_counted(cfg, frame):
    engine, _ = run(cfg, frame, walk(1, 600, 200, 440))          # x=600 is past the line's end (x=520)
    assert (engine.entries, engine.exits) == (0, 0)


def test_flicker_below_min_frames_is_ignored(cfg, frame):
    cfg.detection.min_frames = 5
    script = [det((9, 320, 250)), det((9, 320, 400))]            # 2 frames only
    engine, results = run(cfg, frame, script)
    assert (engine.entries, engine.exits) == (0, 0) and all(len(r.ids) == 0 for r in results)


def test_heatmaps_fill_at_the_mapped_floor_position(cfg, frame):
    engine, results = run(cfg, frame, [det((1, 320, 400))] * 20)
    fx, fy = results[-1].floor_points[0]
    assert 0 <= fx <= 499 and 0 <= fy <= 499
    assert engine.period_heat.canvas.max() > 15                  # dwell accumulated
    assert engine.display_heat.canvas.max() > 0


def test_frames_are_resized_to_frame_size(cfg):
    engine = CountingEngine(cfg, FakeDetector())
    r = engine.process(np.full((1080, 1920, 3), 100, np.uint8))
    assert r.frame.shape == (480, 640, 3)


def test_reset_tracking_and_counts(cfg, frame):
    engine, _ = run(cfg, frame, walk(1, 320, 200, 440))
    engine.reset_tracking()
    assert engine.detector.resets == 1 and engine.entries == 1
    engine.reset_counts()
    assert (engine.entries, engine.exits) == (0, 0) and engine.period_heat.canvas.max() == 0


def test_center_anchor_uses_box_middle(cfg, frame):
    cfg.detection.anchor = "center"
    engine = CountingEngine(cfg, FakeDetector([det((1, 320, 300))]))
    r = engine.process(frame)
    assert r.points[0] == (320.0, 225.0)


def test_look_alike_people_are_merged_by_the_twin_filter_unless_disabled(cfg, frame):
    both = det((1, 150, 300), (2, 450, 300))
    kept = CountingEngine(cfg, FakeDetector([both])).process(frame)
    assert len(kept.ids) == 1                # identical appearance -> treated as a reflection
    cfg.detection.twin_filter = False
    assert len(CountingEngine(cfg, FakeDetector([both])).process(frame).ids) == 2
