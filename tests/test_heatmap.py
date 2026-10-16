import numpy as np

from nexus.heatmap import HeatmapAccumulator, render_heat


def test_peak_is_at_the_point():
    h = HeatmapAccumulator((100, 80))
    h.add(40, 30)
    assert np.unravel_index(h.canvas.argmax(), h.canvas.shape) == (30, 40)


def test_border_clipping_and_far_points():
    h = HeatmapAccumulator((50, 50), radius=10)
    h.add(0, 0)
    h.add(49, 49)
    assert h.canvas[0, 0] > 0 and h.canvas[49, 49] > 0
    before = h.canvas.sum()
    h.add(-200, 25)
    h.add(500, 500)
    assert h.canvas.sum() == before


def test_decay_and_reset():
    h = HeatmapAccumulator((40, 40), decay=0.5)
    h.add(20, 20)
    peak = h.canvas.max()
    h.step_decay()
    assert np.isclose(h.canvas.max(), peak / 2)
    h.reset()
    assert h.canvas.max() == 0


def test_dwell_accumulates():
    h = HeatmapAccumulator((40, 40))
    for _ in range(10):
        h.add(20, 20)
    assert np.isclose(h.canvas.max(), 10.0, atol=1e-4)


def test_render_shape_dtype_and_cold_hot_colours():
    h = HeatmapAccumulator((100, 100))
    for _ in range(100):
        h.add(50, 50)
    img = render_heat(h.canvas, min_peak=1e-6)
    assert img.shape == (100, 100, 3) and img.dtype == np.uint8
    hot, cold = img[50, 50], img[10, 10]
    assert hot[2] > hot[0]          # red channel dominates at the peak (JET)
    assert cold[0] > cold[2]        # blue dominates where nothing happened
