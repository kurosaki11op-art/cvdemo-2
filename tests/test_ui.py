import numpy as np

from nexus.engine import CountingEngine
from nexus.ui import DemoView, _fit

from .conftest import FakeDetector, walk


def test_dashboard_is_two_640x480_panes(cfg, frame):
    engine = CountingEngine(cfg, FakeDetector(walk(1, 320, 200, 440)))
    view = DemoView(engine)
    for _ in range(25):
        out = view.compose(engine.process(frame))
    assert out.shape == (480, 1280, 3)
    kpi_column = out[:, 960:]                                  # the 160px HUD column, right pane
    assert kpi_column.std() > 10                              # text/numbers were drawn


def test_non_square_floor_canvas_is_letterboxed():
    wide = np.full((250, 500, 3), 200, np.uint8)
    out = _fit(wide, (480, 480))
    assert out.shape == (480, 480, 3)
    assert out[0, 0].tolist() == [20, 20, 20] and out[240, 240].tolist() == [200, 200, 200]
