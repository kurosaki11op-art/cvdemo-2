import pytest

from nexus.calibrate import calibrated_config, floor_canvas
from nexus.config import ConfigError, default_config


def test_floor_canvas_keeps_aspect():
    assert floor_canvas(None) == (500, 500)
    assert floor_canvas((6, 3)) == (500, 250)
    assert floor_canvas((3, 6)) == (250, 500)
    with pytest.raises(ValueError):
        floor_canvas((0, 3))


def test_calibrated_config_applies_points_and_scale():
    floor = [(100, 100), (500, 100), (600, 400), (50, 400)]
    cfg = calibrated_config(default_config(), floor, [(0, 300), (600, 300)], floor_size=(6, 3))
    assert cfg.homography.canvas == (500, 250)
    assert tuple(cfg.homography.dst[2]) == (499, 249)
    assert tuple(cfg.tripwire.a) == (0.0, 300.0)


def test_wrong_click_order_is_rejected():
    bowtie = [(100, 100), (500, 400), (500, 100), (100, 400)]       # corners clicked in a Z order
    with pytest.raises(ConfigError, match="convex"):
        calibrated_config(default_config(), bowtie, [(0, 300), (600, 300)])
    with pytest.raises(ValueError):
        calibrated_config(default_config(), bowtie[:3], [(0, 0), (1, 1)])
