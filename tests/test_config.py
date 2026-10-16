import pytest

from nexus.config import ConfigError, config_from_dict, default_config, load_config, save_config


def test_default_is_valid():
    assert default_config().frame_size == (640, 480)


def test_roundtrip(tmp_path):
    cfg = default_config()
    cfg.camera_id = "door-2"
    cfg.tripwire.in_direction = "right_to_left"
    save_config(cfg, tmp_path / "c.yaml")
    loaded = load_config(tmp_path / "c.yaml")
    assert loaded.camera_id == "door-2" and loaded.tripwire.in_direction == "right_to_left"
    assert loaded.homography.canvas == (500, 500)


def test_unknown_key_names_the_typo():
    with pytest.raises(ConfigError, match="tripwire.*confidence|unknown"):
        config_from_dict({"tripwire": {"confidence": 1}})


@pytest.mark.parametrize("patch,msg", [
    ({"detection": {"conf": 0}}, "conf"),
    ({"detection": {"anchor": "head"}}, "anchor"),
    ({"tripwire": {"a": [10, 10], "b": [10, 10]}}, "different"),
    ({"tripwire": {"a": [0, 0], "b": [9999, 10]}}, "outside"),
    ({"tripwire": {"in_direction": "up"}}, "in_direction"),
    ({"service": {"occupancy_reset": "25:99"}}, "occupancy_reset"),
    ({"dashboard": {"port": 0}}, "port"),
    ({"homography": {"src": [[0, 0], [1, 1], [2, 2], [3, 3]]}}, "collinear"),
    ({"homography": {"src": [[0, 0], [100, 0], [0, 100], [100, 100]]}}, "convex"),   # wrong order
    ({"homography": {"src": [[0, 0], [1, 1]]}}, "4 points"),
    ({"camera_id": "bad id!"}, "camera_id"),
])
def test_invalid_values_are_rejected(patch, msg):
    with pytest.raises(ConfigError, match=msg):
        config_from_dict(patch)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    (tmp_path / "bad.yaml").write_text("a: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(tmp_path / "bad.yaml")


@pytest.mark.parametrize("name", ["example.yaml", "docker.yaml"])
def test_shipped_example_configs_are_valid(name):
    from pathlib import Path
    assert load_config(Path(__file__).parent.parent / "configs" / name).camera_id
