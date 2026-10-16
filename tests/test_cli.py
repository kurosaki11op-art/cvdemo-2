import pytest

from nexus.cli import build_parser, main
from nexus.config import load_config


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Default config paths are relative (data/nexus.db): keep them out of the repo."""
    monkeypatch.chdir(tmp_path)


def test_init_config_writes_a_loadable_file(tmp_path, capsys):
    out = tmp_path / "nexus.yaml"
    assert main(["init-config", "--out", str(out), "--camera-id", "door-1", "--source", "0"]) == 0
    cfg = load_config(out)
    assert cfg.camera_id == "door-1" and cfg.source == "0"


def test_run_without_a_source_is_a_clean_error(tmp_path, capsys):
    assert main(["run"]) == 1
    assert "no video source" in capsys.readouterr().err


def test_bad_config_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("detection:\n  conf: 7\n")
    assert main(["run", "--config", str(bad)]) == 1
    assert "conf" in capsys.readouterr().err


def test_missing_video_is_a_clean_error(capsys):
    assert main(["run", "--source", "/no/such/video.mp4"]) == 1
    assert "not found" in capsys.readouterr().err


def test_purge_runs(tmp_path, capsys):
    cfgp = tmp_path / "c.yaml"
    main(["init-config", "--out", str(cfgp)])
    text = cfgp.read_text().replace("data/nexus.db", str(tmp_path / "n.db"))
    cfgp.write_text(text)
    assert main(["purge", "--config", str(cfgp)]) == 0
    assert "Removed 0 rows" in capsys.readouterr().out


def test_all_subcommands_have_help():
    p = build_parser()
    commands = ("demo", "run", "dashboard", "calibrate", "label", "evaluate", "export-model",
                "init-config", "purge", "healthcheck")
    for cmd in commands:
        with pytest.raises(SystemExit) as e:
            p.parse_args([cmd, "--help"])
        assert e.value.code == 0


def test_evaluate_requires_truth():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "--source", "x.mp4"])


def test_evaluate_without_a_video_is_a_clean_error(tmp_path, capsys):
    truth = tmp_path / "t.csv"
    truth.write_text("time_s,direction\n1.0,IN\n")
    assert main(["evaluate", "--truth", str(truth)]) == 1
    assert "no video" in capsys.readouterr().err


def test_healthcheck_follows_the_heartbeat(tmp_path):
    import json
    import time

    cfgp = tmp_path / "c.yaml"
    main(["init-config", "--out", str(cfgp)])
    cfgp.write_text(cfgp.read_text().replace("data/nexus.db", str(tmp_path / "n.db")))
    assert main(["healthcheck", "--config", str(cfgp)]) == 1                     # nothing running
    (tmp_path / "status-camera-1.json").write_text(json.dumps({"state": "running", "ts": time.time()}))
    assert main(["healthcheck", "--config", str(cfgp)]) == 0
