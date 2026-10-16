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
                "init-config", "purge", "healthcheck", "zones", "report", "seed-demo")
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


def test_report_and_seed_demo_end_to_end(tmp_path, capsys):
    cfgp = tmp_path / "c.yaml"
    main(["init-config", "--out", str(cfgp)])
    cfgp.write_text(cfgp.read_text().replace("data/nexus.db", str(tmp_path / "n.db")))
    assert main(["seed-demo", "--config", str(cfgp), "--days", "2", "--seed", "5"]) == 0
    assert "DEMO data" in capsys.readouterr().out
    assert main(["report", "--config", str(cfgp), "--days", "2"]) == 0
    out = capsys.readouterr().out
    assert "Store analytics" in out and "generated demo data" in out and "Zones, most visited first" in out
    assert main(["report", "--config", str(cfgp), "--days", "2", "--format", "json"]) == 0
    import json
    rep = json.loads(capsys.readouterr().out)
    assert rep["overview"]["visitors"] > 100 and rep["demo_data"] is True


def test_seed_demo_refuses_to_overwrite_real_data(tmp_path, capsys):
    import time

    from nexus.sessions import SessionRecord
    from nexus.storage import EventStore
    cfgp = tmp_path / "c.yaml"
    main(["init-config", "--out", str(cfgp)])
    db = tmp_path / "n.db"
    cfgp.write_text(cfgp.read_text().replace("data/nexus.db", str(db)))
    EventStore(db).add_session("camera-1", SessionRecord(time.time(), time.time() + 30))
    assert main(["seed-demo", "--config", str(cfgp)]) == 1
    assert "REAL data" in capsys.readouterr().err
    assert main(["seed-demo", "--config", str(cfgp), "--force", "--days", "1"]) == 0
