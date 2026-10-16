"""Command line: demo window, headless service, dashboard, calibration, labelling, evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import cv2

from .config import Config, ConfigError, default_config, load_config, save_config
from .sources import FrameSource, SourceError

log = logging.getLogger("nexus.cli")


def _parse_line(text: str) -> tuple:
    try:
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
    except ValueError:
        raise ConfigError("--line must be X1,Y1,X2,Y2 (four numbers)") from None
    return (x1, y1), (x2, y2)


def _load(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config) if getattr(args, "config", None) else default_config()
    if getattr(args, "source", None) is not None:
        cfg.source = args.source
    return cfg


# --------------------------------------------------------------------------- demo
def cmd_demo(args: argparse.Namespace) -> int:
    from .engine import CountingEngine
    from .ui import DemoView

    cfg = _load(args)
    if cfg.source is None:
        cfg.source = "0"
    d = cfg.detection
    if args.conf is not None:
        d.conf = args.conf
    if args.device:
        d.device = args.device
    if args.anchor:
        d.anchor = args.anchor
    if args.min_frames is not None:
        d.min_frames = args.min_frames
    if args.no_twin_filter:
        d.twin_filter = False
    if args.no_reid:
        d.reid = False
    if args.in_direction:
        cfg.tripwire.in_direction = args.in_direction
    if args.line:
        cfg.tripwire.a, cfg.tripwire.b = _parse_line(args.line)
    if args.webcam_demo:      # desk test: vertical line in the middle, track the box centre
        w, h = cfg.frame_size
        cfg.tripwire.a, cfg.tripwire.b = (w / 2, h - 100), (w / 2, 100)
        cfg.tripwire.in_direction = "left_to_right"
        d.anchor = "center"
    cfg.validate()

    engine = CountingEngine(cfg)
    view = DemoView(engine)
    src = FrameSource(cfg.source, loop=args.loop or cfg.service.loop).open()
    initial = cfg.service.initial_occupancy
    drops = 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                if src.finished:
                    print("End of stream.")
                    break
                drops += 1
                if drops > 3 or not src.reopen():
                    print("Camera stopped delivering frames.")
                    break
                continue
            drops = 0
            if src.consume_wrap():
                engine.reset_tracking()
            result = engine.process(frame)
            image = view.compose(result, initial)
            if args.scale != 1.0:
                image = cv2.resize(image, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("NEXUS Vision Node", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                engine.reset_counts()
    finally:
        src.release()
        cv2.destroyAllWindows()
    print(f"Final -> Entries: {engine.entries}  Exits: {engine.exits}  "
          f"Occupancy: {max(0, initial + engine.entries - engine.exits)}")
    return 0


# --------------------------------------------------------------------------- service
def cmd_run(args: argparse.Namespace) -> int:
    from .service import Service, setup_logging

    setup_logging(args.log_level, args.log_json)
    cfg = _load(args)
    if cfg.source is None:
        raise ConfigError("no video source: set `source:` in the config or pass --source")
    if args.loop:
        cfg.service.loop = True
    return Service(cfg.validate()).run()


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import serve
    from .service import setup_logging

    setup_logging(args.log_level, args.log_json)
    cfg = _load(args)
    if args.host:
        cfg.dashboard.host = args.host
    if args.port:
        cfg.dashboard.port = args.port
    serve(cfg.validate())
    return 0


def cmd_healthcheck(args: argparse.Namespace) -> int:
    """Exit 0 if the service heartbeat is fresh (for Docker HEALTHCHECK / Kubernetes probes)."""
    from .dashboard import is_healthy

    return 0 if is_healthy(_load(args)) else 1


def cmd_purge(args: argparse.Namespace) -> int:
    from .storage import EventStore

    cfg = _load(args)
    removed = EventStore(cfg.storage.db_path).purge(cfg.storage.retention_days)
    print(f"Removed {removed} rows older than {cfg.storage.retention_days} days.")
    return 0


# --------------------------------------------------------------------------- tools
def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import pick_points, run_calibration

    if args.pick_only:
        pick_points(args.source)
        return 0
    floor = None
    if args.floor_size:
        try:
            w, h = args.floor_size.lower().split("x")
            floor = (float(w), float(h))
        except ValueError:
            raise ConfigError("--floor-size must look like 6x3 (metres, width x depth)") from None
    ok = run_calibration(args.source, args.out, args.config, floor)
    return 0 if ok else 1


def cmd_label(args: argparse.Namespace) -> int:
    from .label import run_labeler

    n = run_labeler(args.source, args.out, args.speed)
    print(f"Saved {n} labelled events to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, load_truth_csv, run_engine_on_video, write_report

    cfg = _load(args)
    if cfg.source is None:
        raise ConfigError("no video: pass --source clip.mp4 (or set `source:` in the config)")
    truth = load_truth_csv(args.truth)
    pred = run_engine_on_video(
        cfg, cfg.source, stride=args.stride,
        progress=lambda i, n: print(f"\r  frame {i}/{n}", end="", file=sys.stderr))
    print(file=sys.stderr)
    report = evaluate(pred, truth, args.tolerance)
    print(report.format_text())
    if args.report:
        write_report(report, args.report)
    if args.min_f1 is not None and (report.f1 or 0.0) < args.min_f1:
        print(f"FAIL: F1 {report.f1} is below --min-f1 {args.min_f1}", file=sys.stderr)
        return 2
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    path = YOLO(args.model).export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"Exported: {path}\nSet `detection.model: {path}` in your config to use it.")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    cfg = default_config()
    cfg.camera_id = args.camera_id
    cfg.source = args.source
    save_config(cfg, args.out)
    print(f"Wrote {args.out}. Next: `nexus calibrate --config {args.out} --source <video|0>`")
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nexus", description="NEXUS Vision Node - footfall counting + heatmap")
    sub = p.add_subparsers(dest="command", required=True)

    def add_cfg(sp: argparse.ArgumentParser, source: bool = True) -> None:
        sp.add_argument("--config", help="YAML config file")
        if source:
            sp.add_argument("--source", default=None, help="video file, webcam index or rtsp:// URL")

    def add_log(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        sp.add_argument("--log-json", action="store_true", help="structured JSON log lines")

    d = sub.add_parser("demo", help="interactive dual-pane window")
    add_cfg(d)
    d.add_argument("--loop", action="store_true", help="loop video files")
    d.add_argument("--conf", type=float, default=None, help="detection confidence")
    d.add_argument("--device", default=None)
    d.add_argument("--anchor", choices=["feet", "center"], default=None)
    d.add_argument("--line", default=None, metavar="X1,Y1,X2,Y2", help="tripwire in frame coordinates")
    d.add_argument("--in-direction", choices=["left_to_right", "right_to_left"], default=None)
    d.add_argument("--webcam-demo", action="store_true", help="desk test: vertical line, track box centre")
    d.add_argument("--min-frames", type=int, default=None)
    d.add_argument("--no-twin-filter", action="store_true", help="keep look-alike boxes (identical uniforms)")
    d.add_argument("--no-reid", action="store_true")
    d.add_argument("--scale", type=float, default=1.0, help="window size factor")
    d.add_argument("--pick-points", action="store_true", help="legacy: print click coordinates and exit")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="headless service: count, store, heartbeat, reconnect")
    add_cfg(r)
    add_log(r)
    r.add_argument("--loop", action="store_true")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("dashboard", help="web dashboard + JSON API")
    add_cfg(w, source=False)
    add_log(w)
    w.add_argument("--host")
    w.add_argument("--port", type=int)
    w.set_defaults(func=cmd_dashboard)

    c = sub.add_parser("calibrate", help="click floor corners + tripwire, write the YAML")
    c.add_argument("--source", required=True)
    c.add_argument("--config", help="base config to start from")
    c.add_argument("--out", default="configs/nexus.yaml")
    c.add_argument("--floor-size", help="real floor patch in metres, e.g. 6x3 (keeps the map to scale)")
    c.add_argument("--pick-only", action="store_true", help="just print click coordinates")
    c.set_defaults(func=cmd_calibrate)

    lb = sub.add_parser("label", help="hand-label real crossings in a clip (ground truth)")
    lb.add_argument("--source", required=True)
    lb.add_argument("--out", required=True)
    lb.add_argument("--speed", type=float, default=1.0)
    lb.set_defaults(func=cmd_label)

    e = sub.add_parser("evaluate", help="measure accuracy against a labelled clip")
    add_cfg(e)
    e.add_argument("--truth", required=True, help="CSV from `nexus label`")
    e.add_argument("--tolerance", type=float, default=1.5, help="match window in seconds")
    e.add_argument("--stride", type=int, default=1, help="process every Nth frame (faster)")
    e.add_argument("--report", help="write the report as JSON")
    e.add_argument("--min-f1", type=float, default=None, help="exit 2 if overall F1 is lower")
    e.set_defaults(func=cmd_evaluate)

    x = sub.add_parser("export-model", help="export YOLO weights to ONNX/TensorRT/OpenVINO")
    x.add_argument("--model", default="yolov8n.pt")
    x.add_argument("--format", default="onnx", help="onnx | engine (TensorRT) | openvino | ncnn ...")
    x.add_argument("--imgsz", type=int, default=640)
    x.add_argument("--half", action="store_true", help="FP16 (GPU targets)")
    x.set_defaults(func=cmd_export)

    i = sub.add_parser("init-config", help="write a starter config file")
    i.add_argument("--out", default="configs/nexus.yaml")
    i.add_argument("--camera-id", default="camera-1")
    i.add_argument("--source", default="0")
    i.set_defaults(func=cmd_init)

    hc = sub.add_parser("healthcheck", help="exit 0 if the service heartbeat is fresh")
    add_cfg(hc, source=False)
    hc.set_defaults(func=cmd_healthcheck)

    pg = sub.add_parser("purge", help="apply the retention policy now")
    add_cfg(pg, source=False)
    pg.set_defaults(func=cmd_purge)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo" and args.pick_points:       # legacy flag from footfall_heatmap.py
        from .calibrate import pick_points
        pick_points(args.source or "0")
        return 0
    try:
        return int(args.func(args) or 0)
    except (ConfigError, SourceError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
