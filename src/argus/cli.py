"""`argus` command line: run, doctor, bootstrap, config, demo.

Flags override config; config never overrides flags. Every override is applied
to a loaded :class:`~argus.config.ArgusConfig` and echoed by `argus config`, so
what a run was actually tuned with is always recoverable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from argus import __version__
from argus.alerts import ConsoleAlertSink
from argus.config import ArgusConfig, CameraConfig, ConfigError, load_config


def _apply_overrides(cfg: ArgusConfig, args: argparse.Namespace) -> ArgusConfig:
    engine = cfg.engine
    if args.engine:
        engine = dataclasses.replace(engine, kind=args.engine)
    if args.allow_cpu_fallback:
        engine = dataclasses.replace(engine, allow_cpu_fallback=True)

    outputs = cfg.outputs
    if args.json_log is not None:
        outputs = dataclasses.replace(outputs, json_log=args.json_log)
    if args.http_port is not None:
        outputs = dataclasses.replace(outputs, http_port=args.http_port)
    if args.overlay_out is not None:
        outputs = dataclasses.replace(outputs, overlay_out=args.overlay_out)
    if args.window:
        outputs = dataclasses.replace(outputs, overlay_window=True)
    if args.quiet:
        outputs = dataclasses.replace(outputs, console=False)

    cameras = cfg.cameras
    if args.camera:
        base = cfg.source_path.parent if cfg.source_path else Path.cwd()
        asset_dir = base.parent if base.name == "configs" else base
        cameras = tuple(
            CameraConfig(
                id=f"cam{i}",
                source=int(spec) if spec.isdigit() else spec,
                enabled=True,
                base_dir=asset_dir,
            )
            for i, spec in enumerate(args.camera)
        )

    vlm = cfg.vlm
    if args.vlm:
        vlm = dataclasses.replace(vlm, kind=args.vlm)

    return dataclasses.replace(
        cfg, engine=engine, outputs=outputs, cameras=cameras, vlm=vlm
    )


def cmd_run(args: argparse.Namespace) -> int:
    from argus.pipeline.runner import ArgusPipeline, FrameClock, WallClock

    cfg = _apply_overrides(load_config(args.config), args)

    clock = None
    if args.clock == "frame":
        clock = FrameClock(args.fps)
    elif args.clock == "wall":
        clock = WallClock()

    sink = ConsoleAlertSink(cfg.outputs.console)
    pipeline = ArgusPipeline(
        cfg, clock=clock, alert_sink=sink, headless=not cfg.outputs.overlay_window
    )

    print(
        f"argus {__version__} | engine={cfg.engine.kind} vlm={cfg.vlm.kind} "
        f"cameras={[c.id for c in cfg.enabled_cameras()]}",
        file=sys.stderr,
    )
    if pipeline.http_port:
        print(
            f"triage endpoint: http://{cfg.outputs.http_host}:{pipeline.http_port}/triage",
            file=sys.stderr,
        )

    ticks = pipeline.run(max_ticks=args.max_ticks)
    print(f"processed {ticks} ticks", file=sys.stderr)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    payload = {
        "source": str(cfg.source_path) if cfg.source_path else None,
        "scoring": dataclasses.asdict(cfg.scoring),
        "engine": dataclasses.asdict(cfg.engine),
        "detector": dataclasses.asdict(cfg.detector),
        "pose": dataclasses.asdict(cfg.pose),
        "super_res": dataclasses.asdict(cfg.super_res),
        "tracking": dataclasses.asdict(cfg.tracking),
        "vlm": dataclasses.asdict(cfg.vlm),
        "outputs": dataclasses.asdict(cfg.outputs),
        "cameras": [
            {"id": c.id, "source": str(c.resolved_source()), "enabled": c.enabled}
            for c in cfg.cameras
        ],
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from argus.doctor import run_doctor

    cfg = _apply_overrides(load_config(args.config), args)
    return run_doctor(cfg)


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from argus.provision import bootstrap

    cfg = load_config(args.config)
    return bootstrap(
        cfg, force=args.force, skip_demo=args.skip_demo, from_dir=args.from_dir
    )


def cmd_demo(args: argparse.Namespace) -> int:
    from demo.make_demo_video import make_demo_video

    path = make_demo_video(args.out, args.frames)
    print(f"wrote {args.frames} frames to {path}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to a config TOML (default: configs/argus.default.toml)")
    parser.add_argument("--engine", choices=["mock", "onnx-cpu", "qnn-npu"], help="Override engine.kind")
    parser.add_argument("--vlm", choices=["mock", "genie"], help="Override vlm.kind")
    parser.add_argument("--camera", action="append", help="Override cameras; repeatable. Index or file path.")
    parser.add_argument("--json-log", help="Override outputs.json_log")
    parser.add_argument("--http-port", type=int, help="Override outputs.http_port")
    parser.add_argument("--overlay-out", help="Override outputs.overlay_out")
    parser.add_argument("--window", action="store_true", help="Show a live overlay window")
    parser.add_argument("--quiet", action="store_true", help="Suppress console alerts")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permit the NPU engine to run on the CPU. Off by default: a silent "
        "CPU fallback makes an NPU latency budget fiction.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argus", description=__doc__)
    parser.add_argument("--version", action="version", version=f"argus {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the multi-camera triage pipeline")
    _add_common(run)
    run.add_argument("--max-ticks", type=int, help="Stop after N ticks")
    run.add_argument(
        "--clock",
        choices=["auto", "frame", "wall"],
        default="auto",
        help="auto = frame clock for file sources, wall clock for live cameras. "
        "The frame clock is what makes a run reproducible.",
    )
    run.add_argument("--fps", type=float, default=15.0, help="Frame-clock rate")
    run.set_defaults(func=cmd_run)

    doctor = sub.add_parser("doctor", help="Diagnose config, artifacts, and engine availability")
    _add_common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    config = sub.add_parser("config", help="Print the effective configuration")
    _add_common(config)
    config.set_defaults(func=cmd_config)

    bootstrap = sub.add_parser("bootstrap", help="Provision models/ and the demo clip")
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--force", action="store_true", help="Re-download present artifacts")
    bootstrap.add_argument("--skip-demo", action="store_true")
    bootstrap.add_argument(
        "--from-dir", type=Path, help="Seed from an existing models/ tree instead of AI Hub"
    )
    bootstrap.set_defaults(func=cmd_bootstrap)

    demo = sub.add_parser("demo", help="Regenerate the synthetic demo clip")
    demo.add_argument("--out", default="demo/trainees_demo.mp4")
    demo.add_argument("--frames", type=int, default=150)
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Engine and camera failures are expected operating conditions with
        # actionable messages -- a traceback buries the sentence that says what
        # to do. Re-raise anything without one so real bugs stay debuggable.
        from argus.engines.base import EngineError
        from argus.pipeline.source import CameraOpenError

        if isinstance(exc, (EngineError, CameraOpenError, FileNotFoundError)):
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
