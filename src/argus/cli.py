"""`argus` command line: run, doctor, config, demo.

Flags override config; config never overrides flags. Every override is applied
to a loaded :class:`~argus.config.ArgusConfig` and echoed by `argus config`, so
what a run was actually tuned with is always recoverable.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from argus import __version__
from argus.alerts import ConsoleAlertSink
from argus.config import ArgusConfig, ConfigError, load_config


def _apply_overrides(cfg: ArgusConfig, args: argparse.Namespace) -> ArgusConfig:
    ingest = cfg.ingest
    if args.ws_host is not None:
        ingest = dataclasses.replace(ingest, ws_host=args.ws_host)
    if args.ws_port is not None:
        ingest = dataclasses.replace(ingest, ws_port=args.ws_port)

    outputs = cfg.outputs
    if args.json_log is not None:
        outputs = dataclasses.replace(outputs, json_log=args.json_log)
    if args.http_port is not None:
        outputs = dataclasses.replace(outputs, http_port=args.http_port)
    if args.quiet:
        outputs = dataclasses.replace(outputs, console=False)

    return dataclasses.replace(cfg, ingest=ingest, outputs=outputs)


def cmd_run(args: argparse.Namespace) -> int:
    from argus.ingest.server import IngestServer

    cfg = _apply_overrides(load_config(args.config), args)
    sink = ConsoleAlertSink(cfg.outputs.console)
    server = IngestServer(cfg, alert_sink=sink)

    print(
        f"argus {__version__} | ws://{cfg.ingest.ws_host}:{cfg.ingest.ws_port} "
        f"| protocol_version={cfg.ingest.protocol_version}",
        file=sys.stderr,
    )
    if cfg.outputs.http_port:
        print(
            f"triage endpoint:   http://{cfg.outputs.http_host}:{cfg.outputs.http_port}/triage",
            file=sys.stderr,
        )
        print(
            f"trainer dashboard: http://{cfg.outputs.http_host}:{cfg.outputs.http_port}/",
            file=sys.stderr,
        )

    try:
        asyncio.run(server.serve_forever(max_ticks=args.max_ticks))
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(
            f"[ERROR] could not bind {cfg.ingest.ws_host}:{cfg.ingest.ws_port}: {exc}\n"
            "        another process may already be listening there -- pick a "
            "different --ws-port, or run `argus doctor` to check first.",
            file=sys.stderr,
        )
        return 1
    print(f"processed {server.ticks} rank ticks", file=sys.stderr)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    payload = {
        "source": str(cfg.source_path) if cfg.source_path else None,
        "scoring": dataclasses.asdict(cfg.scoring),
        "ingest": dataclasses.asdict(cfg.ingest),
        "outputs": dataclasses.asdict(cfg.outputs),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from argus.doctor import run_doctor

    cfg = _apply_overrides(load_config(args.config), args)
    return run_doctor(cfg)


def cmd_demo(args: argparse.Namespace) -> int:
    from argus.synthetic import build_fixture

    fixture = build_fixture(n_ticks=args.ticks)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(
        f"wrote a {args.ticks}-tick, {len(fixture['stations'])}-station fixture to {path}\n"
        f"replay it against a running `argus run` with:\n"
        f"  python demo/replay_client.py --fixture {path}"
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to a config TOML (default: configs/argus.default.toml)")
    parser.add_argument("--ws-host", help="Override ingest.ws_host")
    parser.add_argument("--ws-port", type=int, help="Override ingest.ws_port")
    parser.add_argument("--json-log", help="Override outputs.json_log")
    parser.add_argument("--http-port", type=int, help="Override outputs.http_port")
    parser.add_argument("--quiet", action="store_true", help="Suppress console alerts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argus", description=__doc__)
    parser.add_argument("--version", action="version", version=f"argus {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the WebSocket ingest server + triage ranking")
    _add_common(run)
    run.add_argument("--max-ticks", type=int, help="Stop after N rank ticks (default: run forever)")
    run.set_defaults(func=cmd_run)

    doctor = sub.add_parser("doctor", help="Diagnose config and ingest-server readiness")
    _add_common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    config = sub.add_parser("config", help="Print the effective configuration")
    _add_common(config)
    config.set_defaults(func=cmd_config)

    demo = sub.add_parser("demo", help="Generate a canned multi-station observation fixture")
    demo.add_argument("--out", default="demo/synthetic_stream.json")
    demo.add_argument("--ticks", type=int, default=150)
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


if __name__ == "__main__":
    raise SystemExit(main())
