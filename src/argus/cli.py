"""`argus` command line: run, replay, doctor, config, demo.

Flags override config; config never overrides flags. Every override is applied
to a loaded :class:`~argus.config.ArgusConfig` and echoed by `argus config`, so
what a run was actually tuned with is always recoverable.

Running with no arguments at all is the same as `argus run --open`. That is
for the packaged binary (see `scripts/build_binary.py`): someone who launches
Argus by double-clicking it has not passed a subcommand and should still get a
working server and a console in front of them, rather than a usage error.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import webbrowser
from pathlib import Path

from argus import __version__
from argus.alerts import ConsoleAlertSink
from argus.config import ArgusConfig, ConfigError, load_config

#: Where the console lands when someone launches Argus without choosing a
#: port. Only ever applied on the `--open` path -- `outputs.http_port = 0`
#: still means "no HTTP at all" for an explicit `argus run`, because turning
#: an endpoint on that an operator switched off would be a surprise.
DEFAULT_CONSOLE_PORT = 8080


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
    if getattr(args, "open", False) and not cfg.outputs.http_port:
        cfg = dataclasses.replace(
            cfg, outputs=dataclasses.replace(cfg.outputs, http_port=DEFAULT_CONSOLE_PORT)
        )

    sink = ConsoleAlertSink(cfg.outputs.console)
    try:
        # The HTTP server binds inside this constructor, so a port collision
        # surfaces here rather than in `serve_forever` -- without this arm it
        # comes out as a bare traceback, which is a poor thing to hand someone
        # who just double-clicked the binary.
        server = IngestServer(cfg, alert_sink=sink)
    except OSError as exc:
        print(
            f"[ERROR] could not bind {cfg.outputs.http_host}:{cfg.outputs.http_port} "
            f"for the console: {exc}\n"
            "        another process is already listening there -- pick a different "
            "--http-port, or run `argus doctor` to check first.",
            file=sys.stderr,
        )
        return 1

    console_url = f"http://{cfg.outputs.http_host}:{cfg.outputs.http_port}/"
    print(
        f"argus {__version__} | ws://{cfg.ingest.ws_host}:{cfg.ingest.ws_port} "
        f"| protocol_version={cfg.ingest.protocol_version}",
        file=sys.stderr,
    )
    if cfg.outputs.http_port:
        print(f"triage endpoint:  {console_url}triage", file=sys.stderr)
        print(f"trainer console:  {console_url}", file=sys.stderr)
        if getattr(args, "open", False):
            # The listening socket exists as of the constructor above, so a
            # browser that connects before the serving thread is scheduled
            # waits in the backlog rather than seeing ECONNREFUSED.
            webbrowser.open(console_url)

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


def cmd_replay(args: argparse.Namespace) -> int:
    """Drive a running server with a fixture, standing in for real phones.

    The same code `demo/replay_client.py` runs, exposed as a subcommand so the
    packaged binary can demonstrate itself on a laptop with no checkout.
    """
    from argus.replay import replay_fixture

    return replay_fixture(args.fixture, args.ws_host or "127.0.0.1", args.ws_port or 8765, args.speed)


def cmd_discover(args: argparse.Namespace) -> int:
    """Listen for Argus beacons on the LAN — what a phone does at setup.

    Useful on its own ("is the laptop actually advertising?"), and it is the
    reference implementation of the listening half: the Android client does
    exactly this, so the beacon format can be checked without a phone.
    """
    from argus.discovery import listen

    cfg = _apply_overrides(load_config(args.config), args)
    print(
        f"listening for argus beacons on udp/{cfg.discovery.port} for {args.timeout}s…",
        file=sys.stderr,
    )
    found = listen(cfg.discovery.port, args.timeout, cfg.ingest.protocol_version)
    if not found:
        print(
            "no server found. Check that a laptop is running `argus run` on this "
            "network with discovery.enabled = true, and that the network does not "
            "block broadcast traffic -- a phone can always be given the address by hand.",
            file=sys.stderr,
        )
        return 1
    for beacon in found:
        label = f"  {beacon.get('session_name') or beacon['ws_url']}"
        if beacon.get("session_name"):
            label += f"  {beacon['ws_url']}"
        label += f"  protocol_version={beacon['protocol_version']}"
        if beacon.get("approval") == "manual":
            label += "  [the instructor approves joins]"
        print(label)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    payload = {
        "source": str(cfg.source_path) if cfg.source_path else None,
        "scoring": dataclasses.asdict(cfg.scoring),
        "ingest": dataclasses.asdict(cfg.ingest),
        "outputs": dataclasses.asdict(cfg.outputs),
        "session": dataclasses.asdict(cfg.session),
        "discovery": dataclasses.asdict(cfg.discovery),
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
    # `argus replay` rather than `python demo/replay_client.py`: the packaged
    # binary has no checkout beside it, and pointing someone at a file that is
    # not there is a worse first experience than a slightly longer command.
    print(
        f"wrote a {args.ticks}-tick, {len(fixture['stations'])}-station fixture to {path}\n"
        f"replay it against a running `argus run` with:\n"
        f"  argus replay --fixture {path}\n"
        f"add --speed 1.0 to stream at the fixture's own pace and watch the "
        f"console behave like a real floor."
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
    run.add_argument(
        "--open",
        action="store_true",
        help=f"Open the trainer console in a browser, enabling it on port "
        f"{DEFAULT_CONSOLE_PORT} if no outputs.http_port is configured. "
        "Implied when argus is launched with no arguments at all.",
    )
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser(
        "replay", help="Replay a fixture into a running server, standing in for phones"
    )
    replay.add_argument("--fixture", default="demo/synthetic_stream.json")
    replay.add_argument("--ws-host", default="127.0.0.1")
    replay.add_argument("--ws-port", type=int, default=8765)
    replay.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Playback speed against the fixture's own timestamps (1.0 = real "
        "time). 0 (default) sends as fast as possible -- right for CI and "
        "smoke tests; use 1.0 to watch the console behave like a real floor.",
    )
    replay.set_defaults(func=cmd_replay)

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

    discover = sub.add_parser(
        "discover", help="Listen for Argus servers advertising themselves on this LAN"
    )
    _add_common(discover)
    discover.add_argument(
        "--timeout", type=float, default=3.0, help="Seconds to listen (default: 3)"
    )
    discover.set_defaults(func=cmd_discover)

    return parser


def default_argv(argv: list[str]) -> list[str]:
    """A bare launch means `run --open`.

    argparse would otherwise exit 2 with a usage message, which is the right
    answer in a terminal and the wrong one for the packaged binary, where "no
    arguments" is how it is normally started. An explicit subcommand is
    returned untouched.
    """
    return list(argv) if argv else ["run", "--open"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(default_argv(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
