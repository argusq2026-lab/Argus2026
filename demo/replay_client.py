"""Replay a canned observation fixture over a real WebSocket connection.

This is the direct replacement for the old `demo/make_demo_video.py`: instead
of rendering pixels for a camera pipeline to detect, it stands in for one or
more real phones, so the whole ingest -> triage -> alert path can be
exercised end to end against a live `argus run` with no phone, no on-device
model, and no camera anywhere in the loop.

The replay itself lives in `argus.replay` so the shipped binary can run it
too (`argus replay`); this script is the checkout-friendly entry point and
stays byte-compatible with the CI invocation.

Usage:
    python -m argus.cli demo --out demo/synthetic_stream.json --ticks 150
    python -m argus.cli run --http-port 8080 &
    python demo/replay_client.py --fixture demo/synthetic_stream.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from argus.replay import replay_fixture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="demo/synthetic_stream.json")
    parser.add_argument("--ws-host", default="127.0.0.1")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Playback speed multiplier honouring the fixture's own timestamps "
        "(1.0 = real time). 0 (default) sends as fast as possible -- the right "
        "choice for CI and quick smoke tests.",
    )
    args = parser.parse_args()
    return replay_fixture(args.fixture, args.ws_host, args.ws_port, args.speed)


if __name__ == "__main__":
    raise SystemExit(main())
