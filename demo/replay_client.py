"""Replay a canned observation fixture over a real WebSocket connection.

This is the direct replacement for the old `demo/make_demo_video.py`: instead
of rendering pixels for a camera pipeline to detect, it stands in for one or
more real phones, so the whole ingest -> triage -> alert path can be
exercised end to end against a live `argus run` with no phone, no on-device
model, and no camera anywhere in the loop.

Usage:
    python -m argus.cli demo --out demo/synthetic_stream.json --ticks 150
    python -m argus.cli run --http-port 8080 &
    python demo/replay_client.py --fixture demo/synthetic_stream.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


async def _replay_station(
    station: dict, host: str, port: int, protocol_version: int, speed: float
) -> tuple[str, int, str | None]:
    """Stream one station's messages. Returns (trainee_id, sent, error)."""
    trainee_id = station["trainee_id"]
    uri = f"ws://{host}:{port}"
    try:
        async with websockets.connect(uri) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "protocol_version": protocol_version,
                        "station_id": station["station_id"],
                        "trainee_id": trainee_id,
                    }
                )
            )
            ack = json.loads(await ws.recv())
            if ack.get("type") != "hello_ack":
                return trainee_id, 0, f"handshake rejected: {ack}"

            sent = 0
            last_ts = None
            for message in station["messages"]:
                if last_ts is not None and speed > 0:
                    await asyncio.sleep(max(0.0, (message["ts"] - last_ts) / speed))
                last_ts = message["ts"]
                await ws.send(json.dumps(message))
                sent += 1
            return trainee_id, sent, None
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        return trainee_id, 0, str(exc)


async def main_async(args: argparse.Namespace) -> int:
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    results = await asyncio.gather(
        *(
            _replay_station(station, args.ws_host, args.ws_port, fixture["protocol_version"], args.speed)
            for station in fixture["stations"]
        )
    )

    failed = False
    for trainee_id, sent, error in results:
        if error:
            failed = True
            print(f"[{trainee_id}] FAILED after {sent} messages: {error}", file=sys.stderr)
        else:
            print(f"[{trainee_id}] sent {sent} observations")
    return 1 if failed else 0


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
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
