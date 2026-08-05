"""Replay a canned observation fixture as if it were real phones.

This is the client side of `docs/PROTOCOL.md`, and it is the development loop:
one WebSocket connection per trainee, a `hello`, then the fixture's
`observation` messages at the fixture's own pace. It exercises ingest ->
triage -> alerts -> the trainer console end to end with no phone, no on-device
model, and no camera anywhere in it.

It lives in the package rather than only in `demo/` so the shipped binary can
drive its own demo: a laptop that has Argus but no Python checkout can still
run `argus demo` and then `argus replay` and watch the console fill up.
`demo/replay_client.py` is a thin wrapper over this module, so the two cannot
drift apart.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import websockets


async def replay_station(
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


async def replay_fixture_async(
    fixture: dict, host: str, port: int, speed: float
) -> list[tuple[str, int, str | None]]:
    """Every station in the fixture, concurrently — one connection each."""
    return list(
        await asyncio.gather(
            *(
                replay_station(station, host, port, fixture["protocol_version"], speed)
                for station in fixture["stations"]
            )
        )
    )


def replay_fixture(path: str | Path, host: str, port: int, speed: float) -> int:
    """Replay a fixture file. Returns a process exit code."""
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    results = asyncio.run(replay_fixture_async(fixture, host, port, speed))

    failed = False
    for trainee_id, sent, error in results:
        if error:
            failed = True
            print(f"[{trainee_id}] FAILED after {sent} messages: {error}", file=sys.stderr)
        else:
            print(f"[{trainee_id}] sent {sent} observations")
    return 1 if failed else 0
