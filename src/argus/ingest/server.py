"""The WebSocket ingest server.

Every phone owns one connection and pushes observations into its own
trainee's `TrackState` as they arrive — order matters: an observation is
always pushed before the next rank tick reads it, the same guarantee the old
per-camera frame loop gave the VLM gate. A separate periodic task recomputes
the merged rank on `ingest.rank_interval_s`, decoupled from how fast any one
phone streams, and dispatches it to the same sinks the camera pipeline used
(`argus.outputs`, `argus.alerts`) — those are unchanged, because the alert
boundary never cared where the numbers came from.

`IngestServer.tick()` is synchronous and network-free on purpose: it is the
whole scoring path in one testable call, so tests exercise it directly
without a real socket or a live clock (see tests/test_ingest_server.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import websockets
from websockets.asyncio.server import ServerConnection

from argus.alerts import AlertSink
from argus.config import ArgusConfig
from argus.ingest.protocol import (
    ProtocolError,
    error_message,
    hello_ack_message,
    parse_hello,
    parse_observation,
)
from argus.ingest.session import DuplicateTraineeError, SessionRegistry
from argus.outputs import JsonLogSink, TriageHTTPServer
from argus.triage import TriageRecord, needs_instructor, rank_trainees

logger = logging.getLogger(__name__)

#: WebSocket close codes. 1008 = policy violation (bad/unrecognised message);
#: 1002 = protocol error (no hello within the handshake timeout).
_POLICY_VIOLATION = 1008
_PROTOCOL_ERROR = 1002

#: How long a connection has to send its `hello` before it is dropped.
_HELLO_TIMEOUT_S = 10.0

#: A WebSocket close reason is a control-frame payload, capped at 123 bytes;
#: the full explanation always goes to the client as an `error` message
#: first, so the close reason only needs to be a short pointer to it.
_MAX_CLOSE_REASON = 100


def _close_reason(message: str) -> str:
    return message if len(message) <= _MAX_CLOSE_REASON else message[: _MAX_CLOSE_REASON - 1] + "…"


@dataclass
class TickResult:
    """One rank tick's merged outcome."""

    ts: float
    records: list[TriageRecord]
    alerts: list[TriageRecord]
    active_stations: int
    expired_trainee_ids: list[str] = field(default_factory=list)


class IngestServer:
    """Owns the session registry, the sinks, and the WebSocket listener."""

    def __init__(
        self,
        cfg: ArgusConfig,
        alert_sink: AlertSink | None = None,
        now: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self._alert_sink = alert_sink
        self._now = now
        self._registry = SessionRegistry(cfg.scoring, cfg.ingest.track_ttl_s)
        self._json_sink = JsonLogSink(cfg.outputs.json_log) if cfg.outputs.json_log else None
        self._http: TriageHTTPServer | None = None
        if cfg.outputs.http_port:
            self._http = TriageHTTPServer(cfg.outputs.http_port, cfg.outputs.http_host)
        self.ticks = 0
        self._ws_server: websockets.Server | None = None
        self._rank_task: asyncio.Task | None = None

    @property
    def http_port(self) -> int | None:
        return self._http.port if self._http is not None else None

    @property
    def ws_port(self) -> int:
        """The bound WebSocket port, resolved even when `ingest.ws_port = 0`."""
        if self._ws_server is None:
            raise RuntimeError("the server has not been started yet")
        ports = {sock.getsockname()[1] for sock in self._ws_server.sockets or ()}
        return next(iter(ports))

    @property
    def active_trainee_ids(self) -> list[str]:
        return list(self._registry.tracks())

    # -- the scoring path: synchronous, network-free ------------------------

    def tick(self) -> TickResult:
        """Recompute the merged rank and dispatch it to every sink."""
        now = self._now()
        expired = self._registry.expire_stale(now)
        forget = getattr(self._alert_sink, "forget", None)
        if forget is not None:
            for trainee_id in expired:
                forget(trainee_id)

        records = rank_trainees(self._registry.tracks(), now, self.cfg.scoring)
        alerts = needs_instructor(records, self.cfg.scoring)

        if self._alert_sink is not None:
            for record in alerts:
                self._alert_sink(record)
        if self._json_sink is not None:
            self._json_sink.write(now, records)
        if self._http is not None:
            self._http.update(now, records)

        self.ticks += 1
        return TickResult(now, records, alerts, len(self._registry), expired)

    # -- the WebSocket listener -----------------------------------------------

    async def _handle(self, ws: ServerConnection) -> None:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=_HELLO_TIMEOUT_S)
        except (TimeoutError, asyncio.TimeoutError, websockets.ConnectionClosed):
            await ws.close(_PROTOCOL_ERROR, "no hello received in time")
            return

        try:
            hello = parse_hello(json.loads(raw), self.cfg.ingest.protocol_version)
        except (ProtocolError, json.JSONDecodeError) as exc:
            await ws.send(json.dumps(error_message(str(exc))))
            await ws.close(_POLICY_VIOLATION, _close_reason(str(exc)))
            return

        try:
            self._registry.register(hello.station_id, hello.trainee_id, self._now())
        except DuplicateTraineeError as exc:
            await ws.send(json.dumps(error_message(str(exc))))
            await ws.close(_POLICY_VIOLATION, _close_reason(str(exc)))
            return

        await ws.send(json.dumps(hello_ack_message()))
        logger.info("station %s connected as trainee %s", hello.station_id, hello.trainee_id)

        try:
            async for raw_message in ws:
                try:
                    obs = parse_observation(json.loads(raw_message), self.cfg.scoring.form_error_vocab)
                except (ProtocolError, json.JSONDecodeError) as exc:
                    await ws.send(json.dumps(error_message(str(exc))))
                    await ws.close(_POLICY_VIOLATION, _close_reason(str(exc)))
                    return
                try:
                    self._registry.push_observation(hello.trainee_id, obs, self._now())
                except KeyError:
                    # expire_stale already dropped this session (silent past
                    # ingest.track_ttl_s) while the socket lingered open.
                    msg = f"session for {hello.trainee_id!r} expired; reconnect with a fresh hello"
                    await ws.send(json.dumps(error_message(msg)))
                    await ws.close(_POLICY_VIOLATION, _close_reason(msg))
                    return
        finally:
            self._registry.mark_disconnected(hello.trainee_id)
            logger.info("station %s (trainee %s) disconnected", hello.station_id, hello.trainee_id)

    async def _rank_loop(self) -> None:
        interval = self.cfg.ingest.rank_interval_s
        while True:
            await asyncio.sleep(interval)
            self.tick()

    async def start(self) -> None:
        """Bind the WebSocket listener and start the rank loop.

        Split from `serve_forever` so tests (and `argus doctor`-style
        tooling) can start the server, learn `ws_port`, drive it with a real
        client, and `stop()` it explicitly — no `--max-ticks`-style escape
        hatch needed, unlike the old frame-bounded camera loop.
        """
        if self._http is not None:
            self._http.start()
        self._ws_server = await websockets.serve(
            self._handle, self.cfg.ingest.ws_host, self.cfg.ingest.ws_port
        )
        self._rank_task = asyncio.create_task(self._rank_loop())
        logger.info("ingest server listening on %s:%s", self.cfg.ingest.ws_host, self.ws_port)

    async def stop(self) -> None:
        if self._rank_task is not None:
            self._rank_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rank_task
            self._rank_task = None
        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None
        if self._json_sink is not None:
            self._json_sink.close()
        if self._http is not None:
            self._http.stop()

    async def serve_forever(self, max_ticks: int | None = None) -> None:
        """Run until cancelled — the entry point `argus run` uses.

        `websockets.serve` already accepts connections in the background as
        soon as `start()` returns; this coroutine's only job is to stay alive
        until told to stop. `max_ticks` bounds it to that many rank ticks —
        used by CI and tests, mirroring the old camera loop's `--max-ticks`.
        """
        await self.start()
        try:
            if max_ticks is None:
                await asyncio.Event().wait()  # blocks until this task is cancelled
            else:
                while self.ticks < max_ticks:
                    await asyncio.sleep(min(self.cfg.ingest.rank_interval_s, 0.05))
        finally:
            await self.stop()
