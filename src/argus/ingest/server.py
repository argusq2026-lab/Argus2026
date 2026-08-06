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
from argus.discovery import DiscoveryBeacon, beacon_payload
from argus.ingest.admission import AdmissionQueue, Decision
from argus.ingest.protocol import (
    ProtocolError,
    error_message,
    hello_ack_message,
    join_pending_message,
    parse_hello,
    parse_observation,
)
from argus.ingest.session import DuplicateTraineeError, SessionRegistry
from argus.outputs import ConsoleSettings, JsonLogSink, TriageHTTPServer
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


#: What a phone is told when its join did not go through. Each names the
#: actual cause: "the instructor said no" and "nobody was looking at the
#: console" are different problems for whoever is standing next to the phone,
#: and collapsing them into one message hides which.
_JOIN_REFUSALS = {
    Decision.DENIED: "the instructor declined this join request",
    Decision.TIMED_OUT: (
        "no instructor answered this join request in time; ask them to approve "
        "it on the console, then connect again"
    ),
    Decision.SUPERSEDED: "a newer join request for this trainee replaced this one",
    Decision.WITHDRAWN: "the join request was withdrawn",
}


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
        self._admission = AdmissionQueue()
        self._json_sink = JsonLogSink(cfg.outputs.json_log) if cfg.outputs.json_log else None
        self._http: TriageHTTPServer | None = None
        if cfg.outputs.http_port:
            self._http = TriageHTTPServer(
                cfg.outputs.http_port,
                cfg.outputs.http_host,
                ConsoleSettings(
                    poll_interval_ms=cfg.outputs.console_poll_interval_ms,
                    stale_after_s=cfg.outputs.console_stale_after_s,
                    keypoint_conf_threshold=cfg.scoring.keypoint_conf_threshold,
                    alert_threshold=cfg.scoring.alert_threshold,
                    history_len=cfg.scoring.history_len,
                    track_ttl_s=cfg.ingest.track_ttl_s,
                    session_name=cfg.session.name,
                    approval=cfg.session.approval,
                    default_weights=dict(cfg.scoring.weights),
                    exercise_weights={
                        name: dict(profile)
                        for name, profile in cfg.scoring.exercise_weights.items()
                    },
                ),
                on_join_decision=self._admission.decide,
                allow_remote_control=cfg.outputs.allow_remote_join_control,
            )
        self.ticks = 0
        self._ws_server: websockets.Server | None = None
        self._rank_task: asyncio.Task | None = None
        self._beacon: DiscoveryBeacon | None = None

    def _build_beacon(self) -> DiscoveryBeacon | None:
        """The LAN beacon, or `None` when there is nothing worth advertising.

        Built at `start()` rather than in `__init__` so it can advertise the
        port that was actually bound — `ingest.ws_port = 0` means the OS picks
        one, and a beacon naming port 0 would send every phone nowhere.
        """
        if not self.cfg.discovery.enabled:
            return None
        payload = beacon_payload(
            self.cfg.ingest.ws_host,
            self.ws_port,
            self.cfg.ingest.protocol_version,
            self.cfg.session.name,
            self.cfg.session.approval,
        )
        if payload is None:
            logger.info(
                "discovery beacon not started: nothing phone-reachable to advertise "
                "(ws_host=%s). Phones can still be given the address by hand.",
                self.cfg.ingest.ws_host,
            )
            return None
        logger.info("advertising %s on udp/%s", payload["ws_url"], self.cfg.discovery.port)
        return DiscoveryBeacon(
            payload,
            port=self.cfg.discovery.port,
            interval_s=self.cfg.discovery.interval_s,
            broadcast=self.cfg.discovery.broadcast,
        )

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

        tracks = self._registry.tracks()
        records = rank_trainees(tracks, now, self.cfg.scoring)
        alerts = needs_instructor(records, self.cfg.scoring)

        # Advance each trainee's rolling score here rather than inside
        # `compute_triage`, which stays a pure function of history — that
        # purity is what tests/test_determinism.py protects. Alerts above are
        # computed first and from the *instant* score, deliberately: a fall
        # must fire on the frame it happens, not once a mean catches up.
        for record in records:
            track = tracks.get(record.trainee_id)
            if track is not None:
                track.session.observe_score(
                    record.score, now, self.cfg.scoring.rolling_half_life_s
                )

        if self._alert_sink is not None:
            for record in alerts:
                self._alert_sink(record)
        if self._json_sink is not None:
            self._json_sink.write(now, records)
        # Settle join requests nobody answered. Each waiting phone times itself
        # out too, so this is not what unblocks them -- it is what keeps an
        # unanswered prompt from sitting on the console after the phone behind
        # it has already given up and gone.
        self._admission.expire(now)

        if self._http is not None:
            self._http.update(
                now, records, self._registry.station_views(), self._admission.pending_views()
            )

        self.ticks += 1
        return TickResult(now, records, alerts, len(self._registry), expired)

    def _publish_stations(self) -> None:
        """Push the live station snapshot to the console, between rank ticks.

        The rank tick republishes this too, which is what keeps staleness
        honest: if every phone goes silent, no observation arrives to trigger
        this method, and only the tick's unconditional republish keeps the
        console's clock advancing so the cards visibly age. Without it a
        floor that had gone completely quiet would freeze at its last good
        frame and read as calm.
        """
        if self._http is not None:
            self._http.update_stations(
                self._now(), self._registry.station_views(), self._admission.pending_views()
            )

    # -- admission ------------------------------------------------------------

    async def _admit(self, ws: ServerConnection, hello) -> bool:
        """Decide whether this phone joins. True to proceed to `hello_ack`.

        In `session.approval = "auto"` this is a branch that does nothing,
        which is the point: the default path through the handshake is byte for
        byte what it was before admission existed.
        """
        if self.cfg.session.approval != "manual":
            return True

        timeout_s = self.cfg.session.join_timeout_s
        request = self._admission.submit(
            hello.station_id, hello.trainee_id, hello.display_name, self._now(), timeout_s
        )
        logger.info(
            "join requested by %s (station %s) — awaiting instructor",
            request.label,
            hello.station_id,
        )
        await ws.send(
            json.dumps(join_pending_message(self.cfg.session.name, request.request_id, timeout_s))
        )
        # Surface the prompt now rather than at the next rank tick: this one is
        # a person waiting at a rack, not a number that can be half a second late.
        self._publish_stations()

        # Race the decision against the phone hanging up. Without this arm, a
        # trainee who walked away mid-request leaves a prompt on the console
        # for the whole timeout, and the instructor approves a phone that is
        # not there.
        decided = asyncio.ensure_future(request.wait(timeout_s))
        hung_up = asyncio.ensure_future(ws.wait_closed())
        try:
            await asyncio.wait({decided, hung_up}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            hung_up.cancel()
            if not decided.done():
                self._admission.withdraw(request.request_id)
            decision = await decided

        self._publish_stations()
        if decision is Decision.APPROVED:
            logger.info("join approved for %s", request.label)
            return True

        logger.info("join not granted for %s: %s", request.label, decision.value)
        reason = _JOIN_REFUSALS.get(decision, "join request was not granted")
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps(error_message(reason)))
            await ws.close(_POLICY_VIOLATION, _close_reason(reason))
        return False

    # -- the WebSocket listener -----------------------------------------------

    async def _handle(self, ws: ServerConnection) -> None:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=_HELLO_TIMEOUT_S)
        except (TimeoutError, asyncio.TimeoutError, websockets.ConnectionClosed):
            await ws.close(_PROTOCOL_ERROR, "no hello received in time")
            return

        try:
            hello = parse_hello(
                json.loads(raw), self.cfg.ingest.protocol_version, self.cfg.session.name
            )
        except (ProtocolError, json.JSONDecodeError) as exc:
            await ws.send(json.dumps(error_message(str(exc))))
            await ws.close(_POLICY_VIOLATION, _close_reason(str(exc)))
            return

        # Collisions are settled before the instructor is bothered: asking
        # someone to approve a phone that is going to be refused anyway wastes
        # the one thing a manual gate costs, which is their attention.
        if self._registry.is_connected(hello.trainee_id):
            msg = f"trainee_id {hello.trainee_id!r} is already connected"
            await ws.send(json.dumps(error_message(msg)))
            await ws.close(_POLICY_VIOLATION, _close_reason(msg))
            return

        if not await self._admit(ws, hello):
            return

        try:
            self._registry.register(hello.station_id, hello.trainee_id, self._now())
        except DuplicateTraineeError as exc:
            # Two phones can be approved for one trainee_id in the window
            # between the check above and here. The registry is the authority
            # on identity, not the admission queue, so it still gets the last
            # word.
            await ws.send(json.dumps(error_message(str(exc))))
            await ws.close(_POLICY_VIOLATION, _close_reason(str(exc)))
            return

        await ws.send(json.dumps(hello_ack_message()))
        # Show the station on the console at handshake, not at its first
        # observation: a phone that connects and then sends nothing is a
        # failure worth seeing, and it is invisible if the card only appears
        # once a frame arrives.
        self._publish_stations()
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
                self._publish_stations()
        finally:
            self._registry.mark_disconnected(hello.trainee_id)
            # Republish immediately rather than waiting up to a whole
            # rank_interval_s: a dropped phone is exactly the event a trainer
            # needs promptly, and it is the console's job to show the grace
            # window counting down, not to hide it for half a second.
            self._publish_stations()
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
        self._beacon = self._build_beacon()
        if self._beacon is not None:
            self._beacon.start()
        logger.info("ingest server listening on %s:%s", self.cfg.ingest.ws_host, self.ws_port)

    async def stop(self) -> None:
        if self._beacon is not None:
            self._beacon.stop()
            self._beacon = None
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
