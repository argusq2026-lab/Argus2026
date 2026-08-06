"""The WebSocket ingest server: handshake, protocol enforcement, reconnect
grace, and the sinks it feeds -- the direct successor to the old
`ArgusPipeline` + VLM-gate tests, now over a real socket instead of a mocked
camera.

`IngestServer.tick()` itself is synchronous and covered by
tests/test_determinism.py and tests/test_triage.py via the lower-level
`SessionRegistry` + `rank_trainees` it wraps; this file is about the network
boundary on top of it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest
import websockets

from argus.alerts import AlertSink
from argus.ingest.server import IngestServer
from argus.triage import TriageRecord

pytestmark = pytest.mark.timeout(60)


def _cfg(default_config, tmp_path, **outputs):
    ingest = dataclasses.replace(
        default_config.ingest, ws_host="127.0.0.1", ws_port=0, track_ttl_s=10.0
    )
    return dataclasses.replace(
        default_config,
        ingest=ingest,
        outputs=dataclasses.replace(
            default_config.outputs,
            console=False,
            json_log=outputs.get("json_log", ""),
            http_port=outputs.get("http_port", 0),
        ),
    )


class RecordingSink:
    """An AlertSink that also records `forget` calls, like ConsoleAlertSink."""

    def __init__(self):
        self.alerts: list[TriageRecord] = []
        self.forgotten: list[str] = []

    def __call__(self, record: TriageRecord) -> None:
        self.alerts.append(record)

    def forget(self, trainee_id: str) -> None:
        self.forgotten.append(trainee_id)


async def _connect_and_hello(port: int, station_id: str, trainee_id: str, protocol_version: int = 1):
    ws = await websockets.connect(f"ws://127.0.0.1:{port}")
    await ws.send(
        json.dumps(
            {
                "type": "hello",
                "protocol_version": protocol_version,
                "station_id": station_id,
                "trainee_id": trainee_id,
            }
        )
    )
    ack = json.loads(await ws.recv())
    return ws, ack


def _observation_message(ts: float, **overrides) -> str:
    base = {
        "type": "observation",
        "ts": ts,
        "bbox_xyxy": [0.1, 0.1, 0.5, 0.9],
        "keypoints_xy": [[0.3, 0.2]] * 17,
        "keypoints_conf": [0.9] * 17,
        "form_reason_codes": [],
    }
    base.update(overrides)
    return json.dumps(base)


def test_handshake_and_observation_round_trip(default_config, tmp_path):
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path), now=lambda: 0.0)
        await server.start()
        try:
            ws, ack = await _connect_and_hello(server.ws_port, "s0", "t0")
            assert ack == {"type": "hello_ack", "accepted": True}
            await ws.send(_observation_message(0.0))
            await asyncio.sleep(0.05)

            result = server.tick()
            assert result.active_stations == 1
            assert [r.trainee_id for r in result.records] == ["t0"]
            await ws.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_unknown_protocol_version_is_rejected(default_config, tmp_path):
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ws, ack = await _connect_and_hello(server.ws_port, "s0", "t0", protocol_version=99)
            assert ack["type"] == "error"
            with pytest.raises(websockets.ConnectionClosed):
                await ws.recv()
            assert ws.close_code == 1008
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_use_case_mismatch_is_rejected_at_handshake(default_config, tmp_path):
    """A fitness phone (the only kind that omits `use_case`) must not be
    admitted onto a laptop configured for welding."""
    cfg = _cfg(default_config, tmp_path)
    cfg = dataclasses.replace(cfg, session=dataclasses.replace(cfg.session, use_case="welding"))

    async def run():
        server = IngestServer(cfg)
        await server.start()
        try:
            ws, ack = await _connect_and_hello(server.ws_port, "s0", "t0")
            assert ack["type"] == "error"
            assert "use_case" in ack["message"]
            with pytest.raises(websockets.ConnectionClosed):
                await ws.recv()
            assert ws.close_code == 1008
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_matching_use_case_is_admitted(default_config, tmp_path):
    cfg = _cfg(default_config, tmp_path)
    cfg = dataclasses.replace(cfg, session=dataclasses.replace(cfg.session, use_case="welding"))

    async def run():
        server = IngestServer(cfg)
        await server.start()
        try:
            ws = await websockets.connect(f"ws://127.0.0.1:{server.ws_port}")
            await ws.send(json.dumps({
                "type": "hello", "protocol_version": 1, "station_id": "s0",
                "trainee_id": "t0", "use_case": "welding",
            }))
            ack = json.loads(await ws.recv())
            assert ack == {"type": "hello_ack", "accepted": True}
            await ws.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_an_observation_switching_use_case_mid_stream_is_rejected(default_config, tmp_path):
    """`hello` fixes the connection's use case; an `observation` naming a
    different one is a protocol violation, not a mode change."""
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ws, ack = await _connect_and_hello(server.ws_port, "s0", "t0")
            assert ack == {"type": "hello_ack", "accepted": True}
            await ws.send(json.dumps({"type": "observation", "use_case": "welding", "ts": 0.0}))
            error = json.loads(await ws.recv())
            assert error["type"] == "error"
            assert "use_case" in error["message"]
            with pytest.raises(websockets.ConnectionClosed):
                await ws.recv()
        finally:
            await server.stop()

    asyncio.run(run())


def test_set_use_case_changes_what_future_hellos_are_checked_against(default_config, tmp_path):
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ok, message = server.set_use_case("welding")
            assert ok is True
            assert message == ""
            assert server.cfg.session.use_case == "welding"

            # A fitness phone (the implicit default) is now the mismatch.
            ws, ack = await _connect_and_hello(server.ws_port, "s0", "t0")
            assert ack["type"] == "error"
            assert "use_case" in ack["message"]
        finally:
            await server.stop()

    asyncio.run(run())


def test_set_use_case_rejects_one_this_build_cannot_score(default_config, tmp_path):
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ok, message = server.set_use_case("nursing")
            assert ok is False
            assert "nursing" in message
            # Unchanged: a rejected change must not partially apply.
            assert server.cfg.session.use_case == "fitness"
        finally:
            await server.stop()

    asyncio.run(run())


def test_set_use_case_does_not_affect_an_already_connected_phone(default_config, tmp_path):
    """An instructor switching floors mid-session must not retroactively
    reclassify a trainee someone is already relying on."""
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ws, ack = await _connect_and_hello(server.ws_port, "s0", "t0")
            assert ack == {"type": "hello_ack", "accepted": True}

            ok, _ = server.set_use_case("welding")
            assert ok is True

            # The already-admitted fitness phone keeps working.
            await ws.send(_observation_message(0.0))
            await asyncio.sleep(0.05)
            result = server.tick()
            assert result.active_stations == 1
        finally:
            await server.stop()

    asyncio.run(run())


def test_duplicate_trainee_id_is_rejected(default_config, tmp_path):
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ws1, ack1 = await _connect_and_hello(server.ws_port, "s0", "dup")
            assert ack1 == {"type": "hello_ack", "accepted": True}

            ws2, ack2 = await _connect_and_hello(server.ws_port, "s1", "dup")
            assert ack2["type"] == "error"
            await ws1.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_an_unrecognised_form_code_closes_the_connection(default_config, tmp_path):
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ws, _ = await _connect_and_hello(server.ws_port, "s0", "t0")
            await ws.send(_observation_message(0.0, form_reason_codes=["not_a_real_code"]))
            msg = json.loads(await ws.recv())
            assert msg["type"] == "error"
            with pytest.raises(websockets.ConnectionClosed):
                await ws.recv()
        finally:
            await server.stop()

    asyncio.run(run())


def test_reconnecting_within_the_grace_window_resumes_history(default_config, tmp_path):
    async def run():
        server = IngestServer(_cfg(default_config, tmp_path))
        await server.start()
        try:
            ws1, _ = await _connect_and_hello(server.ws_port, "s0", "t0")
            await ws1.send(_observation_message(0.0))
            await asyncio.sleep(0.05)
            await ws1.close()
            await asyncio.sleep(0.05)

            ws2, ack = await _connect_and_hello(server.ws_port, "s0", "t0")
            assert ack == {"type": "hello_ack", "accepted": True}
            assert len(server._registry.tracks()["t0"].history) == 1
            await ws2.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_alert_sink_and_forget_on_expiry(default_config, tmp_path):
    from argus.synthetic import FALL_START_TICK, synthetic_tick

    async def run():
        cfg = _cfg(default_config, tmp_path)
        cfg = dataclasses.replace(cfg, ingest=dataclasses.replace(cfg.ingest, track_ttl_s=0.01))
        sink: AlertSink = RecordingSink()

        clock = {"t": 0.0}
        server = IngestServer(cfg, alert_sink=sink, now=lambda: clock["t"])
        await server.start()
        try:
            ws, _ = await _connect_and_hello(server.ws_port, "s0", "t0")
            # The synthetic "faller" sequence is the known-good fixture for
            # crossing alert_threshold: the pre-fall stillness plus the fall
            # transient together clear it, same as a real run from tick 0.
            for tick in range(0, FALL_START_TICK + 7):
                obs = synthetic_tick(tick)["faller"]
                await ws.send(
                    _observation_message(
                        obs.ts,
                        bbox_xyxy=list(obs.bbox_xyxy),
                        keypoints_xy=[list(p) for p in obs.keypoints_xy],
                        keypoints_conf=list(obs.keypoints_conf),
                    )
                )
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)

            server.tick()
            assert sink.alerts, "the synthetic fall sequence produced no alert"

            clock["t"] = 100.0  # advance past track_ttl_s while still connected
            server.tick()
            assert "t0" in sink.forgotten
            await ws.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_json_and_http_sinks_receive_ticks(default_config, tmp_path):
    async def run():
        log_path = tmp_path / "triage.jsonl"
        server = IngestServer(_cfg(default_config, tmp_path, json_log=str(log_path), http_port=0))
        await server.start()
        try:
            ws, _ = await _connect_and_hello(server.ws_port, "s0", "t0")
            await ws.send(_observation_message(0.0))
            await asyncio.sleep(0.05)
            server.tick()
            await ws.close()
        finally:
            await server.stop()

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["records"][0]["trainee_id"] == "t0"

    asyncio.run(run())


def test_max_ticks_stops_serve_forever(default_config, tmp_path):
    async def run():
        cfg = _cfg(default_config, tmp_path)
        cfg = dataclasses.replace(cfg, ingest=dataclasses.replace(cfg.ingest, rank_interval_s=0.01))
        server = IngestServer(cfg)
        await asyncio.wait_for(server.serve_forever(max_ticks=3), timeout=10)
        assert server.ticks == 3

    asyncio.run(run())
