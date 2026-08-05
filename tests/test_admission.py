"""Join approval: who gets onto an instructor's floor.

Two things get the most coverage here, because both are ways this feature
could quietly hurt rather than help:

* **Auto is unchanged.** `session.approval = "auto"` is the default, and the
  handshake under it must be exactly what it was before admission existed —
  a gate that slowed down or altered the ordinary path would be a regression
  dressed as a feature.
* **Every request ends.** Approved, denied, superseded, timed out, or
  withdrawn because the phone hung up. A join that never resolves is a phone
  that looks hung and a trainee nobody is watching.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest
import websockets

from argus.ingest.admission import AdmissionQueue, Decision
from argus.ingest.server import IngestServer

pytestmark = pytest.mark.timeout(60)


def _cfg(default_config, **session):
    return dataclasses.replace(
        default_config,
        ingest=dataclasses.replace(
            default_config.ingest, ws_host="127.0.0.1", ws_port=0, track_ttl_s=10.0
        ),
        outputs=dataclasses.replace(default_config.outputs, console=False, http_port=0),
        session=dataclasses.replace(default_config.session, **session),
        discovery=dataclasses.replace(default_config.discovery, enabled=False),
    )


async def _hello(port, trainee_id="t0", **extra):
    ws = await websockets.connect(f"ws://127.0.0.1:{port}")
    await ws.send(
        json.dumps(
            {
                "type": "hello",
                "protocol_version": 1,
                "station_id": "s0",
                "trainee_id": trainee_id,
                **extra,
            }
        )
    )
    return ws


# -- the queue on its own -----------------------------------------------------


def test_a_request_resolves_once_and_the_second_caller_is_told_it_lost():
    """A double-clicked Approve must not overwrite the first outcome."""

    async def run():
        queue = AdmissionQueue()
        request = queue.submit("s0", "t0", "Alex", now=0.0, timeout_s=10.0)
        assert queue.decide(request.request_id, True) is True
        assert queue.decide(request.request_id, False) is False
        assert request.decision is Decision.APPROVED

    asyncio.run(run())


def test_a_reconnecting_phone_supersedes_its_own_stale_request():
    """Refusing the second would lock a phone that dropped and came back out
    of the floor for the whole timeout."""

    async def run():
        queue = AdmissionQueue()
        first = queue.submit("s0", "t0", "", now=0.0, timeout_s=10.0)
        second = queue.submit("s0", "t0", "", now=1.0, timeout_s=10.0)

        assert first.decision is Decision.SUPERSEDED
        assert [r.request_id for r in queue.pending()] == [second.request_id]

    asyncio.run(run())


def test_a_different_trainee_does_not_supersede():
    async def run():
        queue = AdmissionQueue()
        a = queue.submit("s0", "alice", "", now=0.0, timeout_s=10.0)
        b = queue.submit("s1", "bob", "", now=1.0, timeout_s=10.0)
        assert a.decision is None
        assert len(queue.pending()) == 2
        assert [r.trainee_id for r in queue.pending()] == ["alice", "bob"]
        assert b.decision is None

    asyncio.run(run())


def test_expiry_settles_a_request_nobody_answered():
    async def run():
        queue = AdmissionQueue()
        request = queue.submit("s0", "t0", "", now=0.0, timeout_s=10.0)
        assert queue.expire(now=5.0) == []
        assert queue.expire(now=10.0) == [request]
        assert request.decision is Decision.TIMED_OUT
        assert queue.pending() == []

    asyncio.run(run())


def test_deciding_an_unknown_request_is_reported_not_raised():
    """The phone may have hung up between the page drawing the button and
    someone pressing it. That is ordinary, not an error."""
    assert AdmissionQueue().decide("join-404", True) is False


def test_a_view_carries_only_what_an_approval_prompt_needs():
    async def run():
        queue = AdmissionQueue()
        queue.submit("rack-3", "t0", "Alex", now=2.0, timeout_s=30.0)
        [view] = queue.pending_views()
        assert view.station_id == "rack-3"
        assert view.trainee_id == "t0"
        assert view.display_name == "Alex"
        assert view.requested_ts == 2.0
        assert view.expires_ts == 32.0

    asyncio.run(run())


def test_waiting_times_out_rather_than_raising():
    async def run():
        queue = AdmissionQueue()
        request = queue.submit("s0", "t0", "", now=0.0, timeout_s=10.0)
        assert await request.wait(timeout_s=0.05) is Decision.TIMED_OUT

    asyncio.run(run())


# -- the handshake ------------------------------------------------------------


def test_auto_approval_is_the_unchanged_handshake(default_config):
    """The default path must be exactly what it was before admission existed."""

    async def run():
        server = IngestServer(_cfg(default_config, approval="auto"))
        await server.start()
        try:
            ws = await _hello(server.ws_port)
            assert json.loads(await ws.recv()) == {"type": "hello_ack", "accepted": True}
            await ws.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_manual_approval_parks_the_phone_and_says_so(default_config):
    """Silence would get the station restarted by whoever is next to it."""

    async def run():
        server = IngestServer(_cfg(default_config, approval="manual", name="Coach Riley"))
        await server.start()
        try:
            ws = await _hello(server.ws_port, display_name="Alex")
            pending = json.loads(await ws.recv())
            assert pending["type"] == "join_pending"
            assert pending["session_name"] == "Coach Riley"
            assert pending["timeout_s"] == server.cfg.session.join_timeout_s

            [view] = server._admission.pending_views()
            assert view.display_name == "Alex"

            assert server._admission.decide(view.request_id, True) is True
            assert json.loads(await ws.recv()) == {"type": "hello_ack", "accepted": True}
            assert server.active_trainee_ids == ["t0"]
            await ws.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_declined_phone_is_told_why_and_closed(default_config):
    async def run():
        server = IngestServer(_cfg(default_config, approval="manual"))
        await server.start()
        try:
            ws = await _hello(server.ws_port)
            assert json.loads(await ws.recv())["type"] == "join_pending"

            [view] = server._admission.pending_views()
            server._admission.decide(view.request_id, False)

            error = json.loads(await ws.recv())
            assert error["type"] == "error"
            assert "declined" in error["message"]
            with pytest.raises(websockets.ConnectionClosed):
                await ws.recv()
            assert ws.close_code == 1008
            # Declined means not on the floor at all, not on it unscored.
            assert server.active_trainee_ids == []
        finally:
            await server.stop()

    asyncio.run(run())


def test_an_unanswered_request_times_out_with_a_reason(default_config):
    """A prompt nobody is going to answer should end, and say which of "no"
    and "nobody looked" happened -- they are different problems for whoever is
    standing next to the phone."""

    async def run():
        server = IngestServer(_cfg(default_config, approval="manual", join_timeout_s=0.2))
        await server.start()
        try:
            ws = await _hello(server.ws_port)
            assert json.loads(await ws.recv())["type"] == "join_pending"

            error = json.loads(await ws.recv())
            assert error["type"] == "error"
            assert "no instructor answered" in error["message"]
            assert server.active_trainee_ids == []
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_phone_that_hangs_up_stops_being_offered_for_approval(default_config):
    """Otherwise the instructor approves a trainee who already walked away."""

    async def run():
        server = IngestServer(_cfg(default_config, approval="manual", join_timeout_s=30.0))
        await server.start()
        try:
            ws = await _hello(server.ws_port)
            assert json.loads(await ws.recv())["type"] == "join_pending"
            assert len(server._admission.pending()) == 1

            await ws.close()
            for _ in range(100):
                await asyncio.sleep(0.02)
                if not server._admission.pending():
                    break
            assert server._admission.pending() == []
            assert server.active_trainee_ids == []
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_duplicate_id_is_refused_without_bothering_the_instructor(default_config):
    """Approving a phone that is going to be refused anyway spends the one
    thing a manual gate costs, which is the instructor's attention."""

    async def run():
        server = IngestServer(_cfg(default_config, approval="manual"))
        await server.start()
        try:
            first = await _hello(server.ws_port, "t0")
            assert json.loads(await first.recv())["type"] == "join_pending"
            [view] = server._admission.pending_views()
            server._admission.decide(view.request_id, True)
            assert json.loads(await first.recv())["type"] == "hello_ack"

            second = await _hello(server.ws_port, "t0")
            error = json.loads(await second.recv())
            assert error["type"] == "error"
            assert "already connected" in error["message"]
            assert server._admission.pending() == []

            await first.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_phone_naming_another_session_is_refused(default_config):
    """On a floor with two laptops, silently joining the wrong one is a
    trainee monitored by an instructor who is not watching them."""

    async def run():
        server = IngestServer(_cfg(default_config, name="Coach Riley"))
        await server.start()
        try:
            ws = await _hello(server.ws_port, session_name="Coach Sam")
            error = json.loads(await ws.recv())
            assert error["type"] == "error"
            assert "Coach Riley" in error["message"]
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_phone_naming_this_session_is_admitted(default_config):
    async def run():
        server = IngestServer(_cfg(default_config, name="Coach Riley"))
        await server.start()
        try:
            ws = await _hello(server.ws_port, session_name="Coach Riley")
            assert json.loads(await ws.recv())["type"] == "hello_ack"
            await ws.close()
        finally:
            await server.stop()

    asyncio.run(run())


def test_a_phone_that_names_no_session_is_still_admitted(default_config):
    """`session_name` is optional: a phone given the address by hand never
    heard a beacon and cannot know the name."""

    async def run():
        server = IngestServer(_cfg(default_config, name="Coach Riley"))
        await server.start()
        try:
            ws = await _hello(server.ws_port)
            assert json.loads(await ws.recv())["type"] == "hello_ack"
            await ws.close()
        finally:
            await server.stop()

    asyncio.run(run())
