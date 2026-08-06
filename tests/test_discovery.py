"""LAN discovery: the beacon that saves someone typing an IP into every phone.

The round-trip test sends to loopback rather than to the broadcast address on
purpose. Whether a CI runner's network forwards broadcast is not a property of
this code, and a test that depended on it would fail for reasons that have
nothing to do with the beacon being correct.
"""

from __future__ import annotations

import dataclasses
import json
import socket

import pytest

from argus.discovery import (
    BEACON_TYPE,
    DiscoveryBeacon,
    beacon_payload,
    listen,
    local_lan_ip,
    parse_beacon,
)


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# -- what gets advertised -----------------------------------------------------


def test_a_wildcard_bind_advertises_a_routable_address():
    payload = beacon_payload("0.0.0.0", 8765, protocol_version=1)
    if payload is None:
        pytest.skip("host has no non-loopback IPv4 address to advertise")
    assert payload["type"] == BEACON_TYPE
    assert payload["ws_url"].startswith("ws://")
    assert not payload["ws_url"].startswith("ws://127.")
    assert payload["ws_url"].endswith(":8765")
    assert payload["protocol_version"] == 1


def test_an_explicit_host_is_advertised_verbatim():
    payload = beacon_payload("192.168.1.20", 8765, protocol_version=1)
    assert payload["ws_url"] == "ws://192.168.1.20:8765"


def test_a_loopback_bind_advertises_nothing():
    """A beacon naming an address no phone can reach turns "I cannot find the
    server" into "I found it and it does not work", which is worse."""
    assert beacon_payload("127.0.0.1", 8765, protocol_version=1) is None


def test_an_unnamed_session_advertises_no_name():
    """An unnamed session shows as its address, which is honest and works."""
    assert "session_name" not in beacon_payload("192.168.1.20", 8765, 1)


def test_a_named_session_is_advertised_so_a_phone_can_pick_the_right_laptop():
    payload = beacon_payload("192.168.1.20", 8765, 1, session_name="Coach Riley")
    assert payload["session_name"] == "Coach Riley"


def test_the_approval_mode_is_advertised():
    """So a phone can say "the instructor will approve this" rather than
    looking hung while a request sits on someone's screen."""
    assert beacon_payload("192.168.1.20", 8765, 1)["approval"] == "auto"
    assert beacon_payload("192.168.1.20", 8765, 1, approval="manual")["approval"] == "manual"


def test_the_use_case_is_always_advertised():
    """Unlike `session_name`, never blank -- a phone should see "this laptop
    is running welding" before connecting, not after a hello rejection."""
    assert beacon_payload("192.168.1.20", 8765, 1)["use_case"] == "fitness"
    assert beacon_payload("192.168.1.20", 8765, 1, use_case="welding")["use_case"] == "welding"


def test_local_lan_ip_is_never_loopback():
    addr = local_lan_ip()
    assert addr is None or not addr.startswith("127.")


# -- parsing unsolicited traffic ----------------------------------------------


def _datagram(**overrides) -> bytes:
    payload = {"type": BEACON_TYPE, "protocol_version": 1, "ws_url": "ws://10.0.0.5:8765"}
    payload.update(overrides)
    return json.dumps(payload).encode("utf-8")


def test_a_valid_beacon_parses():
    assert parse_beacon(_datagram())["ws_url"] == "ws://10.0.0.5:8765"


@pytest.mark.parametrize(
    "raw",
    [
        b"not json at all",
        b"[1, 2, 3]",
        b"\xff\xfe\x00",
        json.dumps({"type": "something_else", "ws_url": "ws://10.0.0.5:8765"}).encode(),
    ],
    ids=["garbage", "not-an-object", "not-utf8", "wrong-type"],
)
def test_unrelated_traffic_is_dropped_rather_than_raising(raw):
    """This port is on a shared network; anything at all can arrive on it.
    Malformed input is the normal case here, not an error worth surfacing --
    unlike argus.ingest.protocol, which parses a phone that already shook
    hands and so treats a bad message as a real disagreement."""
    assert parse_beacon(raw) is None


def test_a_beacon_without_a_ws_url_is_dropped():
    assert parse_beacon(_datagram(ws_url="http://10.0.0.5:8765")) is None
    assert parse_beacon(_datagram(ws_url=42)) is None


def test_a_mismatched_protocol_version_is_dropped():
    """Better to look like no server than to hand a phone an address whose
    handshake it is going to be rejected at."""
    assert parse_beacon(_datagram(protocol_version=2), expected_protocol_version=1) is None
    assert parse_beacon(_datagram(protocol_version=1), expected_protocol_version=1) is not None


def test_a_non_integer_protocol_version_is_dropped():
    assert parse_beacon(_datagram(protocol_version="1")) is None


# -- the round trip -----------------------------------------------------------


def test_a_listener_hears_a_beacon():
    port = _free_udp_port()
    beacon = DiscoveryBeacon(
        beacon_payload("192.168.1.20", 8765, 1), port=port, interval_s=0.05, broadcast="127.0.0.1"
    )
    beacon.start()
    try:
        found = listen(port, timeout_s=2.0, expected_protocol_version=1)
    finally:
        beacon.stop()

    assert [b["ws_url"] for b in found] == ["ws://192.168.1.20:8765"]


def test_repeated_beacons_collapse_to_one_server():
    """A phone showing the same laptop five times because it broadcast five
    times would be a worse setup experience than typing the address."""
    port = _free_udp_port()
    beacon = DiscoveryBeacon(
        beacon_payload("192.168.1.20", 8765, 1), port=port, interval_s=0.05, broadcast="127.0.0.1"
    )
    beacon.start()
    try:
        found = listen(port, timeout_s=1.0)
    finally:
        beacon.stop()

    assert beacon.sent > 1, "the beacon did not repeat, so this proves nothing"
    assert len(found) == 1


def test_listening_with_nothing_broadcasting_finds_nothing():
    assert listen(_free_udp_port(), timeout_s=0.4) == []


def test_stop_is_idempotent():
    beacon = DiscoveryBeacon({"type": BEACON_TYPE}, port=_free_udp_port())
    beacon.stop()
    beacon.stop()


def test_update_payload_changes_what_the_next_send_broadcasts():
    """An instructor changing `[session] use_case` from the console mid-run
    (see `IngestServer.set_use_case`) must not leave the beacon advertising
    the old one until the process restarts."""
    port = _free_udp_port()
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", port))
    listener.settimeout(2.0)

    beacon = DiscoveryBeacon(
        {"type": BEACON_TYPE, "use_case": "fitness"}, port=port, broadcast="127.0.0.1"
    )
    beacon.update_payload({"type": BEACON_TYPE, "use_case": "welding"})

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        beacon.send_once(sender)
        received, _ = listener.recvfrom(2048)
        assert json.loads(received)["use_case"] == "welding"
    finally:
        sender.close()
        listener.close()


# -- the server's wiring ------------------------------------------------------


def test_the_server_advertises_the_port_it_actually_bound(default_config, monkeypatch):
    """`ingest.ws_port = 0` means the OS picks one. A beacon built from the
    configured value would advertise port 0 and send every phone nowhere."""
    import asyncio

    from argus.ingest.server import IngestServer

    monkeypatch.setattr("argus.discovery.local_lan_ip", lambda: "192.168.1.20")
    cfg = dataclasses.replace(
        default_config,
        ingest=dataclasses.replace(default_config.ingest, ws_host="0.0.0.0", ws_port=0),
        outputs=dataclasses.replace(default_config.outputs, console=False, http_port=0),
        discovery=dataclasses.replace(default_config.discovery, enabled=True),
    )

    async def run():
        server = IngestServer(cfg)
        await server.start()
        try:
            assert server.ws_port != 0
            assert server._beacon is not None
            assert json.loads(server._beacon._datagram.decode())["ws_url"] == (
                f"ws://192.168.1.20:{server.ws_port}"
            )
        finally:
            await server.stop()

    asyncio.run(run())


def test_discovery_can_be_switched_off(default_config):
    import asyncio

    from argus.ingest.server import IngestServer

    cfg = dataclasses.replace(
        default_config,
        ingest=dataclasses.replace(default_config.ingest, ws_host="127.0.0.1", ws_port=0),
        outputs=dataclasses.replace(default_config.outputs, console=False, http_port=0),
        discovery=dataclasses.replace(default_config.discovery, enabled=False),
    )

    async def run():
        server = IngestServer(cfg)
        await server.start()
        try:
            assert server._beacon is None
        finally:
            await server.stop()

    asyncio.run(run())
