"""LAN discovery — how a phone finds the laptop without anyone typing an IP.

Setting a station up used to mean reading the laptop's address off `argus
doctor` and typing `ws://192.168.1.20:8765` into a phone, once per phone,
again whenever the DHCP lease moved. This module removes that step: the
server broadcasts a small UDP beacon saying where its WebSocket listener is,
and a phone listens for one instead of asking a human.

The design is deliberately the least mechanism that solves it:

* **UDP broadcast, not mDNS.** mDNS would mean a dependency on the laptop
  (Argus has exactly one runtime dependency and this is not worth being the
  second) and a responder to keep correct. A beacon is a `sendto` in a loop.
* **Advertise, never accept.** This socket only ever sends. There is no
  listener here for a phone to talk to, so discovery adds no inbound surface
  to the laptop beyond the WebSocket port that was already open.
* **A hint, not an authority.** A phone that hears nothing types an address
  like it always could. Discovery failing degrades to the old workflow rather
  than to a floor that cannot be monitored — which is why a blocked broadcast
  on a locked-down guest network is an inconvenience and not an outage.

The beacon carries no trainee data, and structurally cannot: the payload is
built once at startup from config, before any phone has connected, and this
module never sees a session, a record, or an observation.
"""

from __future__ import annotations

import json
import logging
import socket
import threading

logger = logging.getLogger(__name__)

#: The limited-broadcast address. Not forwarded by routers, which is exactly
#: right: a phone that can reach this laptop's Wi-Fi is on the same segment,
#: and a beacon has no business leaving it. An operator whose network drops
#: limited broadcast can set `discovery.broadcast` to a subnet-directed
#: address (e.g. "192.168.1.255") instead.
LIMITED_BROADCAST = "255.255.255.255"

#: Marks a datagram as ours. A phone must check this before parsing anything
#: else -- any process on the LAN can send to this port, so an unrecognised
#: datagram is noise to drop, not a malformed beacon to report.
BEACON_TYPE = "argus_beacon"


def local_lan_ip() -> str | None:
    """This host's address on the interface that reaches the wider network.

    Opens a UDP socket "to" a public address and reads back the local end.
    UDP is connectionless, so this sends no packet and needs no route to
    actually work — it just asks the kernel which interface it *would* use,
    which is the one a phone on the same Wi-Fi will reach us on. That is more
    reliable than resolving the hostname, which on a laptop with a stale
    `/etc/hosts` or a VPN can answer with an address no phone can route to.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: reserved, never routed
        addr = probe.getsockname()[0]
        return None if addr.startswith("127.") else addr
    except OSError:
        return None
    finally:
        probe.close()


def beacon_payload(
    ws_host: str,
    ws_port: int,
    protocol_version: int,
    session_name: str = "",
    approval: str = "auto",
) -> dict | None:
    """The datagram a phone parses, or `None` if there is nothing to advertise.

    `None` when the WebSocket is bound to loopback, or when no LAN address can
    be found: in either case no phone could connect to what we would be
    advertising, and a beacon pointing at an unreachable address is worse than
    no beacon — it turns "I could not find the server" into "I found it and it
    does not work".
    """
    if ws_host in ("0.0.0.0", "::"):
        advertised = local_lan_ip()
    elif ws_host.startswith("127.") or ws_host == "localhost":
        advertised = None
    else:
        advertised = ws_host
    if not advertised:
        return None

    payload = {
        "type": BEACON_TYPE,
        # The phone checks this before offering to connect, so a version
        # mismatch shows up as "that server speaks protocol 2" during setup
        # rather than as a connection that is refused at the handshake.
        "protocol_version": protocol_version,
        "ws_url": f"ws://{advertised}:{ws_port}",
        # So a phone can say "the instructor will approve this" instead of
        # looking like it hung while a join request sits on someone's screen.
        "approval": approval,
    }
    if session_name:
        # Optional and empty by default. A named session is what lets someone
        # pick the right laptop in a room with three of them; an unnamed one
        # shows as its address, which is honest and still works.
        payload["session_name"] = session_name
    return payload


def parse_beacon(raw: bytes, expected_protocol_version: int | None = None) -> dict | None:
    """Validate a datagram. `None` for anything that is not a usable beacon.

    Returns `None` rather than raising for every rejection: this parses
    unsolicited traffic from a shared network, where malformed input is the
    normal case and not an error worth surfacing. Contrast
    `argus.ingest.protocol`, which raises loudly — that parses messages from a
    phone that has already completed a handshake, where a bad message means a
    real disagreement worth stopping for.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("type") != BEACON_TYPE:
        return None
    ws_url = payload.get("ws_url")
    if not isinstance(ws_url, str) or not ws_url.startswith("ws://"):
        return None
    version = payload.get("protocol_version")
    if not isinstance(version, int):
        return None
    if expected_protocol_version is not None and version != expected_protocol_version:
        return None
    return payload


class DiscoveryBeacon:
    """Broadcasts one payload on an interval until stopped.

    Threaded rather than an asyncio task so a send that blocks on a wedged
    interface cannot delay a rank tick. Discovery is a convenience; the
    scoring loop is not, and the two should not share a thread.
    """

    def __init__(
        self,
        payload: dict,
        port: int,
        interval_s: float = 1.0,
        broadcast: str = LIMITED_BROADCAST,
    ):
        self._datagram = json.dumps(payload).encode("utf-8")
        self._port = port
        self._interval = interval_s
        self._broadcast = broadcast or LIMITED_BROADCAST
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sent = 0

    @property
    def sent(self) -> int:
        return self._sent

    def send_once(self, sock: socket.socket) -> None:
        sock.sendto(self._datagram, (self._broadcast, self._port))
        self._sent += 1

    def _loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                try:
                    self.send_once(sock)
                except OSError as exc:
                    # A network that drops broadcast, or an interface that
                    # went away with the laptop's lid. Log once per occurrence
                    # and keep trying: the operator's fallback is typing the
                    # address, which still works, so this must not take the
                    # ingest server down with it.
                    logger.debug("discovery beacon send failed: %s", exc)
                self._stop.wait(self._interval)
        finally:
            sock.close()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


def listen(port: int, timeout_s: float, expected_protocol_version: int | None = None) -> list[dict]:
    """Collect distinct beacons heard within `timeout_s`. The phone's job, here.

    This is `argus discover`, and it is also the reference implementation of
    the listening half of the protocol — the Android client does exactly this
    (see `android/.../Discovery.kt`). Keeping a working version on the laptop
    means the beacon format can be verified without a phone in the room.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    found: dict[str, dict] = {}
    try:
        sock.bind(("", port))
        sock.settimeout(0.5)
        deadline = threading.Event()
        timer = threading.Timer(timeout_s, deadline.set)
        timer.daemon = True
        timer.start()
        try:
            while not deadline.is_set():
                try:
                    raw, _ = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                beacon = parse_beacon(raw, expected_protocol_version)
                if beacon is not None:
                    found.setdefault(beacon["ws_url"], beacon)
        finally:
            timer.cancel()
    finally:
        sock.close()
    return list(found.values())
