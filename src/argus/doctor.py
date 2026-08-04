"""`argus doctor` — say exactly what is missing and exactly how to fix it.

Every check reports PASS / WARN / FAIL with a one-line remedy. There is no
model runtime or camera to diagnose anymore — the checks that matter now are
whether the config is sane, whether the ingest port can actually be bound,
and how a phone on the same Wi-Fi would reach this machine.
"""

from __future__ import annotations

import platform
import socket
import sys
from dataclasses import dataclass

from argus.config import ArgusConfig

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def _check_host() -> list[Check]:
    return [
        Check(
            "python",
            PASS if sys.version_info >= (3, 11) else FAIL,
            platform.python_version(),
            "" if sys.version_info >= (3, 11) else "Argus requires Python 3.11+.",
        ),
        Check("platform", PASS, f"{platform.system()} {platform.release()} ({platform.machine()})"),
    ]


def _lan_addresses() -> list[str]:
    """Best-effort list of this host's non-loopback IPv4 addresses.

    A phone on the gym's Wi-Fi needs one of these, not 127.0.0.1 -- this is
    purely informational, so a failure to resolve any is a WARN, not a FAIL.
    """
    addresses: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                addresses.add(addr)
    except OSError:
        pass
    return sorted(addresses)


def _check_ingest(cfg: ArgusConfig) -> list[Check]:
    checks: list[Check] = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((cfg.ingest.ws_host, cfg.ingest.ws_port))
        checks.append(
            Check(
                "ws_port bindable",
                PASS,
                f"{cfg.ingest.ws_host}:{cfg.ingest.ws_port}",
            )
        )
    except OSError as exc:
        checks.append(
            Check(
                "ws_port bindable",
                FAIL,
                f"{cfg.ingest.ws_host}:{cfg.ingest.ws_port} -- {exc}",
                "Another process is already listening there, or the host is "
                "invalid. Pick a different ingest.ws_port or stop the other "
                "process.",
            )
        )
    finally:
        sock.close()

    if cfg.ingest.ws_host in ("0.0.0.0", "::"):
        addresses = _lan_addresses()
        detail = (
            f"a phone should connect to one of {addresses}"
            if addresses
            else "could not resolve a LAN-reachable address"
        )
        checks.append(
            Check(
                "phone-reachable address",
                PASS if addresses else WARN,
                detail,
                ""
                if addresses
                else "Connect this machine to the gym's Wi-Fi, or set ingest.ws_host "
                "explicitly if auto-detection is wrong.",
            )
        )
    else:
        checks.append(
            Check(
                "phone-reachable address",
                WARN,
                f"ingest.ws_host = {cfg.ingest.ws_host!r} -- only reachable from that address",
                "Set ingest.ws_host = \"0.0.0.0\" so any phone on the LAN can connect, "
                "unless binding to one interface is deliberate.",
            )
        )

    checks.append(
        Check(
            "protocol_version",
            PASS,
            f"{cfg.ingest.protocol_version} -- must match every phone app's build",
        )
    )
    return checks


def _check_outputs(cfg: ArgusConfig) -> list[Check]:
    checks = []
    if cfg.outputs.http_port:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((cfg.outputs.http_host, cfg.outputs.http_port))
            checks.append(
                Check("http_port bindable", PASS, f"{cfg.outputs.http_host}:{cfg.outputs.http_port}")
            )
        except OSError as exc:
            checks.append(
                Check(
                    "http_port bindable",
                    FAIL,
                    f"{cfg.outputs.http_host}:{cfg.outputs.http_port} -- {exc}",
                    "Another process is already listening there. Pick a different "
                    "outputs.http_port.",
                )
            )
        finally:
            sock.close()
    else:
        checks.append(
            Check(
                "http endpoint",
                WARN,
                "outputs.http_port = 0 (disabled)",
                "The /triage endpoint and trainer dashboard are off. Set "
                "outputs.http_port to enable them.",
            )
        )
    return checks


def run_doctor(cfg: ArgusConfig) -> int:
    """Print every check. Returns 1 if anything FAILed, else 0."""
    checks: list[Check] = []
    checks += _check_host()
    checks += _check_ingest(cfg)
    checks += _check_outputs(cfg)

    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"[{check.status}] {check.name.ljust(width)}  {check.detail}")
        if check.remedy:
            print(f"        -> {check.remedy}")

    failures = sum(1 for c in checks if c.status == FAIL)
    warnings = sum(1 for c in checks if c.status == WARN)
    print(f"\n{len(checks)} checks: {failures} failed, {warnings} warned")
    return 1 if failures else 0
