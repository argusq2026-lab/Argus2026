"""The CLI end to end: `run`, `doctor`, `config`, `demo`, and the replay client
that stands in for a real phone -- the direct successor to the old camera-
pipeline CLI tests in test_pipeline.py.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from argus.synthetic import TRAINEE_IDS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _wait_until_listening(
    port: int, process: subprocess.Popen, timeout: float = 20.0
) -> None:
    """Block until the server accepts a connection on `port`.

    The server binds asynchronously after `Popen` returns, so connecting
    immediately is a race: it passes on a fast machine and fails on a loaded CI
    runner with ECONNREFUSED, which reads like a broken server rather than a
    test that started too early. Poll the port instead of sleeping a guessed
    interval -- a fixed sleep is the same race with a longer fuse.

    Fails fast if the process dies first, surfacing its output rather than
    waiting out the timeout on a server that already exited (a port collision,
    for instance).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            out = process.stdout.read() if process.stdout else ""
            raise AssertionError(
                f"server exited with {process.returncode} before listening on {port}:\n{out}"
            )
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"server did not listen on {port} within {timeout}s")


def _run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    env_src = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "argus.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": env_src},
    )


def test_cli_help_exits_zero():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "run" in result.stdout and "doctor" in result.stdout and "demo" in result.stdout


def test_cli_config_prints_the_effective_tuning():
    result = _run_cli("config")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["scoring"]["weights"]["fall"] == 0.4
    assert payload["ingest"]["ws_port"] == 8765


def test_cli_config_reflects_flag_overrides():
    result = _run_cli("config", "--ws-port", "9000")
    assert json.loads(result.stdout)["ingest"]["ws_port"] == 9000


def test_cli_doctor_runs_and_reports():
    result = _run_cli("doctor", "--ws-port", "18888")
    assert "checks:" in result.stdout
    assert "[PASS] python" in result.stdout


def test_cli_demo_writes_a_multi_station_fixture(tmp_path):
    out = tmp_path / "fixture.json"
    result = _run_cli("demo", "--out", str(out), "--ticks", "20")
    assert result.returncode == 0, result.stderr
    fixture = json.loads(out.read_text(encoding="utf-8"))
    assert len(fixture["stations"]) == len(TRAINEE_IDS)
    assert all(len(s["messages"]) == 20 for s in fixture["stations"])


@pytest.mark.timeout(60)
def test_cli_run_end_to_end_with_the_replay_client(tmp_path):
    fixture = tmp_path / "fixture.json"
    log = tmp_path / "triage.jsonl"
    assert _run_cli("demo", "--out", str(fixture), "--ticks", "25").returncode == 0

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    run_proc = subprocess.Popen(
        [
            sys.executable, "-m", "argus.cli", "run",
            "--ws-port", "18765",
            "--json-log", str(log),
            "--max-ticks", "5",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # The phone cannot connect before the laptop is listening; neither can
        # the client that stands in for it.
        _wait_until_listening(18765, run_proc)

        replay = subprocess.run(
            [sys.executable, "demo/replay_client.py", "--fixture", str(fixture), "--ws-port", "18765"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert replay.returncode == 0, replay.stderr

        out, _ = run_proc.communicate(timeout=30)
        assert run_proc.returncode == 0, out
        assert "processed 5 rank ticks" in out
    finally:
        if run_proc.poll() is None:
            run_proc.kill()

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    ranked = [json.loads(line) for line in lines]
    assert any(line["records"] for line in ranked)
    for line in ranked:
        for record in line["records"]:
            assert set(record) == {"trainee_id", "score", "reason_codes", "ts"}


# -- the subcommands the packaged binary needs -------------------------------
#
# `argus replay` and `argus discover` exist so a laptop with the binary and
# nothing else can drive and diagnose itself: no checkout beside it, no
# `demo/replay_client.py` on disk, no second tool to install. Tested through
# the CLI for that reason -- calling the library directly would not catch the
# subcommand being wired up wrong, which is the whole risk.


def test_default_argv_turns_a_bare_launch_into_run():
    """Someone who double-clicks the binary passes no arguments and should
    get a server and a console, not argparse's usage message."""
    from argus.cli import default_argv

    assert default_argv([]) == ["run", "--open"]


def test_default_argv_leaves_an_explicit_subcommand_alone():
    from argus.cli import default_argv

    for argv in (["doctor"], ["run", "--max-ticks", "1"], ["--version"]):
        assert default_argv(argv) == argv


@pytest.mark.timeout(90)
def test_replay_and_discover_subcommands_against_a_live_server(tmp_path):
    fixture = tmp_path / "fixture.json"
    assert _run_cli("demo", "--out", str(fixture), "--ticks", "20").returncode == 0

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    run_proc = subprocess.Popen(
        [
            sys.executable, "-m", "argus.cli", "run",
            "--ws-port", "18766",
            "--max-ticks", "5",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_until_listening(18766, run_proc)

        replay = _run_cli(
            "replay", "--fixture", str(fixture), "--ws-port", "18766", timeout=30
        )
        assert replay.returncode == 0, replay.stderr
        assert "sent 20 observations" in replay.stdout

        # Discovery is a convenience, and a CI runner may have no non-loopback
        # address to advertise or may drop broadcast entirely. Assert only that
        # the subcommand runs and reports itself honestly either way -- a test
        # that required a beacon to arrive would be testing the runner's
        # network, not this code. tests/test_discovery.py covers the round
        # trip deterministically over loopback.
        discover = _run_cli("discover", "--timeout", "2", timeout=30)
        assert discover.returncode in (0, 1)
        if discover.returncode == 0:
            assert "ws://" in discover.stdout
        else:
            assert "no server found" in discover.stderr

        out, _ = run_proc.communicate(timeout=30)
        assert run_proc.returncode == 0, out
    finally:
        if run_proc.poll() is None:
            run_proc.kill()
