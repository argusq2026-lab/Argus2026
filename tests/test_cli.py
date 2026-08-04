"""The CLI end to end: `run`, `doctor`, `config`, `demo`, and the replay client
that stands in for a real phone -- the direct successor to the old camera-
pipeline CLI tests in test_pipeline.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    assert len(fixture["stations"]) == 3
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
