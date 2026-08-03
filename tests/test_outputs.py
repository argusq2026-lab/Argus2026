"""Output sinks and the console alert boundary."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from argus.alerts import ConsoleAlertSink, format_alert
from argus.outputs import JsonLogSink, TriageHTTPServer
from argus.triage import TriageRecord


def rec(trainee_id="cam0-t0", score=0.75, reasons=("possible_fall",), ts=1.0):
    return TriageRecord(trainee_id, score, tuple(reasons), ts)


# -- JSON lines -------------------------------------------------------------


def test_json_sink_writes_one_line_per_frame(tmp_path):
    sink = JsonLogSink(tmp_path / "out" / "triage.jsonl")
    sink.write(1.0, [rec()])
    sink.write(2.0, [])
    sink.close()

    lines = (tmp_path / "out" / "triage.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["records"][0]["trainee_id"] == "cam0-t0"
    assert json.loads(lines[1])["records"] == []


def test_json_sink_creates_parent_directories(tmp_path):
    sink = JsonLogSink(tmp_path / "deeply" / "nested" / "triage.jsonl")
    sink.write(1.0, [rec()])
    sink.close()
    assert (tmp_path / "deeply" / "nested" / "triage.jsonl").is_file()


def test_json_sink_appends_across_sessions(tmp_path):
    path = tmp_path / "triage.jsonl"
    for _ in range(2):
        sink = JsonLogSink(path)
        sink.write(1.0, [rec()])
        sink.close()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_reason_codes_serialise_as_a_list(tmp_path):
    """They are a tuple in memory (records are frozen) but JSON has no tuple."""
    sink = JsonLogSink(tmp_path / "t.jsonl")
    sink.write(1.0, [rec(reasons=("a", "b"))])
    sink.close()
    payload = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8"))
    assert payload["records"][0]["reason_codes"] == ["a", "b"]


def test_close_is_idempotent(tmp_path):
    sink = JsonLogSink(tmp_path / "t.jsonl")
    sink.close()
    sink.close()


# -- HTTP -------------------------------------------------------------------


@pytest.fixture
def server():
    srv = TriageHTTPServer(0)
    srv.start()
    yield srv
    srv.stop()


def _get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_triage_endpoint_serves_the_latest_rank(server):
    server.update(12.5, [rec(score=0.9)])
    payload = _get(server.port, "/triage")
    assert payload["ts"] == 12.5
    assert payload["records"][0]["score"] == 0.9
    assert set(payload["records"][0]) == {"trainee_id", "score", "reason_codes", "ts"}


def test_endpoint_starts_empty(server):
    assert _get(server.port, "/triage") == {"ts": 0.0, "records": []}


def test_update_replaces_rather_than_accumulates(server):
    server.update(1.0, [rec("a")])
    server.update(2.0, [rec("b")])
    payload = _get(server.port, "/triage")
    assert [r["trainee_id"] for r in payload["records"]] == ["b"]


def test_healthz(server):
    assert _get(server.port, "/healthz") == {"status": "ok"}


def test_nothing_else_is_served(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server.port, "/frames")
    assert exc.value.code == 404


def test_binds_loopback_by_default(server):
    assert server._server.server_address[0] == "127.0.0.1"


def test_update_snapshots_the_list(server):
    """A caller mutating its list afterwards must not change what is served."""
    records = [rec("a")]
    server.update(1.0, records)
    records.append(rec("b"))
    assert len(_get(server.port, "/triage")["records"]) == 1


# -- console ----------------------------------------------------------------


def test_alert_format_names_the_reasons():
    line = format_alert(rec(reasons=("possible_fall", "vlm_anomaly")))
    assert "cam0-t0" in line
    assert "possible_fall, vlm_anomaly" in line
    assert "0.75" in line


def test_console_sink_suppresses_an_unchanged_repeat(capsys):
    sink = ConsoleAlertSink()
    for _ in range(5):
        sink(rec(score=0.75))
    assert capsys.readouterr().err.count("[ALERT]") == 1


def test_console_sink_re_alerts_when_the_situation_changes(capsys):
    sink = ConsoleAlertSink()
    sink(rec(score=0.75))
    sink(rec(score=0.75, reasons=("possible_fall", "vlm_anomaly")))
    sink(rec(score=0.95, reasons=("possible_fall", "vlm_anomaly")))
    assert capsys.readouterr().err.count("[ALERT]") == 3


def test_console_sink_can_be_disabled(capsys):
    sink = ConsoleAlertSink(enabled=False)
    sink(rec())
    assert capsys.readouterr().err == ""
