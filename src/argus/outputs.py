"""Output sinks — JSON-lines log and the HTTP `/triage` endpoint.

Both are carried over from the Argus prototype, where they were covered by
unit tests, with the privacy property tightened rather than loosened.

Every sink here takes `list[TriageRecord]` and nothing else. There is no
parameter on any public function in this module through which a frame, a crop,
or a caption could travel — not because a redaction filter strips them, but
because no such parameter exists. `tests/test_privacy.py` asserts that
structurally, by inspecting the annotations, so adding an output mode that
widened the boundary would fail CI rather than review.

This module deliberately imports neither `cv2` nor `numpy`; that too is
asserted by the privacy test, since an image type cannot cross a boundary a
module cannot name.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from argus.triage import TriageRecord


def to_json_dict(record: TriageRecord) -> dict:
    """Serialise a record. `reason_codes` is a tuple; JSON wants a list."""
    payload = asdict(record)
    payload["reason_codes"] = list(record.reason_codes)
    return payload


class JsonLogSink:
    """Appends one JSON line per frame: {"ts": ..., "records": [...]}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, ts: float, records: list[TriageRecord]) -> None:
        line = json.dumps({"ts": ts, "records": [to_json_dict(r) for r in records]})
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class _TriageRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        if self.path not in ("/triage", "/healthz"):
            self.send_response(404)
            self.end_headers()
            return

        if self.path == "/healthz":
            payload = b'{"status":"ok"}'
        else:
            with self.server.state_lock:  # type: ignore[attr-defined]
                payload = json.dumps(
                    {
                        "ts": self.server.latest_ts,  # type: ignore[attr-defined]
                        "records": [
                            to_json_dict(r)
                            for r in self.server.latest_records  # type: ignore[attr-defined]
                        ],
                    }
                ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # silence the default stderr access log
        pass


class TriageHTTPServer:
    """Background HTTP server exposing the latest triage ranking as JSON.

    `GET /triage` -> {"ts": float, "records": [{trainee_id, score,
    reason_codes, ts}, ...]}. `GET /healthz` -> liveness. Nothing else is
    served: an instructor console polls this, it is not a general-purpose API.

    Binds to 127.0.0.1 by default. The records are already redacted, but the
    endpoint still describes who in a building needs attention, so it is
    loopback-only unless an operator opts out in config.
    """

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self._server = ThreadingHTTPServer((host, port), _TriageRequestHandler)
        self._server.state_lock = threading.Lock()
        self._server.latest_ts = 0.0
        self._server.latest_records = []
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def update(self, ts: float, records: list[TriageRecord]) -> None:
        with self._server.state_lock:
            self._server.latest_ts = ts
            self._server.latest_records = list(records)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
