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


#: The trainer dashboard: a static page that polls `/triage` client-side and
#: renders a table. It reads nothing this module does not already serve, so
#: it widens no boundary -- the four `TriageRecord` fields are still the only
#: thing that ever left the perception layer.
_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Argus — trainer view</title>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 2rem; }
  h1 { font-size: 1.1rem; font-weight: 600; color: #999; }
  table { border-collapse: collapse; width: 100%; max-width: 720px; }
  th, td { text-align: left; padding: 0.4rem 0.8rem; border-bottom: 1px solid #333; }
  th { color: #999; font-weight: 500; font-size: 0.85rem; }
  tr.flagged { background: #3a1414; }
  td.score { font-variant-numeric: tabular-nums; }
  #empty { color: #666; padding: 1rem 0; }
  #ts { color: #666; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>Argus — who needs attention <span id="ts"></span></h1>
<table id="ranked"><thead>
  <tr><th>trainee</th><th>score</th><th>reasons</th></tr>
</thead><tbody></tbody></table>
<div id="empty" style="display:none">No trainee is currently connected.</div>
<script>
async function refresh() {
  const res = await fetch("/triage");
  const payload = await res.json();
  const body = document.querySelector("#ranked tbody");
  body.innerHTML = "";
  document.querySelector("#empty").style.display = payload.records.length ? "none" : "block";
  document.querySelector("#ts").textContent = "(t=" + payload.ts.toFixed(1) + ")";
  for (const r of payload.records) {
    const row = document.createElement("tr");
    if (r.reason_codes.length) row.className = "flagged";
    row.innerHTML =
      "<td>" + r.trainee_id + "</td>" +
      "<td class=score>" + r.score.toFixed(2) + "</td>" +
      "<td>" + (r.reason_codes.join(", ") || "–") + "</td>";
    body.appendChild(row);
  }
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


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
        if self.path == "/":
            self._respond(200, _DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/healthz":
            self._respond(200, b'{"status":"ok"}', "application/json")
        elif self.path == "/triage":
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
            self._respond(200, payload, "application/json")
        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # silence the default stderr access log
        pass


class TriageHTTPServer:
    """Background HTTP server exposing the latest triage ranking as JSON.

    `GET /triage` -> {"ts": float, "records": [{trainee_id, score,
    reason_codes, ts}, ...]}. `GET /healthz` -> liveness. `GET /` -> a static
    dashboard page that polls `/triage` client-side and renders it as a
    table — the trainer's live view of who needs attention, and nothing more
    than a rendering of the same four redacted fields. Nothing else is
    served: this is not a general-purpose API.

    Binds to 127.0.0.1 by default. The records are already redacted, but the
    endpoint still describes who on a floor needs attention, so it is
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
