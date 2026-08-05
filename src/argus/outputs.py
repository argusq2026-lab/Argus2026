"""Output sinks — JSON-lines log, the HTTP `/triage` endpoint, and the
trainer console's `/console` snapshot.

Both sinks are carried over from the Argus prototype, where they were covered
by unit tests, with the privacy property tightened rather than loosened.

**Every alert sink here takes `list[TriageRecord]` and nothing else.** There
is no parameter on any public function in this module through which a frame,
a crop, or a caption could travel — not because a redaction filter strips
them, but because no such parameter exists. `tests/test_privacy.py` asserts
that structurally, by inspecting the annotations, so adding an output mode
that widened the boundary would fail CI rather than review.

`StationView` is the one thing here that is wider than a `TriageRecord`, and
it is worth being exact about what that does and does not mean. A trainer
console that draws a skeleton needs the numeric pose; four scalar fields
cannot express one. So `StationView` carries the live observation — box,
COCO-17 keypoints and their confidences, and the phone's own
exercise/rep/form fields — to exactly one place: `GET /console` on this
module's loopback-bound HTTP server.

What that is *not*:

* not a frame path. No image ever existed past the phone's own camera
  pipeline, so there is nothing here to redact; the same AST check that
  forbids this module from importing `cv2` or `numpy` still applies, and a
  keypoint is a pair of floats whichever endpoint serves it.
* not free text. `form_reason_codes` is closed-vocabulary by the time
  `argus.ingest.protocol` accepts it. `exercise` is the single field a phone
  fills freely; it is length-bounded on the wire, never logged, and the
  console renders it as text and never as markup. It does select the scoring
  weight profile (`ScoringConfig.weights_for`) — a lookup into config, not a
  term in the score — which is why the console is also served the profiles
  themselves, so it can say which checks a given exercise switches off.
* not a widening of the *alert* boundary. `emit_alert`, `JsonLogSink`, and
  `GET /triage` still carry the same four fields they always did — a
  `StationView` cannot reach any of them, because none of them has a
  parameter that names one.

The closed-field-set discipline that guards `TriageRecord` guards this type
too: `tests/test_privacy.py` pins `StationView`'s fields, so widening the
console's view is a visible, reviewable change rather than an incidental one.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from argus.console import CONSOLE_HTML
from argus.triage import TriageRecord


#: Addresses that mean "this machine". A join decision is an access-control
#: decision, so it is accepted only from here unless an operator explicitly
#: opts out — see `OutputsConfig.allow_remote_join_control`.
_LOOPBACK_PREFIXES = ("127.", "::1", "::ffff:127.")


def _is_loopback(address: str) -> bool:
    return address == "::1" or address.startswith(_LOOPBACK_PREFIXES)


def to_json_dict(record: TriageRecord) -> dict:
    """Serialise a record. `reason_codes` is a tuple; JSON wants a list."""
    payload = asdict(record)
    payload["reason_codes"] = list(record.reason_codes)
    return payload


@dataclass(frozen=True)
class StationView:
    """One station's live numeric state, for the trainer console only.

    Frozen and closed for the same reason `TriageRecord` is: the fields below
    are the complete set of what the console can see, and adding one is a
    change to a module `tests/test_privacy.py` inspects. Read this type's
    place in the boundary in the module docstring above before extending it.

    `last_seen_ts` is on the *server's* clock — the same base as the snapshot
    timestamp it is served alongside — so the console can subtract them to get
    a real staleness. It is deliberately not the phone's `ts`: phone and
    laptop clocks are not synchronised and nothing here assumes they are.

    The pose fields are `None` for a station that has completed its handshake
    but not yet sent an observation, so the console can say "connected, no
    frames yet" rather than drawing an empty skeleton and implying a reading.
    """

    station_id: str
    trainee_id: str
    #: Whether the WebSocket is currently open. A station can be disconnected
    #: but still present: `ingest.track_ttl_s` keeps its track alive through a
    #: brief drop, and a trainee mid-grace-window is exactly the case the
    #: console must not draw as if it were live.
    connected: bool
    #: Server-clock time of the most recent observation, or of registration
    #: for a station that has not sent one yet.
    last_seen_ts: float
    #: How many observations are in the rolling history, against
    #: `scoring.history_len`. Several triage features cannot fire until the
    #: window is full, so a console that showed a half-warm station as simply
    #: "nothing wrong" would be overstating what the scorer had looked at.
    observations: int
    bbox_xyxy: tuple[float, float, float, float] | None = None
    keypoints_xy: tuple[tuple[float, float], ...] | None = None
    keypoints_conf: tuple[float, ...] | None = None
    #: Closed-vocabulary, enforced by `argus.ingest.protocol`.
    form_reason_codes: tuple[str, ...] = ()
    #: Selects the scoring weight profile, and is displayed. `""` means the
    #: phone reported none, which scores on the default weights.
    exercise: str = ""
    #: Display-only: nothing scores these two.
    rep_count: int | None = None
    form_ok: bool | None = None


def station_to_json_dict(view: StationView) -> dict:
    """Serialise a station view. Tuples become lists; JSON has no tuple."""
    payload = asdict(view)
    payload["form_reason_codes"] = list(view.form_reason_codes)
    if view.bbox_xyxy is not None:
        payload["bbox_xyxy"] = list(view.bbox_xyxy)
    if view.keypoints_xy is not None:
        payload["keypoints_xy"] = [list(p) for p in view.keypoints_xy]
    if view.keypoints_conf is not None:
        payload["keypoints_conf"] = list(view.keypoints_conf)
    return payload


@dataclass(frozen=True)
class PendingJoinView:
    """One phone waiting at the door, as the console renders it.

    Closed for the same reason the types above are. An approval prompt is the
    one place a human is asked to make a decision about a person, so what it
    is allowed to say about them should be a short, reviewed list rather than
    whatever the phone felt like sending.

    `display_name` is phone-chosen and length-bounded by
    `argus.ingest.protocol`; like every other phone-chosen string it is
    rendered as text and never as markup.
    """

    request_id: str
    station_id: str
    trainee_id: str
    display_name: str
    requested_ts: float
    expires_ts: float


def pending_to_json_dict(view: PendingJoinView) -> dict:
    return asdict(view)


@dataclass(frozen=True)
class ConsoleSettings:
    """The config values the console page needs to render honestly.

    Served in the snapshot rather than baked into the page, so the console
    obeys `configs/argus.default.toml` like everything else: it polls at the
    configured cadence, calls a station stale at the configured threshold,
    and — importantly — gates which keypoints it draws on the *same*
    `keypoint_conf_threshold` the scorer gates on, so the skeleton a trainer
    sees is the pose the rank was actually computed from.
    """

    poll_interval_ms: int = 200
    stale_after_s: float = 2.0
    #: Mirrors `scoring.*` and `ingest.track_ttl_s`; see above.
    keypoint_conf_threshold: float = 0.3
    alert_threshold: float = 0.5
    history_len: int = 30
    track_ttl_s: float = 10.0
    #: Whose floor this is, and whether phones are admitted automatically.
    #: The console shows the mode because "no phones have joined" means two
    #: very different things under `auto` and under `manual`.
    session_name: str = ""
    approval: str = "auto"
    #: The scoring weight vectors, default and per-exercise, so the page can
    #: say *which checks are switched off* for a given trainee.
    #:
    #: This is the console's half of a trade the scorer makes deliberately. A
    #: correct plank is horizontal and motionless, so
    #: `[scoring.exercise_weights.plank]` zeroes `fall` and `stillness` —
    #: which also means a trainee who collapses mid-plank raises neither. That
    #: is the right call for the score and a terrible thing to leave implicit:
    #: without this, a plank card reading "nothing flagged" is
    #: indistinguishable from one where those checks were actually run.
    default_weights: dict[str, float] = field(default_factory=dict)
    exercise_weights: dict[str, dict[str, float]] = field(default_factory=dict)


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
            self._respond(200, CONSOLE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/healthz":
            self._respond(200, b'{"status":"ok"}', "application/json")
        elif self.path == "/triage":
            self._respond(200, self._triage_payload(), "application/json")
        elif self.path == "/console":
            self._respond(200, self._console_payload(), "application/json")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 (stdlib method name)
        """The one mutating route: admitting or refusing a phone.

        Guarded on the *client* address rather than on what the server was
        bound to, so it keeps working for the instructor's own browser even
        when an operator has opened `outputs.http_host` to serve the console
        on a second screen — while never letting the rest of the LAN decide
        who is allowed to monitor a trainee.
        """
        if self.path != "/join/decide":
            # JSON rather than a bare 404: every other reply on this route is
            # JSON, and a caller that has to special-case the error shape is a
            # caller that will get it wrong.
            self._respond(404, b'{"error":"no such route"}', "application/json")
            return

        if not _is_loopback(self.client_address[0]) and not self.server.allow_remote_control:  # type: ignore[attr-defined]
            self._respond(
                403,
                b'{"error":"join decisions are accepted from this machine only"}',
                "application/json",
            )
            return

        decide = self.server.on_join_decision  # type: ignore[attr-defined]
        if decide is None:
            self._respond(
                503, b'{"error":"no admission queue is attached"}', "application/json"
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            request_id = body["request_id"]
            approve = body["approve"]
            if not isinstance(request_id, str) or not isinstance(approve, bool):
                raise ValueError("request_id must be a string and approve a bool")
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            self._respond(
                400,
                json.dumps({"error": f"malformed decision: {exc}"}).encode("utf-8"),
                "application/json",
            )
            return

        # A stale id is the ordinary case, not an error: the phone may have
        # hung up or timed out between the page drawing the button and someone
        # pressing it. Say so plainly so the console can re-render instead of
        # showing a decision that did not land.
        settled = decide(request_id, approve)
        self._respond(
            200 if settled else 409,
            json.dumps(
                {"settled": settled}
                if settled
                else {"settled": False, "error": "that request is no longer waiting"}
            ).encode("utf-8"),
            "application/json",
        )

    def _triage_payload(self) -> bytes:
        """The ranked records, and nothing else — the original contract."""
        with self.server.state_lock:  # type: ignore[attr-defined]
            return json.dumps(
                {
                    "ts": self.server.latest_ts,  # type: ignore[attr-defined]
                    "records": [
                        to_json_dict(r)
                        for r in self.server.latest_records  # type: ignore[attr-defined]
                    ],
                }
            ).encode("utf-8")

    def _console_payload(self) -> bytes:
        """Everything the trainer console needs, read atomically.

        Rank and stations come out under one lock acquisition on purpose: two
        endpoints polled separately would let the console draw a skeleton from
        one instant beside a score from another, and a trainer would have no
        way to tell that a card disagreed with itself.

        `ts` is when the station snapshot was taken and `rank_ts` is when the
        rank was last recomputed — different clocks-on-the-same-base, because
        stations refresh on every observation and the rank only on
        `ingest.rank_interval_s`. Staleness is `ts - station.last_seen_ts`.
        """
        with self.server.state_lock:  # type: ignore[attr-defined]
            return json.dumps(
                {
                    "ts": self.server.latest_station_ts,  # type: ignore[attr-defined]
                    "rank_ts": self.server.latest_ts,  # type: ignore[attr-defined]
                    "records": [
                        to_json_dict(r)
                        for r in self.server.latest_records  # type: ignore[attr-defined]
                    ],
                    "stations": [
                        station_to_json_dict(s)
                        for s in self.server.latest_stations  # type: ignore[attr-defined]
                    ],
                    "pending": [
                        pending_to_json_dict(p)
                        for p in self.server.latest_pending  # type: ignore[attr-defined]
                    ],
                    "config": self.server.console_config,  # type: ignore[attr-defined]
                }
            ).encode("utf-8")

    def _respond(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # silence the default stderr access log
        pass


class TriageHTTPServer:
    """Background HTTP server: the triage ranking, and the trainer console.

    Four routes, and deliberately only four — this is not a general-purpose
    API:

    * `GET /triage` -> `{"ts": float, "records": [{trainee_id, score,
      reason_codes, ts}, ...]}`. The four redacted fields, unchanged.
    * `GET /console` -> the same records plus `stations` (live
      `StationView`s) and `config`, read under one lock so a console never
      renders a skeleton and a score from two different instants.
    * `GET /healthz` -> liveness.
    * `GET /` -> the trainer console page, which polls `/console`.

    Binds to 127.0.0.1 by default, and that default matters more now than it
    did: `/triage` describes who on a floor needs attention, and `/console`
    additionally carries live body keypoints. Both are numeric and neither
    has ever touched a frame, but an operator who opens `outputs.http_host`
    beyond loopback is publishing a live pose stream on the LAN — see
    docs/CONSOLE.md.
    """

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        console: ConsoleSettings | None = None,
        on_join_decision: Callable[[str, bool], bool] | None = None,
        allow_remote_control: bool = False,
    ):
        self._server = ThreadingHTTPServer((host, port), _TriageRequestHandler)
        self._server.state_lock = threading.Lock()
        self._server.latest_ts = 0.0
        self._server.latest_station_ts = 0.0
        self._server.latest_records = []
        self._server.latest_stations = []
        self._server.latest_pending = []
        # Immutable for the server's lifetime, so they need no lock; built
        # once here rather than per request.
        self._server.console_config = asdict(console or ConsoleSettings())
        self._server.on_join_decision = on_join_decision
        self._server.allow_remote_control = allow_remote_control
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def update(
        self,
        ts: float,
        records: list[TriageRecord],
        stations: list[StationView] | None = None,
        pending: list[PendingJoinView] | None = None,
    ) -> None:
        """Publish a full snapshot: a rank tick and the stations behind it."""
        with self._server.state_lock:
            self._server.latest_ts = ts
            self._server.latest_records = list(records)
            self._server.latest_station_ts = ts
            self._server.latest_stations = list(stations or [])
            self._server.latest_pending = list(pending or [])

    def update_stations(
        self,
        ts: float,
        stations: list[StationView],
        pending: list[PendingJoinView] | None = None,
    ) -> None:
        """Refresh only the live station view, leaving the rank alone.

        Called as observations arrive, between rank ticks, so the console's
        skeletons move at the phones' own rate instead of stepping at
        `ingest.rank_interval_s`. `latest_ts` is deliberately not touched:
        `/triage`'s timestamp means "when the rank was computed", and
        advancing it here would claim a recomputation that did not happen.
        """
        with self._server.state_lock:
            self._server.latest_station_ts = ts
            self._server.latest_stations = list(stations)
            self._server.latest_pending = list(pending or [])

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
