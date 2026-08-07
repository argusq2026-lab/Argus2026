"""Wire message parsing and validation. See `docs/PROTOCOL.md` for the spec.

Every message is a JSON object. Validation here is deliberately strict and
raises rather than defaults, the same "nothing degrades quietly" posture as
`argus.config`: a phone and a laptop that disagree on the protocol version or
the form-error vocabulary are a deployment bug, not something to paper over
by ignoring the field that doesn't parse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from argus.triage import NUM_KEYPOINTS, FrameObservation

#: Longest a `use_case` may claim to be. Same reasoning as `_MAX_EXERCISE_LEN`
#: below: it selects a parser/scorer by dict lookup, never grows into
#: anything else, so an oversized value is a malformed message.
_MAX_USE_CASE_LEN = 32

#: `use_case` a phone omits is `"fitness"` — every phone in the field today
#: predates this field, and treating its absence as anything other than the
#: one use case that has ever existed would refuse every one of them.
_DEFAULT_USE_CASE = "fitness"


#: `exercise` is the one field on the wire whose value a phone chooses freely
#: rather than drawing from a closed vocabulary, so it is the one place free
#: text could enter. It is bounded here rather than trusted: a classifier
#: label is a short token ("squat", "burpee"), and anything longer than this
#: is a malformed message, not a plausible label. The bound keeps the field
#: from becoming a side channel; `argus.outputs` keeps it from becoming a
#: score, and the console renders it as text, never as markup. Like
#: `_HELLO_TIMEOUT_S` in `argus.ingest.server`, this is a protocol constant —
#: what the wire format permits — not an operator-tunable threshold.
_MAX_EXERCISE_LEN = 64

#: Same reasoning as `_MAX_EXERCISE_LEN`, for the label a phone offers the
#: instructor on an approval prompt. Bounded so a join request cannot arrive
#: carrying a paragraph, and rendered as text by the console, never as markup.
_MAX_DISPLAY_NAME_LEN = 64

#: Same reasoning as `_MAX_EXERCISE_LEN`, for nursing's `procedure` label.
_MAX_PROCEDURE_LEN = 64


class ProtocolError(ValueError):
    """A message is malformed, mis-versioned, or uses an unrecognised code."""


@dataclass(frozen=True)
class HelloMessage:
    """The first message on every connection."""

    station_id: str
    trainee_id: str
    exercise_plan: str = ""
    #: What to call this phone on the instructor's approval prompt. Optional,
    #: phone-chosen, length-bounded, display-only — an instructor deciding
    #: whether to admit "Alex — rack 3" is making a better decision than one
    #: shown an opaque device id, and that is the whole of its purpose.
    display_name: str = ""
    #: The session the phone believes it is joining, if it learned one from a
    #: beacon. The server rejects a mismatch rather than admitting the phone
    #: anyway: on a floor with two laptops, silently joining the wrong one is
    #: a trainee monitored by an instructor who is not watching them.
    session_name: str = ""
    #: What the phone believes it is running. The server rejects a mismatch
    #: against `[session] use_case` for the same reason it rejects a
    #: `session_name` mismatch: a fitness phone admitted onto a floor an
    #: instructor set up for welding is monitored by a scorer that will
    #: never fire, which is a quieter failure than a rejected connection.
    use_case: str = "fitness"


def _require(raw: Mapping[str, Any], key: str, kind: type) -> Any:
    if key not in raw:
        raise ProtocolError(f"missing required field: {key!r}")
    value = raw[key]
    if kind is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, kind):
        raise ProtocolError(f"field {key!r} must be a {kind.__name__}, got {type(value).__name__}")
    return value


def _parse_use_case(raw: Mapping[str, Any]) -> str:
    """`use_case`, shared by `hello` and `observation`: bounded, non-empty,
    defaulting to `"fitness"` for a message that omits it or sends `null`.
    Only the type and length are enforced here — whether a use case with
    that name is one this server actually runs is a different question,
    answered by whoever calls this (a hello-vs-session-config check, an
    observation-vs-`_OBSERVATION_PARSERS` lookup), not by this helper.
    """
    value = raw.get("use_case", _DEFAULT_USE_CASE)
    if value is None:
        value = _DEFAULT_USE_CASE
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"use_case must be a non-empty string, got {value!r}")
    if len(value) > _MAX_USE_CASE_LEN:
        raise ProtocolError(f"use_case must be at most {_MAX_USE_CASE_LEN} characters")
    return value


def parse_hello(
    raw: Mapping[str, Any],
    expected_protocol_version: int,
    session_name: str = "",
    session_use_case: str = "fitness",
) -> HelloMessage:
    """Validate a `hello` message. Raises `ProtocolError` on any mismatch.

    `session_name` is this server's own. A phone that names a *different*
    session is rejected: it found some other laptop's beacon and is about to
    put a trainee on a console nobody is watching.

    `session_use_case` is `[session] use_case` — what this floor is running.
    Unlike `session_name`, this check is strict even when the phone said
    nothing: a phone that omits `use_case` defaults to `"fitness"` the same
    as an observation does, and a laptop configured for anything else must
    still refuse it rather than silently admitting a fitness phone onto a
    non-fitness floor.
    """
    if raw.get("type") != "hello":
        raise ProtocolError(f"expected the first message to be type 'hello', got {raw.get('type')!r}")

    version = _require(raw, "protocol_version", int)
    if version != expected_protocol_version:
        raise ProtocolError(
            f"protocol_version {version} is not supported by this server "
            f"(expects {expected_protocol_version}); update the phone app or "
            "the server's ingest.protocol_version"
        )

    station_id = _require(raw, "station_id", str)
    trainee_id = _require(raw, "trainee_id", str)
    if not station_id:
        raise ProtocolError("station_id must not be empty")
    if not trainee_id:
        raise ProtocolError("trainee_id must not be empty")

    exercise_plan = raw.get("exercise_plan", "")
    if not isinstance(exercise_plan, str):
        raise ProtocolError("exercise_plan must be a string")

    display_name = raw.get("display_name", "")
    if not isinstance(display_name, str):
        raise ProtocolError("display_name must be a string")
    if len(display_name) > _MAX_DISPLAY_NAME_LEN:
        raise ProtocolError(
            f"display_name must be at most {_MAX_DISPLAY_NAME_LEN} characters "
            f"(got {len(display_name)}); it is a label, not free text"
        )

    claimed_session = raw.get("session_name", "")
    if not isinstance(claimed_session, str):
        raise ProtocolError("session_name must be a string")
    if claimed_session and session_name and claimed_session != session_name:
        raise ProtocolError(
            f"this server is running session {session_name!r}, not "
            f"{claimed_session!r}; the phone found a different laptop's beacon"
        )

    use_case = _parse_use_case(raw)
    if use_case != session_use_case:
        raise ProtocolError(
            f"this server is running a {session_use_case!r} session, but the "
            f"phone is configured for {use_case!r} — check the app's use case "
            "matches this laptop's [session] use_case"
        )

    return HelloMessage(
        station_id=station_id,
        trainee_id=trainee_id,
        exercise_plan=exercise_plan,
        display_name=display_name,
        session_name=claimed_session,
        use_case=use_case,
    )


def parse_observation(
    raw: Mapping[str, Any],
    form_error_vocab: Mapping[str, float],
    expected_use_case: str | None = None,
) -> FrameObservation:
    """Validate an `observation` message and translate it into a `FrameObservation`.

    `type` and `use_case` are the only fields every use case shares; the rest
    of the message belongs to whichever use case's parser is registered in
    `_OBSERVATION_PARSERS` below, so a station running a use case this server
    does not implement is rejected here rather than failing deeper inside a
    parser expecting fields (`bbox_xyxy`, `keypoints_xy`, ...) that only
    fitness's phones ever send. See `docs/PROTOCOL.md` for the extension
    point this is: a new use case adds one parser and one registry entry, it
    does not touch the fields another use case already validates.

    `expected_use_case` is the use case this *connection* already agreed to
    at `hello` (`argus.ingest.server` passes `hello.use_case`). A message
    naming a different one is rejected here rather than silently switching
    the trainee's track to a scorer nobody validated the handshake against —
    the same "one connection, one use case" guarantee `hello` establishes,
    enforced on every message rather than trusted once. `None` skips the
    check, which is what every existing call in the test suite does; it is
    not a way for a real connection to opt out.
    """
    if raw.get("type") != "observation":
        raise ProtocolError(f"expected type 'observation', got {raw.get('type')!r}")

    use_case = _parse_use_case(raw)
    if expected_use_case is not None and use_case != expected_use_case:
        raise ProtocolError(
            f"this connection agreed to use_case {expected_use_case!r} at hello, "
            f"but this observation names {use_case!r}"
        )

    parser = _OBSERVATION_PARSERS.get(use_case)
    if parser is None:
        raise ProtocolError(
            f"unsupported use_case {use_case!r}; this server only implements "
            f"{sorted(_OBSERVATION_PARSERS)}"
        )
    return parser(raw, use_case, form_error_vocab)


def _parse_pose_fields(raw: Mapping[str, Any]):
    """The pose triple — `bbox_xyxy`, `keypoints_xy`, `keypoints_conf`.

    Shared by fitness and nursing because a COCO-17 pose is genuinely the same
    measurement in both, not because one was made to fit the other: what
    differs between them is everything *around* the pose (an exercise label and
    a rep counter, versus a procedure), and those stay in their own parsers.
    A use case that reads no pose at all — welding today — never calls this.
    """
    bbox = _require(raw, "bbox_xyxy", list)
    if len(bbox) != 4 or not all(isinstance(v, (int, float)) for v in bbox):
        raise ProtocolError("bbox_xyxy must be a list of 4 numbers")
    bbox_xyxy = tuple(float(v) for v in bbox)

    kp_xy_raw = _require(raw, "keypoints_xy", list)
    if len(kp_xy_raw) != NUM_KEYPOINTS:
        raise ProtocolError(f"keypoints_xy must have {NUM_KEYPOINTS} entries, got {len(kp_xy_raw)}")
    keypoints_xy = []
    for pair in kp_xy_raw:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ProtocolError("each keypoints_xy entry must be a [x, y] pair")
        keypoints_xy.append((float(pair[0]), float(pair[1])))

    kp_conf_raw = _require(raw, "keypoints_conf", list)
    if len(kp_conf_raw) != NUM_KEYPOINTS or not all(isinstance(v, (int, float)) for v in kp_conf_raw):
        raise ProtocolError(f"keypoints_conf must be {NUM_KEYPOINTS} numbers, got {len(kp_conf_raw)}")
    keypoints_conf = [float(v) for v in kp_conf_raw]

    return bbox_xyxy, keypoints_xy, keypoints_conf


def _parse_procedure(raw: Mapping[str, Any]) -> str | None:
    """Which procedure a nursing station is performing.

    Bounded and type-checked here but *not* checked against the set of
    procedures this build can score — that lookup belongs to
    `argus.triage.compute_triage_nursing`, which answers it by scoring a flat
    0.0 rather than by refusing the connection. Same reasoning as `exercise`:
    a ward running a procedure this laptop has no scorer for should still see
    its station on the console, not be turned away at the door.
    """
    value = raw.get("procedure")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProtocolError(f"procedure must be a string, got {type(value).__name__}")
    if not value:
        return None
    if len(value) > _MAX_PROCEDURE_LEN:
        raise ProtocolError(
            f"procedure must be at most {_MAX_PROCEDURE_LEN} characters "
            f"(got {len(value)}); it is a label, not free text"
        )
    return value


def _parse_nursing_observation(
    raw: Mapping[str, Any], use_case: str, form_error_vocab: Mapping[str, float]
) -> FrameObservation:
    """Nursing's `observation` payload: pose plus which procedure is running.

    Shares fitness's pose fields and none of the rest. There is no `exercise`,
    no `rep_count` and no `form_reason_codes` here: a nursing station's faults
    are *derived on the laptop* from the movement itself (see
    `argus.triage.compute_triage_cpr`), not classified on the phone against a
    vocabulary. That difference is the point — fitness trusts the phone's
    classifier for form, nursing measures the rhythm itself.
    """
    ts = _require(raw, "ts", float)
    bbox_xyxy, keypoints_xy, keypoints_conf = _parse_pose_fields(raw)
    return FrameObservation(
        ts=ts,
        use_case=use_case,
        procedure=_parse_procedure(raw),
        bbox_xyxy=bbox_xyxy,
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
    )


def _parse_fitness_observation(
    raw: Mapping[str, Any], use_case: str, form_error_vocab: Mapping[str, float]
) -> FrameObservation:
    """Fitness's `observation` payload: pose, exercise, rep/form fields.

    `form_reason_codes` must be drawn from `form_error_vocab` (the config's
    `[scoring.form_error_vocab]`) — a code outside that set is a
    protocol/version mismatch between the phone and the laptop, so it is
    rejected here rather than silently scored as zero.
    """
    ts = _require(raw, "ts", float)
    bbox_xyxy, keypoints_xy, keypoints_conf = _parse_pose_fields(raw)

    codes_raw = raw.get("form_reason_codes", [])
    if not isinstance(codes_raw, list) or not all(isinstance(c, str) for c in codes_raw):
        raise ProtocolError("form_reason_codes must be a list of strings")
    unknown = [c for c in codes_raw if c not in form_error_vocab]
    if unknown:
        raise ProtocolError(
            f"form_reason_codes {unknown} not in the configured "
            "[scoring.form_error_vocab]; phone and server vocabularies have diverged"
        )

    return FrameObservation(
        ts=ts,
        use_case=use_case,
        bbox_xyxy=bbox_xyxy,
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        form_reason_codes=tuple(codes_raw),
        exercise=_parse_exercise(raw),
        rep_count=_parse_rep_count(raw),
        form_ok=_parse_form_ok(raw),
    )


def _parse_welding_observation(
    raw: Mapping[str, Any], use_case: str, form_error_vocab: Mapping[str, float]
) -> FrameObservation:
    """Welding's `observation` payload: a placeholder, deliberately.

    There is no welding classifier yet and no welding data to define real
    fields around — see `argus.triage.compute_triage_welding`, the scorer
    this feeds, for the same point made about scoring. So this validates
    only the envelope every use case shares (`ts`) plus one opaque `payload`
    object whose contents are carried through uninterpreted rather than
    validated field-by-field, because there is nothing yet to validate them
    against. When a real welding classifier exists, this function is where
    its actual fields (torch angle, travel speed, whatever it turns out to
    be) get named and checked — not bolted onto `payload` forever.
    """
    ts = _require(raw, "ts", float)

    payload = raw.get("payload", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ProtocolError(f"payload must be an object, got {type(payload).__name__}")

    return FrameObservation(ts=ts, use_case=use_case, payload=payload)


#: Every use case this server can parse an `observation` for. A new use case
#: (welding, nursing, ...) is added here with its own parser reading its own
#: payload shape — it does not extend or branch inside `_parse_fitness_
#: observation`, whose fields (`bbox_xyxy`, `exercise`, `rep_count`, ...)
#: are fitness's own and mean nothing to another use case.
_OBSERVATION_PARSERS: dict[
    str, Callable[[Mapping[str, Any], str, Mapping[str, float]], FrameObservation]
] = {
    "fitness": _parse_fitness_observation,
    "nursing": _parse_nursing_observation,
    "welding": _parse_welding_observation,
}


def _parse_exercise(raw: Mapping[str, Any]) -> str:
    """The phone's classified exercise label.

    Selects the scoring weight profile (`ScoringConfig.weights_for`), but
    stays deliberately open: unlike `form_reason_codes`, whose closed
    vocabulary is what keeps the form feature auditable, an exercise this
    server has no profile for scores on the default weights rather than
    being rejected. Only its type and length are enforced here.
    """
    value = raw.get("exercise", "")
    if value is None:      # absent or explicitly null: no exercise reported
        return ""
    if not isinstance(value, str):
        raise ProtocolError(f"exercise must be a string, got {type(value).__name__}")
    if len(value) > _MAX_EXERCISE_LEN:
        raise ProtocolError(
            f"exercise must be at most {_MAX_EXERCISE_LEN} characters "
            f"(got {len(value)}); it is a classifier label, not free text"
        )
    return value


def _parse_rep_count(raw: Mapping[str, Any]) -> int | None:
    """Running rep count for the current set. Optional, display-only.

    `bool` is rejected explicitly: it is a subclass of `int`, so a phone
    sending `true` here would otherwise be silently displayed as 1 rep.
    """
    value = raw.get("rep_count")
    # An explicit `null` means the same as omitting it: not reported. A held
    # exercise has no rep count to report, and refusing the connection over a
    # phone that says so plainly would be strictness pointed the wrong way --
    # nothing is being silently defaulted, the field is simply absent.
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"rep_count must be an int, got {type(value).__name__}")
    if value < 0:
        raise ProtocolError(f"rep_count must be non-negative, got {value}")
    return value


def _parse_form_ok(raw: Mapping[str, Any]) -> bool | None:
    """The phone's own correct/incorrect verdict. Optional, display-only.

    Never used to derive whether form is flagged — that comes from
    `form_reason_codes` being non-empty, which is closed-vocabulary and
    therefore scoreable. `None` means the phone did not send a verdict, which
    the console shows as unknown rather than as a pass.
    """
    value = raw.get("form_ok")
    if value is None:      # absent or explicitly null: the phone has no verdict
        return None
    if not isinstance(value, bool):
        raise ProtocolError(f"form_ok must be a bool, got {type(value).__name__}")
    return value


def parse_idle(raw: Mapping[str, Any]) -> float:
    """Validate an `idle` message and return its timestamp.

    "I am here and watching, and there is nobody in frame." Without it a
    station pointed at an empty rack is indistinguishable from a dead phone:
    both send nothing, so `ingest.track_ttl_s` evicts the healthy one, its
    next message is refused, and it reconnects — a flap that starts the moment
    a station is set up before its trainee arrives, which is the normal case.

    Deliberately *not* an observation with the subject fields nulled. An
    observation asserts a reading about a person; this asserts that there is
    no person to read. Making the difference a null inside a message everything
    else treats as a measurement is how a null gets scored as a zero.
    """
    if raw.get("type") != "idle":
        raise ProtocolError(f"expected type 'idle', got {raw.get('type')!r}")
    return _require(raw, "ts", float)


def hello_ack_message() -> dict:
    return {"type": "hello_ack", "accepted": True}


def join_pending_message(session_name: str, request_id: str, timeout_s: float) -> dict:
    """Sent instead of `hello_ack` when the instructor has to approve first.

    A phone that got no reply at all would look hung, and a station that looks
    hung gets restarted by whoever is standing next to it — repeatedly, each
    restart queueing another request. Saying "you are in a queue, for at most
    this long" is what stops that.
    """
    return {
        "type": "join_pending",
        "session_name": session_name,
        "request_id": request_id,
        "timeout_s": timeout_s,
    }


def error_message(reason: str) -> dict:
    return {"type": "error", "message": reason}
