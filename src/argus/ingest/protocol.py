"""Wire message parsing and validation. See `docs/PROTOCOL.md` for the spec.

Every message is a JSON object. Validation here is deliberately strict and
raises rather than defaults, the same "nothing degrades quietly" posture as
`argus.config`: a phone and a laptop that disagree on the protocol version or
the form-error vocabulary are a deployment bug, not something to paper over
by ignoring the field that doesn't parse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from argus.triage import NUM_KEYPOINTS, FrameObservation


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


def _require(raw: Mapping[str, Any], key: str, kind: type) -> Any:
    if key not in raw:
        raise ProtocolError(f"missing required field: {key!r}")
    value = raw[key]
    if kind is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, kind):
        raise ProtocolError(f"field {key!r} must be a {kind.__name__}, got {type(value).__name__}")
    return value


def parse_hello(
    raw: Mapping[str, Any],
    expected_protocol_version: int,
    session_name: str = "",
) -> HelloMessage:
    """Validate a `hello` message. Raises `ProtocolError` on any mismatch.

    `session_name` is this server's own. A phone that names a *different*
    session is rejected: it found some other laptop's beacon and is about to
    put a trainee on a console nobody is watching.
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

    return HelloMessage(
        station_id=station_id,
        trainee_id=trainee_id,
        exercise_plan=exercise_plan,
        display_name=display_name,
        session_name=claimed_session,
    )


def parse_observation(raw: Mapping[str, Any], form_error_vocab: Mapping[str, float]) -> FrameObservation:
    """Validate an `observation` message and translate it into a `FrameObservation`.

    `form_reason_codes` must be drawn from `form_error_vocab` (the config's
    `[scoring.form_error_vocab]`) — a code outside that set is a
    protocol/version mismatch between the phone and the laptop, so it is
    rejected here rather than silently scored as zero.
    """
    if raw.get("type") != "observation":
        raise ProtocolError(f"expected type 'observation', got {raw.get('type')!r}")

    ts = _require(raw, "ts", float)

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
        bbox_xyxy=bbox_xyxy,
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        form_reason_codes=tuple(codes_raw),
        exercise=_parse_exercise(raw),
        rep_count=_parse_rep_count(raw),
        form_ok=_parse_form_ok(raw),
    )


def _parse_exercise(raw: Mapping[str, Any]) -> str:
    """The phone's classified exercise label. Optional, display-only."""
    value = raw.get("exercise", "")
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
    if "rep_count" not in raw:
        return None
    value = raw["rep_count"]
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
    if "form_ok" not in raw:
        return None
    value = raw["form_ok"]
    if not isinstance(value, bool):
        raise ProtocolError(f"form_ok must be a bool, got {type(value).__name__}")
    return value


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
