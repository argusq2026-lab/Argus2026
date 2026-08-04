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


class ProtocolError(ValueError):
    """A message is malformed, mis-versioned, or uses an unrecognised code."""


@dataclass(frozen=True)
class HelloMessage:
    """The first message on every connection."""

    station_id: str
    trainee_id: str
    exercise_plan: str = ""


def _require(raw: Mapping[str, Any], key: str, kind: type) -> Any:
    if key not in raw:
        raise ProtocolError(f"missing required field: {key!r}")
    value = raw[key]
    if kind is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, kind):
        raise ProtocolError(f"field {key!r} must be a {kind.__name__}, got {type(value).__name__}")
    return value


def parse_hello(raw: Mapping[str, Any], expected_protocol_version: int) -> HelloMessage:
    """Validate a `hello` message. Raises `ProtocolError` on any mismatch."""
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

    return HelloMessage(station_id=station_id, trainee_id=trainee_id, exercise_plan=exercise_plan)


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
    )


def hello_ack_message() -> dict:
    return {"type": "hello_ack", "accepted": True}


def error_message(reason: str) -> dict:
    return {"type": "error", "message": reason}
