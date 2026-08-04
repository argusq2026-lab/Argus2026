"""Wire message validation — the ingest server's "nothing degrades quietly"."""

from __future__ import annotations

import pytest

from argus.ingest.protocol import (
    ProtocolError,
    error_message,
    hello_ack_message,
    parse_hello,
    parse_observation,
)

VOCAB = {"knee_valgus": 0.8, "insufficient_depth": 0.6}


def _obs(**overrides) -> dict:
    base = {
        "type": "observation",
        "ts": 1.0,
        "bbox_xyxy": [0.1, 0.1, 0.5, 0.9],
        "keypoints_xy": [[0.3, 0.2]] * 17,
        "keypoints_conf": [0.9] * 17,
        "form_reason_codes": [],
    }
    base.update(overrides)
    return base


# -- hello --------------------------------------------------------------


def test_valid_hello_parses():
    hello = parse_hello(
        {"type": "hello", "protocol_version": 1, "station_id": "s0", "trainee_id": "t0"}, 1
    )
    assert hello.station_id == "s0"
    assert hello.trainee_id == "t0"
    assert hello.exercise_plan == ""


def test_hello_carries_an_optional_exercise_plan():
    hello = parse_hello(
        {
            "type": "hello", "protocol_version": 1, "station_id": "s0",
            "trainee_id": "t0", "exercise_plan": "squat",
        },
        1,
    )
    assert hello.exercise_plan == "squat"


def test_hello_must_be_the_first_message_type():
    with pytest.raises(ProtocolError, match="hello"):
        parse_hello({"type": "observation"}, 1)


def test_hello_rejects_a_mismatched_protocol_version():
    with pytest.raises(ProtocolError, match="protocol_version"):
        parse_hello(
            {"type": "hello", "protocol_version": 2, "station_id": "s0", "trainee_id": "t0"}, 1
        )


@pytest.mark.parametrize("field", ["station_id", "trainee_id", "protocol_version"])
def test_hello_rejects_a_missing_required_field(field):
    raw = {"type": "hello", "protocol_version": 1, "station_id": "s0", "trainee_id": "t0"}
    del raw[field]
    with pytest.raises(ProtocolError):
        parse_hello(raw, 1)


def test_hello_rejects_an_empty_trainee_id():
    with pytest.raises(ProtocolError, match="trainee_id"):
        parse_hello({"type": "hello", "protocol_version": 1, "station_id": "s0", "trainee_id": ""}, 1)


# -- observation ----------------------------------------------------------


def test_valid_observation_parses_into_a_frame_observation():
    obs = parse_observation(_obs(), VOCAB)
    assert obs.ts == 1.0
    assert obs.bbox_xyxy == (0.1, 0.1, 0.5, 0.9)
    assert len(obs.keypoints_xy) == 17
    assert len(obs.keypoints_conf) == 17
    assert obs.form_reason_codes == ()


def test_observation_carries_recognised_form_codes():
    obs = parse_observation(_obs(form_reason_codes=["knee_valgus"]), VOCAB)
    assert obs.form_reason_codes == ("knee_valgus",)


def test_observation_must_be_the_right_type():
    with pytest.raises(ProtocolError, match="observation"):
        parse_observation(_obs(type="hello"), VOCAB)


def test_observation_rejects_a_short_bbox():
    with pytest.raises(ProtocolError, match="bbox_xyxy"):
        parse_observation(_obs(bbox_xyxy=[0.1, 0.1, 0.5]), VOCAB)


def test_observation_rejects_the_wrong_keypoint_count():
    with pytest.raises(ProtocolError, match="keypoints_xy"):
        parse_observation(_obs(keypoints_xy=[[0.1, 0.1]] * 10), VOCAB)


def test_observation_rejects_mismatched_confidence_count():
    with pytest.raises(ProtocolError, match="keypoints_conf"):
        parse_observation(_obs(keypoints_conf=[0.9] * 5), VOCAB)


def test_observation_rejects_a_code_outside_the_vocabulary():
    with pytest.raises(ProtocolError, match="form_error_vocab"):
        parse_observation(_obs(form_reason_codes=["not_a_real_code"]), VOCAB)


def test_observation_rejects_a_missing_field():
    raw = _obs()
    del raw["ts"]
    with pytest.raises(ProtocolError, match="ts"):
        parse_observation(raw, VOCAB)


# -- server -> client messages ------------------------------------------------


def test_hello_ack_message_shape():
    assert hello_ack_message() == {"type": "hello_ack", "accepted": True}


def test_error_message_shape():
    assert error_message("bad thing") == {"type": "error", "message": "bad thing"}
