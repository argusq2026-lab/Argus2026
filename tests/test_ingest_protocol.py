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


# -- the informational fields -------------------------------------------------
#
# `exercise`, `rep_count`, and `form_ok` are display-only: the trainer console
# shows them and nothing scores them. That is exactly why they still have to
# be validated here -- a field nothing scores is a field nobody notices is
# wrong, and it reaches a human's screen either way.


def test_observation_carries_the_informational_fields():
    obs = parse_observation(_obs(exercise="squat", rep_count=12, form_ok=False), VOCAB)
    assert obs.exercise == "squat"
    assert obs.rep_count == 12
    assert obs.form_ok is False


def test_informational_fields_are_optional():
    """A phone that sends none of them is well-formed, not degraded."""
    obs = parse_observation(_obs(), VOCAB)
    assert obs.exercise == ""
    assert obs.rep_count is None
    assert obs.form_ok is None


def test_a_missing_form_ok_is_unknown_rather_than_a_pass():
    """`None` and `False` must stay distinguishable all the way to the
    console: "the phone did not say" is not "the phone said it was fine"."""
    assert parse_observation(_obs(), VOCAB).form_ok is None
    assert parse_observation(_obs(form_ok=False), VOCAB).form_ok is False


def test_observation_rejects_a_non_string_exercise():
    with pytest.raises(ProtocolError, match="exercise"):
        parse_observation(_obs(exercise=7), VOCAB)


def test_observation_rejects_an_overlong_exercise():
    """The one field a phone fills freely is bounded, so it cannot become a
    free-text channel into the trainer's view."""
    with pytest.raises(ProtocolError, match="exercise"):
        parse_observation(_obs(exercise="x" * 500), VOCAB)


def test_observation_rejects_a_non_int_rep_count():
    with pytest.raises(ProtocolError, match="rep_count"):
        parse_observation(_obs(rep_count="12"), VOCAB)


def test_observation_rejects_a_boolean_rep_count():
    """`bool` is a subclass of `int`, so `true` would otherwise display as 1."""
    with pytest.raises(ProtocolError, match="rep_count"):
        parse_observation(_obs(rep_count=True), VOCAB)


def test_observation_rejects_a_negative_rep_count():
    with pytest.raises(ProtocolError, match="rep_count"):
        parse_observation(_obs(rep_count=-1), VOCAB)


def test_observation_rejects_a_non_bool_form_ok():
    with pytest.raises(ProtocolError, match="form_ok"):
        parse_observation(_obs(form_ok="yes"), VOCAB)


# -- server -> client messages ------------------------------------------------


def test_hello_ack_message_shape():
    assert hello_ack_message() == {"type": "hello_ack", "accepted": True}


def test_error_message_shape():
    assert error_message("bad thing") == {"type": "error", "message": "bad thing"}


def test_an_explicit_null_rep_count_means_not_reported():
    """A held exercise has no rep count. A phone that says so plainly is
    well-formed; refusing the connection over it would be strictness pointed
    the wrong way, since nothing is being silently defaulted."""
    assert parse_observation(_obs(rep_count=None), VOCAB).rep_count is None


def test_an_explicit_null_form_ok_is_still_unknown():
    assert parse_observation(_obs(form_ok=None), VOCAB).form_ok is None


def test_an_explicit_null_exercise_scores_on_the_defaults():
    assert parse_observation(_obs(exercise=None), VOCAB).exercise == ""
