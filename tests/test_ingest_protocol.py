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


# -- hello use_case: server and phone must agree ---------------------------


def _hello(**overrides) -> dict:
    base = {"type": "hello", "protocol_version": 1, "station_id": "s0", "trainee_id": "t0"}
    base.update(overrides)
    return base


def test_hello_defaults_to_fitness_and_matches_a_fitness_session():
    hello = parse_hello(_hello(), 1, session_use_case="fitness")
    assert hello.use_case == "fitness"


def test_hello_matches_an_explicit_use_case():
    hello = parse_hello(_hello(use_case="welding"), 1, session_use_case="welding")
    assert hello.use_case == "welding"


def test_hello_rejects_a_use_case_mismatch():
    """A fitness phone connecting to a welding session must be refused, the
    same as a `session_name` mismatch -- not admitted and scored by nothing."""
    with pytest.raises(ProtocolError, match="use_case"):
        parse_hello(_hello(), 1, session_use_case="welding")


def test_hello_rejects_a_legacy_phone_against_a_non_fitness_session():
    """A phone that predates this field omits `use_case` entirely and
    defaults to `"fitness"` -- that default must still be checked, not
    treated as "said nothing, so assume it agrees"."""
    with pytest.raises(ProtocolError, match="fitness"):
        parse_hello({"type": "hello", "protocol_version": 1, "station_id": "s0", "trainee_id": "t0"},
                    1, session_use_case="welding")


def test_hello_rejects_a_non_string_use_case():
    with pytest.raises(ProtocolError, match="use_case"):
        parse_hello(_hello(use_case=7), 1, session_use_case="fitness")


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


# -- use_case dispatch ---------------------------------------------------
#
# `use_case` selects which parser reads the rest of the message (see
# `docs/PROTOCOL.md`). Only "fitness" is implemented today; everything else
# here is about that dispatch staying a clean rejection rather than a
# confusing failure three fields into a parser that expected a different
# message shape.


def test_observation_defaults_to_fitness_when_use_case_is_absent():
    """Every phone in the field predates this field; omitting it must not
    refuse them."""
    obs = parse_observation(_obs(), VOCAB)
    assert obs.use_case == "fitness"


def test_observation_accepts_an_explicit_fitness_use_case():
    obs = parse_observation(_obs(use_case="fitness"), VOCAB)
    assert obs.use_case == "fitness"
    assert obs.bbox_xyxy == (0.1, 0.1, 0.5, 0.9)


def test_observation_rejects_an_unimplemented_use_case():
    with pytest.raises(ProtocolError, match="use_case"):
        parse_observation(_obs(use_case="lab"), VOCAB)


def test_observation_rejects_a_non_string_use_case():
    with pytest.raises(ProtocolError, match="use_case"):
        parse_observation(_obs(use_case=7), VOCAB)


def test_observation_rejects_an_empty_use_case():
    with pytest.raises(ProtocolError, match="use_case"):
        parse_observation(_obs(use_case=""), VOCAB)


def test_observation_rejects_an_overlong_use_case():
    with pytest.raises(ProtocolError, match="use_case"):
        parse_observation(_obs(use_case="x" * 100), VOCAB)


def test_an_explicit_null_use_case_falls_back_to_fitness():
    """Same posture as `exercise=None`: an explicit null is not reported,
    not a malformed value."""
    assert parse_observation(_obs(use_case=None), VOCAB).use_case == "fitness"


# -- welding: a placeholder use case --------------------------------------
#
# There is no welding classifier, so its parser validates only the envelope
# (`ts`) and carries an opaque `payload` through uninterpreted. See
# `argus.triage.compute_triage_welding` for why the scorer is equally inert.


def test_welding_observation_parses_with_no_payload():
    obs = parse_observation({"type": "observation", "use_case": "welding", "ts": 5.0}, VOCAB)
    assert obs.use_case == "welding"
    assert obs.payload == {}


def test_welding_observation_carries_an_opaque_payload_through_uninterpreted():
    raw = {
        "type": "observation",
        "use_case": "welding",
        "ts": 5.0,
        "payload": {"torch_angle_deg": 12.5, "anything": "at all"},
    }
    obs = parse_observation(raw, VOCAB)
    assert obs.payload == {"torch_angle_deg": 12.5, "anything": "at all"}


def test_welding_observation_rejects_a_non_object_payload():
    raw = {"type": "observation", "use_case": "welding", "ts": 5.0, "payload": "not an object"}
    with pytest.raises(ProtocolError, match="payload"):
        parse_observation(raw, VOCAB)


def test_welding_observation_still_requires_ts():
    raw = {"type": "observation", "use_case": "welding"}
    with pytest.raises(ProtocolError, match="ts"):
        parse_observation(raw, VOCAB)


def test_welding_observation_ignores_fitness_only_fields():
    """A welding message need not — and does not have to — carry `bbox_xyxy`
    or any other fitness field; welding's parser never looks for them."""
    obs = parse_observation({"type": "observation", "use_case": "welding", "ts": 5.0}, VOCAB)
    assert obs.bbox_xyxy is None
    assert obs.keypoints_xy is None


# -- expected_use_case: an observation must match what hello agreed to ------


def test_observation_matching_expected_use_case_parses():
    obs = parse_observation(_obs(), VOCAB, expected_use_case="fitness")
    assert obs.use_case == "fitness"


def test_observation_rejects_a_use_case_hello_did_not_agree_to():
    """The connection agreed to fitness at hello; a message switching to
    welding mid-stream is a protocol violation, not a mode change."""
    raw = {"type": "observation", "use_case": "welding", "ts": 5.0}
    with pytest.raises(ProtocolError, match="use_case"):
        parse_observation(raw, VOCAB, expected_use_case="fitness")


def test_expected_use_case_none_skips_the_check():
    """The default: existing callers that never pass `expected_use_case`
    keep working exactly as before."""
    obs = parse_observation(_obs(use_case="fitness"), VOCAB)
    assert obs.use_case == "fitness"


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


# -- nursing ------------------------------------------------------------
#
# Nursing shares fitness's pose fields and none of the rest. The tests below
# are as much about what it *does not* accept: a nursing station's faults are
# derived on the laptop from the movement, so there is no phone-classified
# form vocabulary to carry, and admitting one would invite a second source of
# truth about whether a compression was good.


def _nursing_obs(**overrides) -> dict:
    base = {
        "type": "observation",
        "use_case": "nursing",
        "ts": 1.0,
        "procedure": "cpr",
        "bbox_xyxy": [0.1, 0.1, 0.5, 0.9],
        "keypoints_xy": [[0.3, 0.2]] * 17,
        "keypoints_conf": [0.9] * 17,
    }
    base.update(overrides)
    return base


def test_a_nursing_observation_carries_pose_and_procedure():
    obs = parse_observation(_nursing_obs(), VOCAB)
    assert obs.use_case == "nursing"
    assert obs.procedure == "cpr"
    assert len(obs.keypoints_xy) == 17
    assert obs.bbox_xyxy == (0.1, 0.1, 0.5, 0.9)


def test_a_nursing_observation_needs_a_pose():
    """Unlike welding's opaque payload, nursing's scorer reads keypoints, so a
    missing pose is a malformed message rather than a station with nothing to
    say."""
    incomplete = _nursing_obs()
    del incomplete["keypoints_xy"]
    with pytest.raises(ProtocolError, match="keypoints_xy"):
        parse_observation(incomplete, VOCAB)


@pytest.mark.parametrize("value", [None, ""])
def test_a_nursing_station_may_decline_to_name_a_procedure(value):
    """Scored as a flat 0.0 by `compute_triage_nursing` rather than refused —
    a ward running something this build cannot score should still appear."""
    assert parse_observation(_nursing_obs(procedure=value), VOCAB).procedure is None


def test_a_nursing_observation_rejects_a_non_string_procedure():
    with pytest.raises(ProtocolError, match="procedure"):
        parse_observation(_nursing_obs(procedure=7), VOCAB)


def test_a_nursing_observation_bounds_the_procedure_label():
    with pytest.raises(ProtocolError, match="procedure"):
        parse_observation(_nursing_obs(procedure="c" * 65), VOCAB)


def test_fitness_fields_are_not_read_from_a_nursing_observation():
    """A nursing station that sent a rep count or a form code would be sending
    fitness's vocabulary; those fields are simply not part of its payload, and
    must not leak into the observation the scorer sees."""
    obs = parse_observation(
        _nursing_obs(rep_count=12, exercise="squat", form_reason_codes=["knee_valgus"]),
        VOCAB,
    )
    assert obs.rep_count is None
    assert obs.exercise is None
    assert obs.form_reason_codes == ()
