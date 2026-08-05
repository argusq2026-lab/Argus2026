"""Per-exercise weight profiles, and the plank misfire they exist to fix.

The scorer's fall and stillness features encode an assumption that held for
every exercise Argus had seen: a trainee who is horizontal, or who has stopped
moving, is in trouble. A plank is both, correctly. Before
`[scoring.exercise_weights.plank]` existed, a textbook plank scored 0.42
against a 0.5 alert threshold and displayed `prolonged_stillness,
off_task_orientation` on the trainer's dashboard — a wrong explanation
attached to a score that was itself mostly noise.

These tests pin both halves: that the profile fixes it, and that the fix is
scoped to exercises that ask for it.
"""

from __future__ import annotations

import dataclasses

import pytest

from argus.config import ConfigError, ScoringConfig
from argus.triage import (
    REASON_FORM_ERROR,
    REASON_OFF_TASK,
    REASON_STILLNESS,
    TrackState,
    compute_triage,
)
from tests.conftest import make_observation, make_plank_observation


def hold(scoring, ticks: int, **kwargs) -> TrackState:
    """A full history window of one held plank."""
    track = TrackState(history_len=scoring.history_len)
    for i in range(ticks):
        track.push(make_plank_observation(ts=i * 0.066, **kwargs), scoring)
    return track


# -- the misfire this exists to fix -----------------------------------------


def test_a_correct_plank_scores_zero_and_explains_nothing(scoring):
    record = compute_triage("t", hold(scoring, scoring.history_len), 2.0, scoring)
    assert record.score == 0.0
    assert record.reason_codes == ()


def test_the_same_posture_without_the_exercise_label_still_misfires(scoring):
    """The geometry really does look like a fall; only the label saves it.

    This is the regression that matters: a phone that stops sending
    `exercise` does not degrade to "slightly worse plank scoring", it degrades
    to a correct plank accruing most of an alert. It is asserted here so the
    cost of dropping that field is visible in the suite rather than on a floor.
    """
    record = compute_triage("t", hold(scoring, scoring.history_len, exercise=None), 2.0, scoring)
    assert record.score > 0.4
    assert REASON_STILLNESS in record.reason_codes
    assert REASON_OFF_TASK in record.reason_codes


def test_an_exercise_with_no_profile_scores_on_the_default_weights(scoring):
    """`exercise` is a free-form label, not a closed vocabulary."""
    labelled = compute_triage("t", hold(scoring, scoring.history_len, exercise="burpee"), 2.0, scoring)
    unlabelled = compute_triage("t", hold(scoring, scoring.history_len, exercise=None), 2.0, scoring)
    assert labelled.score == unlabelled.score
    assert labelled.reason_codes == unlabelled.reason_codes


# -- a bad plank still gets an instructor -----------------------------------


@pytest.mark.parametrize("code", ["hips_sagging", "hips_piked"])
def test_a_bad_plank_alerts(scoring, code):
    track = hold(scoring, scoring.history_len, form_reason_codes=(code,))
    record = compute_triage("t", track, 2.0, scoring)
    assert record.score >= scoring.alert_threshold, f"{code} did not reach the threshold"
    assert record.reason_codes == (REASON_FORM_ERROR,)


def test_sagging_outranks_piking(scoring):
    """Lumbar hyperextension is the injury mechanism; piking is just weak."""
    sagging = compute_triage(
        "t", hold(scoring, scoring.history_len, form_reason_codes=("hips_sagging",)), 2.0, scoring
    )
    piked = compute_triage(
        "t", hold(scoring, scoring.history_len, form_reason_codes=("hips_piked",)), 2.0, scoring
    )
    assert sagging.score > piked.score


def test_a_bad_plank_outranks_a_correct_one(scoring):
    correct = compute_triage("a", hold(scoring, scoring.history_len), 2.0, scoring)
    bad = compute_triage(
        "b", hold(scoring, scoring.history_len, form_reason_codes=("hips_sagging",)), 2.0, scoring
    )
    assert bad.score > correct.score


# -- zero weight suppresses the reason code, not just the number ------------


def test_a_zero_weighted_feature_emits_no_reason_code(scoring):
    """A reason must explain the score it is attached to.

    Stillness reads 1.0 for a held plank and contributes nothing, because the
    plank profile weights it 0. Reporting `prolonged_stillness` anyway would
    be an explanation for a number that does not exist.
    """
    track = hold(scoring, scoring.history_len)
    assert track.last_exercise == "plank"
    record = compute_triage("t", track, 2.0, scoring)
    assert REASON_STILLNESS not in record.reason_codes
    assert scoring.weights_for("plank")["stillness"] == 0.0


def test_form_error_reason_survives_because_its_weight_is_non_zero(scoring):
    track = hold(scoring, scoring.history_len, form_reason_codes=("hips_sagging",))
    record = compute_triage("t", track, 2.0, scoring)
    assert record.reason_codes == (REASON_FORM_ERROR,)


# -- the exercise label itself ----------------------------------------------


def test_the_latest_exercise_wins(scoring):
    """A trainee dropping into a plank is scored as planking immediately."""
    track = TrackState(history_len=scoring.history_len)
    for i in range(5):
        track.push(make_observation(ts=float(i), exercise="squat"), scoring)
    assert track.last_exercise == "squat"
    track.push(make_plank_observation(ts=5.0), scoring)
    assert track.last_exercise == "plank"


def test_weights_for_is_case_insensitive(scoring):
    assert scoring.weights_for("PLANK") == scoring.weights_for("plank")


def test_weights_for_none_is_the_default_set(scoring):
    assert scoring.weights_for(None) == scoring.weights


# -- profiles are held to the same contract as the default weights ----------


def _scoring_with(profiles: dict) -> ScoringConfig:
    return ScoringConfig(
        weights={"fall": 0.4, "stillness": 0.2, "occlusion": 0.15, "off_task": 0.1, "form_error": 0.15},
        form_error_vocab={"knee_valgus": 0.8},
        exercise_weights=profiles,
    )


def test_a_profile_that_does_not_sum_to_one_is_rejected():
    """Otherwise its score is not comparable against `alert_threshold`."""
    with pytest.raises(ConfigError, match="must sum to 1.0"):
        _scoring_with({"plank": {"fall": 0.0, "stillness": 0.0, "occlusion": 0.1,
                                 "off_task": 0.0, "form_error": 0.5}})


def test_a_partial_profile_is_rejected():
    """A sparse patch would leave "what does a plank score on" split in two."""
    with pytest.raises(ConfigError, match="missing required weight"):
        _scoring_with({"plank": {"form_error": 1.0}})


def test_a_profile_naming_an_unread_feature_is_rejected():
    with pytest.raises(ConfigError, match="unknown weight"):
        _scoring_with({"plank": {"fall": 0.0, "stillness": 0.0, "occlusion": 0.15,
                                 "off_task": 0.0, "form_error": 0.75, "hip_angle": 0.1}})


def test_a_negative_profile_weight_is_rejected():
    with pytest.raises(ConfigError, match="non-negative"):
        _scoring_with({"plank": {"fall": -0.1, "stillness": 0.0, "occlusion": 0.15,
                                 "off_task": 0.0, "form_error": 0.95}})


def test_the_error_names_the_offending_profile():
    """An operator with several profiles needs to know which one is wrong."""
    with pytest.raises(ConfigError, match=r"scoring\.exercise_weights\.plank"):
        _scoring_with({"plank": {"form_error": 1.0}})


def test_no_profiles_is_valid():
    """The section is optional; its absence is not a degraded state."""
    cfg = _scoring_with({})
    assert cfg.weights_for("plank") == cfg.weights


# -- the shipped profile ----------------------------------------------------


def test_the_shipped_plank_profile_zeroes_the_features_that_misread_it(scoring):
    profile = scoring.weights_for("plank")
    assert profile["fall"] == 0.0
    assert profile["stillness"] == 0.0
    assert profile["off_task"] == 0.0
    assert profile["occlusion"] > 0.0, "an unseen trainee is still worth flagging"
    assert profile["form_error"] > 0.0


def test_the_scorer_cannot_mutate_a_profile(scoring):
    """Frozen tuning is what keeps two runs of the same history identical."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        scoring.alert_threshold = 0.9
