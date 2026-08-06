"""Scoring functions. Carried over from the prototype suite, config-driven."""

from __future__ import annotations

import dataclasses
from collections import deque

import pytest

from argus.triage import (
    REASON_FALL,
    REASON_FORM_ERROR,
    REASON_OCCLUSION,
    REASON_OFF_TASK,
    REASON_STILLNESS,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    TrackState,
    compute_triage,
    compute_triage_fitness,
    known_use_cases,
    needs_instructor,
    rank_trainees,
    score_fall,
    score_form_codes,
    score_occlusion,
    score_off_task,
    score_stillness,
)
from tests.conftest import make_observation, standing_pose_kp_xy


# -- fall -------------------------------------------------------------------


def test_score_fall_no_history_is_zero(scoring):
    assert score_fall(deque(), scoring) == (0.0, False)


def test_score_fall_detects_drop_and_aspect_flip(scoring, track_state):
    for i in range(4):
        track_state.push(make_observation(ts=float(i), torso_y=140.0), scoring)
    track_state.push(
        make_observation(ts=4.0, bbox=(80.0, 260.0, 220.0, 320.0), torso_y=240.0),
        scoring,
    )
    score, hit = score_fall(track_state.history, scoring)
    assert score > 0.5
    assert hit is True


def test_score_fall_ignores_low_confidence_torso(scoring, track_state):
    """A drop inferred from keypoints we do not trust must not fire."""
    for i in range(4):
        track_state.push(make_observation(ts=float(i), kp_conf=[0.1] * 17), scoring)
    track_state.push(
        make_observation(ts=4.0, torso_y=240.0, kp_conf=[0.1] * 17), scoring
    )
    score, hit = score_fall(track_state.history, scoring)
    assert hit is False
    assert score == 0.0


# -- stillness --------------------------------------------------------------


def test_stillness_requires_a_full_window(scoring, track_state):
    for i in range(scoring.history_len):
        track_state.push(make_observation(ts=float(i)), scoring)
    fraction, hit = score_stillness(track_state.history, scoring)
    assert fraction == 1.0
    assert hit is True


def test_stillness_not_triggered_on_short_history(scoring, track_state):
    for i in range(3):
        track_state.push(make_observation(ts=float(i)), scoring)
    _, hit = score_stillness(track_state.history, scoring)
    assert hit is False


def test_stillness_zero_while_walking(scoring, track_state):
    for i in range(scoring.history_len):
        x = 100.0 + 20.0 * i
        track_state.push(make_observation(ts=float(i), bbox=(x, 100.0, x + 60, 240.0)), scoring)
    fraction, hit = score_stillness(track_state.history, scoring)
    assert fraction == 0.0
    assert hit is False


# -- occlusion --------------------------------------------------------------


def test_occlusion_needs_both_hands_and_face_hidden(scoring, track_state):
    for i in range(scoring.history_len):
        conf = [0.9] * 17
        for idx in (KP_NOSE, KP_LEFT_WRIST, KP_RIGHT_WRIST):
            conf[idx] = 0.05
        track_state.push(make_observation(ts=float(i), kp_conf=conf), scoring)
    fraction, hit = score_occlusion(track_state.history, scoring)
    assert fraction == 1.0
    assert hit is True


def test_occlusion_not_triggered_when_face_is_visible(scoring, track_state):
    for i in range(scoring.history_len):
        conf = [0.9] * 17
        conf[KP_LEFT_WRIST] = conf[KP_RIGHT_WRIST] = 0.05
        track_state.push(make_observation(ts=float(i), kp_conf=conf), scoring)
    _, hit = score_occlusion(track_state.history, scoring)
    assert hit is False


# -- off-task ---------------------------------------------------------------


def test_facing_the_station_scores_zero_off_task(scoring, track_state):
    """The baseline pose is camera-facing, and the config's reference is 180."""
    for i in range(scoring.history_len):
        track_state.push(make_observation(ts=float(i)), scoring)
    score, hit = score_off_task(track_state.history, scoring)
    assert score == pytest.approx(0.0)
    assert hit is False


def test_turned_ninety_degrees_scores_full_deviation(scoring, track_state):
    for i in range(scoring.history_len):
        kp = standing_pose_kp_xy()
        kp[KP_LEFT_SHOULDER] = (130.0, 120.0)   # shoulder line now vertical
        kp[KP_RIGHT_SHOULDER] = (130.0, 160.0)
        track_state.push(make_observation(ts=float(i), kp_xy=kp), scoring)
    score, hit = score_off_task(track_state.history, scoring)
    assert score == pytest.approx(1.0)
    assert hit is True


def test_off_task_reference_angle_can_be_overridden_per_camera(scoring, track_state):
    for i in range(scoring.history_len):
        track_state.push(make_observation(ts=float(i)), scoring)
    facing = score_off_task(track_state.history, scoring, reference_angle_deg=180.0)
    rotated = score_off_task(track_state.history, scoring, reference_angle_deg=90.0)
    assert facing[0] == pytest.approx(0.0)
    assert rotated[0] == pytest.approx(1.0)


# -- form-error vocabulary ----------------------------------------------------


def test_form_code_scores_its_configured_weight(scoring):
    assert score_form_codes(["knee_valgus"], scoring) == scoring.form_error_vocab["knee_valgus"]


def test_highest_scoring_code_wins(scoring):
    codes = ["insufficient_depth", "knee_valgus"]
    assert score_form_codes(codes, scoring) == max(scoring.form_error_vocab[c] for c in codes)


def test_no_codes_scores_zero(scoring):
    assert score_form_codes([], scoring) == 0.0


def test_a_code_outside_the_vocabulary_scores_zero(scoring):
    """`argus.ingest.protocol` rejects this before it ever reaches the scorer;
    this module still defaults safely if it somehow arrived anyway."""
    assert score_form_codes(["not_a_real_code"], scoring) == 0.0


def test_form_codes_are_scored_on_push(scoring, track_state):
    track_state.push(make_observation(ts=0.0, form_reason_codes=("knee_valgus",)), scoring)
    assert track_state.last_form_error_score == scoring.form_error_vocab["knee_valgus"]


def test_only_the_latest_observations_codes_apply(scoring, track_state):
    """Mirrors the old VLM-sample cadence: a code is scored per observation,
    not accumulated -- a trainee who corrects their form stops being flagged."""
    track_state.push(make_observation(ts=0.0, form_reason_codes=("knee_valgus",)), scoring)
    track_state.push(make_observation(ts=1.0, form_reason_codes=()), scoring)
    assert track_state.last_form_error_score == 0.0


# -- combination ------------------------------------------------------------


def test_reason_codes_explain_the_score(scoring, track_state):
    """Stillness + occlusion + off-task + a form-error code, stacked, must
    together clear the alert threshold -- each reason contributes its own
    weight, and the combination is what an instructor's alert explains."""
    for i in range(scoring.history_len):
        kp = standing_pose_kp_xy()
        kp[KP_LEFT_SHOULDER] = (130.0, 120.0)  # shoulder line rotated: off-task
        kp[KP_RIGHT_SHOULDER] = (130.0, 160.0)
        conf = [0.9] * 17
        for idx in (KP_NOSE, KP_LEFT_WRIST, KP_RIGHT_WRIST):
            conf[idx] = 0.05
        codes = ("knee_valgus",) if i == scoring.history_len - 1 else ()
        track_state.push(
            make_observation(ts=float(i), kp_xy=kp, kp_conf=conf, form_reason_codes=codes), scoring
        )

    record = compute_triage("t0", track_state, 1.0, scoring)
    assert REASON_STILLNESS in record.reason_codes
    assert REASON_OCCLUSION in record.reason_codes
    assert REASON_OFF_TASK in record.reason_codes
    assert REASON_FORM_ERROR in record.reason_codes
    assert REASON_FALL not in record.reason_codes
    assert record.score >= scoring.alert_threshold


def test_score_is_bounded_by_the_weight_sum(scoring, track_state):
    for i in range(scoring.history_len):
        track_state.push(make_observation(ts=float(i)), scoring)
    record = compute_triage("t0", track_state, 1.0, scoring)
    assert 0.0 <= record.score <= 1.0


# -- use_case dispatch -------------------------------------------------------
#
# `compute_triage` is a thin dispatcher over `track.use_case` (see
# `docs/PROTOCOL.md`); "fitness" is the only scorer registered today.


def test_known_use_cases_matches_the_scorer_registry(scoring):
    """`argus.config.SessionConfig` validates `[session] use_case` against
    this; it must name exactly what `compute_triage` can actually dispatch
    to, not a hand-maintained list that could drift from it."""
    known = known_use_cases()
    assert known == {"fitness", "welding"}
    for use_case in known:
        track = TrackState(history_len=scoring.history_len)
        track.use_case = use_case
        compute_triage("t0", track, 1.0, scoring)  # must not raise KeyError


def test_compute_triage_dispatches_to_the_fitness_scorer(scoring, track_state):
    for i in range(scoring.history_len):
        track_state.push(make_observation(ts=float(i)), scoring)
    assert track_state.use_case == "fitness"
    assert compute_triage("t0", track_state, 1.0, scoring) == compute_triage_fitness(
        "t0", track_state, 1.0, scoring
    )


def test_compute_triage_dispatches_to_the_welding_scorer(scoring):
    from argus.triage import FrameObservation, compute_triage_welding

    track = TrackState(history_len=scoring.history_len)
    track.push(FrameObservation(ts=1.0, use_case="welding", payload={"torch_angle_deg": 40.0}), scoring)
    assert track.use_case == "welding"
    assert compute_triage("t0", track, 1.0, scoring) == compute_triage_welding(
        "t0", track, 1.0, scoring
    )


def test_welding_scorer_is_always_neutral_regardless_of_payload(scoring):
    """No welding classifier exists to define what a bad weld looks like, so
    this scorer must not invent a threshold — it always reports 0.0 with no
    reason codes, whatever the payload says."""
    from argus.triage import FrameObservation, compute_triage_welding

    track = TrackState(history_len=scoring.history_len)
    for angle in (0.0, 45.0, 179.0):
        track.push(FrameObservation(ts=angle, use_case="welding", payload={"torch_angle_deg": angle}), scoring)
    record = compute_triage_welding("t0", track, 1.0, scoring)
    assert record.score == 0.0
    assert record.reason_codes == ()


def test_compute_triage_rejects_a_track_with_no_registered_scorer(scoring, track_state):
    track_state.use_case = "nursing"
    with pytest.raises(KeyError):
        compute_triage("t0", track_state, 1.0, scoring)


def test_rank_is_descending_with_id_tiebreak(scoring):
    tracks = {}
    for name in ("t_b", "t_a", "t_c"):
        state = TrackState(history_len=scoring.history_len)
        for i in range(scoring.history_len):
            state.push(make_observation(ts=float(i)), scoring)
        tracks[name] = state
    records = rank_trainees(tracks, 1.0, scoring)
    scores = [r.score for r in records]
    assert scores == sorted(scores, reverse=True)
    # identical scores => stable alphabetical order
    assert [r.trainee_id for r in records] == ["t_a", "t_b", "t_c"]


def test_needs_instructor_applies_the_threshold(scoring):
    from argus.triage import TriageRecord

    records = [
        TriageRecord("hi", scoring.alert_threshold, (), 0.0),
        TriageRecord("lo", scoring.alert_threshold - 0.01, (), 0.0),
    ]
    assert [r.trainee_id for r in needs_instructor(records, scoring)] == ["hi"]


def test_records_are_immutable():
    from argus.triage import TriageRecord

    record = TriageRecord("t0", 0.5, ("possible_fall",), 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.score = 0.9  # type: ignore[misc]


def test_retuning_weights_changes_the_score_without_touching_code(scoring, track_state):
    for i in range(scoring.history_len):
        track_state.push(make_observation(ts=float(i)), scoring)
    baseline = compute_triage("t0", track_state, 1.0, scoring).score

    retuned = dataclasses.replace(
        scoring,
        weights={"fall": 0.0, "stillness": 1.0, "occlusion": 0.0,
                 "off_task": 0.0, "form_error": 0.0},
    )
    assert compute_triage("t0", track_state, 1.0, retuned).score != baseline
