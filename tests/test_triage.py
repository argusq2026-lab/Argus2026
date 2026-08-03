"""Scoring functions. Carried over from the prototype suite, config-driven."""

from __future__ import annotations

import dataclasses
from collections import deque

import pytest

from argus.triage import (
    REASON_FALL,
    REASON_OCCLUSION,
    REASON_OFF_TASK,
    REASON_STILLNESS,
    REASON_VLM,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    TrackState,
    compute_triage,
    needs_instructor,
    rank_trainees,
    score_fall,
    score_occlusion,
    score_off_task,
    score_stillness,
    score_vlm_caption,
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


# -- VLM vocabulary ---------------------------------------------------------


def test_vocabulary_match_is_case_insensitive(scoring):
    assert score_vlm_caption("A trainee has FALLEN near the press", scoring) == 1.0


def test_highest_scoring_vocabulary_hit_wins(scoring):
    assert score_vlm_caption("no gloves and smoke visible", scoring) == 1.0


def test_free_text_without_vocabulary_scores_zero(scoring):
    assert score_vlm_caption("a person stands calmly by a workbench", scoring) == 0.0


def test_caption_is_scored_but_never_retained(scoring, track_state):
    track_state.apply_caption("smoke near the bench", scoring)
    assert track_state.last_vlm_anomaly_score == 1.0
    assert not hasattr(track_state, "caption")
    assert all(obs.vlm_caption is None for obs in track_state.history)


# -- combination ------------------------------------------------------------


def test_reason_codes_explain_the_score(scoring, track_state):
    for i in range(scoring.history_len):
        conf = [0.9] * 17
        for idx in (KP_NOSE, KP_LEFT_WRIST, KP_RIGHT_WRIST):
            conf[idx] = 0.05
        track_state.push(make_observation(ts=float(i), kp_conf=conf), scoring)
    track_state.apply_caption("trainee unresponsive", scoring)

    record = compute_triage("t0", track_state, 1.0, scoring)
    assert REASON_STILLNESS in record.reason_codes
    assert REASON_OCCLUSION in record.reason_codes
    assert REASON_VLM in record.reason_codes
    assert REASON_FALL not in record.reason_codes
    assert REASON_OFF_TASK not in record.reason_codes
    assert record.score >= scoring.alert_threshold


def test_score_is_bounded_by_the_weight_sum(scoring, track_state):
    for i in range(scoring.history_len):
        track_state.push(make_observation(ts=float(i)), scoring)
    record = compute_triage("t0", track_state, 1.0, scoring)
    assert 0.0 <= record.score <= 1.0


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
                 "off_task": 0.0, "vlm_anomaly": 0.0},
    )
    assert compute_triage("t0", track_state, 1.0, retuned).score != baseline
