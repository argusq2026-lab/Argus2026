"""Shared fixtures.

The COCO-17 pose fixtures below are carried over from the prototype's suite,
with one correction: a camera-facing trainee has their *left* shoulder at the
larger image x, because COCO labels joints from the subject's perspective. The
prototype's baseline had it the other way round, which is why its
`off_task_reference_angle_deg` default of 0.0 looked correct in tests and would
have flagged every attentive trainee on real keypoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from argus.config import ArgusConfig, ScoringConfig, load_config  # noqa: E402
from argus.triage import FrameObservation, TrackState  # noqa: E402


@pytest.fixture(scope="session")
def default_config() -> ArgusConfig:
    return load_config()


@pytest.fixture
def scoring(default_config: ArgusConfig) -> ScoringConfig:
    return default_config.scoring


def standing_pose_kp_xy(torso_y: float = 140.0) -> list[tuple[float, float]]:
    """An upright, camera-facing COCO-17 pose.

    The shoulder line is level and the subject's left shoulder sits at the
    larger x, so `score_off_task` reads 180 degrees -- the config's
    "facing the station" reference. Override individual indices for a rotated
    or dropped pose; do not edit this baseline.
    """
    kp = [(130.0, 100.0)] * 17
    kp[5] = (150.0, torso_y)          # left shoulder  (image right)
    kp[6] = (110.0, torso_y)          # right shoulder (image left)
    kp[9] = (160.0, torso_y + 40.0)   # left wrist
    kp[10] = (100.0, torso_y + 40.0)  # right wrist
    kp[11] = (145.0, torso_y + 60.0)  # left hip
    kp[12] = (115.0, torso_y + 60.0)  # right hip
    return kp


def make_observation(
    ts: float,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 160.0, 240.0),
    kp_xy: list[tuple[float, float]] | None = None,
    kp_conf: list[float] | None = None,
    form_reason_codes: tuple[str, ...] = (),
    torso_y: float = 140.0,
    exercise: str | None = None,
) -> FrameObservation:
    """A well-formed 17-keypoint observation, fully confident by default."""
    return FrameObservation(
        ts=ts,
        bbox_xyxy=bbox,
        keypoints_xy=kp_xy if kp_xy is not None else standing_pose_kp_xy(torso_y),
        keypoints_conf=kp_conf if kp_conf is not None else [0.9] * 17,
        form_reason_codes=form_reason_codes,
        exercise=exercise,
    )


def plank_pose_kp_xy() -> list[tuple[float, float]]:
    """A horizontal, motionless COCO-17 pose — a correct plank.

    Deliberately the shape the standing-trainee features misread: the torso
    runs along x rather than y, so the shoulder line is far from the
    station-facing reference, and the bounding box in `make_plank_observation`
    is wider than it is tall, which is the `fall` feature's aspect-flip
    trigger. Both are correct readings of the geometry and wrong readings of
    the situation, which is what `[scoring.exercise_weights.plank]` exists for.
    """
    kp = [(0.20, 0.42)] * 17
    kp[5], kp[6] = (0.30, 0.45), (0.30, 0.55)     # shoulders
    kp[7], kp[8] = (0.26, 0.55), (0.26, 0.65)     # elbows
    kp[9], kp[10] = (0.24, 0.60), (0.24, 0.70)    # wrists
    kp[11], kp[12] = (0.55, 0.47), (0.55, 0.57)   # hips
    kp[13], kp[14] = (0.70, 0.48), (0.70, 0.58)   # knees
    kp[15], kp[16] = (0.85, 0.49), (0.85, 0.59)   # ankles
    return kp


def make_plank_observation(
    ts: float,
    form_reason_codes: tuple[str, ...] = (),
    exercise: str | None = "plank",
) -> FrameObservation:
    """One tick of a held plank. Wide, short box; horizontal, static pose."""
    return FrameObservation(
        ts=ts,
        bbox_xyxy=(0.18, 0.40, 0.88, 0.62),
        keypoints_xy=plank_pose_kp_xy(),
        keypoints_conf=[0.9] * 17,
        form_reason_codes=form_reason_codes,
        exercise=exercise,
    )


def make_bicep_observation(
    ts: float,
    form_reason_codes: tuple[str, ...] = (),
    exercise: str | None = "bicep",
) -> FrameObservation:
    """One tick of a bicep curl: an ordinary standing, camera-facing pose.

    Unlike plank, nothing about a correct curl's geometry misreads `fall` or
    `off_task` (docs/VALIDATION.md), so the default standing pose stands in
    directly -- these tests exercise the weight profile, not a geometry claim.
    """
    return make_observation(ts, form_reason_codes=form_reason_codes, exercise=exercise)


def make_lunge_observation(
    ts: float,
    form_reason_codes: tuple[str, ...] = (),
    exercise: str | None = "lunge",
) -> FrameObservation:
    """One tick of a lunge: an ordinary standing, camera-facing pose.

    `[scoring.exercise_weights.lunge]` zeroes `fall` because a full-depth
    lunge viewed side-on plausibly reads bbox-wider-than-tall
    (docs/VALIDATION.md), not because this fixture's geometry demonstrates it
    -- no footage exists to model that pose from. These tests exercise the
    weight profile lookup, the same as bicep's.
    """
    return make_observation(ts, form_reason_codes=form_reason_codes, exercise=exercise)


@pytest.fixture
def track_state(scoring: ScoringConfig) -> TrackState:
    return TrackState(history_len=scoring.history_len)
