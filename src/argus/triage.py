"""Deterministic triage scorer for Argus.

Pure functions of numeric pose/bbox/form-classification history — no model
calls, no randomness, no wall-clock reads other than the caller-supplied
timestamp. The same input history always produces the same rank. Every phone
in front of a trainee runs its own on-device pose and form/exercise
classifier and streams only structured numeric results here (see
`argus.ingest` and `docs/PROTOCOL.md`); nothing upstream of this module is
free text or imagery, so there is nothing non-deterministic left to
neutralize — unlike the prototype's VLM-caption design, this scorer's inputs
were never anything but numbers.

Privacy by wiring: nothing in this module ever holds a frame, a crop, or free
text — only numeric keypoints, boxes, and a fixed-vocabulary form-error score.
That is what makes :class:`TriageRecord` safe to cross the alert boundary in
:mod:`argus.alerts`.

The five-feature algorithm is carried over verbatim from the Argus prototype;
the weights, thresholds, vocabulary, and history length arrive as a
:class:`~argus.config.ScoringConfig` argument instead of being module
constants, so tuning is a config edit rather than a code edit.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from argus.config import ScoringConfig

# Pose keypoint indices this module cares about, in COCO-17 order. Only
# indices are ever used -- never pixel data. The phone's on-device pose model
# is responsible for delivering COCO-17 regardless of what layout it emits
# internally (see docs/PROTOCOL.md).
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
HAND_KEYPOINTS = (KP_LEFT_WRIST, KP_RIGHT_WRIST)
NUM_KEYPOINTS = 17

REASON_FALL = "possible_fall"
REASON_STILLNESS = "prolonged_stillness"
REASON_OCCLUSION = "hands_face_occluded"
REASON_OFF_TASK = "off_task_orientation"
REASON_FORM_ERROR = "form_error"


@dataclass
class FrameObservation:
    """One tick's numeric observation for one trainee. No pixels, no text.

    `form_reason_codes` are the phone's own on-device form/exercise
    classifier output — a closed vocabulary agreed with `ingest.form_error_
    vocab`, not free text. The ingest layer rejects a code outside that
    vocabulary before it ever reaches this dataclass (see
    `argus.ingest.protocol`); this module only ever scores what it is given.

    `exercise` selects the weight profile (`ScoringConfig.weights_for`).
    Unlike a form-error code it is *not* a closed vocabulary — an exercise
    with no configured profile scores on the default weights rather than
    being rejected, because the protocol has always described this field as a
    free-form label.

    `rep_count` and `form_ok` are the protocol's purely informational fields.
    **Nothing in this module reads them**, and nothing may start to: beyond
    `exercise` above, the rank is a pure function of the numeric pose/box
    history plus the closed-vocabulary form codes, and scoring a
    phone-maintained rep counter would make it a function of an unauditable
    device-side counter. They exist so the trainer console can display them
    (see `argus.outputs.StationView`), and for no other reason.
    """

    ts: float
    bbox_xyxy: tuple[float, float, float, float]
    keypoints_xy: list[tuple[float, float]]  # len 17, COCO order
    keypoints_conf: list[float]  # len 17
    form_reason_codes: tuple[str, ...] = ()
    exercise: str | None = None
    #: Display-only. See the note above — the scorer must never read these.
    rep_count: int | None = None
    form_ok: bool | None = None


@dataclass
class TrackState:
    """Rolling per-trainee history. Owned by the caller, one per trainee id."""

    history_len: int = 30
    history: deque[FrameObservation] = field(default_factory=deque)
    last_form_error_score: float = 0.0
    #: The exercise most recently reported, which selects the weight profile.
    #: Latest-wins rather than a majority over the window: a trainee who has
    #: just dropped into a plank should be scored as planking immediately, not
    #: after the history fills.
    last_exercise: str | None = None

    def __post_init__(self) -> None:
        # deque(maxlen=...) cannot be expressed as a field default that depends
        # on another field, so bind it here.
        if self.history.maxlen != self.history_len:
            self.history = deque(self.history, maxlen=self.history_len)

    def push(self, obs: FrameObservation, cfg: ScoringConfig) -> None:
        self.history.append(obs)
        self.last_form_error_score = score_form_codes(obs.form_reason_codes, cfg)
        self.last_exercise = obs.exercise


@dataclass(frozen=True)
class TriageRecord:
    """Redacted output — the ONLY thing allowed to cross the alert boundary.

    Frozen so a sink cannot mutate a record another sink has already seen, and
    so the four fields below are the complete, closed set of what leaves the
    perception layer. There is deliberately no field that could carry a frame,
    a crop, a caption, or a bounding box.
    """

    trainee_id: str
    score: float
    reason_codes: tuple[str, ...]
    ts: float


def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _bbox_aspect(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    h = max(y1 - y0, 1e-6)
    w = max(x1 - x0, 1e-6)
    return w / h


def score_fall(
    history: Iterable[FrameObservation], cfg: ScoringConfig
) -> tuple[float, bool]:
    """Sudden vertical drop of the hip/shoulder centroid + bbox aspect flip.

    Returns (score in [0,1], triggered).
    """
    obs_list = list(history)
    if len(obs_list) < 2:
        return 0.0, False

    recent = obs_list[-6:]  # ~0.4 s window at 15 FPS
    first, last = recent[0], recent[-1]

    def torso_y(obs: FrameObservation) -> float | None:
        idxs = (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)
        ys = [
            obs.keypoints_xy[i][1]
            for i in idxs
            if obs.keypoints_conf[i] >= cfg.keypoint_conf_threshold
        ]
        return sum(ys) / len(ys) if ys else None

    y0, y1 = torso_y(first), torso_y(last)
    bbox_h_first = max(first.bbox_xyxy[3] - first.bbox_xyxy[1], 1e-6)

    vertical_drop_ratio = 0.0
    if y0 is not None and y1 is not None:
        vertical_drop_ratio = max(0.0, (y1 - y0) / bbox_h_first)

    aspect_flip = _bbox_aspect(last.bbox_xyxy) > 1.3  # wider than tall

    score = min(1.0, vertical_drop_ratio / 0.6)  # a 60%+ body-height drop -> 1.0
    if aspect_flip:
        score = min(1.0, score + 0.3)

    return score, score >= 0.6


def score_stillness(
    history: Iterable[FrameObservation], cfg: ScoringConfig
) -> tuple[float, bool]:
    """Fraction of the history window with near-zero centroid displacement."""
    obs_list = list(history)
    if len(obs_list) < 2:
        return 0.0, False

    still_frames = 0
    for prev, cur in zip(obs_list, obs_list[1:]):
        cx0, cy0 = _centroid(prev.bbox_xyxy)
        cx1, cy1 = _centroid(cur.bbox_xyxy)
        disp = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
        if disp < cfg.stillness_motion_threshold_frac:
            still_frames += 1

    fraction = still_frames / max(len(obs_list) - 1, 1)
    triggered = fraction >= 0.9 and len(obs_list) >= cfg.history_len
    return fraction, triggered


def score_occlusion(
    history: Iterable[FrameObservation], cfg: ScoringConfig
) -> tuple[float, bool]:
    """Fraction of the window where both hands *and* face are low-confidence."""
    obs_list = list(history)
    if not obs_list:
        return 0.0, False

    occluded_frames = 0
    for obs in obs_list:
        hands_visible = any(
            obs.keypoints_conf[i] >= cfg.keypoint_conf_threshold for i in HAND_KEYPOINTS
        )
        face_visible = obs.keypoints_conf[KP_NOSE] >= cfg.keypoint_conf_threshold
        if not hands_visible and not face_visible:
            occluded_frames += 1

    fraction = occluded_frames / len(obs_list)
    triggered = fraction >= 0.8 and len(obs_list) >= cfg.history_len
    return fraction, triggered


def score_off_task(
    history: Iterable[FrameObservation],
    cfg: ScoringConfig,
    reference_angle_deg: float | None = None,
) -> tuple[float, bool]:
    """Shoulder-line orientation deviation from the expected station-facing angle.

    `reference_angle_deg` overrides the config default so a camera whose
    stations face a different way can be corrected per-source.
    """
    reference = (
        cfg.off_task_reference_angle_deg
        if reference_angle_deg is None
        else reference_angle_deg
    )
    obs_list = list(history)

    deviations = []
    for obs in obs_list:
        ls = obs.keypoints_xy[KP_LEFT_SHOULDER]
        rs = obs.keypoints_xy[KP_RIGHT_SHOULDER]
        lc = obs.keypoints_conf[KP_LEFT_SHOULDER]
        rc = obs.keypoints_conf[KP_RIGHT_SHOULDER]
        if lc < cfg.keypoint_conf_threshold or rc < cfg.keypoint_conf_threshold:
            continue
        angle = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
        deviations.append(abs(((angle - reference + 180) % 360) - 180))

    if not deviations:
        return 0.0, False

    mean_dev = sum(deviations) / len(deviations)
    score = min(1.0, mean_dev / 90.0)
    triggered = score >= 0.5 and len(deviations) >= cfg.history_len // 2
    return score, triggered


def score_form_codes(codes: Iterable[str], cfg: ScoringConfig) -> float:
    """Highest weight among the phone-reported form-error codes present.

    `codes` is the phone's own closed-vocabulary classifier output, not free
    text — there is no matching left to do, only a lookup. A code the config
    does not recognise contributes 0 here (an evolving vocabulary should not
    break scoring); the ingest layer is where an unrecognised code is treated
    as a protocol error instead (see `argus.ingest.protocol`).
    """
    hits = [cfg.form_error_vocab[c] for c in codes if c in cfg.form_error_vocab]
    return max(hits) if hits else 0.0


def compute_triage(
    trainee_id: str,
    track: TrackState,
    ts: float,
    cfg: ScoringConfig,
    reference_angle_deg: float | None = None,
) -> TriageRecord:
    """Combine the deterministic features into one ranked, explainable record.

    The weight vector is chosen by the trainee's current exercise, so a
    feature a movement makes meaningless can be zeroed for that movement
    alone. A zero weight also suppresses the feature's reason code: a signal
    that contributes nothing to the score must not appear on the trainer's
    dashboard as a reason, or the explanation stops matching the number it
    claims to explain.
    """
    fall, fall_hit = score_fall(track.history, cfg)
    still, still_hit = score_stillness(track.history, cfg)
    occl, occl_hit = score_occlusion(track.history, cfg)
    off_task, off_task_hit = score_off_task(track.history, cfg, reference_angle_deg)
    form_error = track.last_form_error_score

    w = cfg.weights_for(track.last_exercise)
    score = (
        w["fall"] * fall
        + w["stillness"] * still
        + w["occlusion"] * occl
        + w["off_task"] * off_task
        + w["form_error"] * form_error
    )

    reasons: list[str] = []
    if fall_hit and w["fall"] > 0:
        reasons.append(REASON_FALL)
    if still_hit and w["stillness"] > 0:
        reasons.append(REASON_STILLNESS)
    if occl_hit and w["occlusion"] > 0:
        reasons.append(REASON_OCCLUSION)
    if off_task_hit and w["off_task"] > 0:
        reasons.append(REASON_OFF_TASK)
    if form_error >= 0.5 and w["form_error"] > 0:
        reasons.append(REASON_FORM_ERROR)

    return TriageRecord(
        trainee_id=trainee_id,
        score=round(score, 4),
        reason_codes=tuple(reasons),
        ts=ts,
    )


def rank_trainees(
    tracks: dict[str, TrackState],
    ts: float,
    cfg: ScoringConfig,
    top_k: int | None = None,
    reference_angles: dict[str, float] | None = None,
) -> list[TriageRecord]:
    """Deterministic descending rank; ties broken by trainee_id for stability.

    `reference_angles` maps trainee_id -> the station-facing angle of the
    phone that trainee is streaming from, so the merged rank across every
    connected station still scores off-task orientation per-station.
    """
    angles = reference_angles or {}
    records = [
        compute_triage(tid, track, ts, cfg, angles.get(tid))
        for tid, track in tracks.items()
    ]
    records.sort(key=lambda r: (-r.score, r.trainee_id))
    return records[:top_k] if top_k is not None else records


def needs_instructor(
    records: list[TriageRecord], cfg: ScoringConfig
) -> list[TriageRecord]:
    return [r for r in records if r.score >= cfg.alert_threshold]
