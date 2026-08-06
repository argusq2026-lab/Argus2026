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
#: Not "a rep was wrong" but "they cannot hold the movement". Distinct from
#: `form_error` because they call for different things from an instructor:
#: one is a cue, the other is stop-and-reset.
REASON_PERSISTENT_FORM = "persistent_form_fault"


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

    `rep_count` is read by `SessionMetrics` — and by nothing else. It counts
    the trainee's work so a form fault can be expressed as a *rate* ("3 of 14
    reps") rather than as whichever frame happened to arrive last, and a plank
    with no reps at all is measured in seconds instead. The line that has not
    moved is the one that matters: **it never enters the score that fires an
    alert.** `compute_triage` does not read it, so a phone with a broken
    counter can misorder the calm end of the instructor's queue and cannot
    cause or suppress an alert. That is a real dependency on a device-side
    value, bounded to display and ordering, and it is stated rather than
    hidden — the closed-vocabulary `form_reason_codes` remain the only thing
    the phone says that the score itself believes.

    `form_ok` is read by nothing. The server derives whether form is flagged
    from `form_reason_codes` being non-empty, because those can be checked
    against a vocabulary and a bare boolean cannot.
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
class SessionMetrics:
    """What a trainee has *done* this session, as against what they are doing
    in the last two seconds.

    The triage score is instantaneous by design — it answers "who needs a
    human right now", and a fall must not be averaged away. But it is a poor
    thing to read: it is computed from a ~2 s window and recomputed every
    `rank_interval_s`, so an instructor watching the console sees a number
    that moves constantly and cannot be compared with the same trainee's
    number a minute ago. This is the other half: a rolling account of the
    session that changes slowly, on purpose.

    Work is measured in whatever unit the exercise actually has. Reps for a
    squat or a curl; **seconds for a plank**, which has no reps at all and
    whose whole quality is how long it was held well. A single `fault_rate`
    over "flagged work / observed work" then means the same thing for both.

    Durations come from the phone's own `ts`, not the server's clock. That
    looks like it contradicts "phone and laptop clocks are not synchronised",
    and does not: a *delta between two frames from one phone* is exactly the
    measurement that phone's clock is good for. What is forbidden is
    comparing timestamps across devices, and nothing here does.
    """

    frames: int = 0
    #: Seconds of this trainee actually being observed, gaps excluded.
    active_s: float = 0.0
    reps: int = 0
    #: Reps during which the phone reported at least one form-error code.
    reps_flagged: int = 0
    hold_s: float = 0.0
    hold_flagged_s: float = 0.0
    #: How many frames carried each code. A tally, not a rate — the rate
    #: needs a denominator and `fault_rate` is where that judgement lives.
    code_counts: dict[str, int] = field(default_factory=dict)
    #: Time-decayed mean of the instantaneous score: the number an instructor
    #: reads. Advanced by `observe_score` on the rank tick, never here.
    rolling_score: float = 0.0
    #: The worst instant the session has seen. A rolling mean forgets, and
    #: "this trainee was briefly in real trouble" should survive being
    #: forgotten by the mean.
    peak_score: float = 0.0
    #: Decayed share of observed time carrying *any* form code, and the
    #: decayed mean severity over the same window. Together they answer the
    #: question an instructor actually asks — "can this person hold the
    #: movement, or not?" — which no single frame can. Measured on time
    #: rather than on reps deliberately: the phone's rep counter must stay out
    #: of anything that can raise an alert.
    form_fault_fraction: float = 0.0
    form_severity_mean: float = 0.0

    # Cursors. Leading underscore because they are bookkeeping rather than
    # anything a console should render.
    _last_ts: float | None = None
    _last_rep_count: int | None = None
    _rep_flagged: bool = False
    _rolling_ts: float | None = None
    _persist_seeded: bool = False

    def observe(self, obs: FrameObservation, cfg: ScoringConfig) -> None:
        """Fold one observation into the session's running account."""
        self.frames += 1
        flagged = bool(obs.form_reason_codes)
        for code in obs.form_reason_codes:
            self.code_counts[code] = self.code_counts.get(code, 0) + 1

        # A gap longer than a plausible frame interval is a reconnect, a
        # backgrounded app, or a phone that was put down — not work done. It
        # is skipped rather than accumulated, so "held for 4 minutes" cannot
        # be earned by a station that was away for three of them.
        dt = 0.0
        if self._last_ts is not None:
            delta = obs.ts - self._last_ts
            if 0.0 < delta <= cfg.max_frame_gap_s:
                dt = delta
        self._last_ts = obs.ts
        self.active_s += dt

        # "Can this person hold the movement?" -- decayed over its own,
        # longer half-life than the urgency mean, so a trainee who fixes their
        # form stops being flagged within a minute or two rather than carrying
        # a bad first set for the rest of the session.
        severity = score_form_codes(obs.form_reason_codes, cfg)
        if not self._persist_seeded:
            self.form_fault_fraction = 1.0 if flagged else 0.0
            self.form_severity_mean = severity
            self._persist_seeded = True
        elif dt > 0:
            alpha = 1.0 - 0.5 ** (dt / cfg.form_persistence_half_life_s)
            self.form_fault_fraction += alpha * ((1.0 if flagged else 0.0) - self.form_fault_fraction)
            self.form_severity_mean += alpha * (severity - self.form_severity_mean)

        if obs.rep_count is None:
            # No rep counter: a held exercise, where work is time.
            self.hold_s += dt
            if flagged:
                self.hold_flagged_s += dt
            return

        if flagged:
            self._rep_flagged = True
        if self._last_rep_count is not None:
            advanced = obs.rep_count - self._last_rep_count
            if advanced < 0:
                # The counter went backwards: the phone started a new set.
                # Credit the reps of the new set, not a negative count.
                advanced = obs.rep_count
            if advanced > 0:
                self.reps += advanced
                # One flag per completed rep. If several reps landed between
                # frames we saw, the ones we did not see are not accused.
                if self._rep_flagged:
                    self.reps_flagged += 1
                self._rep_flagged = False
        self._last_rep_count = obs.rep_count

    def observe_score(self, score: float, ts: float, half_life_s: float) -> None:
        """Advance the rolling mean to `ts`. Called once per rank tick.

        Deliberately separate from `observe`, and deliberately not inside
        `compute_triage`: the scorer stays a pure function of history, which
        is the property `tests/test_determinism.py` exists to protect. The
        stateful part is here, where it can be seen and tested on its own.

        Decays by half-life rather than by tick count, so the number means the
        same thing whether ticks arrive every 0.5 s or every 2 s.
        """
        self.peak_score = max(self.peak_score, score)
        if self._rolling_ts is None:
            self.rolling_score = score
        else:
            dt = max(0.0, ts - self._rolling_ts)
            alpha = 1.0 - 0.5 ** (dt / half_life_s)
            self.rolling_score += alpha * (score - self.rolling_score)
        self._rolling_ts = ts

    def persistent_form_score(self, cfg: ScoringConfig) -> float:
        """How bad it is that this trainee *cannot hold the movement*.

        The instructor's actual question is not "is this frame wrong" but "can
        they do it at all", and the weighted sum could not answer it. Form is
        weighted 0.15 in the default vector, so a trainee getting **every
        single squat rep wrong** produced 0.15 x 0.8 = 0.12 against a 0.5
        threshold: no matter how badly or how long they struggled, nobody was
        ever sent. That is not a tuning nit, it is the coaching case failing
        silently, and the profile weight cannot fix it — raising `form_error`
        enough to alert would make one bad frame outrank a fall.

        So persistence is scored on its own terms and joins the score as a
        floor (`max`), not as another weighted term. Once a trainee has been
        wrong for the *majority* of a meaningful stretch, they are escalated
        at the severity of the fault they keep making — `hips_sagging` at 0.8
        alerts, a 0.4 `incomplete_lockout` still does not, which is right.

        Dividing the mean severity by the fault fraction recovers "how bad it
        is *when* it is wrong", so being wrong 60% of the time with a 0.8 fault
        escalates at 0.8 rather than at 0.48 — being wrong most of the time is
        the trigger, and the severity is the fault's, not an average diluted by
        the good reps.
        """
        if self.active_s < cfg.form_persistence_min_s:
            return 0.0
        if self.form_fault_fraction < cfg.form_persistence_threshold:
            return 0.0
        return min(1.0, self.form_severity_mean / self.form_fault_fraction)

    def fault_rate(self, cfg: ScoringConfig) -> float | None:
        """Flagged work over observed work, or `None` if too little is done.

        `None` rather than 0.0 or 1.0 on thin evidence: one bad rep out of one
        is not a 100% fault rate, it is not yet a rate at all, and showing it
        as one would put a trainee at the top of a queue on a single frame.
        """
        if self.reps >= cfg.min_reps_for_fault_rate:
            return self.reps_flagged / self.reps
        if self.hold_s >= cfg.min_hold_s_for_fault_rate:
            return self.hold_flagged_s / self.hold_s
        return None


@dataclass
class TrackState:
    """Rolling per-trainee history. Owned by the caller, one per trainee id."""

    history_len: int = 30
    history: deque[FrameObservation] = field(default_factory=deque)
    last_form_error_score: float = 0.0
    #: The session-long account behind the instant score. See `SessionMetrics`.
    session: SessionMetrics = field(default_factory=SessionMetrics)
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
        self.session.observe(obs, cfg)


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

    # A trainee who cannot hold the movement is escalated on their own terms
    # rather than through the weighted sum, which cannot express it -- see
    # `SessionMetrics.persistent_form_score`. A floor, not a sixth term: it
    # can only raise a score, never dilute the four features that answer
    # "is this person in danger right now".
    persistent = (
        track.session.persistent_form_score(cfg) if w["form_error"] > 0 else 0.0
    )
    score = max(score, persistent)

    reasons: list[str] = []
    if fall_hit and w["fall"] > 0:
        reasons.append(REASON_FALL)
    if still_hit and w["stillness"] > 0:
        reasons.append(REASON_STILLNESS)
    if occl_hit and w["occlusion"] > 0:
        reasons.append(REASON_OCCLUSION)
    if off_task_hit and w["off_task"] > 0:
        reasons.append(REASON_OFF_TASK)
    # One or the other, never both: "cannot hold the movement" already says
    # everything "this rep was wrong" would, and an instructor reading two
    # codes for one problem learns to skim them.
    if persistent > 0:
        reasons.append(REASON_PERSISTENT_FORM)
    elif form_error >= 0.5 and w["form_error"] > 0:
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
