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
from statistics import median, pstdev
from typing import Callable, Iterable

from argus.config import ScoringConfig

# Pose keypoint indices this module cares about, in COCO-17 order. Only
# indices are ever used -- never pixel data. The phone's on-device pose model
# is responsible for delivering COCO-17 regardless of what layout it emits
# internally (see docs/PROTOCOL.md).
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
HAND_KEYPOINTS = (KP_LEFT_WRIST, KP_RIGHT_WRIST)
NUM_KEYPOINTS = 17

#: One arm, shoulder -> elbow -> wrist. Nursing's CPR scorer reads the angle at
#: the elbow off these; a side-on camera sees one arm and not the other, so
#: they are tried in turn rather than averaged.
ARM_CHAINS = (
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
)

REASON_FALL = "possible_fall"
REASON_STILLNESS = "prolonged_stillness"
REASON_OCCLUSION = "hands_face_occluded"
REASON_OFF_TASK = "off_task_orientation"
REASON_FORM_ERROR = "form_error"
#: Not "a rep was wrong" but "they cannot hold the movement". Distinct from
#: `form_error` because they call for different things from an instructor:
#: one is a cue, the other is stop-and-reset.
REASON_PERSISTENT_FORM = "persistent_form_fault"

# -- nursing / CPR ------------------------------------------------------------
#
# Unlike fitness's thresholds, which `docs/VALIDATION.md` records as an
# unfitted hypothesis, the compression-rate band below is *external*: the AHA
# publishes 100-120/min for adult CPR, so this scorer is measuring against
# somebody else's number rather than one invented here. What remains
# unvalidated is whether a phone camera recovers that rate faithfully, which
# is a measurement question and is recorded as such.

REASON_CPR_RATE_SLOW = "cpr_rate_slow"
REASON_CPR_RATE_FAST = "cpr_rate_fast"
REASON_CPR_ARMS_BENT = "cpr_arms_bent"
REASON_CPR_ERRATIC = "cpr_cadence_erratic"


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
    #: `None` for a use case with no pose (e.g. today's welding placeholder,
    #: `compute_triage_welding` below) — only `compute_triage_fitness` and its
    #: four pose-based scorers ever read these three fields, and `use_case`
    #: dispatch guarantees a non-fitness track never reaches them.
    bbox_xyxy: tuple[float, float, float, float] | None = None
    keypoints_xy: list[tuple[float, float]] | None = None  # len 17, COCO order
    keypoints_conf: list[float] | None = None  # len 17
    form_reason_codes: tuple[str, ...] = ()
    exercise: str | None = None
    #: Display-only. See the note above — the scorer must never read these.
    rep_count: int | None = None
    form_ok: bool | None = None
    #: Which use case this observation belongs to (see `docs/PROTOCOL.md`).
    #: Selects the entry in `_SCORERS` that `compute_triage` dispatches to.
    #: Defaults to `"fitness"` because every other field on this dataclass —
    #: `bbox_xyxy`, `keypoints_xy`, `exercise` — is fitness's own payload
    #: shape; a non-fitness use case is expected to define its own
    #: observation type rather than force its data through these fields.
    use_case: str = "fitness"
    #: Which procedure a nursing station is performing — `"cpr"` today.
    #: Selects the entry in `_NURSING_PROCEDURES` exactly as `exercise`
    #: selects a fitness weight profile: nursing is the *use case*, CPR is one
    #: skill inside it, and collapsing the two would mean a new use case for
    #: every procedure a ward ever trains. `None` for any other use case, and
    #: for a nursing station that has not said which procedure it is running.
    procedure: str | None = None
    #: Opaque, use-case-owned data that isn't one of the fitness fields
    #: above. Fitness observations never set this (it is `{}`); it exists so
    #: a use case with no typed fields of its own yet — welding today — has
    #: somewhere to carry whatever its parser accepted, without forcing a
    #: dataclass field to be added here for every future use case's payload.
    payload: dict = field(default_factory=dict)


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
    #: Whether the phone currently has anyone in frame. False after an `idle`
    #: message. A station whose trainee has walked away still holds their last
    #: two seconds of pose, and scoring it would report `prolonged_stillness`
    #: about an empty rack — the history describes somebody who is not there.
    subject_present: bool = True
    #: The exercise most recently reported, which selects the weight profile.
    #: Latest-wins rather than a majority over the window: a trainee who has
    #: just dropped into a plank should be scored as planking immediately, not
    #: after the history fills.
    last_exercise: str | None = None
    #: The use case this track's observations belong to, latest-wins like
    #: `last_exercise`. Selects the entry in `_SCORERS` that `compute_triage`
    #: dispatches to. A track that has never seen an observation defaults to
    #: `"fitness"`, matching `FrameObservation.use_case`'s default.
    use_case: str = "fitness"
    #: The procedure most recently reported, latest-wins like `last_exercise`
    #: and for the same reason: a nursing station that switches from bagging
    #: to compressions should be scored as compressing from the next frame,
    #: not once a window has refilled.
    last_procedure: str | None = None

    def __post_init__(self) -> None:
        # deque(maxlen=...) cannot be expressed as a field default that depends
        # on another field, so bind it here.
        if self.history.maxlen != self.history_len:
            self.history = deque(self.history, maxlen=self.history_len)

    def push(self, obs: FrameObservation, cfg: ScoringConfig) -> None:
        self.history.append(obs)
        self.last_form_error_score = score_form_codes(obs.form_reason_codes, cfg)
        self.last_exercise = obs.exercise
        self.use_case = obs.use_case
        self.last_procedure = obs.procedure
        self.subject_present = True
        self.session.observe(obs, cfg)

    def note_idle(self) -> None:
        """The phone is watching an empty station. Nobody to score."""
        self.subject_present = False
        self.last_form_error_score = 0.0


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


def compute_triage_fitness(
    trainee_id: str,
    track: TrackState,
    ts: float,
    cfg: ScoringConfig,
    reference_angle_deg: float | None = None,
) -> TriageRecord:
    """Fitness's triage scorer: the five-feature algorithm above.

    The weight vector is chosen by the trainee's current exercise, so a
    feature a movement makes meaningless can be zeroed for that movement
    alone. A zero weight also suppresses the feature's reason code: a signal
    that contributes nothing to the score must not appear on the trainer's
    dashboard as a reason, or the explanation stops matching the number it
    claims to explain.
    """
    # An empty station is not a calm trainee, and it is not a trainee at all.
    # Its stale history would otherwise score `prolonged_stillness` about a
    # rack nobody is standing at. The console shows the state separately; what
    # matters here is that nothing is *asserted* about a person who is absent.
    if not track.subject_present:
        return TriageRecord(trainee_id=trainee_id, score=0.0, reason_codes=(), ts=ts)

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


def compute_triage_welding(
    trainee_id: str,
    track: TrackState,
    ts: float,
    cfg: ScoringConfig,
    reference_angle_deg: float | None = None,
) -> TriageRecord:
    """Welding's triage scorer: a placeholder, deliberately.

    There is no welding classifier and no welding data to fit thresholds
    against, unlike fitness's five features, which — even unvalidated
    against a real trainee (`docs/VALIDATION.md`) — are at least a stated
    hypothesis about what a fall or a stalled rep looks like numerically.
    Inventing a torch-angle or travel-speed threshold here would not be a
    smaller version of that hypothesis, it would be a number with nothing
    behind it presented on an instructor's screen as a signal. So this
    scorer asserts nothing: every welding station reports a flat 0.0 with no
    reason codes, regardless of what its `payload` carries. It exists to let
    a welding station connect, stream, and appear on the console end to end
    — proving the `use_case` wiring — without claiming to watch for
    anything until a real classifier defines what "wrong" means for a weld.
    """
    return TriageRecord(trainee_id=trainee_id, score=0.0, reason_codes=(), ts=ts)


# -- nursing: CPR compression quality ----------------------------------------
#
# The constants below are properties of the *measurement*, not knobs an
# instructor tunes -- the same distinction `_HELLO_TIMEOUT_S` draws in
# `argus.ingest.server`. What a ward might legitimately change (the target
# band, the window, how bent an elbow is too bent) lives in `ScoringConfig`.

#: Rates we will look for at all. Wider than the AHA target band on purpose:
#: a trainee compressing at 70/min must be told "70, too slow", not "no signal"
#: -- those are very different things for an instructor to act on.
_CPR_SEARCH_MIN_BPM = 50.0
_CPR_SEARCH_MAX_BPM = 180.0

#: Drift-removal window. Must span several compression periods, or the moving
#: average subtracts the compressions along with the drift.
_CPR_TREND_WINDOW_S = 2.0

#: How periodic the wrist must be before we will name a rate, as normalised
#: autocorrelation at the winning lag. Noise scores near zero. This is what
#: stops a rescuer standing still from being assigned a plausible-looking rate.
_CPR_MIN_PERIODICITY = 0.35

#: Smallest wrist excursion, as a fraction of frame height, that counts as a
#: compression rather than as camera shake.
_CPR_MIN_AMPLITUDE = 0.005

#: Samples per compression cycle below which a rate cannot be trusted. Nyquist
#: says two are enough to know an oscillation exists; *locating* its period is
#: a stronger demand, and below three the octave wins and the reported rate
#: silently halves. Consequence: resolving 120/min needs >= 6 Hz, so the 5 Hz
#: `docs/PROTOCOL.md` permits is not enough for CPR.
_CPR_MIN_SAMPLES_PER_PERIOD = 3.0

#: How close to the best autocorrelation score a *shorter* lag must come before
#: we prefer it. A periodic signal correlates with itself at every multiple of
#: its period, so the highest-scoring lag is often twice the true one -- and
#: reporting 55/min for a real 110/min does not merely miss, it inverts the
#: verdict an instructor acts on.
_CPR_OCTAVE_TOLERANCE = 0.85


def _cpr_window(history: Iterable[FrameObservation], cfg: ScoringConfig) -> list[FrameObservation]:
    """The trailing `cpr_window_s` seconds of pose-bearing history.

    Windowed by *timestamp*, not by frame count. A phone streaming at 28 Hz and
    one throttled to 10 Hz must be judged over the same amount of wall time, or
    the same trainee scores differently on different hardware.
    """
    posed = [
        o for o in history
        if o.keypoints_xy is not None and o.keypoints_conf is not None
    ]
    if not posed:
        return []
    cutoff = posed[-1].ts - cfg.cpr_window_s
    return [o for o in posed if o.ts >= cutoff]


def _better_wrist(window: list[FrameObservation], cfg: ScoringConfig):
    """(timestamps, y-values) for whichever wrist the camera actually saw.

    A side-on view -- the one that shows compression travel best -- puts one
    arm behind the other, so one wrist is typically tracked well and the other
    barely at all. Averaging a tracked wrist with a guessed one turns a good
    signal into a mediocre one, so the better-tracked side wins outright.
    """
    best_ts: list[float] = []
    best_ys: list[float] = []
    for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
        ts = [o.ts for o in window if o.keypoints_conf[idx] >= cfg.keypoint_conf_threshold]
        ys = [
            o.keypoints_xy[idx][1]
            for o in window
            if o.keypoints_conf[idx] >= cfg.keypoint_conf_threshold
        ]
        if len(ts) > len(best_ts):
            best_ts, best_ys = ts, ys
    return best_ts, best_ys


def _moving_average(xs: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(xs)
    half = window // 2
    out = []
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def _parabolic_offset(y_prev: float, y_mid: float, y_next: float) -> float:
    """Sub-sample position of a peak straddled by three samples.

    Fitting a parabola through the three and taking its vertex is what lets a
    discrete sample grid resolve a period that falls between two samples.
    """
    denom = y_prev - 2.0 * y_mid + y_next
    if denom == 0:
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return offset if -1.0 < offset < 1.0 else 0.0


def _resample_uniform(ts: list[float], xs: list[float], dt: float):
    """Interpolate an irregular series onto a uniform grid, which
    autocorrelation assumes and a network-delivered stream never provides."""
    if len(ts) < 2:
        return []
    n = int((ts[-1] - ts[0]) / dt) + 1
    if n < 2:
        return []
    out, j = [], 0
    for i in range(n):
        t = ts[0] + i * dt
        while j + 2 < len(ts) and ts[j + 1] < t:
            j += 1
        span = ts[j + 1] - ts[j]
        frac = 0.0 if span <= 0 else (t - ts[j]) / span
        out.append(xs[j] + frac * (xs[j + 1] - xs[j]))
    return out


def _autocorr_rate(xs: list[float], dt: float) -> tuple[float | None, float]:
    """Compressions per minute by normalised autocorrelation, and how periodic
    the signal was. The second value is the guard: it is near zero for noise.
    """
    n = len(xs)
    if n < 16:
        return None, 0.0
    mean_x = sum(xs) / n
    centered = [x - mean_x for x in xs]
    variance = sum(v * v for v in centered) / n
    if variance <= 0:
        return None, 0.0

    # Round the fast end up and the slow end down, so every searched lag stays
    # inside the plausible band; truncating both lets the shortest lag report a
    # rate faster than the band allows.
    lo = max(1, math.ceil((60.0 / _CPR_SEARCH_MAX_BPM) / dt))
    hi = min(n // 2, int((60.0 / _CPR_SEARCH_MIN_BPM) / dt))
    if hi <= lo:
        return None, 0.0

    curve = {
        lag: (sum(centered[i] * centered[i + lag] for i in range(n - lag)) / (n - lag)) / variance
        for lag in range(lo, hi + 1)
    }
    lags = sorted(curve)
    peak = max(curve.values())
    if peak <= 0:
        return None, 0.0

    # Walk up from the shortest lag and take the first local maximum that comes
    # close to the best: that picks the fundamental over its octave.
    best_lag = next(
        (
            lag
            for i, lag in enumerate(lags)
            if 0 < i < len(lags) - 1
            and curve[lag] >= curve[lags[i - 1]]
            and curve[lag] >= curve[lags[i + 1]]
            and curve[lag] >= _CPR_OCTAVE_TOLERANCE * peak
        ),
        max(curve, key=lambda k: curve[k]),
    )
    strength = curve[best_lag]

    refined = float(best_lag)
    if lo < best_lag < hi:
        refined += _parabolic_offset(curve[best_lag - 1], strength, curve[best_lag + 1])
    if refined <= 0:
        return None, strength
    return 60.0 / (refined * dt), strength


def _peak_cadence(ts: list[float], wave: list[float], amplitude: float):
    """Inter-compression regularity, by counting peaks.

    Autocorrelation answers *what rate*; this answers *how steady*, which it
    cannot. An average of 110/min made of alternating 80s and 140s is not good
    CPR, and a scorer reporting only the mean would call it perfect.
    """
    min_separation = 60.0 / _CPR_SEARCH_MAX_BPM
    prominence = amplitude * 0.5
    peaks: list[float] = []
    for i in range(1, len(wave) - 1):
        if wave[i] < prominence or wave[i] < wave[i - 1] or wave[i] < wave[i + 1]:
            continue
        local_dt = (ts[i + 1] - ts[i - 1]) / 2.0
        refined = ts[i] + _parabolic_offset(wave[i - 1], wave[i], wave[i + 1]) * local_dt
        if peaks and refined - peaks[-1] < min_separation:
            continue
        peaks.append(refined)
    if len(peaks) < 3:
        return None, None
    intervals = [b - a for a, b in zip(peaks, peaks[1:])]
    med = median(intervals)
    if med <= 0:
        return None, None
    return 60.0 / med, pstdev(intervals) / med


def estimate_compression_rate(window: list[FrameObservation], cfg: ScoringConfig) -> dict:
    """Compression rate and cadence from a window of pose history.

    Returns a dict rather than a number because the caller needs the evidence
    behind the answer: `periodicity`, `undersampled` and `agreement` are what
    decide whether `bpm` should be believed or discarded. A missing `bpm` is
    the normal, honest outcome for a station that is set up but idle.
    """
    ts, ys = _better_wrist(window, cfg)
    if len(ts) < 8:
        return {"bpm": None, "reason": "too few tracked wrist samples"}

    span = ts[-1] - ts[0]
    if span <= 0:
        return {"bpm": None, "reason": "no elapsed time"}
    hz = (len(ts) - 1) / span
    dt = 1.0 / hz

    # The fastest rate this stream can resolve. Above it, a rate is liable to
    # come back octave-halved -- and told "60/min, too slow" an instructor
    # would coach a trainee already at the top of the band to go faster still.
    resolvable_max_bpm = hz * 60.0 / _CPR_MIN_SAMPLES_PER_PERIOD
    out = {
        "hz": hz,
        "resolvable_max_bpm": resolvable_max_bpm,
        "undersampled": resolvable_max_bpm < cfg.cpr_rate_max_bpm,
    }
    if out["undersampled"]:
        return {**out, "bpm": None, "reason": f"{hz:.1f} Hz cannot resolve the target band"}

    wave = [y - b for y, b in zip(ys, _moving_average(ys, max(3, int(_CPR_TREND_WINDOW_S * hz))))]
    amplitude = pstdev(wave) if len(wave) > 1 else 0.0
    out["amplitude"] = amplitude
    if amplitude < _CPR_MIN_AMPLITUDE:
        return {**out, "bpm": None, "reason": "the wrist barely moved"}

    resampled = _resample_uniform(ts, wave, dt)
    bpm, periodicity = _autocorr_rate(resampled, dt) if resampled else (None, 0.0)
    out["periodicity"] = periodicity
    if bpm is None or periodicity < _CPR_MIN_PERIODICITY:
        return {**out, "bpm": None, "reason": "the motion is not periodic"}

    bpm_peaks, cv = _peak_cadence(ts, wave, amplitude)
    return {
        **out,
        "bpm": bpm,
        "cv": cv,
        "agreement": None if bpm_peaks is None else abs(bpm - bpm_peaks) / max(bpm, bpm_peaks),
        "reason": None,
    }


def score_cpr_arms(window: list[FrameObservation], cfg: ScoringConfig) -> tuple[float, bool]:
    """How far from locked the rescuer's elbows are. 180 degrees is straight.

    Only ever consulted while compressions are actually being detected: an
    elbow angle measured on somebody kneeling and talking says nothing about
    their CPR, and scoring it would flag a pause as a fault.
    """
    angles = []
    for obs in window:
        for s, e, w in ARM_CHAINS:
            conf = obs.keypoints_conf
            if min(conf[s], conf[e], conf[w]) < cfg.keypoint_conf_threshold:
                continue
            a, b, c = obs.keypoints_xy[s], obs.keypoints_xy[e], obs.keypoints_xy[w]
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 == 0 or n2 == 0:
                continue
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))

    if not angles:
        return 0.0, False
    angle = median(angles)
    deficit = max(0.0, cfg.cpr_min_elbow_angle_deg - angle)
    headroom = max(1.0, 180.0 - cfg.cpr_min_elbow_angle_deg)
    return min(1.0, deficit / headroom), deficit > 0.0


def compute_triage_cpr(
    trainee_id: str,
    track: TrackState,
    ts: float,
    cfg: ScoringConfig,
    reference_angle_deg: float | None = None,
) -> TriageRecord:
    """Score CPR chest compressions against the AHA's published targets.

    Deliberately scores only what a phone camera can honestly measure. **Depth
    and hand placement are out of scope**: recovering a 5-6 cm chest travel
    from an uncalibrated monocular camera needs a scale reference the frame
    does not contain, and a published feasibility study using the same class of
    pose model found frequency agreed closely with an instrumented manikin
    while depth was "overall not accurate". Reporting a depth number anyway
    would be the one failure this scorer cannot afford -- see
    `docs/VALIDATION.md`.

    The worst single fault sets the score, rather than a weighted sum. Each
    fault here is independently a reason to send someone: averaging would let a
    dangerously slow rate be diluted by well-locked elbows, and an instructor
    reading "0.4" would not know which half of it to act on.
    """
    if not track.subject_present:
        return TriageRecord(trainee_id=trainee_id, score=0.0, reason_codes=(), ts=ts)

    window = _cpr_window(track.history, cfg)
    rate = estimate_compression_rate(window, cfg)
    bpm = rate.get("bpm")

    # No compressions detected is not a fault. A station set up before the
    # trainee starts, or paused between cycles, is the normal case; asserting
    # about it would put every idle nursing station on the instructor's queue.
    if bpm is None:
        return TriageRecord(trainee_id=trainee_id, score=0.0, reason_codes=(), ts=ts)

    reasons: list[str] = []
    scores = [0.0]

    if bpm < cfg.cpr_rate_min_bpm:
        reasons.append(REASON_CPR_RATE_SLOW)
        scores.append(min(1.0, (cfg.cpr_rate_min_bpm - bpm) / cfg.cpr_rate_full_deviation_bpm))
    elif bpm > cfg.cpr_rate_max_bpm:
        reasons.append(REASON_CPR_RATE_FAST)
        scores.append(min(1.0, (bpm - cfg.cpr_rate_max_bpm) / cfg.cpr_rate_full_deviation_bpm))

    cv = rate.get("cv")
    if cv is not None and cv > cfg.cpr_cadence_cv_threshold:
        reasons.append(REASON_CPR_ERRATIC)
        scores.append(min(1.0, cv / cfg.cpr_cadence_cv_threshold - 1.0))

    arms, arms_bent = score_cpr_arms(window, cfg)
    if arms_bent:
        reasons.append(REASON_CPR_ARMS_BENT)
        scores.append(arms)

    return TriageRecord(
        trainee_id=trainee_id,
        score=round(max(scores), 4),
        reason_codes=tuple(reasons),
        ts=ts,
    )


#: Procedures a nursing station can be scored for. CPR is the one that is
#: built; a ward's other trained skills (bagging, recovery position, transfers)
#: each add an entry here rather than a whole use case of their own.
_NURSING_PROCEDURES: dict[
    str, Callable[[str, TrackState, float, ScoringConfig, float | None], TriageRecord]
] = {
    "cpr": compute_triage_cpr,
}


def known_procedures() -> frozenset[str]:
    """Every nursing procedure this build can score."""
    return frozenset(_NURSING_PROCEDURES)


def compute_triage_nursing(
    trainee_id: str,
    track: TrackState,
    ts: float,
    cfg: ScoringConfig,
    reference_angle_deg: float | None = None,
) -> TriageRecord:
    """Dispatch to the scorer for whichever procedure this station is running.

    A nursing station that has not named a procedure, or names one this build
    does not implement, scores a flat 0.0 rather than being forced through
    CPR's scorer -- the same posture as `compute_triage_welding`. Watching a
    trainee do one thing and grading them at another is worse than not grading
    them, because it looks like a measurement.
    """
    scorer = _NURSING_PROCEDURES.get(track.last_procedure or "")
    if scorer is None:
        return TriageRecord(trainee_id=trainee_id, score=0.0, reason_codes=(), ts=ts)
    return scorer(trainee_id, track, ts, cfg, reference_angle_deg)


#: Every use case this server can score a `TrackState` for. A new use case's
#: scorer is added here rather than by branching inside `compute_triage_
#: fitness`, whose five features (`fall`, `stillness`, `occlusion`,
#: `off_task`, `form_error`) are read off fitness's own pose/bbox history and
#: mean nothing against another use case's evidence.
_SCORERS: dict[
    str, Callable[[str, TrackState, float, ScoringConfig, float | None], TriageRecord]
] = {
    "fitness": compute_triage_fitness,
    "nursing": compute_triage_nursing,
    "welding": compute_triage_welding,
}


#: History a nursing track keeps, in frames. Far longer than fitness's ~2s
#: window, because a compression rate cannot be read off one second of wrist —
#: at 100/min that is a single cycle, and at the 28 Hz a phone actually
#: delivers, `[scoring] history_len`'s 30 frames *is* one second. Sized
#: generously in frames and then windowed by timestamp in `_cpr_window`, so the
#: buffer holds enough at any plausible frame rate (512 frames is ~17s at
#: 30 Hz) while the scorer still judges a fixed amount of wall time. The cost
#: is 512 small dataclasses per station, which is nothing against the cost of
#: measuring a rhythm through a window shorter than the rhythm.
NURSING_HISTORY_LEN = 512


def history_len_for(use_case: str, cfg: ScoringConfig) -> int:
    """How many frames of history a track for `use_case` should keep.

    `[scoring] history_len` is tuned for fitness's five features, which read a
    posture over ~2 seconds. A use case whose evidence is a *rhythm* needs
    materially more history, and scoring CPR on a 30-frame buffer would look
    like it was working while measuring nothing at all.
    """
    return NURSING_HISTORY_LEN if use_case == "nursing" else cfg.history_len


def known_use_cases() -> frozenset[str]:
    """Every use case this build of Argus can actually score.

    `argus.config.SessionConfig` validates `[session] use_case` against this
    at load time — rejecting an operator's typo (or a use case that sounds
    plausible but was never wired up) before a laptop spends a session
    accepting phones it cannot score. It reads `_SCORERS` rather than
    duplicating its keys in a second list, so the two cannot drift apart.
    """
    return frozenset(_SCORERS)


def compute_triage(
    trainee_id: str,
    track: TrackState,
    ts: float,
    cfg: ScoringConfig,
    reference_angle_deg: float | None = None,
) -> TriageRecord:
    """Dispatch to the scorer registered for `track.use_case`.

    `argus.ingest.protocol` already refuses an `observation` naming a use
    case with no registered parser, so in practice `track.use_case` is always
    a key present here; the `KeyError` below is a defensive backstop for a
    `TrackState` built directly (as tests do) rather than a path expected in
    production.
    """
    scorer = _SCORERS[track.use_case]
    return scorer(trainee_id, track, ts, cfg, reference_angle_deg)


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
