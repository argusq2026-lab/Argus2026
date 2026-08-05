"""The synthetic trainee scene — one definition, shared by the demo replay
fixture and the tests.

There is no camera or model to mock anymore: a phone's on-device pose model
and form/exercise classifier already deliver numeric observations (see
`docs/PROTOCOL.md`), so the fixture just *is* a sequence of those
observations, directly in `argus.triage.FrameObservation` shape. This is what
lets `argus demo` + `demo/replay_client.py` exercise the whole ingest ->
triage -> alert path with no phone, no model, and no camera, the same role
the old pixel-rendered demo clip played for the camera pipeline.

Coordinates are normalized to [0, 1] of an implied phone frame, matching the
wire protocol's convention — there is no letterboxed detector canvas to
invert, because there is no detector.

This is a pipeline fixture, not an accuracy benchmark: the numbers are
hand-authored to exercise specific triage features, not measured from a real
trainee. See docs/VALIDATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.triage import FrameObservation

#: Tick at which the third trainee starts to fall, and how long it takes.
FALL_START_TICK = 20
FALL_DURATION_TICKS = 6

#: Every synthetic trainee id this scene ever emits, in a stable order.
#:
#: The two plank stations are here because a plank is the case the scorer's
#: standing-trainee assumptions get wrong: `good_plank` is horizontal and
#: motionless, which `score_fall` and `score_stillness` read as an emergency,
#: and only `[scoring.exercise_weights.plank]` stops it alerting. A demo
#: without one shows a system that looks correct because nothing on the floor
#: was lying down.
TRAINEE_IDS = ("walker", "still", "faller", "good_plank", "sagging_plank")

#: Which exercise each station reports. This selects the server's scoring
#: weight profile, so it is part of the scene, not decoration.
TRAINEE_EXERCISE = {
    "walker": "squat",
    "still": "squat",
    "faller": "squat",
    "good_plank": "plank",
    "sagging_plank": "plank",
}

#: The form codes each station reports, drawn from `[scoring.form_error_vocab]`.
#: Only the sagging plank is doing anything wrong.
TRAINEE_FORM_CODES: dict[str, tuple[str, ...]] = {
    "sagging_plank": ("hips_sagging",),
}

#: The scene's stand-in for an on-device form classifier.
#:
#: No phone reports `form_reason_codes` yet — that classifier is edge work
#: that has not landed — so the `form_error` feature would otherwise be inert
#: everywhere, including in the one fixture the console is developed against.
#: One synthetic trainee reports a code on a fixed, repeating slice of each
#: cycle, so the console's form-error path is exercisable today and looks the
#: same when a real classifier starts filling the field. Intermittent rather
#: than latched on purpose: a real classifier flags the bad reps, not the set.
FORM_ERROR_TRAINEE = "walker"
FORM_ERROR_CODE = "insufficient_depth"
FORM_ERROR_CYCLE_TICKS = 30
FORM_ERROR_ONSET_TICK = 20

#: The exercise every synthetic trainee is nominally performing, and how many
#: ticks one rep takes. Informational only — see `FrameObservation`.
SCENE_EXERCISE = "squat"
TICKS_PER_REP = 5


def _form_reason_codes(trainee_id: str, tick: int) -> tuple[str, ...]:
    """The scene's form codes for one trainee at one tick. Deterministic."""
    if trainee_id != FORM_ERROR_TRAINEE:
        return ()
    if tick % FORM_ERROR_CYCLE_TICKS < FORM_ERROR_ONSET_TICK:
        return ()
    return (FORM_ERROR_CODE,)


@dataclass(frozen=True)
class _Box:
    """A synthetic bounding box, centre + size, normalized to [0, 1]."""

    cx: float
    cy: float
    w: float
    h: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        )


def _boxes(tick: int) -> dict[str, _Box]:
    """The scene at one tick: one walking, one motionless, one falling.

    Each exercises a different triage feature — `prolonged_stillness` for the
    motionless one, `possible_fall` for the third, and a clean baseline for
    the walker. The numbers are carried over unchanged from the old
    camera-canvas fixture; only the coordinate system's meaning changed, from
    "normalized to the letterboxed detector canvas" to "normalized to the
    phone's own frame" — both are already fractions in [0, 1], so the values
    that made `score_fall` and `score_stillness` fire before still do.
    """
    walk_cx = min(0.18 + 0.006 * tick, 0.42)
    boxes = {
        "walker": _Box(walk_cx, 0.50, 0.09, 0.32),
        "still": _Box(0.60, 0.50, 0.09, 0.32),
        # Both planks: wide, short, and unmoving. Deliberately the same shape
        # the faller ends up in, and the same stillness the "still" trainee
        # has -- the geometry cannot tell them apart, which is the point.
        "good_plank": _Box(0.30, 0.82, 0.30, 0.09),
        "sagging_plank": _Box(0.70, 0.82, 0.30, 0.09),
    }
    if tick < FALL_START_TICK:
        boxes["faller"] = _Box(0.84, 0.50, 0.09, 0.32)
    else:
        t = min((tick - FALL_START_TICK) / FALL_DURATION_TICKS, 1.0)
        boxes["faller"] = _Box(
            0.84,
            0.50 + 0.22 * t,
            0.09 + 0.21 * t,  # widens
            0.32 - 0.22 * t,  # and flattens: the aspect flip score_fall keys on
        )
    return boxes


def _pose(box: _Box) -> tuple[list[tuple[float, float]], list[float]]:
    """An upright, camera-facing COCO-17 pose scaled to `box`.

    Only the joints the scorer reads (nose, shoulders, wrists, hips) get real
    positions and full confidence; the rest — eyes, ears, knees, ankles — are
    reported at confidence 0.0, mirroring a real upper-body-only pose export's
    coverage. The left shoulder sits at the larger image x: COCO and
    MediaPipe both label joints from the *subject's* perspective, so that is
    what "facing the camera" looks like (see
    `argus.config.ScoringConfig.off_task_reference_angle_deg`).
    """
    x0, y0, x1, y1 = box.xyxy
    left_x = x1 - box.w * 0.15
    right_x = x0 + box.w * 0.15
    shoulder_y = y0 + box.h * 0.25
    wrist_y = y0 + box.h * 0.55
    hip_y = y0 + box.h * 0.60

    kp_xy = [(box.cx, y0)] * 17
    kp_conf = [0.0] * 17
    kp_xy[0] = (box.cx, y0 + box.h * 0.08)  # nose
    kp_xy[5] = (left_x, shoulder_y)  # left shoulder
    kp_xy[6] = (right_x, shoulder_y)  # right shoulder
    kp_xy[9] = (left_x, wrist_y)  # left wrist
    kp_xy[10] = (right_x, wrist_y)  # right wrist
    kp_xy[11] = (left_x, hip_y)  # left hip
    kp_xy[12] = (right_x, hip_y)  # right hip
    for idx in (0, 5, 6, 9, 10, 11, 12):
        kp_conf[idx] = 0.9
    return kp_xy, kp_conf


def synthetic_tick(tick: int, fps: float = 15.0) -> dict[str, FrameObservation]:
    """One tick's numeric observation for every synthetic trainee.

    `fps` only sets the timestamp cadence — the scene itself is authored in
    ticks, same as the old frame-indexed fixture.
    """
    ts = tick / fps
    observations = {}
    for trainee_id, box in _boxes(tick).items():
        kp_xy, kp_conf = _pose(box)
        # Two independent form-code mechanisms, one per trainee: the sagging
        # plank's is a static, always-on fault (it never has correct form),
        # the walker's is `_form_reason_codes`' cyclic squat demo (a real
        # classifier flags the bad reps, not the whole set). They never
        # target the same trainee_id, so this is a union, not a priority.
        codes = TRAINEE_FORM_CODES.get(trainee_id) or _form_reason_codes(trainee_id, tick)
        observations[trainee_id] = FrameObservation(
            ts=ts,
            bbox_xyxy=box.xyxy,
            keypoints_xy=kp_xy,
            keypoints_conf=kp_conf,
            form_reason_codes=codes,
            # Load-bearing (selects the scoring weight profile), so each
            # trainee needs its own value rather than the scene-wide
            # SCENE_EXERCISE constant -- a plank scored as a squat is exactly
            # the misfire TRAINEE_EXERCISE/[scoring.exercise_weights] exist
            # to prevent.
            exercise=TRAINEE_EXERCISE[trainee_id],
            rep_count=tick // TICKS_PER_REP,
            form_ok=not codes,
        )
    return observations


def build_fixture(n_ticks: int = 150, fps: float = 15.0, protocol_version: int = 1) -> dict:
    """A canned multi-station wire-protocol fixture.

    Exactly what `n_ticks` of real phones would have sent — one `hello`
    handshake per trainee plus its `observation` messages — laid out per
    station so `demo/replay_client.py` can open one WebSocket connection per
    trainee and drive a real, running `argus run` end to end with no phone,
    no model, and no camera. `argus demo` writes this to a JSON file; nothing
    here is unique to being written to disk, so tests can call it directly.
    """
    per_trainee: dict[str, list[dict]] = {trainee_id: [] for trainee_id in TRAINEE_IDS}
    for tick in range(n_ticks):
        for trainee_id, obs in synthetic_tick(tick, fps).items():
            per_trainee[trainee_id].append(
                {
                    "type": "observation",
                    "ts": obs.ts,
                    "bbox_xyxy": list(obs.bbox_xyxy),
                    "keypoints_xy": [list(p) for p in obs.keypoints_xy],
                    "keypoints_conf": list(obs.keypoints_conf),
                    "exercise": obs.exercise,
                    "rep_count": obs.rep_count,
                    "form_ok": obs.form_ok,
                    "form_reason_codes": list(obs.form_reason_codes),
                }
            )
    return {
        "protocol_version": protocol_version,
        "stations": [
            {
                "station_id": f"demo-{trainee_id}",
                "trainee_id": trainee_id,
                "messages": messages,
            }
            for trainee_id, messages in per_trainee.items()
        ],
    }
