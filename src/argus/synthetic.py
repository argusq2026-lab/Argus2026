"""The synthetic scene — one definition, used by both the mock and the demo clip.

The mock backend emits detections for these people and `demo/make_demo_video.py`
draws them, so the boxes the pipeline tracks land on the pixels the fixture
actually contains. That matters for more than tidiness: the tracker's
re-identification signature is a colour histogram of the crop, so a mock whose
boxes float over unrelated background produces meaningless signatures and
manufactures identity churn that the real system would not have.

Coordinates are normalised to the **letterboxed detector canvas**, because that
is the space YOLO-X emits boxes in. `canvas_to_frame` inverts the letterbox for
a given frame size, which is how the demo renderer places the same people.

This is a pipeline fixture, not an accuracy benchmark: the shapes are drawn,
not photographed. See docs/VALIDATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Frame size the demo clip is rendered at.
SCENE_FRAME_SIZE = (640, 480)

#: Tick at which the third trainee starts to fall, and how long it takes.
FALL_START_FRAME = 20
FALL_DURATION_FRAMES = 6


@dataclass(frozen=True)
class SyntheticPerson:
    """One synthetic trainee, in canvas-normalised coordinates."""

    name: str
    cx: float
    cy: float
    w: float
    h: float
    score: float
    #: Distinct BGR fill, so each trainee has a separable appearance signature.
    colour_bgr: tuple[int, int, int]

    @property
    def xyxy_norm(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        )


def synthetic_people(frame_index: int) -> list[SyntheticPerson]:
    """The scene at one tick: one walking, one motionless, one falling.

    Each exercises a different triage feature — `prolonged_stillness` for the
    motionless one, `possible_fall` for the third, and a clean baseline for the
    walker.
    """
    walk_cx = min(0.18 + 0.006 * frame_index, 0.42)
    people = [
        SyntheticPerson("walker", walk_cx, 0.50, 0.09, 0.32, 0.91, (60, 180, 60)),
        SyntheticPerson("still", 0.60, 0.50, 0.09, 0.32, 0.88, (200, 140, 40)),
    ]

    if frame_index < FALL_START_FRAME:
        people.append(SyntheticPerson("faller", 0.84, 0.50, 0.09, 0.32, 0.86, (50, 50, 210)))
    else:
        t = min((frame_index - FALL_START_FRAME) / FALL_DURATION_FRAMES, 1.0)
        people.append(
            SyntheticPerson(
                "faller",
                0.84,
                0.50 + 0.22 * t,
                0.09 + 0.21 * t,   # widens
                0.32 - 0.22 * t,   # and flattens: the aspect flip score_fall keys on
                0.86,
                (50, 50, 210),
            )
        )
    return people


def canvas_to_frame(
    xyxy_norm: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    canvas: int,
) -> tuple[float, float, float, float]:
    """Map canvas-normalised xyxy into source-frame pixels.

    Inverts the centred letterbox in :func:`argus.vision.preprocess.letterbox`,
    so a box drawn here is the box the detector's post-processing recovers.
    """
    scale = canvas / max(frame_w, frame_h)
    nh, nw = round(frame_h * scale), round(frame_w * scale)
    top, left = (canvas - nh) // 2, (canvas - nw) // 2
    x0, y0, x1, y1 = (v * canvas for v in xyxy_norm)
    return (
        (x0 - left) / scale,
        (y0 - top) / scale,
        (x1 - left) / scale,
        (y1 - top) / scale,
    )
