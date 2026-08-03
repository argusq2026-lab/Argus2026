"""BlazePose landmark layout -> COCO-17, the layout the triage scorer indexes.

This module exists because of a contract mismatch that would otherwise be
silent and total. `triage.py` addresses keypoints by COCO index — nose 0,
shoulders 5/6, wrists 9/10, hips 11/12 — but the shipped
`pose_landmark_detector.bin` emits **25 BlazePose landmarks**, whose indices
mean entirely different joints. Feeding BlazePose indices to a COCO-indexed
scorer reads the left eye as a shoulder and a mouth corner as a wrist: every
pose-derived feature would be wrong, and every one of them would still return
a plausible number.

The 25-point export is upper-body BlazePose, so COCO's knees and ankles
(13-16) have **no source landmark**. They are emitted at confidence 0.0 rather
than at a guessed position. That is safe for the current feature set — fall
uses shoulders and hips, occlusion uses nose and wrists, off-task uses
shoulders — but any future lower-body feature must check for it, which is why
:data:`UNMAPPED_COCO_INDICES` is public and tested.
"""

from __future__ import annotations

import numpy as np

from argus.triage import NUM_KEYPOINTS

#: Landmark count of the shipped upper-body BlazePose export.
BLAZEPOSE_NUM_LANDMARKS = 25

#: BlazePose landmark order, indices 0-24.
BLAZEPOSE_NAMES = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
)

#: COCO-17 order, the layout `triage.py` indexes.
COCO_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

#: COCO index -> BlazePose index, or None where the 25-point export has no
#: corresponding landmark. Both layouts use the *subject's* left/right.
COCO_FROM_BLAZEPOSE: tuple[int | None, ...] = (
    BLAZEPOSE_NAMES.index("nose"),            # 0  nose
    BLAZEPOSE_NAMES.index("left_eye"),        # 1  left_eye
    BLAZEPOSE_NAMES.index("right_eye"),       # 2  right_eye
    BLAZEPOSE_NAMES.index("left_ear"),        # 3  left_ear
    BLAZEPOSE_NAMES.index("right_ear"),       # 4  right_ear
    BLAZEPOSE_NAMES.index("left_shoulder"),   # 5  left_shoulder
    BLAZEPOSE_NAMES.index("right_shoulder"),  # 6  right_shoulder
    BLAZEPOSE_NAMES.index("left_elbow"),      # 7  left_elbow
    BLAZEPOSE_NAMES.index("right_elbow"),     # 8  right_elbow
    BLAZEPOSE_NAMES.index("left_wrist"),      # 9  left_wrist
    BLAZEPOSE_NAMES.index("right_wrist"),     # 10 right_wrist
    BLAZEPOSE_NAMES.index("left_hip"),        # 11 left_hip
    BLAZEPOSE_NAMES.index("right_hip"),       # 12 right_hip
    None,                                     # 13 left_knee   -- upper-body export
    None,                                     # 14 right_knee  -- upper-body export
    None,                                     # 15 left_ankle  -- upper-body export
    None,                                     # 16 right_ankle -- upper-body export
)

#: COCO indices with no source landmark; always confidence 0.0.
UNMAPPED_COCO_INDICES: tuple[int, ...] = tuple(
    i for i, src in enumerate(COCO_FROM_BLAZEPOSE) if src is None
)


def blazepose_to_coco(
    landmarks_xy: np.ndarray, visibility: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remap (25, 2) BlazePose points + (25,) visibility to COCO-17.

    Returns ``(xy (17, 2) float32, conf (17,) float32)``. Unmapped COCO joints
    get position (0, 0) and confidence 0.0 — the scorer's confidence gate then
    excludes them, so they never contribute a fabricated position.
    """
    if landmarks_xy.shape != (BLAZEPOSE_NUM_LANDMARKS, 2):
        raise ValueError(
            f"expected ({BLAZEPOSE_NUM_LANDMARKS}, 2) landmarks, got {landmarks_xy.shape}"
        )
    if visibility.shape != (BLAZEPOSE_NUM_LANDMARKS,):
        raise ValueError(
            f"expected ({BLAZEPOSE_NUM_LANDMARKS},) visibility, got {visibility.shape}"
        )

    xy = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    conf = np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
    for coco_idx, src in enumerate(COCO_FROM_BLAZEPOSE):
        if src is None:
            continue
        xy[coco_idx] = landmarks_xy[src]
        conf[coco_idx] = visibility[src]
    return xy, conf


def empty_coco_keypoints() -> tuple[np.ndarray, np.ndarray]:
    """All-zero COCO-17 keypoints, used when pose estimation finds nobody.

    Zero confidence everywhere means the scorer treats the trainee as fully
    occluded rather than as a confidently-detected motionless person.
    """
    return (
        np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
        np.zeros((NUM_KEYPOINTS,), dtype=np.float32),
    )
