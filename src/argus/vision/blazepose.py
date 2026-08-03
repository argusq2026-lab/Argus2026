"""BlazePose SSD anchor generation and tensor decoding.

The shipped MediaPipe-Pose export is a two-stage chain, not a heatmap model:

1. ``pose_detector.bin`` — 128x128 NHWC SSD. Two anchor heads, 512 anchors at
   stride 8 (16x16 grid x 2) and 384 at stride 16 (8x8 grid x 6) = 896 total,
   which is exactly the ``box_scores_1``/``box_scores_2`` split in
   metadata.json. Each anchor regresses 12 values: a box plus four alignment
   keypoints.
2. ``pose_landmark_detector.bin`` — 256x256 NHWC, emitting a presence score
   and 25 landmarks of (x, y, z, visibility) normalised to the ROI.

There are no heatmaps anywhere in this chain, so the prototype's
``decode_heatmaps_cpu()`` had nothing to decode.

**Quantization caveat, and it is a real one.** ``box_scores_*`` were calibrated
to a range that is entirely non-positive: scale 5.55 with zero_point 255 means
the largest representable logit is exactly 0.0 (score 0.5) and the next
quantization step down is -5.55 (score 0.0039). The detector's confidence is
therefore effectively two-valued. That is a symptom of the placeholder INT8
calibration set — see docs/VALIDATION.md — not of this decoder. Re-quantize
with real trainee-floor footage before reading anything into a pose-detector
confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Values per anchor in ``box_coords_*``: dx, dy, w, h, then 4 (x, y) keypoints.
COORDS_PER_ANCHOR = 12
#: Alignment keypoints the detector regresses (hip mid, full-body, shoulder mid, upper-body).
NUM_ALIGNMENT_KEYPOINTS = 4


@dataclass(frozen=True)
class AnchorLayout:
    """One SSD head: a grid size and the number of anchors per cell."""

    grid: int
    anchors_per_cell: int

    @property
    def count(self) -> int:
        return self.grid * self.grid * self.anchors_per_cell


#: The 128x128 BlazePose detector's two heads, in output order. Derived from
#: MediaPipe's SsdAnchorsCalculator options (strides [8, 16, 16, 16],
#: aspect_ratios [1.0], interpolated scale => 2 anchors per layer per cell).
DETECTOR_HEADS: tuple[AnchorLayout, ...] = (
    AnchorLayout(grid=16, anchors_per_cell=2),  # stride 8  -> 512
    AnchorLayout(grid=8, anchors_per_cell=6),   # stride 16 -> 384 (3 layers x 2)
)


@lru_cache(maxsize=4)
def ssd_anchor_centres(
    heads: tuple[AnchorLayout, ...] = DETECTOR_HEADS,
) -> tuple[np.ndarray, ...]:
    """Per-head anchor centres, normalised to [0, 1].

    ``fixed_anchor_size`` is set in the BlazePose config, so every anchor has
    width and height 1.0 and only the centre matters.
    """
    out = []
    for head in heads:
        ys, xs = np.meshgrid(
            np.arange(head.grid, dtype=np.float32),
            np.arange(head.grid, dtype=np.float32),
            indexing="ij",
        )
        centres = np.stack([(xs + 0.5) / head.grid, (ys + 0.5) / head.grid], axis=-1)
        centres = np.repeat(
            centres.reshape(-1, 2), head.anchors_per_cell, axis=0
        )  # (grid*grid*apc, 2)
        out.append(centres.astype(np.float32))
    return tuple(out)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, so a large negative logit cannot overflow."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))),
                    np.exp(np.clip(x, -60, 60)) / (1.0 + np.exp(np.clip(x, -60, 60))))


@dataclass(frozen=True)
class PoseDetection:
    """One decoded person ROI, in coordinates normalised to the detector input."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    alignment_xy: np.ndarray  # (4, 2)


def decode_detector_head(
    scores_logit: np.ndarray,
    coords: np.ndarray,
    anchor_centres: np.ndarray,
    input_size: int,
    scores_are_logits: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one SSD head into (boxes_xyxy_norm, scores, alignment_xy).

    `scores_logit` is (N, 1) or (N,), `coords` is (N, 12), both already
    dequantized to float. Returned boxes are normalised to [0, 1].
    """
    raw_scores = np.asarray(scores_logit, dtype=np.float32).reshape(-1)
    scores = sigmoid(raw_scores) if scores_are_logits else raw_scores

    c = np.asarray(coords, dtype=np.float32).reshape(-1, COORDS_PER_ANCHOR)
    if len(c) != len(anchor_centres):
        raise ValueError(
            f"anchor count mismatch: {len(c)} coords vs {len(anchor_centres)} anchors"
        )

    # MediaPipe divides every regressed value by the input edge length; anchor
    # width/height are fixed at 1.0, so the anchor contributes only its centre.
    scaled = c / float(input_size)
    cx = scaled[:, 0] + anchor_centres[:, 0]
    cy = scaled[:, 1] + anchor_centres[:, 1]
    w = scaled[:, 2]
    h = scaled[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    kp = scaled[:, 4:].reshape(-1, NUM_ALIGNMENT_KEYPOINTS, 2)
    kp = kp + anchor_centres[:, None, :]
    return boxes.astype(np.float32), scores.astype(np.float32), kp.astype(np.float32)


def decode_landmarks(
    landmarks: np.ndarray, visibility_is_logit: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Split a dequantized (25, 4) landmark tensor into normalised xy + visibility.

    x and y are normalised to the 256x256 ROI. The z channel is dropped: Argus
    scores in the image plane and an unvalidated INT8 depth estimate would add
    noise, not signal.
    """
    lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 4)
    xy = lm[:, :2]
    vis = lm[:, 3]
    if visibility_is_logit:
        vis = sigmoid(vis)
    return xy, np.clip(vis, 0.0, 1.0).astype(np.float32)
