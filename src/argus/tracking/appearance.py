"""Colour-histogram appearance signatures for re-identification.

`trainee_id` is a triage key: an instructor is dispatched to *a specific
person*, so an ID that jumps between two trainees who crossed paths is a
correctness bug, not a cosmetic one. Motion alone cannot tell those two apart
— at the crossing point both hypotheses fit the motion model equally well.

This is a deliberately model-free signature: an HSV histogram of the torso
region, which needs no fourth NPU graph, adds no inference latency, and is
computable on any host. It is weaker than a learned re-ID embedding and will
confuse two trainees in identical PPE. That trade is recorded in
docs/VALIDATION.md; upgrading to an OSNet-class embedder is a drop-in
replacement for :func:`signature`.

Hue and saturation only — the value channel is dropped so a trainee walking
through a shadow keeps their signature.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Fraction of the box height treated as torso. Skips the head (small, and
#: mostly skin/hair across trainees) and the legs (frequently occluded by
#: equipment), keeping the region most likely to carry distinguishing colour.
TORSO_TOP = 0.15
TORSO_BOTTOM = 0.65


def torso_region(crop_bgr: np.ndarray) -> np.ndarray:
    """The horizontal band of a person crop used for the signature."""
    h = crop_bgr.shape[0]
    top = int(h * TORSO_TOP)
    bottom = max(int(h * TORSO_BOTTOM), top + 1)
    return crop_bgr[top:bottom]


def signature(crop_bgr: np.ndarray, bins: int) -> np.ndarray | None:
    """A normalised 2-D hue/saturation histogram, flattened. None if unusable."""
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return None
    region = torso_region(crop_bgr)
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0.0:
        return None
    return (hist / total).flatten().astype(np.float32)


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Bhattacharyya distance in [0, 1]. Returns 0.5 when either side is unknown.

    A neutral 0.5 means an unknown signature neither helps nor blocks a match:
    the motion term decides, which is the correct behaviour for a track that
    has only ever been seen through an occlusion.
    """
    if a is None or b is None:
        return 0.5
    coefficient = float(np.sum(np.sqrt(np.clip(a * b, 0.0, None))))
    return float(np.sqrt(max(0.0, 1.0 - min(coefficient, 1.0))))


def blend(
    stored: np.ndarray | None, observed: np.ndarray | None, momentum: float
) -> np.ndarray | None:
    """Exponential moving average of a track's signature.

    High momentum keeps the identity anchored to how the trainee looked when
    first cleanly seen, rather than letting it drift into whatever occluded it.
    """
    if observed is None:
        return stored
    if stored is None:
        return observed
    merged = momentum * stored + (1.0 - momentum) * observed
    total = float(merged.sum())
    return merged / total if total > 0 else stored
