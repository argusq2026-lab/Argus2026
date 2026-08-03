"""On-device overlay drawing.

This lives *inside* the privacy boundary: it annotates a frame that already
exists in the capture loop and renders it to a local window. It is not a sink
— `argus.outputs` and `argus.alerts` are, and neither can accept a frame.

One caveat worth stating rather than burying: `outputs.overlay_out` writes
annotated frames to a video file, which does persist raw imagery to disk. That
is an operator decision, so it is off by default and called out in the config
comment and README.
"""

from __future__ import annotations

import cv2
import numpy as np

from argus.triage import TriageRecord

_FLAGGED = (0, 0, 255)
_NORMAL = (0, 200, 0)
_COASTING = (0, 165, 255)


def draw_overlay(
    frame: np.ndarray,
    boxes_by_id: dict[str, tuple[float, float, float, float]],
    records_by_id: dict[str, TriageRecord],
    alert_threshold: float,
    coasting_ids: frozenset[str] = frozenset(),
) -> np.ndarray:
    """Annotate a copy of `frame` with boxes, ids, and scores.

    Coasting tracks — ones the Kalman filter is predicting through an
    occlusion — are drawn in a third colour, so an instructor can see that an
    identity is being maintained rather than wonder why a box has no person
    under it.
    """
    annotated = frame.copy()
    for track_id, bbox in sorted(boxes_by_id.items()):
        x0, y0, x1, y1 = (int(v) for v in bbox)
        record = records_by_id.get(track_id)
        if track_id in coasting_ids:
            color = _COASTING
        elif record is not None and record.score >= alert_threshold:
            color = _FLAGGED
        else:
            color = _NORMAL
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        label = track_id if record is None else f"{track_id} {record.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x0, max(y0 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


class OverlayWriter:
    """Optional annotated-video writer. Lazily sized to the first frame."""

    def __init__(self, path: str, fps: float):
        self._path = path
        self._fps = fps if fps > 0 else 15.0
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
