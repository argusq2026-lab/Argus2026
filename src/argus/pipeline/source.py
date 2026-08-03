"""Camera sources — one per configured `[[cameras]]` entry.

"Many eyes" is the product premise, so N sources is the base case and one
source is the degenerate case of it. Each source carries its own tracker state
and its own station-facing reference angle; a merged global rank is assembled
across all of them (see :mod:`argus.pipeline.runner`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from argus.config import CameraConfig


class CameraOpenError(RuntimeError):
    """A configured source could not be opened."""


@dataclass
class Frame:
    """One frame with the identity of the camera that produced it."""

    camera_id: str
    index: int
    image_bgr: np.ndarray


class CameraSource:
    """A `cv2.VideoCapture` plus the config that describes it."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.camera_id = cfg.id
        source = cfg.resolved_source()
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise CameraOpenError(
                f"camera {cfg.id!r}: could not open source {source!r}"
            )
        self._index = 0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 15.0
        self.exhausted = False

    @property
    def is_live(self) -> bool:
        """True for a camera index, False for a file — which is what decides
        whether a run can use a deterministic frame clock."""
        return isinstance(self.cfg.source, int)

    @property
    def frame_area(self) -> float:
        return float(self.width * self.height)

    @property
    def reference_angle_deg(self) -> float | None:
        return self.cfg.reference_angle_deg

    def read(self) -> Frame | None:
        """Next frame, or None once the source is exhausted."""
        ok, image = self._capture.read()
        if not ok or image is None:
            self.exhausted = True
            return None
        frame = Frame(self.camera_id, self._index, image)
        self._index += 1
        return frame

    def release(self) -> None:
        self._capture.release()


def open_sources(cameras: tuple[CameraConfig, ...]) -> list[CameraSource]:
    """Open every enabled camera, releasing any already opened on failure."""
    opened: list[CameraSource] = []
    try:
        for cfg in cameras:
            opened.append(CameraSource(cfg))
    except Exception:
        for source in opened:
            source.release()
        raise
    if not opened:
        raise CameraOpenError(
            "no enabled cameras in config; add a [[cameras]] entry with enabled = true"
        )
    return opened
