"""Person detection — YOLO-X w8a8.

Written against the artifact's real contract, which differs from the
prototype's assumption in every particular:

* input is **NCHW (1, 3, 640, 640) uint8**, not NHWC int8;
* there are **three** outputs — ``boxes (1, 8400, 4)``, ``scores (1, 8400)``,
  ``class_idx (1, 8400)`` — not one fused ``(1, N, 6)`` tensor;
* boxes arrive **already decoded to xyxy** in the 640x640 letterboxed space,
  quantized with scale 4.4157 / zero_point 51 (hence the representable
  negative coordinates for boxes that hang off an edge);
* 8400 == 80² + 40² + 20², the anchor-free grid for a 640 input at strides
  8/16/32.

NMS is not in the graph, so it runs here on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argus.config import DetectorConfig
from argus.engines.base import ModelRunner
from argus.vision.nms import nms_cpu
from argus.vision.preprocess import letterbox, to_nchw_uint8, undo_letterbox


@dataclass(frozen=True)
class Detection:
    """One person box in source-frame pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_index: int = 0


class PersonDetector:
    """Runs the detector graph and post-processes to person boxes."""

    def __init__(self, runner: ModelRunner, cfg: DetectorConfig):
        self._runner = runner
        self._cfg = cfg
        spec = runner.spec
        self._input = spec.input()
        if len(self._input.shape) != 4 or self._input.shape[1] != 3:
            raise ValueError(
                f"{spec.file_name}: expected NCHW (1, 3, H, W) input, got {self._input.shape}"
            )
        self._size = int(self._input.shape[2])
        if self._input.shape[3] != self._size:
            raise ValueError(
                f"{spec.file_name}: expected a square input, got {self._input.shape}"
            )
        self._boxes = spec.output("boxes")
        self._scores = spec.output("scores")
        self._class_idx = spec.output("class_idx")

    @property
    def input_size(self) -> int:
        return self._size

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        canvas, scale, offset = letterbox(
            frame_bgr, self._size, self._cfg.letterbox_pad_value
        )
        tensor = to_nchw_uint8(canvas, rgb=True)
        outputs = self._runner.run({self._input.name: tensor})

        boxes = self._boxes.dequantize(outputs["boxes"])[0]        # (8400, 4) xyxy
        scores = self._scores.dequantize(outputs["scores"])[0]      # (8400,)
        classes = outputs["class_idx"][0].astype(np.int32)          # raw index

        keep_mask = (scores >= self._cfg.score_threshold) & (
            classes == self._cfg.person_class_index
        )
        if not keep_mask.any():
            return []

        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        classes = classes[keep_mask]

        keep = nms_cpu(boxes, scores, self._cfg.nms_iou_threshold)
        keep = keep[: self._cfg.max_detections]

        boxes = undo_letterbox(boxes[keep], scale, offset)
        h, w = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(h))

        out: list[Detection] = []
        for box, score, cls in zip(boxes, scores[keep], classes[keep]):
            x0, y0, x1, y1 = (float(v) for v in box)
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                continue  # degenerate after clipping
            out.append(Detection((x0, y0, x1, y1), float(score), int(cls)))
        return out

    def close(self) -> None:
        self._runner.close()
