"""Pose estimation — the two-stage MediaPipe BlazePose chain.

``pose_detector.bin`` (128x128 NHWC) locates a person and regresses four
alignment keypoints; ``pose_landmark_detector.bin`` (**256x256** NHWC — the two
binaries do not share an input size) turns the aligned ROI into 25 landmarks.
Those 25 BlazePose landmarks are remapped to COCO-17 before they reach the
scorer; see :mod:`argus.vision.keypoints` for why that remap is load-bearing.

Two deliberate simplifications of MediaPipe's reference graph, both stated
rather than hidden:

* **No ROI rotation.** MediaPipe rotates the landmark ROI to the hip->shoulder
  axis. Argus takes an axis-aligned square around the same centre and scale.
  A trainee lying down is therefore fed to the landmark stage upright-boxed,
  which is the exact case fall detection cares about — accuracy here is
  unvalidated until real footage exists (docs/VALIDATION.md).
* **No temporal ROI tracking.** MediaPipe reuses the previous frame's
  landmarks to skip the detector. Argus re-runs both stages every sampled
  frame, trading 421 us of NPU time for statelessness, which keeps the
  pipeline reproducible frame-for-frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argus.config import PoseConfig, SuperResConfig
from argus.engines.base import ModelRunner
from argus.vision.blazepose import (
    DETECTOR_HEADS,
    NUM_ALIGNMENT_KEYPOINTS,
    PoseDetection,
    decode_detector_head,
    decode_landmarks,
    ssd_anchor_centres,
)
from argus.vision.keypoints import blazepose_to_coco, empty_coco_keypoints
from argus.vision.nms import nms_cpu
from argus.vision.preprocess import (
    crop_padded,
    resize_exact,
    square_roi,
    to_nhwc_uint8,
)
from argus.vision.superres import SuperResolver

#: Alignment keypoint indices used to place the landmark ROI: the mid-hip
#: centre and the full-body scale point, per MediaPipe's pose_detection graph.
ALIGN_CENTRE = 0
ALIGN_SCALE = 1


@dataclass(frozen=True)
class PoseResult:
    """COCO-17 keypoints in **frame** pixel coordinates, plus provenance."""

    keypoints_xy: np.ndarray  # (17, 2) float32
    keypoints_conf: np.ndarray  # (17,) float32
    presence: float
    roi_xyxy: tuple[float, float, float, float]
    #: "blazepose_detector" when stage 1 placed the ROI, "person_box" when it
    #: found nothing and the YOLO-X box was used instead. Surfaced so a
    #: degraded pose path is observable rather than silent.
    roi_source: str
    super_res_applied: bool = False


class PoseEstimator:
    """Two-stage detector -> landmark chain with optional super-resolution."""

    def __init__(
        self,
        detector_runner: ModelRunner,
        landmark_runner: ModelRunner,
        cfg: PoseConfig,
        super_resolver: SuperResolver | None = None,
        super_res_cfg: SuperResConfig | None = None,
    ):
        self._det = detector_runner
        self._lm = landmark_runner
        self._cfg = cfg
        self._sr = super_resolver
        self._sr_cfg = super_res_cfg

        det_in = detector_runner.spec.input()
        if len(det_in.shape) != 4 or det_in.shape[3] != 3:
            raise ValueError(
                f"{detector_runner.spec.file_name}: expected NHWC (1, H, W, 3), "
                f"got {det_in.shape}"
            )
        self._det_input = det_in
        self._det_size = int(det_in.shape[1])

        lm_in = landmark_runner.spec.input()
        if len(lm_in.shape) != 4 or lm_in.shape[3] != 3:
            raise ValueError(
                f"{landmark_runner.spec.file_name}: expected NHWC (1, H, W, 3), "
                f"got {lm_in.shape}"
            )
        self._lm_input = lm_in
        self._lm_size = int(lm_in.shape[1])

        self._score_specs = tuple(
            detector_runner.spec.output(f"box_scores_{i + 1}")
            for i in range(len(DETECTOR_HEADS))
        )
        self._coord_specs = tuple(
            detector_runner.spec.output(f"box_coords_{i + 1}")
            for i in range(len(DETECTOR_HEADS))
        )
        self._lm_score_spec = landmark_runner.spec.output("scores")
        self._lm_spec = landmark_runner.spec.output("landmarks")

    # -- stage 1 ------------------------------------------------------------

    def detect_pose_roi(self, crop_bgr: np.ndarray) -> PoseDetection | None:
        """Locate a person inside `crop_bgr`, in coordinates normalised to it."""
        resized = resize_exact(crop_bgr, (self._det_size, self._det_size))
        outputs = self._det.run({self._det_input.name: to_nhwc_uint8(resized, rgb=True)})

        centres = ssd_anchor_centres(DETECTOR_HEADS)
        all_boxes, all_scores, all_align = [], [], []
        for i, (score_spec, coord_spec) in enumerate(
            zip(self._score_specs, self._coord_specs)
        ):
            boxes, scores, align = decode_detector_head(
                score_spec.dequantize(outputs[score_spec.name])[0],
                coord_spec.dequantize(outputs[coord_spec.name])[0],
                centres[i],
                self._det_size,
                scores_are_logits=self._cfg.detector_scores_are_logits,
            )
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_align.append(align)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        align = np.concatenate(all_align, axis=0)

        mask = scores >= self._cfg.detector_score_threshold
        if not mask.any():
            return None
        boxes, scores, align = boxes[mask], scores[mask], align[mask]

        keep = nms_cpu(boxes, scores, self._cfg.detector_nms_iou_threshold)
        if not keep:
            return None
        best = keep[0]
        return PoseDetection(
            bbox_xyxy=tuple(float(v) for v in boxes[best]),
            score=float(scores[best]),
            alignment_xy=align[best],
        )

    # -- ROI placement ------------------------------------------------------

    def _landmark_roi(
        self,
        person_bbox: tuple[float, float, float, float],
        stage1_roi: tuple[float, float, float, float],
        detection: PoseDetection | None,
    ) -> tuple[tuple[float, float, float, float], str]:
        """Where to crop for the landmark stage, and what decided it."""
        if detection is None:
            return square_roi(person_bbox, self._cfg.roi_scale), "person_box"

        rx0, ry0, rx1, ry1 = stage1_roi
        rw, rh = rx1 - rx0, ry1 - ry0

        align = detection.alignment_xy
        if align.shape[0] < NUM_ALIGNMENT_KEYPOINTS:
            return square_roi(person_bbox, self._cfg.roi_scale), "person_box"

        cx = rx0 + float(align[ALIGN_CENTRE, 0]) * rw
        cy = ry0 + float(align[ALIGN_CENTRE, 1]) * rh
        sx = rx0 + float(align[ALIGN_SCALE, 0]) * rw
        sy = ry0 + float(align[ALIGN_SCALE, 1]) * rh
        half = float(np.hypot(sx - cx, sy - cy)) * self._cfg.roi_scale

        if not np.isfinite(half) or half < 1.0:
            # Degenerate alignment (collapsed keypoints). Fall back to the
            # detector's own box rather than emitting a 1-pixel ROI.
            bx0, by0, bx1, by1 = detection.bbox_xyxy
            box = (rx0 + bx0 * rw, ry0 + by0 * rh, rx0 + bx1 * rw, ry0 + by1 * rh)
            return square_roi(box, self._cfg.roi_scale), "blazepose_detector"

        return (cx - half, cy - half, cx + half, cy + half), "blazepose_detector"

    # -- full chain ---------------------------------------------------------

    def estimate(
        self,
        frame_bgr: np.ndarray,
        person_bbox: tuple[float, float, float, float],
        frame_area: float,
    ) -> PoseResult:
        """Run both stages for one person box; returns COCO-17 in frame pixels."""
        stage1_roi = square_roi(person_bbox, self._cfg.roi_scale)
        stage1_crop, stage1_roi = crop_padded(frame_bgr, stage1_roi)
        if stage1_crop.size == 0:
            xy, conf = empty_coco_keypoints()
            return PoseResult(xy, conf, 0.0, stage1_roi, "empty_crop")

        detection = self.detect_pose_roi(stage1_crop)
        roi, roi_source = self._landmark_roi(person_bbox, stage1_roi, detection)
        lm_crop, roi = crop_padded(frame_bgr, roi)
        if lm_crop.size == 0:
            xy, conf = empty_coco_keypoints()
            return PoseResult(xy, conf, 0.0, roi, "empty_crop")

        # Super-resolution feeds the landmark stage, not the detector: stage 1
        # downsamples to 128x128 anyway, so a 4x upscale there is wasted NPU
        # time, while stage 2's 25-landmark regression is what actually
        # benefits from a distant trainee's crop having real detail.
        sr_applied = False
        if self._sr is not None and self._sr.should_upscale(lm_crop, frame_area):
            lm_crop = self._sr.upscale(lm_crop)
            sr_applied = True

        resized = resize_exact(lm_crop, (self._lm_size, self._lm_size))
        outputs = self._lm.run({self._lm_input.name: to_nhwc_uint8(resized, rgb=True)})

        presence = float(
            np.asarray(self._lm_score_spec.dequantize(outputs["scores"])).reshape(-1)[0]
        )
        if presence < self._cfg.landmark_presence_threshold:
            xy, conf = empty_coco_keypoints()
            return PoseResult(xy, conf, presence, roi, roi_source, sr_applied)

        lm_xy_norm, visibility = decode_landmarks(
            self._lm_spec.dequantize(outputs["landmarks"])[0],
            visibility_is_logit=self._cfg.landmark_visibility_is_logit,
        )

        rx0, ry0, rx1, ry1 = roi
        frame_xy = np.empty_like(lm_xy_norm)
        frame_xy[:, 0] = rx0 + lm_xy_norm[:, 0] * (rx1 - rx0)
        frame_xy[:, 1] = ry0 + lm_xy_norm[:, 1] * (ry1 - ry0)

        xy, conf = blazepose_to_coco(frame_xy, visibility)
        return PoseResult(xy, conf, presence, roi, roi_source, sr_applied)

    def close(self) -> None:
        self._det.close()
        self._lm.close()
        if self._sr is not None:
            self._sr.close()
