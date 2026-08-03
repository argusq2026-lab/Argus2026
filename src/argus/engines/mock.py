"""Mock backend — synthetic tensors at the real contract.

The mock does **not** short-circuit the pipeline. It produces quantized
tensors in exactly the shapes, dtypes, and quantization ranges the real
artifacts declare, so the real letterbox, the real YOLO-X box decode, the real
BlazePose SSD anchor decode, and the real 25-landmark -> COCO-17 remap all run
against it. A mock that returned finished `Detection` objects would exercise
none of that, which is how the prototype's contracts stayed wrong.

It needs no `models/` tree, so a fresh clone can run and be tested immediately.

Determinism: each runner advances a call counter, so a given sequence of calls
always produces the same tensors. Two runs over the same video are
byte-identical; two calls on the same frame are not, by design — the counter is
what animates the synthetic scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from argus.engines.base import EngineBackend, ModelRunner
from argus.engines.metadata import ModelSpec
from argus.synthetic import synthetic_people
from argus.vision.blazepose import COORDS_PER_ANCHOR, DETECTOR_HEADS

#: A camera-facing person, in BlazePose's 25-landmark order, normalised to the
#: ROI. Coordinates are anatomically correct: MediaPipe labels joints from the
#: *subject's* perspective, so a person facing the camera has their left
#: shoulder on the image's right (larger x). Getting this backwards inverts the
#: shoulder-line angle, which is exactly what `off_task_reference_angle_deg`
#: has to agree with.
_UPRIGHT_LANDMARKS_XY = np.array(
    [
        (0.500, 0.120),  # 0  nose
        (0.520, 0.100),  # 1  left_eye_inner
        (0.545, 0.100),  # 2  left_eye
        (0.570, 0.100),  # 3  left_eye_outer
        (0.480, 0.100),  # 4  right_eye_inner
        (0.455, 0.100),  # 5  right_eye
        (0.430, 0.100),  # 6  right_eye_outer
        (0.590, 0.120),  # 7  left_ear
        (0.410, 0.120),  # 8  right_ear
        (0.530, 0.170),  # 9  mouth_left
        (0.470, 0.170),  # 10 mouth_right
        (0.620, 0.300),  # 11 left_shoulder
        (0.380, 0.300),  # 12 right_shoulder
        (0.680, 0.450),  # 13 left_elbow
        (0.320, 0.450),  # 14 right_elbow
        (0.700, 0.580),  # 15 left_wrist
        (0.300, 0.580),  # 16 right_wrist
        (0.710, 0.620),  # 17 left_pinky
        (0.290, 0.620),  # 18 right_pinky
        (0.720, 0.620),  # 19 left_index
        (0.280, 0.620),  # 20 right_index
        (0.700, 0.610),  # 21 left_thumb
        (0.300, 0.610),  # 22 right_thumb
        (0.570, 0.620),  # 23 left_hip
        (0.430, 0.620),  # 24 right_hip
    ],
    dtype=np.float32,
)


class MockRunner:
    """Emits contract-shaped synthetic tensors for one artifact."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._calls = 0

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        n = self._calls
        self._calls += 1
        model = self.spec.model_id
        if model == "yolox":
            return self._yolox(n)
        if self.spec.file_name == "pose_detector.bin":
            return self._pose_detector()
        if self.spec.file_name == "pose_landmark_detector.bin":
            return self._pose_landmark()
        if model == "quicksrnetmedium":
            return self._super_res(inputs)
        return self._zeros()

    # -- per-model synthesis ------------------------------------------------

    def _yolox(self, frame_index: int) -> dict[str, np.ndarray]:
        boxes_spec = self.spec.output("boxes")
        scores_spec = self.spec.output("scores")
        class_spec = self.spec.output("class_idx")
        n_anchors = boxes_spec.shape[1]
        size = 640.0  # canvas edge the boxes are expressed in

        boxes = np.zeros((n_anchors, 4), dtype=np.float32)
        scores = np.zeros((n_anchors,), dtype=np.float32)
        classes = np.zeros((n_anchors,), dtype=np.uint8)

        for i, person in enumerate(synthetic_people(frame_index)):
            boxes[i] = tuple(v * size for v in person.xyxy_norm)
            scores[i] = person.score
            classes[i] = 0  # person

        return {
            "boxes": boxes_spec.quantize(boxes)[np.newaxis, ...],
            "scores": scores_spec.quantize(scores)[np.newaxis, ...],
            "class_idx": classes.astype(class_spec.np_dtype)[np.newaxis, ...],
        }

    def _pose_detector(self) -> dict[str, np.ndarray]:
        """One high-confidence anchor whose decode covers most of the crop."""
        out: dict[str, np.ndarray] = {}
        input_size = float(self.spec.input().shape[1])

        for head_idx, head in enumerate(DETECTOR_HEADS):
            score_spec = self.spec.output(f"box_scores_{head_idx + 1}")
            coord_spec = self.spec.output(f"box_coords_{head_idx + 1}")
            count = score_spec.shape[1]

            # A logit of -60 sigmoids to ~0; the winning anchor gets 0.0, which
            # is the largest value this output's quantization can represent.
            logits = np.full((count, 1), -60.0, dtype=np.float32)
            coords = np.zeros((count, COORDS_PER_ANCHOR), dtype=np.float32)

            if head_idx == 0:
                cell = head.grid // 2
                anchor = (cell * head.grid + cell) * head.anchors_per_cell
                anchor_cx = (cell + 0.5) / head.grid
                anchor_cy = (cell + 0.5) / head.grid
                logits[anchor, 0] = 0.0

                def enc(x: float, y: float) -> tuple[float, float]:
                    return (
                        (x - anchor_cx) * input_size,
                        (y - anchor_cy) * input_size,
                    )

                dx, dy = enc(0.5, 0.5)
                coords[anchor, 0] = dx
                coords[anchor, 1] = dy
                coords[anchor, 2] = 0.9 * input_size  # w
                coords[anchor, 3] = 0.9 * input_size  # h
                # Alignment keypoints: mid-hip centre, then the full-body scale
                # point above it. Their separation sets the landmark ROI size.
                for kp_idx, (kx, ky) in enumerate(
                    [(0.5, 0.6), (0.5, 0.2), (0.5, 0.3), (0.5, 0.6)]
                ):
                    ex, ey = enc(kx, ky)
                    coords[anchor, 4 + kp_idx * 2] = ex
                    coords[anchor, 5 + kp_idx * 2] = ey

            out[score_spec.name] = score_spec.quantize(logits)[np.newaxis, ...]
            out[coord_spec.name] = coord_spec.quantize(coords)[np.newaxis, ...]
        return out

    def _pose_landmark(self) -> dict[str, np.ndarray]:
        score_spec = self.spec.output("scores")
        lm_spec = self.spec.output("landmarks")
        n_lm = lm_spec.shape[1]

        lm = np.zeros((n_lm, 4), dtype=np.float32)
        lm[:, :2] = _UPRIGHT_LANDMARKS_XY[:n_lm]
        lm[:, 2] = 0.0  # z, unused
        lm[:, 3] = 0.9  # visibility
        return {
            "scores": score_spec.quantize(np.array([0.95], dtype=np.float32)),
            "landmarks": lm_spec.quantize(lm)[np.newaxis, ...],
        }

    def _super_res(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out_spec = self.spec.outputs[0]
        _, _, out_h, out_w = out_spec.shape
        source = next(iter(inputs.values()))
        _, _, in_h, in_w = source.shape
        fy, fx = out_h // in_h, out_w // in_w
        upscaled = np.repeat(np.repeat(source, fy, axis=2), fx, axis=3)
        return {out_spec.name: upscaled.astype(out_spec.np_dtype)}

    def _zeros(self) -> dict[str, np.ndarray]:
        return {
            s.name: np.zeros(s.shape, dtype=s.np_dtype) for s in self.spec.outputs
        }

    def close(self) -> None:
        pass


class MockBackend(EngineBackend):
    """Loads nothing from disk — `path` is accepted and ignored."""

    kind = "mock"

    def load(self, path: Path, spec: ModelSpec) -> ModelRunner:
        return MockRunner(spec)
