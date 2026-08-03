"""Real-artifact tests — the gate the brief asks merges to pass.

Each shipped artifact is checked against its own `metadata.json`, and against
the built-in reference contracts the mock and the pre/post code are written to.
Two independent levels:

* **Always** (no artifacts needed): the built-in contracts are internally
  consistent — the anchor layout matches the declared output shapes, the pose
  binaries' input sizes really do differ, and so on.
* **When `models/` is provisioned**: metadata.json is compared field-for-field
  with the reference, ONNX artifacts are actually loaded and run, and their
  session I/O is asserted against the manifest. `.bin` artifacts additionally
  run for real under `-m npu` on hardware.

`models/` is gitignored, so the artifact-dependent tests skip cleanly on a
fresh clone and in CI, and fail loudly the moment a re-export changes a shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argus.engines import reference_specs
from argus.engines.metadata import load_model_specs

ONNX_ROLES = ("detector", "super_res")
BIN_ROLES = ("pose_detector", "pose_landmark")


def _artifact(default_config, role: str) -> Path:
    return Path(default_config.models.path(role))


def _require(path: Path, what: str):
    if not path.is_file():
        pytest.skip(f"{what} not provisioned ({path}); run `argus bootstrap`")


# ---------------------------------------------------------------------------
# Contract self-consistency — runs everywhere, needs no artifacts
# ---------------------------------------------------------------------------


def test_detector_grid_matches_a_640_input():
    """8400 == 80^2 + 40^2 + 20^2, the anchor-free grid at strides 8/16/32."""
    spec = reference_specs.YOLOX
    size = spec.input().shape[2]
    expected = sum((size // stride) ** 2 for stride in (8, 16, 32))
    assert spec.output("boxes").shape == (1, expected, 4)
    assert spec.output("scores").shape == (1, expected)
    assert spec.output("class_idx").shape == (1, expected)


def test_detector_has_three_outputs_not_one_fused_tensor():
    """The prototype decoded a (1, N, 6) tensor that does not exist."""
    assert [s.name for s in reference_specs.YOLOX.outputs] == ["boxes", "scores", "class_idx"]


def test_detector_input_is_nchw_uint8():
    spec = reference_specs.YOLOX.input()
    assert spec.shape == (1, 3, 640, 640)
    assert spec.dtype == "uint8"
    assert spec.scale == pytest.approx(1 / 255, rel=1e-6)
    assert spec.zero_point == 0


def test_pose_stages_have_different_input_sizes():
    """A real contract detail the brief itself had wrong: only the *detector*
    is 128x128; the landmark stage is 256x256."""
    assert reference_specs.POSE_DETECTOR.input().shape == (1, 128, 128, 3)
    assert reference_specs.POSE_LANDMARK.input().shape == (1, 256, 256, 3)


def test_pose_binaries_are_nhwc_unlike_the_detector():
    for spec in (reference_specs.POSE_DETECTOR, reference_specs.POSE_LANDMARK):
        assert spec.input().shape[3] == 3, "NHWC"
    assert reference_specs.YOLOX.input().shape[1] == 3, "NCHW"


def test_pose_anchor_heads_sum_to_the_declared_scores():
    from argus.vision.blazepose import DETECTOR_HEADS

    declared = [
        reference_specs.POSE_DETECTOR.output("box_scores_1").shape[1],
        reference_specs.POSE_DETECTOR.output("box_scores_2").shape[1],
    ]
    assert [head.count for head in DETECTOR_HEADS] == declared


def test_pose_landmark_output_is_25_points_not_a_heatmap():
    """There are no heatmaps in this chain, so there is nothing to ArgMax."""
    spec = reference_specs.POSE_LANDMARK.output("landmarks")
    assert spec.shape == (1, 25, 4)
    from argus.vision.keypoints import BLAZEPOSE_NUM_LANDMARKS

    assert spec.shape[1] == BLAZEPOSE_NUM_LANDMARKS


def test_pose_detector_confidence_range_is_degenerate():
    """Documents the placeholder-calibration artefact rather than hiding it:
    the top quantization step is logit 0.0 (score 0.5) and the next is
    -5.55 (score 0.0039), so confidence is effectively two-valued."""
    from argus.vision.blazepose import sigmoid

    spec = reference_specs.POSE_DETECTOR.output("box_scores_1")
    best = float(spec.dequantize(np.array([255], dtype=np.uint8))[0])
    next_down = float(spec.dequantize(np.array([254], dtype=np.uint8))[0])
    assert sigmoid(np.array([best]))[0] == pytest.approx(0.5)
    assert sigmoid(np.array([next_down]))[0] < 0.01


def test_super_res_is_a_fixed_four_times_upscale():
    spec = reference_specs.QUICKSRNET
    _, _, in_h, in_w = spec.input().shape
    _, _, out_h, out_w = spec.output("upscaled_image").shape
    assert (in_h, in_w) == (128, 128)
    assert (out_h, out_w) == (512, 512)
    assert out_h // in_h == 4


# ---------------------------------------------------------------------------
# metadata.json agreement — needs models/
# ---------------------------------------------------------------------------


@pytest.mark.artifacts
@pytest.mark.parametrize("role", sorted(reference_specs.BY_ROLE))
def test_metadata_matches_the_reference_contract(default_config, role):
    reference = reference_specs.BY_ROLE[role]
    metadata_key = reference_specs.METADATA_KEY_BY_ROLE[role]
    path = Path(default_config.models.path(metadata_key))
    _require(path, f"metadata.json for {role}")

    actual = load_model_specs(path)[reference.file_name]
    assert actual.inputs == reference.inputs, f"{role}: input contract drifted"
    assert actual.outputs == reference.outputs, f"{role}: output contract drifted"
    assert actual.runtime == reference.runtime
    assert actual.precision == reference.precision


# ---------------------------------------------------------------------------
# Real execution — ONNX artifacts run on any host
# ---------------------------------------------------------------------------


@pytest.mark.artifacts
@pytest.mark.parametrize("role", ONNX_ROLES)
def test_onnx_artifact_runs_and_matches_its_declared_shapes(default_config, role):
    """Load the real file, feed it a real tensor, assert what comes back."""
    from argus.engines.base import ValidatingRunner
    from argus.engines.onnx_cpu import OnnxCpuBackend

    path = _artifact(default_config, role)
    _require(path, f"{role} artifact")

    spec = reference_specs.BY_ROLE[role]
    runner = ValidatingRunner(OnnxCpuBackend().load(path, spec))
    try:
        feed = {
            s.name: np.zeros(s.shape, dtype=s.np_dtype) for s in spec.inputs
        }
        outputs = runner.run(feed)  # ValidatingRunner asserts the contract
        for out_spec in spec.outputs:
            got = outputs[out_spec.name]
            assert got.shape == out_spec.shape, (
                f"{role}: {out_spec.name} shape {got.shape} != "
                f"metadata.json {out_spec.shape}"
            )
            assert got.dtype == out_spec.np_dtype
    finally:
        runner.close()


@pytest.mark.artifacts
def test_detector_end_to_end_on_the_real_artifact(default_config):
    """The full pre/post path against the real graph, not just its shapes."""
    from argus.engines.base import ValidatingRunner
    from argus.engines.onnx_cpu import OnnxCpuBackend
    from argus.vision.detect import PersonDetector

    path = _artifact(default_config, "detector")
    _require(path, "detector artifact")

    detector = PersonDetector(
        ValidatingRunner(OnnxCpuBackend().load(path, reference_specs.YOLOX)),
        default_config.detector,
    )
    try:
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detections = detector.detect(frame)  # must not raise
        assert isinstance(detections, list)
        for det in detections:
            x0, y0, x1, y1 = det.bbox_xyxy
            assert 0.0 <= x0 < x1 <= 640.0
            assert 0.0 <= y0 < y1 <= 480.0
            assert 0.0 <= det.score <= 1.0
            assert det.class_index == default_config.detector.person_class_index
    finally:
        detector.close()


@pytest.mark.artifacts
def test_super_res_artifact_is_structurally_valid(default_config):
    """The QuickSRNet artifact is valid, despite ORT's message saying otherwise.

    ORT fails to load it at `extended` optimization with "two nodes with same
    node name (/model/cnn/0/Conv)". This pins that the *artifact* is not the
    problem — the checker passes and every node name is unique — so the
    collision is created by ORT's QDQ fusion. If this test ever fails, the
    export really did regress and re-exporting is the right remedy.
    """
    import onnx

    from argus.provision import duplicate_node_names

    path = _artifact(default_config, "super_res")
    _require(path, "super-res artifact")

    assert duplicate_node_names(path) == []
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)


@pytest.mark.artifacts
def test_super_res_loads_despite_the_ort_fusion_collision(default_config, capsys):
    """The backend must produce a working session anyway, and say what it did."""
    from argus.engines.onnx_cpu import OnnxCpuBackend

    path = _artifact(default_config, "super_res")
    _require(path, "super-res artifact")

    runner = OnnxCpuBackend("extended").load(path, reference_specs.QUICKSRNET)
    try:
        outputs = runner.run(
            {"image": np.zeros((1, 3, 128, 128), dtype=np.uint8)}
        )
        assert outputs["upscaled_image"].shape == (1, 3, 512, 512)
    finally:
        runner.close()
    assert "QDQ fusion" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Real execution — context binaries need the NPU
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.artifacts
@pytest.mark.parametrize("role", BIN_ROLES)
def test_context_binary_runs_on_the_npu(default_config, role):
    import dataclasses

    from argus.engines.base import ValidatingRunner
    from argus.engines.qnn_npu import QnnNpuBackend

    path = _artifact(default_config, role)
    _require(path, f"{role} artifact")

    spec = reference_specs.BY_ROLE[role]
    cfg = dataclasses.replace(default_config.engine, kind="qnn-npu")
    runner = ValidatingRunner(QnnNpuBackend(cfg).load(path, spec))
    try:
        feed = {s.name: np.zeros(s.shape, dtype=s.np_dtype) for s in spec.inputs}
        outputs = runner.run(feed)
        for out_spec in spec.outputs:
            assert outputs[out_spec.name].shape == out_spec.shape
            assert outputs[out_spec.name].dtype == out_spec.np_dtype
    finally:
        runner.close()
