"""The engine abstraction and its contract enforcement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argus.config import EngineConfig
from argus.engines import reference_specs
from argus.engines.base import ContractViolation, ValidatingRunner, check_tensor
from argus.engines.factory import build_backend, build_vision_stack, resolve_spec
from argus.engines.metadata import MetadataError, TensorSpec, load_model_specs
from argus.engines.mock import MockBackend
from argus.engines.onnx_cpu import OnnxCpuBackend
from argus.engines.qnn_context import (
    build_epcontext_model,
    check_qairt_compatibility,
    find_qairt_sdk,
)


# -- metadata ---------------------------------------------------------------


def test_dequantize_matches_the_declared_parameters():
    spec = TensorSpec("boxes", (1, 4), "uint8", scale=4.0, zero_point=51)
    q = np.array([[51, 52, 0, 255]], dtype=np.uint8)
    np.testing.assert_allclose(spec.dequantize(q), [[0.0, 4.0, -204.0, 816.0]])


def test_quantize_is_the_inverse_and_clips_to_the_dtype():
    spec = TensorSpec("x", (3,), "uint8", scale=0.5, zero_point=100)
    values = np.array([0.0, 10.0, 1e6], dtype=np.float32)
    q = spec.quantize(values)
    assert q.dtype == np.uint8
    assert q.tolist() == [100, 120, 255]


def test_unquantized_tensor_passes_through():
    spec = TensorSpec("class_idx", (3,), "uint8")
    assert not spec.is_quantized
    np.testing.assert_array_equal(spec.dequantize(np.array([0, 1, 2], np.uint8)), [0.0, 1.0, 2.0])


def test_missing_metadata_names_the_remedy(tmp_path):
    with pytest.raises(MetadataError, match="argus bootstrap"):
        load_model_specs(tmp_path / "nope.json")


def test_malformed_metadata_is_rejected(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(MetadataError, match="model_files"):
        load_model_specs(path)


# -- contract checking ------------------------------------------------------


def test_check_tensor_rejects_a_wrong_dtype():
    spec = TensorSpec("image", (1, 3, 8, 8), "uint8")
    with pytest.raises(ContractViolation, match="dtype"):
        check_tensor(spec, np.zeros((1, 3, 8, 8), np.int8), where="test")


def test_check_tensor_rejects_a_wrong_shape():
    spec = TensorSpec("image", (1, 3, 8, 8), "uint8")
    with pytest.raises(ContractViolation, match="does not match"):
        check_tensor(spec, np.zeros((1, 8, 8, 3), np.uint8), where="test")


def test_check_tensor_rejects_a_wrong_rank():
    spec = TensorSpec("image", (1, 3, 8, 8), "uint8")
    with pytest.raises(ContractViolation, match="rank"):
        check_tensor(spec, np.zeros((3, 8, 8), np.uint8), where="test")


def test_validating_runner_catches_a_bad_input():
    runner = ValidatingRunner(MockBackend().load(Path("x"), reference_specs.YOLOX))
    with pytest.raises(ContractViolation):
        runner.run({"image": np.zeros((1, 640, 640, 3), np.uint8)})  # NHWC, wrong


# -- mock backend -----------------------------------------------------------


@pytest.mark.parametrize("role", sorted(reference_specs.BY_ROLE))
def test_mock_emits_tensors_at_the_real_contract(role):
    """A mock that drifts from the contract is worse than no mock."""
    spec = reference_specs.BY_ROLE[role]
    runner = ValidatingRunner(MockBackend().load(Path("unused"), spec))
    inputs = {s.name: np.zeros(s.shape, dtype=s.np_dtype) for s in spec.inputs}
    outputs = runner.run(inputs)  # ValidatingRunner asserts both directions
    assert set(outputs) == {s.name for s in spec.outputs}
    for out_spec in spec.outputs:
        assert outputs[out_spec.name].shape == out_spec.shape
        assert outputs[out_spec.name].dtype == out_spec.np_dtype


def test_mock_detector_yields_three_people_that_decode():
    from argus.config import DetectorConfig
    from argus.vision.detect import PersonDetector

    detector = PersonDetector(
        ValidatingRunner(MockBackend().load(Path("x"), reference_specs.YOLOX)),
        DetectorConfig(),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert len(detections) == 3
    for det in detections:
        x0, y0, x1, y1 = det.bbox_xyxy
        assert x1 > x0 and y1 > y0
        assert 0.0 <= x0 <= 640 and 0.0 <= y1 <= 480


def test_mock_is_deterministic_for_a_given_call_sequence():
    def sequence():
        runner = MockBackend().load(Path("x"), reference_specs.YOLOX)
        feed = {"image": np.zeros((1, 3, 640, 640), np.uint8)}
        return [runner.run(feed)["boxes"].copy() for _ in range(5)]

    for first, second in zip(sequence(), sequence()):
        np.testing.assert_array_equal(first, second)


def test_mock_pose_chain_produces_upright_coco_keypoints():
    from argus.config import PoseConfig
    from argus.vision.pose import PoseEstimator
    from argus.triage import KP_LEFT_SHOULDER, KP_NOSE, KP_RIGHT_SHOULDER

    backend = MockBackend()
    estimator = PoseEstimator(
        ValidatingRunner(backend.load(Path("x"), reference_specs.POSE_DETECTOR)),
        ValidatingRunner(backend.load(Path("y"), reference_specs.POSE_LANDMARK)),
        PoseConfig(),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = estimator.estimate(frame, (200.0, 100.0, 280.0, 340.0), 480 * 640)

    assert result.roi_source == "blazepose_detector"
    assert result.keypoints_xy.shape == (17, 2)
    assert result.keypoints_conf[KP_NOSE] > 0.3
    # Camera-facing: the subject's left shoulder sits at the larger image x.
    assert result.keypoints_xy[KP_LEFT_SHOULDER][0] > result.keypoints_xy[KP_RIGHT_SHOULDER][0]
    # Knees and ankles have no source landmark in the 25-point export.
    for idx in (13, 14, 15, 16):
        assert result.keypoints_conf[idx] == 0.0


def test_mock_super_res_upscales_four_times():
    from argus.config import SuperResConfig
    from argus.vision.superres import SuperResolver

    resolver = SuperResolver(
        ValidatingRunner(MockBackend().load(Path("x"), reference_specs.QUICKSRNET)),
        SuperResConfig(),
    )
    assert resolver.input_hw == (128, 128)
    upscaled = resolver.upscale(np.full((30, 20, 3), 128, dtype=np.uint8))
    assert upscaled.shape == (512, 512, 3)


def test_super_res_gate_only_fires_on_small_crops():
    from argus.config import SuperResConfig
    from argus.vision.superres import SuperResolver

    resolver = SuperResolver(
        ValidatingRunner(MockBackend().load(Path("x"), reference_specs.QUICKSRNET)),
        SuperResConfig(min_bbox_area_frac=0.02),
    )
    frame_area = 640 * 480.0
    assert resolver.should_upscale(np.zeros((40, 20, 3), np.uint8), frame_area)
    assert not resolver.should_upscale(np.zeros((400, 200, 3), np.uint8), frame_area)


# -- backend selection ------------------------------------------------------


def test_engine_kind_selects_the_backend():
    assert isinstance(build_backend(EngineConfig(kind="mock")), MockBackend)
    assert isinstance(build_backend(EngineConfig(kind="onnx-cpu")), OnnxCpuBackend)


def test_cpu_backend_refuses_a_context_binary_rather_than_faking_it():
    with pytest.raises(Exception, match="no CPU path"):
        OnnxCpuBackend().load(Path("pose_detector.bin"), reference_specs.POSE_DETECTOR)


def test_cpu_backend_names_bootstrap_when_an_artifact_is_missing(tmp_path):
    with pytest.raises(Exception, match="argus bootstrap"):
        OnnxCpuBackend().load(tmp_path / "absent.onnx", reference_specs.YOLOX)


def test_mock_mode_needs_no_models_tree(default_config):
    """A fresh clone must be runnable before `argus bootstrap`."""
    for role in reference_specs.BY_ROLE:
        assert resolve_spec(role, default_config) is not None
    stack = build_vision_stack(default_config)
    try:
        assert stack.backend_kind == "mock"
    finally:
        stack.close()


def test_real_backend_refuses_to_guess_a_contract(default_config, tmp_path):
    import dataclasses

    cfg = dataclasses.replace(
        default_config,
        engine=dataclasses.replace(default_config.engine, kind="onnx-cpu"),
        models=dataclasses.replace(default_config.models, root=str(tmp_path)),
    )
    with pytest.raises(Exception, match="metadata.json"):
        resolve_spec("detector", cfg)


# -- QNN context binaries ---------------------------------------------------


def test_epcontext_wrapper_declares_the_binary_io(tmp_path):
    import onnx

    binary = tmp_path / "pose_detector.bin"
    binary.write_bytes(b"not-a-real-context-binary")
    out = build_epcontext_model(binary, reference_specs.POSE_DETECTOR, tmp_path / "w.onnx")

    model = onnx.load(str(out))
    assert [n.op_type for n in model.graph.node] == ["EPContext"]
    assert [i.name for i in model.graph.input] == ["image"]
    assert sorted(o.name for o in model.graph.output) == sorted(
        s.name for s in reference_specs.POSE_DETECTOR.outputs
    )
    attrs = {a.name: a for a in model.graph.node[0].attribute}
    assert attrs["ep_cache_context"].s.decode() == binary.name
    assert attrs["embed_mode"].i == 0
    assert attrs["source"].s.decode() == "QNN"


def test_epcontext_wrapper_requires_the_binary(tmp_path):
    with pytest.raises(Exception, match="not found"):
        build_epcontext_model(
            tmp_path / "missing.bin", reference_specs.POSE_DETECTOR, tmp_path / "w.onnx"
        )


def test_qairt_version_skew_is_reported_not_swallowed(tmp_path):
    """The shipped artifacts are 2.45.0; a 2.32.x SDK cannot load them."""
    sdk = tmp_path / "2.32.6.250402"
    sdk.mkdir()
    warning = check_qairt_compatibility(reference_specs.POSE_DETECTOR, sdk)
    assert warning is not None
    assert "2.45.0" in warning and "2.32.6" in warning


def test_matching_qairt_versions_produce_no_warning(tmp_path):
    sdk = tmp_path / "2.45.0.260326154327"
    sdk.mkdir()
    assert check_qairt_compatibility(reference_specs.POSE_DETECTOR, sdk) is None


def test_sdk_discovery_prefers_an_explicit_setting(tmp_path):
    assert find_qairt_sdk(str(tmp_path)) == tmp_path
    assert find_qairt_sdk(str(tmp_path / "absent")) is None
