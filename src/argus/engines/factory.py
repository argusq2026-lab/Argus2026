"""Builds the vision stack from config. The one place a backend is chosen.

Backend selection is explicit and total: `engine.kind` picks a backend, that
backend loads every role, and a failure in any role aborts construction. There
is no arrangement in which Argus starts with pose missing, or with one stage on
the NPU and another silently on the CPU, because a partially-loaded pipeline
still produces scores — just meaningless ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from argus.config import ArgusConfig, EngineConfig
from argus.engines import reference_specs
from argus.engines.base import EngineBackend, EngineError, ModelRunner, ValidatingRunner
from argus.engines.metadata import ModelSpec, load_model_specs
from argus.engines.mock import MockBackend
from argus.engines.onnx_cpu import OnnxCpuBackend
from argus.vision.detect import PersonDetector
from argus.vision.pose import PoseEstimator
from argus.vision.superres import SuperResolver

#: Role -> the [models] key holding its artifact path.
ROLE_PATH_KEY = {
    "detector": "detector",
    "pose_detector": "pose_detector",
    "pose_landmark": "pose_landmark",
    "super_res": "super_res",
}


@dataclass
class VisionStack:
    """Every loaded stage, plus which backend produced them."""

    detector: PersonDetector
    pose: PoseEstimator
    super_res: SuperResolver | None
    backend_kind: str

    def close(self) -> None:
        self.detector.close()
        self.pose.close()  # also closes super_res, which pose owns


def build_backend(cfg: EngineConfig) -> EngineBackend:
    if cfg.kind == "mock":
        return MockBackend()
    if cfg.kind == "onnx-cpu":
        return OnnxCpuBackend(cfg.graph_optimization_level)
    if cfg.kind == "qnn-npu":
        # Imported lazily: constructing it probes for onnxruntime-qnn and the
        # QAIRT SDK, which must not happen on a mock-mode import.
        from argus.engines.qnn_npu import QnnNpuBackend

        return QnnNpuBackend(cfg)
    raise EngineError(f"unknown engine.kind: {cfg.kind!r}")


def resolve_spec(role: str, cfg: ArgusConfig) -> ModelSpec:
    """The tensor contract for a role.

    Real backends read the artifact's own ``metadata.json`` — the artifact is
    authoritative about itself. Mock mode falls back to the built-in reference
    contract so a fresh clone with no ``models/`` still runs; those two are
    asserted equal by ``tests/test_artifacts.py`` whenever both exist.
    """
    if role not in reference_specs.BY_ROLE:
        raise EngineError(f"unknown model role: {role!r}")
    reference = reference_specs.BY_ROLE[role]

    metadata_key = reference_specs.METADATA_KEY_BY_ROLE[role]
    try:
        metadata_path = cfg.models.path(metadata_key)
    except Exception:
        metadata_path = None

    if metadata_path is None or not Path(metadata_path).is_file():
        if cfg.engine.kind == "mock":
            return reference
        raise EngineError(
            f"metadata.json for role {role!r} not found at {metadata_path}. "
            "Run `argus bootstrap` to provision models/, or set "
            "engine.kind = 'mock' to develop without artifacts."
        )

    specs = load_model_specs(metadata_path)
    if reference.file_name not in specs:
        raise EngineError(
            f"{metadata_path} describes {sorted(specs)} but role {role!r} needs "
            f"{reference.file_name!r}"
        )
    return specs[reference.file_name]


def load_runner(
    backend: EngineBackend, role: str, cfg: ArgusConfig
) -> ModelRunner:
    """Load one role's artifact, contract-checked on every call thereafter."""
    spec = resolve_spec(role, cfg)
    path = Path(cfg.models.path(ROLE_PATH_KEY[role]))
    return ValidatingRunner(backend.load(path, spec))


def build_vision_stack(cfg: ArgusConfig) -> VisionStack:
    """Load the detector, both pose stages, and (optionally) super-resolution."""
    backend = build_backend(cfg.engine)

    detector = PersonDetector(load_runner(backend, "detector", cfg), cfg.detector)

    super_res: SuperResolver | None = None
    if cfg.super_res.enabled:
        super_res = SuperResolver(
            load_runner(backend, "super_res", cfg), cfg.super_res
        )

    pose = PoseEstimator(
        load_runner(backend, "pose_detector", cfg),
        load_runner(backend, "pose_landmark", cfg),
        cfg.pose,
        super_resolver=super_res,
        super_res_cfg=cfg.super_res,
    )

    return VisionStack(
        detector=detector,
        pose=pose,
        super_res=super_res,
        backend_kind=cfg.engine.kind,
    )
